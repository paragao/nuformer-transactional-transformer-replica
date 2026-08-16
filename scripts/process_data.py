"""Process raw transaction data into training-ready numpy arrays.

Reads parquet files from data generation, tokenizes transactions into
sequences, and saves as numpy memmap files ready for the DataLoader.

Outputs:
    data/processed/train_sequences.npy  - (N_train, max_seq_len) int32
    data/processed/val_sequences.npy    - (N_val, max_seq_len) int32
    data/processed/train_labels.npy     - (N_train,) int8
    data/processed/val_labels.npy       - (N_val,) int8
    data/processed/train_features.npy   - (N_train, 291) float32
    data/processed/val_features.npy     - (N_val, 291) float32

Usage:
    PYTHONPATH=. python scripts/process_data.py --input-dir data/raw/dev --max-seq-len 2048
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
import multiprocessing as mp

import numpy as np
import polars as pl

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.tokenization.special_tokens import SpecialTokens
from src.tokenization.bpe_tokenizer import DescriptionTokenizer
from src.tokenization.sequence_builder import SequenceBuilder


def load_data(input_dir: Path) -> tuple[pl.DataFrame, pl.DataFrame, pl.DataFrame]:
    """Load parquet files."""
    transactions = pl.read_parquet(input_dir / "transactions.parquet")
    labels = pl.read_parquet(input_dir / "labels.parquet")
    features = pl.read_parquet(input_dir / "tabular_features.parquet")
    return transactions, labels, features


def build_tokenizer(transactions: pl.DataFrame, tokenizer_path: str = None) -> DescriptionTokenizer:
    """Load pre-trained or train BPE tokenizer on transaction descriptions."""
    if tokenizer_path and Path(tokenizer_path).exists():
        print(f"  Loading pre-trained tokenizer from {tokenizer_path}")
        return DescriptionTokenizer.from_pretrained(tokenizer_path)

    descriptions = transactions["description"].unique().to_list()
    print(f"  Training BPE on {len(descriptions):,} unique descriptions...")
    tokenizer = DescriptionTokenizer(vocab_size=24000)
    tokenizer.train(descriptions)
    return tokenizer


def _process_user_batch(args):
    """Process a batch of users (for parallel execution).

    Each worker gets: (user_ids_batch, txn_ranges, txn_amounts, txn_timestamps,
                       txn_descriptions, label_map, feature_map, n_features,
                       sequence_builder, max_seq_len)
    """
    (user_ids_batch, txn_ranges, txn_amounts, txn_timestamps,
     txn_descriptions, label_map, feature_map, n_features,
     sequence_builder, max_seq_len) = args

    sequences = []
    labels = []
    features = []
    skipped = 0

    for uid in user_ids_batch:
        if uid not in txn_ranges:
            skipped += 1
            continue

        start, end = txn_ranges[uid]
        n_txns = end - start
        if n_txns < 5:
            skipped += 1
            continue

        # Build transaction list
        txns = []
        for j in range(start, end):
            ts = txn_timestamps[j]
            txns.append({
                "amount": float(txn_amounts[j]),
                "timestamp": datetime.fromisoformat(ts) if isinstance(ts, str) else ts,
                "description": txn_descriptions[j],
            })

        # Build and pad token sequence
        sequence = sequence_builder.build_sequence(txns)
        sequence = sequence_builder.pad_sequence(sequence, max_seq_len)

        sequences.append(sequence)
        labels.append(int(label_map.get(uid, False)))
        features.append(feature_map.get(uid, np.zeros(n_features, dtype=np.float32)))

    return sequences, labels, features, skipped


def process_users(
    transactions: pl.DataFrame,
    labels: pl.DataFrame,
    features: pl.DataFrame,
    sequence_builder: SequenceBuilder,
    max_seq_len: int,
    train_end_date: str = "2023-12-31",
    num_workers: int = 0,
) -> dict:
    """Process all users into train/val splits.

    Optimized: sorts once by (user_id, timestamp) then iterates groups
    with zero per-user DataFrame operations. Supports multiprocessing.
    """
    # Get user list from labels
    user_ids = labels["user_id"].to_list()
    label_map = dict(zip(
        labels["user_id"].to_list(),
        labels["activated_credit_card"].to_list(),
    ))

    # Get features as numpy (indexed by user_id)
    feature_cols = [c for c in features.columns if c != "user_id"]
    feature_user_ids = features["user_id"].to_list()
    feature_map = {}
    feature_array = features.select(feature_cols).to_numpy().astype(np.float32)
    for i, uid in enumerate(feature_user_ids):
        feature_map[uid] = feature_array[i]

    # Sort transactions once by user_id + timestamp (fast in polars)
    print("  Sorting transactions by user_id + timestamp...")
    transactions = transactions.sort(["user_id", "timestamp"])

    # Pre-compute date components vectorized in polars (MUCH faster than per-row datetime.fromisoformat)
    print("  Pre-computing date components (vectorized)...")
    ts_col = transactions["timestamp"]
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()
    transactions = transactions.with_columns([
        ts_col.dt.month().alias("_month"),
        ts_col.dt.day().alias("_day"),
        ts_col.dt.weekday().alias("_weekday"),  # 1=Monday ... 7=Sunday in polars
    ])

    # Extract columns as numpy/lists for fast iteration
    print("  Extracting columns for fast iteration...")
    txn_user_ids = transactions["user_id"].to_numpy()
    txn_amounts = transactions["amount"].to_numpy()
    txn_descriptions = transactions["description"].to_list()
    txn_months = transactions["_month"].to_numpy()
    txn_days = transactions["_day"].to_numpy()
    txn_weekdays = transactions["_weekday"].to_numpy()  # polars: 1=Mon, need 0=Mon for Python

    # Find group boundaries using numpy (much faster than polars group_by + iter)
    print("  Finding user group boundaries...")
    change_mask = np.concatenate([[True], txn_user_ids[1:] != txn_user_ids[:-1]])
    group_starts = np.where(change_mask)[0]
    group_ends = np.concatenate([group_starts[1:], [len(txn_user_ids)]])

    # Build user -> (start, end) index map
    user_txn_ranges = {}
    for idx in range(len(group_starts)):
        uid = txn_user_ids[group_starts[idx]]
        user_txn_ranges[uid] = (group_starts[idx], group_ends[idx])

    print(f"  Found {len(user_txn_ranges):,} users with transactions")

    # Determine workers
    t_start = time.time()

    if num_workers <= 0:
        num_workers = 1  # serial is safest for large shared arrays

    # Process users (serial — avoids pickling 479M-row arrays)
    print(f"  Processing {len(user_ids):,} users (serial, optimized)...")

    all_sequences = []
    all_labels = []
    all_features = []
    skipped = 0

    for i, uid in enumerate(user_ids):
        if uid not in user_txn_ranges:
            skipped += 1
            continue

        start, end = user_txn_ranges[uid]
        n_txns = end - start
        if n_txns < 5:
            skipped += 1
            continue

        # Build transaction list using pre-computed date components (no datetime parsing)
        txns = []
        for j in range(start, end):
            txns.append({
                "amount": float(txn_amounts[j]),
                "month": int(txn_months[j]),
                "day": int(txn_days[j]),
                "weekday": int(txn_weekdays[j]) - 1,  # polars 1-indexed -> python 0-indexed
                "description": txn_descriptions[j],
            })

        # Build and pad token sequence (uses optimized path)
        sequence = sequence_builder.build_sequence_fast(txns)
        sequence = sequence_builder.pad_sequence(sequence, max_seq_len)

        all_sequences.append(sequence)
        all_labels.append(int(label_map.get(uid, False)))
        all_features.append(feature_map.get(uid, np.zeros(len(feature_cols), dtype=np.float32)))

        if (i + 1) % 5000 == 0:
            elapsed = time.time() - t_start
            rate = (i + 1) / elapsed
            eta = (len(user_ids) - i - 1) / rate
            print(f"    {i + 1:,}/{len(user_ids):,} users | "
                  f"{rate:.0f} users/s | ETA {eta/60:.0f} min")

    print(f"  Processed {len(all_sequences):,} users (skipped {skipped})")

    # Split into train/val (80/20 random split with fixed seed)
    rng = np.random.default_rng(42)
    n = len(all_sequences)
    indices = rng.permutation(n)
    split_idx = int(n * 0.8)

    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    sequences_arr = np.array(all_sequences, dtype=np.int32)
    labels_arr = np.array(all_labels, dtype=np.int8)
    features_arr = np.array(all_features, dtype=np.float32)

    return {
        "train_sequences": sequences_arr[train_idx],
        "val_sequences": sequences_arr[val_idx],
        "train_labels": labels_arr[train_idx],
        "val_labels": labels_arr[val_idx],
        "train_features": features_arr[train_idx],
        "val_features": features_arr[val_idx],
    }


def save_arrays(data: dict, output_dir: Path):
    """Save numpy arrays."""
    output_dir.mkdir(parents=True, exist_ok=True)

    for name, arr in data.items():
        path = output_dir / f"{name}.npy"
        np.save(path, arr)
        print(f"  Saved {path} ({arr.shape}, {arr.dtype}, {arr.nbytes / 1e6:.1f} MB)")


def main():
    parser = argparse.ArgumentParser(description="Process raw data into training arrays")
    parser.add_argument("--input-dir", type=str, default="data/raw/dev")
    parser.add_argument("--output-dir", type=str, default="data/processed")
    parser.add_argument("--max-seq-len", type=int, default=2048)
    parser.add_argument("--bpe-vocab-size", type=int, default=24000)
    parser.add_argument("--workers", type=int, default=0,
                        help="Number of parallel workers (0 = auto-detect CPU count)")
    parser.add_argument("--tokenizer", type=str, default=None,
                        help="Path to pre-trained tokenizer.json (if not provided, trains from data)")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    print(f"=== Data Processing ===")
    print(f"Input:  {input_dir}")
    print(f"Output: {output_dir}")
    print(f"Seq len: {args.max_seq_len}")
    print(f"Workers: {args.workers if args.workers > 0 else 'auto'}")
    print()

    # Load data
    print("[1/4] Loading parquet files...")
    t0 = time.time()
    transactions, labels, features = load_data(input_dir)
    print(f"  Transactions: {len(transactions):,} rows")
    print(f"  Users: {len(labels):,}")
    print(f"  Features: {features.shape}")
    print(f"  Loaded in {time.time()-t0:.1f}s")

    # Train tokenizer
    print("\n[2/4] Training BPE tokenizer...")
    t0 = time.time()
    desc_tokenizer = build_tokenizer(transactions, tokenizer_path=args.tokenizer)
    print(f"  BPE vocab: {desc_tokenizer.vocab_size} tokens")
    print(f"  Trained in {time.time()-t0:.1f}s")

    # Build sequence builder
    sequence_builder = SequenceBuilder(
        description_tokenizer=desc_tokenizer,
        max_seq_len=args.max_seq_len,
        max_desc_tokens=8,
    )

    # Process users
    print(f"\n[3/4] Tokenizing user sequences...")
    t0 = time.time()
    data = process_users(
        transactions, labels, features,
        sequence_builder, args.max_seq_len,
        num_workers=args.workers,
    )
    print(f"  Tokenized in {time.time()-t0:.1f}s")
    print(f"  Train: {len(data['train_sequences']):,} users")
    print(f"  Val:   {len(data['val_sequences']):,} users")

    # Save
    print(f"\n[4/4] Saving arrays...")
    save_arrays(data, output_dir)

    # Summary
    print(f"\n=== Processing Complete ===")
    total_mb = sum(arr.nbytes for arr in data.values()) / 1e6
    print(f"Total output size: {total_mb:.1f} MB")
    train_pos = data["train_labels"].sum()
    val_pos = data["val_labels"].sum()
    print(f"Train positive rate: {train_pos}/{len(data['train_labels'])} "
          f"({train_pos/len(data['train_labels'])*100:.1f}%)")
    print(f"Val positive rate: {val_pos}/{len(data['val_labels'])} "
          f"({val_pos/len(data['val_labels'])*100:.1f}%)")


if __name__ == "__main__":
    main()
