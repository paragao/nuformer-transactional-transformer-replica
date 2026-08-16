"""Batch inference for nuFormer Joint Fusion model.

Accepts raw (non-tokenized) transaction data + tabular features,
tokenizes on-the-fly using the saved BPE tokenizer, and outputs
fraud probabilities per user.

Input: transactions.parquet + tabular_features.parquet (same format as generate_data.py output)
Output: predictions.parquet with [user_id, fraud_probability, predicted_label, confidence]

Usage:
    python scripts/batch_inference.py \\
        --checkpoint ckpt/joint_fusion_v2/best.pt \\
        --tokenizer tokenizer/tokenizer.json \\
        --transactions data/raw_300k/transactions.parquet \\
        --features data/raw_300k/tabular_features.parquet \\
        --output predictions.parquet \\
        --batch-size 128
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

# Allow torch.load to unpickle configs saved from training scripts
from src.training.joint_fusion import JointFusionConfig
from src.training.pretrain import PretrainConfig
sys.modules["__main__"].JointFusionConfig = JointFusionConfig
sys.modules["__main__"].PretrainConfig = PretrainConfig


def load_model(checkpoint_path: str, device: torch.device) -> torch.nn.Module:
    """Load nuFormer model from joint fusion checkpoint."""
    from src.models.nuformer import NuFormer, NuFormerConfig
    from src.models.transformer import TransformerConfig
    from src.models.dcnv2 import DCNv2Config
    from src.models.lora import apply_lora

    print(f"  Loading checkpoint: {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Reconstruct config from checkpoint
    config = ckpt.get("config", None)
    if config is not None:
        # Extract architecture params from stored JointFusionConfig
        transformer_config = TransformerConfig(
            vocab_size=getattr(config, "vocab_size", 24078),
            d_model=getattr(config, "d_model", 1024),
            n_layers=getattr(config, "n_layers", 24),
            n_heads=getattr(config, "n_heads", 16),
            d_ff=getattr(config, "d_ff", 4096),
            max_seq_len=getattr(config, "max_seq_len", 2048),
            dropout=0.0,
        )
        plr_dim = getattr(config, "_plr_dim", 8)
        plr_freq = getattr(config, "_plr_frequencies", 4)
        dcnv2_config = DCNv2Config(
            input_dim=getattr(config, "num_tabular_features", 291),
            cross_layers=getattr(config, "dcnv2_cross_layers", 3),
            deep_layers=getattr(config, "dcnv2_deep_dims", [512, 256]),
            output_dim=getattr(config, "dcnv2_output_dim", 128),
            dropout=0.0,  # No dropout at inference
            use_plr=True,
            plr_dim=plr_dim,
            plr_frequencies=plr_freq,
        )
        lora_rank = getattr(config, "lora_rank", 16)
        lora_alpha = getattr(config, "lora_alpha", 32.0)
    else:
        # Fallback defaults (v1 config)
        transformer_config = TransformerConfig(
            vocab_size=24078, d_model=1024, n_layers=24,
            n_heads=16, d_ff=4096, max_seq_len=2048, dropout=0.0,
        )
        dcnv2_config = DCNv2Config(
            input_dim=291, cross_layers=3, deep_layers=[512, 256],
            output_dim=128, dropout=0.0, use_plr=True, plr_dim=8, plr_frequencies=4,
        )
        lora_rank = 16
        lora_alpha = 32.0

    nuformer_config = NuFormerConfig(
        transformer=transformer_config,
        dcnv2=dcnv2_config,
        fusion_dropout=0.0,  # No dropout at inference
    )

    model = NuFormer(nuformer_config)

    # Apply LoRA structure (needed to match state dict keys)
    model.transformer = apply_lora(model.transformer, rank=lora_rank, alpha=lora_alpha)

    # Load weights
    model.load_state_dict(ckpt["model"], strict=False)
    model.to(device)
    model.eval()

    # Print info
    best_auc = ckpt.get("best_auc", "N/A")
    step = ckpt.get("step", "N/A")
    print(f"  Model loaded (step={step}, best_auc={best_auc})")
    print(f"  Parameters: {sum(p.numel() for p in model.parameters()):,}")

    return model


def build_sequence_builder(tokenizer_path: str):
    """Load tokenizer and create SequenceBuilder."""
    from src.tokenization.bpe_tokenizer import DescriptionTokenizer
    from src.tokenization.sequence_builder import SequenceBuilder

    print(f"  Loading tokenizer: {tokenizer_path}")
    tokenizer = DescriptionTokenizer.from_pretrained(tokenizer_path)
    print(f"  BPE vocab: {tokenizer.get_vocab_size()}, total: {tokenizer.total_vocab_size()}")

    seq_builder = SequenceBuilder(
        description_tokenizer=tokenizer,
        max_seq_len=2048,
        max_desc_tokens=8,
    )
    return seq_builder


def load_transactions(path: str) -> dict[str, list[dict]]:
    """Load transactions grouped by user_id, sorted by timestamp."""
    import polars as pl

    print(f"  Loading transactions: {path}")
    df = pl.read_parquet(path, columns=["user_id", "timestamp", "amount", "description"])

    # Parse timestamps and extract date components (vectorized — fast)
    ts_col = df["timestamp"]
    if ts_col.dtype == pl.Utf8:
        ts_col = ts_col.str.to_datetime()
    df = df.with_columns([
        ts_col.dt.month().alias("month"),
        ts_col.dt.day().alias("day"),
        ts_col.dt.weekday().alias("weekday"),  # polars: 1=Monday
    ])
    df = df.sort(["user_id", "timestamp"])

    # Extract columns as numpy/lists for fast iteration (same pattern as process_data.py)
    print(f"  Extracting columns for fast iteration...")
    user_ids_arr = df["user_id"].to_numpy()
    amounts = df["amount"].to_numpy()
    months = df["month"].to_numpy()
    days = df["day"].to_numpy()
    weekdays = df["weekday"].to_numpy()
    descriptions = df["description"].to_list()

    # Find group boundaries using numpy
    n_rows = len(user_ids_arr)
    change_mask = np.concatenate([[True], user_ids_arr[1:] != user_ids_arr[:-1]])
    group_starts = np.where(change_mask)[0]
    group_ends = np.concatenate([group_starts[1:], [n_rows]])

    # Build user -> transactions dict
    user_transactions = {}
    for idx in range(len(group_starts)):
        start, end = group_starts[idx], group_ends[idx]
        uid = user_ids_arr[start]
        txns = []
        for j in range(start, end):
            txns.append({
                "amount": float(amounts[j]),
                "month": int(months[j]),
                "day": int(days[j]) ,
                "weekday": int(weekdays[j]) - 1,  # polars 1-indexed → python 0-indexed
                "description": descriptions[j],
            })
        user_transactions[uid] = txns

    total_txns = sum(len(v) for v in user_transactions.values())
    print(f"  Loaded {len(user_transactions):,} users, {total_txns:,} transactions")
    return user_transactions


def load_features(path: str) -> tuple[list[str], np.ndarray]:
    """Load tabular features, return (user_ids, feature_matrix)."""
    import polars as pl

    print(f"  Loading features: {path}")
    df = pl.read_parquet(path)

    user_ids = df["user_id"].to_list()
    feature_cols = [c for c in df.columns if c != "user_id"]
    features = df.select(feature_cols).to_numpy().astype(np.float32)

    print(f"  Features shape: {features.shape} ({len(feature_cols)} dims)")
    return user_ids, features


def tokenize_users(
    user_transactions: dict[str, list[dict]],
    user_ids: list[str],
    sequence_builder,
    max_seq_len: int = 2048,
) -> tuple[np.ndarray, np.ndarray]:
    """Tokenize all users and return sequences + attention masks.

    Only processes users that appear in both user_ids (from features) and
    user_transactions. Returns arrays aligned with user_ids order.
    """
    from src.tokenization.special_tokens import SpecialTokens

    print(f"  Tokenizing {len(user_ids):,} users...")
    t0 = time.time()

    sequences = []
    attention_masks = []
    valid_indices = []

    for i, uid in enumerate(user_ids):
        txns = user_transactions.get(uid)
        if txns is None or len(txns) < 1:
            # User has no transactions — use empty sequence (all PAD)
            seq = [SpecialTokens.PAD_TOKEN] * max_seq_len
        else:
            seq = sequence_builder.build_sequence_fast(txns)
            seq = sequence_builder.pad_sequence(seq, max_seq_len)

        mask = [0 if t == SpecialTokens.PAD_TOKEN else 1 for t in seq]
        sequences.append(seq)
        attention_masks.append(mask)
        valid_indices.append(i)

        if (i + 1) % 10000 == 0:
            elapsed = time.time() - t0
            rate = (i + 1) / elapsed
            print(f"    {i + 1:,}/{len(user_ids):,} users ({rate:.0f} users/s)")

    elapsed = time.time() - t0
    print(f"  Tokenized {len(sequences):,} users in {elapsed:.1f}s "
          f"({len(sequences)/elapsed:.0f} users/s)")

    return (
        np.array(sequences, dtype=np.int32),
        np.array(attention_masks, dtype=np.int32),
    )


@torch.no_grad()
def run_inference(
    model: torch.nn.Module,
    sequences: np.ndarray,
    features: np.ndarray,
    attention_masks: np.ndarray,
    batch_size: int,
    device: torch.device,
) -> np.ndarray:
    """Run batched inference, return fraud probabilities."""
    all_probs = []
    n = len(sequences)

    print(f"  Running inference on {n:,} users (batch_size={batch_size})...")
    t0 = time.time()

    for i in range(0, n, batch_size):
        batch_seq = torch.from_numpy(sequences[i:i+batch_size]).long().to(device)
        batch_feat = torch.from_numpy(features[i:i+batch_size]).float().to(device)
        batch_mask = torch.from_numpy(attention_masks[i:i+batch_size]).long().to(device)

        with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=device.type == "cuda"):
            output = model(batch_seq, batch_feat, batch_mask)

        probs = F.softmax(output["logits"].float(), dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())

    elapsed = time.time() - t0
    probabilities = np.concatenate(all_probs)
    print(f"  Inference complete: {elapsed:.1f}s ({n/elapsed:.0f} users/s)")

    return probabilities


def save_predictions(
    user_ids: list[str],
    probabilities: np.ndarray,
    output_path: str,
    threshold: float = 0.5,
) -> None:
    """Save predictions as parquet."""
    import polars as pl

    predicted_labels = (probabilities >= threshold).astype(int)
    confidence = np.where(
        predicted_labels == 1, probabilities, 1.0 - probabilities
    )

    df = pl.DataFrame({
        "user_id": user_ids,
        "fraud_probability": probabilities.tolist(),
        "predicted_label": predicted_labels.tolist(),
        "confidence": confidence.tolist(),
    })

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)

    if output_path.endswith(".csv"):
        df.write_csv(output)
    else:
        df.write_parquet(output)

    print(f"  Saved predictions: {output}")
    print(f"  Total users: {len(user_ids):,}")
    print(f"  Predicted fraud: {predicted_labels.sum():,} ({predicted_labels.mean()*100:.1f}%)")
    print(f"  Mean probability: {probabilities.mean():.4f}")
    print(f"  Threshold: {threshold}")


def main():
    """Entry point."""
    parser = argparse.ArgumentParser(
        description="nuFormer batch inference from raw transactions"
    )
    parser.add_argument("--checkpoint", type=str, required=True,
                        help="Path to joint fusion checkpoint (.pt)")
    parser.add_argument("--tokenizer", type=str, default="tokenizer/tokenizer.json",
                        help="Path to saved tokenizer.json")
    parser.add_argument("--transactions", type=str, required=True,
                        help="Path to transactions.parquet")
    parser.add_argument("--features", type=str, required=True,
                        help="Path to tabular_features.parquet")
    parser.add_argument("--output", type=str, default="predictions.parquet",
                        help="Output path for predictions")
    parser.add_argument("--batch-size", type=int, default=128,
                        help="Inference batch size")
    parser.add_argument("--threshold", type=float, default=0.5,
                        help="Classification threshold for predicted_label")
    parser.add_argument("--device", type=str, default="auto",
                        help="Device: 'auto', 'cuda', 'cpu'")
    args = parser.parse_args()

    print("=" * 60)
    print("  nuFormer Batch Inference")
    print("=" * 60)
    print(f"  Checkpoint:    {args.checkpoint}")
    print(f"  Tokenizer:     {args.tokenizer}")
    print(f"  Transactions:  {args.transactions}")
    print(f"  Features:      {args.features}")
    print(f"  Output:        {args.output}")
    print(f"  Batch size:    {args.batch_size}")
    print(f"  Threshold:     {args.threshold}")
    print("=" * 60)

    t_total = time.time()

    # Device
    if args.device == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(args.device)
    print(f"\n  Device: {device}")

    # Step 1: Load model
    print("\n[1/5] Loading model...")
    model = load_model(args.checkpoint, device)

    # Step 2: Load tokenizer
    print("\n[2/5] Loading tokenizer...")
    sequence_builder = build_sequence_builder(args.tokenizer)

    # Step 3: Load data
    print("\n[3/5] Loading data...")
    user_transactions = load_transactions(args.transactions)
    user_ids, features = load_features(args.features)

    # Step 4: Tokenize
    print("\n[4/5] Tokenizing...")
    sequences, attention_masks = tokenize_users(
        user_transactions, user_ids, sequence_builder
    )

    # Step 5: Inference
    print("\n[5/5] Running inference...")
    probabilities = run_inference(
        model, sequences, features, attention_masks, args.batch_size, device
    )

    # Save
    print("\n" + "=" * 60)
    print("  Results")
    print("=" * 60)
    save_predictions(user_ids, probabilities, args.output, args.threshold)

    elapsed = time.time() - t_total
    print(f"\n  Total time: {elapsed:.1f}s")
    print("=" * 60)


if __name__ == "__main__":
    main()
