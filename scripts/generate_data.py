"""Main data generation script.

Generates synthetic financial transaction data:
1. User personas (correlated demographics)
2. Transaction histories (realistic distributions)
3. Tabular features (291 dimensions)
4. Labels (credit card activation)

Usage:
    python scripts/generate_data.py --num-users 1000 --output-dir data/validation
    python scripts/generate_data.py --num-users 100000 --output-dir data/raw
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

try:
    import polars as pl
except ImportError:
    pl = None

from src.synthetic_data.schemas import DatasetConfig
from src.synthetic_data.personas import PersonaGenerator
from src.synthetic_data.transaction_generator import TransactionGenerator
from src.synthetic_data.tabular_features import TabularFeatureComputer
from src.synthetic_data.label_generator import LabelGenerator


def generate_dataset(config: DatasetConfig) -> None:
    """Generate complete synthetic dataset."""
    output_dir = Path(config.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"=== Synthetic Data Generation ===")
    print(f"Users: {config.num_users:,}")
    print(f"Date range: {config.start_date} to {config.end_date}")
    print(f"Output: {output_dir}")
    print()

    # Step 1: Generate personas
    print("[1/4] Generating user personas...")
    t0 = time.time()
    persona_gen = PersonaGenerator(seed=config.seed)
    personas = persona_gen.generate(config.num_users)
    print(f"  Generated {len(personas):,} personas in {time.time() - t0:.1f}s")

    # Step 2: Generate transactions
    print("[2/4] Generating transactions...")
    t0 = time.time()
    txn_gen = TransactionGenerator(config, seed=config.seed + 1)
    all_transactions = []
    transactions_by_user = {}

    for i, persona in enumerate(personas):
        user_txns = txn_gen.generate_for_user(persona)
        all_transactions.extend(user_txns)
        transactions_by_user[persona.user_id] = user_txns

        if (i + 1) % 1000 == 0 or i == len(personas) - 1:
            print(f"  Processed {i + 1:,}/{len(personas):,} users "
                  f"({len(all_transactions):,} transactions)")

    elapsed = time.time() - t0
    print(f"  Generated {len(all_transactions):,} transactions in {elapsed:.1f}s "
          f"({len(all_transactions) / elapsed:.0f} txns/sec)")

    # Step 3: Compute tabular features
    print("[3/4] Computing tabular features...")
    t0 = time.time()
    feature_computer = TabularFeatureComputer(reference_date=config.train_end_date)
    features_list = []

    for i, persona in enumerate(personas):
        user_txns = transactions_by_user[persona.user_id]
        features = feature_computer.compute(persona, user_txns)
        features["user_id"] = persona.user_id
        features_list.append(features)

        if (i + 1) % 5000 == 0 or i == len(personas) - 1:
            print(f"  Computed features for {i + 1:,}/{len(personas):,} users")

    print(f"  Features computed in {time.time() - t0:.1f}s "
          f"({len(features_list[0]) - 1} dimensions)")

    # Step 4: Generate labels
    print("[4/4] Generating labels...")
    t0 = time.time()
    label_gen = LabelGenerator(target_positive_rate=config.positive_rate, seed=config.seed + 2)
    labels = label_gen.generate_labels_batch(personas, transactions_by_user)
    positive_count = sum(labels.values())
    print(f"  Labels: {positive_count:,} positive / {len(labels):,} total "
          f"({positive_count / len(labels) * 100:.1f}%)")
    print(f"  Generated in {time.time() - t0:.1f}s")

    # Save outputs
    print("\nSaving outputs...")
    _save_outputs(output_dir, personas, all_transactions, features_list, labels, config)

    print(f"\n=== Generation Complete ===")
    print(f"Transactions: {len(all_transactions):,}")
    print(f"Users: {len(personas):,}")
    print(f"Features: {len(features_list[0]) - 1} dims")
    print(f"Positive rate: {positive_count / len(labels) * 100:.1f}%")
    print(f"Output: {output_dir}")


def _save_outputs(
    output_dir: Path,
    personas: list,
    transactions: list,
    features_list: list[dict],
    labels: dict[str, bool],
    config: DatasetConfig,
) -> None:
    """Save all outputs as parquet files."""
    if pl is None:
        # Fallback to CSV if polars not available
        _save_csv(output_dir, personas, transactions, features_list, labels)
        return

    # Personas
    persona_records = [p.model_dump() for p in personas]
    for record in persona_records:
        record["label"] = labels.get(record["user_id"], False)
    df_personas = pl.DataFrame(persona_records)
    df_personas.write_parquet(output_dir / "personas.parquet")
    print(f"  Saved personas: {output_dir / 'personas.parquet'}")

    # Transactions
    txn_records = [
        {
            "user_id": t.user_id,
            "timestamp": t.timestamp.isoformat(),
            "amount": t.amount,
            "description": t.description,
            "category": t.category.value,
            "merchant_id": t.merchant_id,
            "merchant_category_code": t.merchant_category_code,
            "transaction_type": t.transaction_type.value,
            "installments": t.installments,
            "status": t.status.value,
        }
        for t in transactions
    ]
    df_txns = pl.DataFrame(txn_records)
    df_txns.write_parquet(output_dir / "transactions.parquet")
    print(f"  Saved transactions: {output_dir / 'transactions.parquet'} ({len(txn_records):,} rows)")

    # Tabular features
    df_features = pl.DataFrame(features_list)
    df_features.write_parquet(output_dir / "tabular_features.parquet")
    print(f"  Saved features: {output_dir / 'tabular_features.parquet'}")

    # Labels (separate file for convenience)
    df_labels = pl.DataFrame([
        {"user_id": uid, "activated_credit_card": label}
        for uid, label in labels.items()
    ])
    df_labels.write_parquet(output_dir / "labels.parquet")
    print(f"  Saved labels: {output_dir / 'labels.parquet'}")

    # Stats summary
    stats = {
        "num_users": len(personas),
        "num_transactions": len(transactions),
        "positive_rate": sum(labels.values()) / len(labels),
        "avg_txns_per_user": len(transactions) / len(personas),
        "num_features": len(features_list[0]) - 1,  # minus user_id
        "config": config.model_dump(),
    }
    import json
    with open(output_dir / "stats.json", "w") as f:
        json.dump(stats, f, indent=2, default=str)
    print(f"  Saved stats: {output_dir / 'stats.json'}")


def _save_csv(output_dir, personas, transactions, features_list, labels):
    """Fallback CSV saving when polars is not available."""
    import csv

    # Save personas
    with open(output_dir / "personas.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(personas[0].model_dump().keys()) + ["label"])
        writer.writeheader()
        for p in personas:
            row = p.model_dump()
            row["label"] = labels.get(row["user_id"], False)
            writer.writerow(row)

    print(f"  Saved personas CSV: {output_dir / 'personas.csv'}")
    print(f"  (Install polars for faster parquet output)")


def main():
    parser = argparse.ArgumentParser(description="Generate synthetic financial transaction data")
    parser.add_argument("--num-users", type=int, default=1000, help="Number of users")
    parser.add_argument("--output-dir", type=str, default="data/validation", help="Output directory")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--start-date", type=str, default="2022-01-01")
    parser.add_argument("--end-date", type=str, default="2024-06-30")
    parser.add_argument("--positive-rate", type=float, default=0.15)
    args = parser.parse_args()

    config = DatasetConfig(
        num_users=args.num_users,
        output_dir=args.output_dir,
        seed=args.seed,
        start_date=args.start_date,
        end_date=args.end_date,
        positive_rate=args.positive_rate,
    )

    generate_dataset(config)


if __name__ == "__main__":
    main()
