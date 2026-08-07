"""Tests for synthetic data generation pipeline."""

import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.synthetic_data.schemas import DatasetConfig, TransactionCategory, IncomeBracket
from src.synthetic_data.personas import PersonaGenerator
from src.synthetic_data.transaction_generator import TransactionGenerator
from src.synthetic_data.tabular_features import TabularFeatureComputer
from src.synthetic_data.label_generator import LabelGenerator


def test_persona_generation():
    """Test persona generation produces valid, diverse personas."""
    gen = PersonaGenerator(seed=42)
    personas = gen.generate(100)

    assert len(personas) == 100
    assert all(p.user_id.startswith("user_") for p in personas)
    assert all(500 <= p.monthly_income <= 50000 for p in personas)
    assert all(300 <= p.credit_score <= 850 for p in personas)
    assert all(0 <= p.spending_discipline <= 1 for p in personas)
    assert all(0 <= p.digital_affinity <= 1 for p in personas)

    # Check diversity
    income_brackets = set(p.income_bracket for p in personas)
    age_groups = set(p.age_group for p in personas)
    assert len(income_brackets) >= 3, "Should have diverse income brackets"
    assert len(age_groups) >= 3, "Should have diverse age groups"

    print("  test_persona_generation PASSED")


def test_transaction_generation():
    """Test transaction generation produces realistic patterns."""
    config = DatasetConfig(
        num_users=10,
        min_transactions_per_user=50,
        max_transactions_per_user=500,
        start_date="2023-01-01",
        end_date="2024-01-01",
    )
    persona_gen = PersonaGenerator(seed=42)
    personas = persona_gen.generate(10)

    txn_gen = TransactionGenerator(config, seed=42)
    all_txns = []
    for persona in personas:
        txns = txn_gen.generate_for_user(persona)
        all_txns.extend(txns)
        assert len(txns) >= config.min_transactions_per_user
        assert len(txns) <= config.max_transactions_per_user

    # Check amounts
    amounts = [t.amount for t in all_txns]
    assert any(a > 0 for a in amounts), "Should have inflows (salary)"
    assert any(a < 0 for a in amounts), "Should have outflows (spending)"

    # Check categories
    categories = set(t.category for t in all_txns)
    assert len(categories) >= 8, f"Should have diverse categories, got {len(categories)}"

    # Check temporal ordering per user
    for persona in personas:
        user_txns = [t for t in all_txns if t.user_id == persona.user_id]
        timestamps = [t.timestamp for t in user_txns]
        assert timestamps == sorted(timestamps), "Transactions should be time-ordered"

    # Check salary is positive
    salary_txns = [t for t in all_txns if t.category == TransactionCategory.SALARY]
    assert all(t.amount > 0 for t in salary_txns), "Salary should be positive (inflow)"

    print("  test_transaction_generation PASSED")


def test_tabular_features():
    """Test tabular feature computation produces 291 dims."""
    config = DatasetConfig(num_users=5, start_date="2023-01-01", end_date="2024-01-01")
    persona_gen = PersonaGenerator(seed=42)
    personas = persona_gen.generate(5)

    txn_gen = TransactionGenerator(config, seed=42)
    feature_computer = TabularFeatureComputer(reference_date="2024-01-01")

    for persona in personas:
        txns = txn_gen.generate_for_user(persona)
        features = feature_computer.compute(persona, txns)

        assert len(features) == 291, f"Expected 291 features, got {len(features)}"
        assert "user_id" not in features or len(features) == 291
        # Check no NaN
        for k, v in features.items():
            assert not np.isnan(v), f"Feature {k} is NaN"
            assert not np.isinf(v), f"Feature {k} is Inf"

    print("  test_tabular_features PASSED")


def test_label_generation():
    """Test label generation produces calibrated positive rate."""
    config = DatasetConfig(num_users=1000, start_date="2023-01-01", end_date="2024-01-01")
    persona_gen = PersonaGenerator(seed=42)
    personas = persona_gen.generate(1000)

    txn_gen = TransactionGenerator(config, seed=42)
    transactions_by_user = {}
    for persona in personas:
        transactions_by_user[persona.user_id] = txn_gen.generate_for_user(persona)

    label_gen = LabelGenerator(target_positive_rate=0.15, seed=42)
    labels = label_gen.generate_labels_batch(personas, transactions_by_user)

    positive_rate = sum(labels.values()) / len(labels)
    assert 0.10 <= positive_rate <= 0.25, f"Positive rate {positive_rate:.3f} outside [0.10, 0.25]"

    print(f"  test_label_generation PASSED (positive_rate={positive_rate:.3f})")


def test_data_distributions():
    """Test that generated data follows expected statistical distributions."""
    config = DatasetConfig(num_users=100, start_date="2023-01-01", end_date="2024-01-01")
    persona_gen = PersonaGenerator(seed=42)
    personas = persona_gen.generate(100)

    txn_gen = TransactionGenerator(config, seed=42)
    all_txns = []
    for persona in personas:
        all_txns.extend(txn_gen.generate_for_user(persona))

    # Check amount distribution: most outflows should be < $500
    outflows = [abs(t.amount) for t in all_txns if t.amount < 0]
    p50 = np.percentile(outflows, 50)
    p95 = np.percentile(outflows, 95)
    assert p50 < 200, f"Median outflow ${p50:.2f} should be < $200"
    assert p95 < 5000, f"95th percentile outflow ${p95:.2f} should be < $5000"

    # Check temporal: weekday should have more transactions
    weekday_count = sum(1 for t in all_txns if t.timestamp.weekday() < 5)
    weekend_count = sum(1 for t in all_txns if t.timestamp.weekday() >= 5)
    weekday_ratio = weekday_count / (weekday_count + weekend_count)
    assert 0.55 <= weekday_ratio <= 0.85, f"Weekday ratio {weekday_ratio:.3f} unexpected"

    print(f"  test_data_distributions PASSED (median=${p50:.0f}, p95=${p95:.0f}, "
          f"weekday_ratio={weekday_ratio:.2f})")


if __name__ == "__main__":
    print("Running synthetic data tests...")
    print()
    test_persona_generation()
    test_transaction_generation()
    test_tabular_features()
    test_label_generation()
    test_data_distributions()
    print()
    print("All tests PASSED!")
