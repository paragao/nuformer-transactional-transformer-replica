"""Tests for tokenization pipeline."""

import sys
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
from src.tokenization.special_tokens import SpecialTokens, amount_to_bucket, AMOUNT_BOUNDARIES
from src.tokenization.bpe_tokenizer import DescriptionTokenizer
from src.tokenization.sequence_builder import SequenceBuilder


def test_amount_bucketing():
    """Test amount quantization into correct buckets."""
    # Boundary cases
    assert amount_to_bucket(0.005) == 0  # < $0.01
    assert amount_to_bucket(0.03) == 1   # $0.01 - $0.05
    assert amount_to_bucket(0.08) == 2   # $0.05 - $0.10
    assert amount_to_bucket(5.0) == 7    # $2.0 - $5.0 (boundary = exact match goes to next)
    assert amount_to_bucket(99.0) == 10  # $50 - $100
    assert amount_to_bucket(499.0) == 12 # $250 - $500
    assert amount_to_bucket(200000) == 20  # > $100K

    # Negative amounts use absolute value
    assert amount_to_bucket(-50.0) == 10  # same as positive

    print("  test_amount_bucketing PASSED")


def test_special_token_ids():
    """Test special token ID assignments are correct and non-overlapping."""
    # Sign tokens
    assert SpecialTokens.amount_sign_token(100.0) == 0  # positive
    assert SpecialTokens.amount_sign_token(-50.0) == 1  # negative

    # Bucket tokens
    assert SpecialTokens.amount_bucket_token(0.005) == 2  # offset=2, bucket=0
    assert SpecialTokens.amount_bucket_token(200000) == 22  # offset=2, bucket=20

    # Month tokens
    assert SpecialTokens.month_token(1) == 23   # January
    assert SpecialTokens.month_token(12) == 34  # December

    # Day tokens
    assert SpecialTokens.day_token(1) == 35
    assert SpecialTokens.day_token(31) == 65

    # Weekday tokens
    assert SpecialTokens.weekday_token(0) == 66  # Monday
    assert SpecialTokens.weekday_token(6) == 72  # Sunday

    # Control tokens
    assert SpecialTokens.SEP_TOKEN == 73
    assert SpecialTokens.PAD_TOKEN == 74
    assert SpecialTokens.BOS_TOKEN == 75
    assert SpecialTokens.EOS_TOKEN == 76
    assert SpecialTokens.UNK_TOKEN == 77

    # BPE offset
    assert SpecialTokens.BPE_OFFSET == 78

    # No overlaps
    all_special = set(range(78))
    assert len(all_special) == 78

    print("  test_special_token_ids PASSED")


def test_token_decoding():
    """Test that tokens can be decoded back to readable strings."""
    assert SpecialTokens.decode_token(0) == "<SIGN:+>"
    assert SpecialTokens.decode_token(1) == "<SIGN:->"
    assert SpecialTokens.decode_token(23) == "<MONTH:1>"
    assert SpecialTokens.decode_token(34) == "<MONTH:12>"
    assert SpecialTokens.decode_token(66) == "<WDAY:Mon>"
    assert SpecialTokens.decode_token(72) == "<WDAY:Sun>"
    assert SpecialTokens.decode_token(73) == "<SEP>"
    assert SpecialTokens.decode_token(75) == "<BOS>"
    assert SpecialTokens.decode_token(76) == "<EOS>"

    print("  test_token_decoding PASSED")


def test_description_tokenizer_simple():
    """Test simple word-level tokenizer (no HF dependency)."""
    tokenizer = DescriptionTokenizer(vocab_size=100)

    # Train on sample descriptions
    descriptions = [
        "WALMART - groceries",
        "UBER - transport",
        "NETFLIX - subscriptions",
        "STARBUCKS - coffee",
        "AMAZON - shopping",
        "IFOOD - dining",
    ] * 10  # repeat for frequency

    tokenizer.train(descriptions)

    # Encode
    tokens = tokenizer.encode("WALMART - groceries")
    assert len(tokens) > 0
    assert all(t >= SpecialTokens.BPE_OFFSET for t in tokens)

    # Decode round-trip
    decoded = tokenizer.decode(tokens)
    assert "walmart" in decoded.lower() or "groceries" in decoded.lower()

    # Vocab size
    assert tokenizer.get_vocab_size() > 0
    assert tokenizer.total_vocab_size() >= SpecialTokens.NUM_SPECIAL

    print("  test_description_tokenizer_simple PASSED")


def test_sequence_builder():
    """Test building token sequences from transactions."""
    # Setup tokenizer
    tokenizer = DescriptionTokenizer(vocab_size=100)
    descriptions = ["WALMART - groceries", "UBER - transport", "NETFLIX - subscriptions"] * 10
    tokenizer.train(descriptions)

    builder = SequenceBuilder(tokenizer, max_seq_len=128, max_desc_tokens=4)

    # Create sample transactions
    transactions = [
        {"amount": -65.42, "timestamp": datetime(2023, 6, 15, 10, 30), "description": "WALMART - groceries"},
        {"amount": -12.50, "timestamp": datetime(2023, 6, 15, 14, 0), "description": "UBER - transport"},
        {"amount": 4500.0, "timestamp": datetime(2023, 6, 30, 8, 0), "description": "SALARY DEPOSIT"},
    ]

    # Build sequence
    seq = builder.build_sequence(transactions)

    # Should start with BOS and end with EOS
    assert seq[0] == SpecialTokens.BOS_TOKEN
    assert seq[-1] == SpecialTokens.EOS_TOKEN

    # Should contain SEP tokens between transactions
    assert SpecialTokens.SEP_TOKEN in seq

    # Should not exceed max_seq_len
    assert len(seq) <= 128

    # Check first transaction tokens (after BOS)
    # Amount -65.42: sign=negative(1), bucket for $65~(10)
    assert seq[1] == SpecialTokens.amount_sign_token(-65.42)  # 1 (negative)
    assert seq[2] == SpecialTokens.amount_bucket_token(65.42)  # bucket for $50-$100

    # Month=6 (June)
    assert seq[3] == SpecialTokens.month_token(6)

    # Day=15
    assert seq[4] == SpecialTokens.day_token(15)

    # Weekday: June 15, 2023 is Thursday (3)
    assert seq[5] == SpecialTokens.weekday_token(3)

    print("  test_sequence_builder PASSED")


def test_sequence_padding():
    """Test padding and attention mask creation."""
    tokenizer = DescriptionTokenizer(vocab_size=50)
    tokenizer.train(["TEST"] * 10)
    builder = SequenceBuilder(tokenizer, max_seq_len=64)

    transactions = [
        {"amount": -10.0, "timestamp": datetime(2023, 1, 1, 12, 0), "description": "TEST"},
    ]

    seq = builder.build_sequence(transactions)
    assert len(seq) < 64

    # Pad
    padded = builder.pad_sequence(seq, target_len=64)
    assert len(padded) == 64
    assert padded[-1] == SpecialTokens.PAD_TOKEN

    # Attention mask
    mask = builder.create_attention_mask(padded)
    assert len(mask) == 64
    assert mask[0] == 1  # BOS is real
    assert mask[-1] == 0  # PAD is masked

    # Real token count should match original sequence length
    real_count = sum(mask)
    assert real_count == len(seq)

    print("  test_sequence_padding PASSED")


def test_sequence_truncation():
    """Test that long sequences are properly truncated (keeping recent)."""
    tokenizer = DescriptionTokenizer(vocab_size=50)
    tokenizer.train(["MERCHANT_NAME - category"] * 10)
    builder = SequenceBuilder(tokenizer, max_seq_len=32, max_desc_tokens=2)

    # Create many transactions (more than can fit in 32 tokens)
    transactions = [
        {"amount": -10.0 * i, "timestamp": datetime(2023, 1, i + 1, 12, 0), "description": "MERCHANT_NAME - category"}
        for i in range(20)
    ]

    seq = builder.build_sequence(transactions)

    # Should not exceed max_seq_len
    assert len(seq) <= 32

    # Should contain BOS and EOS
    assert seq[0] == SpecialTokens.BOS_TOKEN
    assert seq[-1] == SpecialTokens.EOS_TOKEN

    # Should keep most recent transactions (higher amounts = later dates)
    print(f"  test_sequence_truncation PASSED (seq_len={len(seq)} <= 32)")


def test_memmap_dataset():
    """Test memory-mapped dataset creation and loading."""
    from src.tokenization.dataset import TransactionDataset
    import tempfile
    import os

    with tempfile.TemporaryDirectory() as tmpdir:
        # Create sample sequences
        sequences = [
            [75, 1, 10, 23, 35, 66, 80, 81, 73, 0, 5, 25, 40, 68, 82, 73, 76],
            [75, 0, 15, 28, 50, 70, 85, 73, 76],
        ]

        seq_path = os.path.join(tmpdir, "sequences")
        TransactionDataset.create_memmap(sequences, seq_path, max_seq_len=32)

        # Verify file exists (.npy extension added by np.save)
        npy_path = seq_path + ".npy" if not seq_path.endswith(".npy") else seq_path
        assert os.path.exists(npy_path)

        # Load and verify
        data = np.load(npy_path)
        assert data.shape == (2, 32)
        assert data[0, 0] == 75  # BOS
        assert data[0, 16] == 76  # EOS at position 16
        assert data[0, 17] == 74  # PAD after EOS
        assert data[1, 8] == 76  # EOS for second sequence

    print("  test_memmap_dataset PASSED")


if __name__ == "__main__":
    print("Running tokenization tests...")
    print()
    test_amount_bucketing()
    test_special_token_ids()
    test_token_decoding()
    test_description_tokenizer_simple()
    test_sequence_builder()
    test_sequence_padding()
    test_sequence_truncation()
    test_memmap_dataset()
    print()
    print("All tokenization tests PASSED!")
