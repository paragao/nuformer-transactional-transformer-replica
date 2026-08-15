"""Special token definitions for transaction tokenization.

Implements the paper's tokenization scheme:
- Amount sign: 2 tokens (positive/negative)
- Amount bucket: 21 tokens (log-scale quantized bins)
- Month: 12 tokens
- Day: 31 tokens
- Weekday: 7 tokens
- Separator: 1 token
- Control tokens: PAD, BOS, EOS, UNK
"""

from __future__ import annotations

import numpy as np


# === Amount Bucket Boundaries (log-scale, 21 bins) ===
# Covers $0.01 to $100,000+ with denser bins at common amounts
AMOUNT_BOUNDARIES = [
    0.01, 0.05, 0.10, 0.50, 1.0, 2.0, 5.0, 10.0, 25.0, 50.0,
    100.0, 250.0, 500.0, 1000.0, 2500.0, 5000.0, 10000.0,
    25000.0, 50000.0, 100000.0,
]  # 20 boundaries -> 21 buckets


def amount_to_bucket(amount: float) -> int:
    """Quantize absolute amount into bucket index (0-20)."""
    abs_amt = abs(amount)
    for i, boundary in enumerate(AMOUNT_BOUNDARIES):
        if abs_amt < boundary:
            return i
    return len(AMOUNT_BOUNDARIES)  # bucket 20 = $100K+


# === Special Token Registry ===
class SpecialTokens:
    """Registry of all special tokens with their IDs.

    Token layout (total special tokens = 78):
        [0-1]   Amount sign (2)
        [2-22]  Amount bucket (21)
        [23-34] Month (12)
        [35-65] Day (31)
        [66-72] Weekday (7)
        [73]    SEP (transaction separator)
        [74]    PAD
        [75]    BOS (beginning of sequence)
        [76]    EOS (end of sequence)
        [77]    UNK (unknown)

    BPE tokens start at offset 78.
    """

    # Token ranges
    AMOUNT_SIGN_OFFSET = 0
    AMOUNT_SIGN_COUNT = 2

    AMOUNT_BUCKET_OFFSET = 2
    AMOUNT_BUCKET_COUNT = 21

    MONTH_OFFSET = 23
    MONTH_COUNT = 12

    DAY_OFFSET = 35
    DAY_COUNT = 31

    WEEKDAY_OFFSET = 66
    WEEKDAY_COUNT = 7

    # Control tokens
    SEP_TOKEN = 73
    PAD_TOKEN = 74
    BOS_TOKEN = 75
    EOS_TOKEN = 76
    UNK_TOKEN = 77

    # BPE tokens start here
    BPE_OFFSET = 78

    # Total special tokens
    NUM_SPECIAL = 78

    @classmethod
    def amount_sign_token(cls, amount: float) -> int:
        """Get token ID for amount sign (0=positive/inflow, 1=negative/outflow)."""
        return cls.AMOUNT_SIGN_OFFSET + (0 if amount >= 0 else 1)

    @classmethod
    def amount_bucket_token(cls, amount: float) -> int:
        """Get token ID for quantized amount bucket."""
        bucket = amount_to_bucket(amount)
        return cls.AMOUNT_BUCKET_OFFSET + bucket

    @classmethod
    def month_token(cls, month: int) -> int:
        """Get token ID for month (1-12 -> tokens 23-34)."""
        assert 1 <= month <= 12, f"Month must be 1-12, got {month}"
        return cls.MONTH_OFFSET + (month - 1)

    @classmethod
    def day_token(cls, day: int) -> int:
        """Get token ID for day of month (1-31 -> tokens 35-65)."""
        assert 1 <= day <= 31, f"Day must be 1-31, got {day}"
        return cls.DAY_OFFSET + (day - 1)

    @classmethod
    def weekday_token(cls, weekday: int) -> int:
        """Get token ID for day of week (0=Monday to 6=Sunday -> tokens 66-72)."""
        assert 0 <= weekday <= 6, f"Weekday must be 0-6, got {weekday}"
        return cls.WEEKDAY_OFFSET + weekday

    @classmethod
    def decode_token(cls, token_id: int) -> str:
        """Decode a special token ID to human-readable string."""
        if token_id == cls.SEP_TOKEN:
            return "<SEP>"
        elif token_id == cls.PAD_TOKEN:
            return "<PAD>"
        elif token_id == cls.BOS_TOKEN:
            return "<BOS>"
        elif token_id == cls.EOS_TOKEN:
            return "<EOS>"
        elif token_id == cls.UNK_TOKEN:
            return "<UNK>"
        elif cls.AMOUNT_SIGN_OFFSET <= token_id < cls.AMOUNT_SIGN_OFFSET + cls.AMOUNT_SIGN_COUNT:
            sign = "+" if (token_id - cls.AMOUNT_SIGN_OFFSET) == 0 else "-"
            return f"<SIGN:{sign}>"
        elif cls.AMOUNT_BUCKET_OFFSET <= token_id < cls.AMOUNT_BUCKET_OFFSET + cls.AMOUNT_BUCKET_COUNT:
            bucket = token_id - cls.AMOUNT_BUCKET_OFFSET
            if bucket == 0:
                return "<AMT:$0-0.01>"
            elif bucket < len(AMOUNT_BOUNDARIES):
                low = AMOUNT_BOUNDARIES[bucket - 1]
                high = AMOUNT_BOUNDARIES[bucket]
                return f"<AMT:${low}-{high}>"
            else:
                return f"<AMT:${AMOUNT_BOUNDARIES[-1]}+>"
        elif cls.MONTH_OFFSET <= token_id < cls.MONTH_OFFSET + cls.MONTH_COUNT:
            month = token_id - cls.MONTH_OFFSET + 1
            return f"<MONTH:{month}>"
        elif cls.DAY_OFFSET <= token_id < cls.DAY_OFFSET + cls.DAY_COUNT:
            day = token_id - cls.DAY_OFFSET + 1
            return f"<DAY:{day}>"
        elif cls.WEEKDAY_OFFSET <= token_id < cls.WEEKDAY_OFFSET + cls.WEEKDAY_COUNT:
            weekday = token_id - cls.WEEKDAY_OFFSET
            names = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
            return f"<WDAY:{names[weekday]}>"
        elif token_id >= cls.BPE_OFFSET:
            return f"<BPE:{token_id - cls.BPE_OFFSET}>"
        else:
            return f"<UNK:{token_id}>"

    @classmethod
    def vocab_size(cls, bpe_vocab_size: int) -> int:
        """Total vocabulary size (special + BPE)."""
        return cls.NUM_SPECIAL + bpe_vocab_size
