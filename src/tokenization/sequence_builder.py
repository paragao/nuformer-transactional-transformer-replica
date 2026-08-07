"""Sequence builder: converts transaction lists into token sequences.

Implements the paper's tokenization:
    tau(t) = [sign_token, amt_token, month_token, day_token, weekday_token] + BPE(desc)

User sequence: BOS + tau(t_1) + SEP + tau(t_2) + SEP + ... + tau(t_n) + EOS
Truncated to max_seq_len (most recent transactions kept).
"""

from __future__ import annotations

from datetime import datetime
from typing import Optional

import numpy as np

from .special_tokens import SpecialTokens
from .bpe_tokenizer import DescriptionTokenizer


class SequenceBuilder:
    """Build token sequences from user transaction histories."""

    def __init__(
        self,
        description_tokenizer: DescriptionTokenizer,
        max_seq_len: int = 2048,
        max_desc_tokens: int = 8,
    ):
        self.desc_tokenizer = description_tokenizer
        self.max_seq_len = max_seq_len
        self.max_desc_tokens = max_desc_tokens

    def tokenize_transaction(self, amount: float, timestamp: datetime, description: str) -> list[int]:
        """Tokenize a single transaction into token IDs.

        Returns: [sign, bucket, month, day, weekday, desc_tok_1, ..., desc_tok_n]
        """
        tokens = [
            SpecialTokens.amount_sign_token(amount),
            SpecialTokens.amount_bucket_token(amount),
            SpecialTokens.month_token(timestamp.month),
            SpecialTokens.day_token(timestamp.day),
            SpecialTokens.weekday_token(timestamp.weekday()),
        ]

        # BPE encode description (truncate to max_desc_tokens)
        desc_tokens = self.desc_tokenizer.encode(description)
        tokens.extend(desc_tokens[:self.max_desc_tokens])

        return tokens

    def build_sequence(
        self,
        transactions: list[dict],
        add_bos: bool = True,
        add_eos: bool = True,
    ) -> list[int]:
        """Build a full token sequence from a list of transactions.

        Args:
            transactions: List of dicts with keys: amount, timestamp, description
                         Assumed to be sorted by timestamp (oldest first).
            add_bos: Whether to prepend BOS token.
            add_eos: Whether to append EOS token.

        Returns:
            Token ID sequence, truncated to max_seq_len (keeping most recent).
        """
        # Tokenize all transactions
        all_token_groups = []
        for txn in transactions:
            tokens = self.tokenize_transaction(
                amount=txn["amount"],
                timestamp=txn["timestamp"],
                description=txn["description"],
            )
            all_token_groups.append(tokens)

        # Build sequence from most recent (reverse, fill budget, then reverse back)
        sequence = []
        budget = self.max_seq_len - (1 if add_bos else 0) - (1 if add_eos else 0)

        # Start from most recent transaction
        for tokens in reversed(all_token_groups):
            # tokens + SEP
            needed = len(tokens) + 1  # +1 for SEP
            if len(sequence) + needed > budget:
                break
            # Prepend: SEP + tokens (we'll reverse at the end)
            sequence.extend(reversed(tokens))
            sequence.append(SpecialTokens.SEP_TOKEN)

        # Reverse to get chronological order
        sequence.reverse()

        # Remove leading SEP if present
        if sequence and sequence[0] == SpecialTokens.SEP_TOKEN:
            sequence = sequence[1:]

        # Add BOS/EOS
        if add_bos:
            sequence = [SpecialTokens.BOS_TOKEN] + sequence
        if add_eos:
            sequence = sequence + [SpecialTokens.EOS_TOKEN]

        return sequence

    def pad_sequence(self, sequence: list[int], target_len: Optional[int] = None) -> list[int]:
        """Pad sequence to target_len with PAD tokens."""
        target = target_len or self.max_seq_len
        if len(sequence) >= target:
            return sequence[:target]
        padding = [SpecialTokens.PAD_TOKEN] * (target - len(sequence))
        return sequence + padding

    def create_attention_mask(self, sequence: list[int]) -> list[int]:
        """Create attention mask (1 for real tokens, 0 for PAD)."""
        return [0 if t == SpecialTokens.PAD_TOKEN else 1 for t in sequence]

    def decode_sequence(self, token_ids: list[int]) -> str:
        """Decode a token sequence to human-readable string."""
        parts = []
        for tid in token_ids:
            if tid >= SpecialTokens.BPE_OFFSET:
                parts.append(self.desc_tokenizer.decode([tid]))
            else:
                parts.append(SpecialTokens.decode_token(tid))
        return " ".join(parts)
