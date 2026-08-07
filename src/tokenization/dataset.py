"""PyTorch Dataset for tokenized transaction sequences.

Supports:
- Pre-training: Next Token Prediction (NTP) with causal masking
- Fine-tuning: Sequence classification (final token -> label)
- Joint fusion: Sequence + tabular features -> label
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np


class TransactionDataset:
    """PyTorch-compatible dataset for transaction sequences.

    Stores tokenized sequences as memory-mapped numpy arrays for
    efficient random access without loading everything into RAM.

    Works without torch import (for data preparation steps).
    For PyTorch training, wrap with torch.utils.data.Dataset.
    """

    def __init__(
        self,
        sequences_path: str,
        labels_path: Optional[str] = None,
        features_path: Optional[str] = None,
        max_seq_len: int = 2048,
    ):
        self.max_seq_len = max_seq_len
        self.sequences = np.load(sequences_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r") if labels_path else None
        self.features = np.load(features_path, mmap_mode="r") if features_path else None

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict:
        """Get a single sample.

        Returns dict with:
            - input_ids: token sequence (max_seq_len,)
            - attention_mask: 1 for real tokens, 0 for PAD
            - labels: (optional) binary classification label
            - tabular_features: (optional) 291-dim feature vector
        """
        seq = np.array(self.sequences[idx], dtype=np.int64)

        # Create attention mask (assume PAD_TOKEN=74)
        attention_mask = (seq != 74).astype(np.int64)

        sample = {
            "input_ids": seq,
            "attention_mask": attention_mask,
        }

        if self.labels is not None:
            sample["labels"] = np.array(self.labels[idx], dtype=np.int64)

        if self.features is not None:
            sample["tabular_features"] = np.array(self.features[idx], dtype=np.float32)

        return sample

    @staticmethod
    def create_memmap(
        sequences: list[list[int]],
        output_path: str,
        max_seq_len: int = 2048,
        pad_token: int = 74,
    ) -> None:
        """Create numpy array from token sequences and save to disk.

        Pads/truncates all sequences to max_seq_len.
        """
        n = len(sequences)
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        # Create array (pad with pad_token)
        arr = np.full((n, max_seq_len), pad_token, dtype=np.int32)

        # Write sequences
        for i, seq in enumerate(sequences):
            length = min(len(seq), max_seq_len)
            arr[i, :length] = seq[:length]

        # Save as .npy
        np.save(str(output_path), arr)

        # Save metadata
        meta = {"n_samples": n, "max_seq_len": max_seq_len, "dtype": "int32"}
        import json
        with open(str(output_path) + ".meta.json", "w") as f:
            json.dump(meta, f)

    @staticmethod
    def create_labels_memmap(labels: list[int], output_path: str) -> None:
        """Create memory-mapped numpy array for labels."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        arr = np.array(labels, dtype=np.int32)
        np.save(str(output_path), arr)

    @staticmethod
    def create_features_memmap(features: list[list[float]], output_path: str) -> None:
        """Create memory-mapped numpy array for tabular features."""
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        arr = np.array(features, dtype=np.float32)
        np.save(str(output_path), arr)
