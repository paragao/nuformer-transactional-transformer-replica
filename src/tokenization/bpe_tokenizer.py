"""BPE tokenizer for transaction descriptions.

Trains a byte-pair encoding tokenizer on transaction description text.
Uses HuggingFace tokenizers library for fast tokenization.
Falls back to a simple whitespace tokenizer if tokenizers is not available.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from .special_tokens import SpecialTokens


class DescriptionTokenizer:
    """BPE tokenizer for transaction descriptions.

    If HuggingFace `tokenizers` library is available, uses BPE.
    Otherwise, falls back to a simple word-level tokenizer.
    """

    def __init__(self, vocab_size: int = 16000, tokenizer_path: Optional[str] = None):
        self.vocab_size = vocab_size
        self.bpe_offset = SpecialTokens.BPE_OFFSET
        self._tokenizer = None
        self._word_to_id: dict[str, int] = {}
        self._id_to_word: dict[int, str] = {}
        self._next_id = 0
        self._use_hf = False
        self._encode_cache: dict[str, list[int]] = {}

        if tokenizer_path and Path(tokenizer_path).exists():
            self._load(tokenizer_path)

    @classmethod
    def from_pretrained(cls, path: str) -> "DescriptionTokenizer":
        """Load a pre-trained tokenizer from a saved file.

        Args:
            path: Path to saved tokenizer.json file

        Returns:
            Loaded DescriptionTokenizer ready for encode/decode

        Raises:
            FileNotFoundError: If the tokenizer file doesn't exist
        """
        if not Path(path).exists():
            raise FileNotFoundError(f"Tokenizer not found at: {path}")
        instance = cls(tokenizer_path=path)
        return instance

    def train(self, texts: list[str], save_path: Optional[str] = None) -> None:
        """Train tokenizer on a corpus of description texts."""
        try:
            self._train_hf(texts, save_path)
            self._use_hf = True
        except ImportError:
            self._train_simple(texts)
            self._use_hf = False

    def encode(self, text: str) -> list[int]:
        """Encode text to token IDs (offset by BPE_OFFSET). Cached."""
        cached = self._encode_cache.get(text)
        if cached is not None:
            return cached

        if self._use_hf and self._tokenizer is not None:
            ids = self._tokenizer.encode(text).ids
            result = [token_id + self.bpe_offset for token_id in ids]
        else:
            result = self._encode_simple(text)

        self._encode_cache[text] = result
        return result

    def decode(self, token_ids: list[int]) -> str:
        """Decode token IDs back to text."""
        # Remove BPE offset
        raw_ids = [tid - self.bpe_offset for tid in token_ids if tid >= self.bpe_offset]

        if self._use_hf and self._tokenizer is not None:
            return self._tokenizer.decode(raw_ids)
        else:
            return self._decode_simple(raw_ids)

    def get_vocab_size(self) -> int:
        """Return actual vocabulary size."""
        if self._use_hf and self._tokenizer is not None:
            return self._tokenizer.get_vocab_size()
        return len(self._word_to_id)

    def total_vocab_size(self) -> int:
        """Return total vocab including special tokens."""
        return SpecialTokens.vocab_size(self.get_vocab_size())

    def save(self, path: str) -> None:
        """Save tokenizer to file."""
        if self._use_hf and self._tokenizer is not None:
            self._tokenizer.save(path)
        else:
            import json
            with open(path, "w") as f:
                json.dump({"word_to_id": self._word_to_id, "vocab_size": self.vocab_size}, f)

    def _load(self, path: str) -> None:
        """Load tokenizer from file."""
        try:
            from tokenizers import Tokenizer
            self._tokenizer = Tokenizer.from_file(path)
            self._use_hf = True
        except (ImportError, Exception):
            import json
            with open(path) as f:
                data = json.load(f)
            self._word_to_id = data["word_to_id"]
            self._id_to_word = {v: k for k, v in self._word_to_id.items()}
            self._next_id = max(self._word_to_id.values()) + 1 if self._word_to_id else 0
            self._use_hf = False

    def _train_hf(self, texts: list[str], save_path: Optional[str]) -> None:
        """Train HuggingFace BPE tokenizer."""
        from tokenizers import Tokenizer
        from tokenizers.models import BPE
        from tokenizers.trainers import BpeTrainer
        from tokenizers.pre_tokenizers import Whitespace

        tokenizer = Tokenizer(BPE(unk_token="[UNK]"))
        tokenizer.pre_tokenizer = Whitespace()

        trainer = BpeTrainer(
            vocab_size=self.vocab_size,
            special_tokens=["[UNK]", "[PAD]"],
            min_frequency=2,
            show_progress=True,
        )

        tokenizer.train_from_iterator(texts, trainer=trainer)
        self._tokenizer = tokenizer

        if save_path:
            tokenizer.save(save_path)

    def _train_simple(self, texts: list[str]) -> None:
        """Train simple word-level tokenizer (fallback)."""
        from collections import Counter

        word_counts = Counter()
        for text in texts:
            words = text.lower().replace("-", " ").replace("_", " ").split()
            word_counts.update(words)

        # Keep top vocab_size words
        most_common = word_counts.most_common(self.vocab_size - 1)
        self._word_to_id = {"<UNK>": 0}
        for i, (word, _) in enumerate(most_common, start=1):
            self._word_to_id[word] = i

        self._id_to_word = {v: k for k, v in self._word_to_id.items()}
        self._next_id = len(self._word_to_id)

    def _encode_simple(self, text: str) -> list[int]:
        """Simple word-level encoding."""
        words = text.lower().replace("-", " ").replace("_", " ").split()
        ids = []
        for word in words:
            token_id = self._word_to_id.get(word, 0)  # 0 = UNK
            ids.append(token_id + self.bpe_offset)
        return ids

    def _decode_simple(self, token_ids: list[int]) -> str:
        """Simple word-level decoding."""
        words = [self._id_to_word.get(tid, "<UNK>") for tid in token_ids]
        return " ".join(words)
