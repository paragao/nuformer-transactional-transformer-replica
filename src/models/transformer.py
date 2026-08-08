"""Causal Transformer with NoPE (No Positional Embeddings).

Implements the paper's architecture:
- Decoder-only transformer (GPT-style)
- No positional embeddings (causal mask provides implicit position)
- FlashAttention support (falls back to scaled dot-product attention)
- torch.compile compatible

330M Configuration:
    d_model=1024, n_layers=24, n_heads=16, d_ff=4096
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass
class TransformerConfig:
    """Configuration for the causal transformer."""

    vocab_size: int = 24078  # 78 special + 24000 BPE
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 4096
    dropout: float = 0.1
    max_seq_len: int = 4096
    layer_norm_eps: float = 1e-5
    use_flash_attention: bool = True
    activation: str = "gelu"  # gelu or swiglu


class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization."""

    def __init__(self, dim: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        norm = torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)
        return x * norm * self.weight


class CausalSelfAttention(nn.Module):
    """Multi-head causal self-attention (NoPE - no positional embeddings)."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        assert config.d_model % config.n_heads == 0

        self.n_heads = config.n_heads
        self.d_head = config.d_model // config.n_heads
        self.d_model = config.d_model
        self.use_flash = config.use_flash_attention

        # QKV projection (fused for efficiency)
        self.qkv_proj = nn.Linear(config.d_model, 3 * config.d_model, bias=False)
        self.out_proj = nn.Linear(config.d_model, config.d_model, bias=False)
        self.attn_dropout = nn.Dropout(config.dropout)
        self.resid_dropout = nn.Dropout(config.dropout)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        B, T, C = x.shape

        # QKV projection
        qkv = self.qkv_proj(x)
        q, k, v = qkv.chunk(3, dim=-1)

        # Reshape for multi-head attention: (B, T, C) -> (B, n_heads, T, d_head)
        q = q.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        k = k.view(B, T, self.n_heads, self.d_head).transpose(1, 2)
        v = v.view(B, T, self.n_heads, self.d_head).transpose(1, 2)

        # Use PyTorch's scaled_dot_product_attention (supports FlashAttention backend)
        attn_out = F.scaled_dot_product_attention(
            q, k, v,
            attn_mask=attention_mask,
            dropout_p=self.attn_dropout.p if self.training else 0.0,
            is_causal=True if attention_mask is None else False,
        )

        # Reshape back: (B, n_heads, T, d_head) -> (B, T, C)
        attn_out = attn_out.transpose(1, 2).contiguous().view(B, T, C)

        return self.resid_dropout(self.out_proj(attn_out))


class FeedForward(nn.Module):
    """Feed-forward network with GELU activation."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.fc1 = nn.Linear(config.d_model, config.d_ff, bias=False)
        self.fc2 = nn.Linear(config.d_ff, config.d_model, bias=False)
        self.dropout = nn.Dropout(config.dropout)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.fc2(self.activation(self.fc1(x))))


class TransformerBlock(nn.Module):
    """Single transformer block: Attention + FFN with pre-norm."""

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.attn_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.attn = CausalSelfAttention(config)
        self.ffn_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)
        self.ffn = FeedForward(config)

    def forward(self, x: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), attention_mask)
        x = x + self.ffn(self.ffn_norm(x))
        return x


class TransactionTransformer(nn.Module):
    """Causal transformer for transaction sequences (NoPE).

    Architecture:
    - Token embedding (no positional embedding)
    - N transformer blocks (pre-norm)
    - Final layer norm
    - LM head (for pre-training) or embedding output (for fine-tuning)
    """

    def __init__(self, config: TransformerConfig):
        super().__init__()
        self.config = config

        # Token embedding (NoPE: no positional embedding)
        self.token_embedding = nn.Embedding(config.vocab_size, config.d_model)

        # Transformer blocks
        self.blocks = nn.ModuleList([TransformerBlock(config) for _ in range(config.n_layers)])

        # Final normalization
        self.final_norm = RMSNorm(config.d_model, eps=config.layer_norm_eps)

        # LM head (tied with embedding for pre-training)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        # Weight tying
        self.lm_head.weight = self.token_embedding.weight

        # Embedding dropout
        self.embed_dropout = nn.Dropout(config.dropout)

        # Initialize weights
        self.apply(self._init_weights)

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with scaled normal distribution."""
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        return_embeddings: bool = False,
    ) -> dict[str, torch.Tensor]:
        """Forward pass.

        Args:
            input_ids: (B, T) token IDs
            attention_mask: (B, T) binary mask (1=attend, 0=ignore)
            return_embeddings: If True, return hidden states instead of logits

        Returns:
            dict with 'logits' (B, T, V) or 'embeddings' (B, T, D)
        """
        B, T = input_ids.shape

        # Token embedding (NoPE: no positional encoding)
        x = self.embed_dropout(self.token_embedding(input_ids))

        # Use causal attention (padding is handled via ignore_index in loss)
        # Don't create combined masks to avoid NaN from all-masked positions
        attn_mask = None

        # Transformer blocks
        for block in self.blocks:
            x = block(x, attn_mask)

        # Final norm
        x = self.final_norm(x)

        if return_embeddings:
            return {"embeddings": x}

        # LM head for next-token prediction
        logits = self.lm_head(x)
        return {"logits": logits}

    def get_user_embedding(self, input_ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Get the final token embedding (user embedding for fine-tuning).

        Returns the embedding of the last non-PAD token for each sequence.
        Shape: (B, d_model)
        """
        output = self.forward(input_ids, attention_mask, return_embeddings=True)
        embeddings = output["embeddings"]  # (B, T, D)

        if attention_mask is not None:
            # Find last non-pad position
            lengths = attention_mask.sum(dim=1).long() - 1  # (B,)
            lengths = lengths.clamp(min=0)
            user_emb = embeddings[torch.arange(embeddings.size(0)), lengths]
        else:
            # Use last position
            user_emb = embeddings[:, -1, :]

        return user_emb

    def _make_attention_mask(self, padding_mask: torch.Tensor, seq_len: int) -> Optional[torch.Tensor]:
        """Create combined causal + padding attention mask.

        When all tokens are real (no padding), returns None to use efficient causal mode.
        """
        # Check if any padding exists
        if padding_mask.all():
            return None  # No padding, use is_causal=True

        # Create causal mask: (1, 1, T, T)
        causal = torch.triu(
            torch.full((seq_len, seq_len), float("-inf"), device=padding_mask.device),
            diagonal=1,
        ).unsqueeze(0).unsqueeze(0)

        # Padding mask: (B, 1, 1, T) - mask out padding keys
        pad_mask = padding_mask.unsqueeze(1).unsqueeze(2)  # (B, 1, 1, T)
        pad_mask = (1.0 - pad_mask.float()) * float("-inf")

        # Combine: causal + padding
        return causal + pad_mask

    def num_parameters(self, exclude_embedding: bool = False) -> int:
        """Count total parameters."""
        total = sum(p.numel() for p in self.parameters())
        if exclude_embedding:
            total -= self.token_embedding.weight.numel()
        return total
