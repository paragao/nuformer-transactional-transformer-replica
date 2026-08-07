"""nuFormer: Joint Fusion Model.

Combines:
1. Causal Transformer (transaction embeddings)
2. DCNv2 (tabular feature processing)
3. MLP prediction head

End-to-end training: Transformer + DCNv2 + MLP trained jointly.
Key design from paper:
- Transaction embeddings are normalized (LayerNorm) before concatenation
- DCNv2 processes embedded tabular features, projects to low-dim
- Concatenation (not addition) of feature and transaction embeddings
- Regularization: weight decay + dropout on DCNv2 cross layers
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .transformer import TransactionTransformer, TransformerConfig
from .dcnv2 import DCNv2, DCNv2Config


@dataclass
class NuFormerConfig:
    """Configuration for the full nuFormer model."""

    # Transformer config
    transformer: TransformerConfig = None

    # DCNv2 config
    dcnv2: DCNv2Config = None

    # Fusion config
    transaction_emb_dim: int = 1024  # = d_model
    feature_emb_dim: int = 128  # DCNv2 output dim
    fusion_hidden_dim: int = 256
    num_classes: int = 2  # binary classification
    fusion_dropout: float = 0.1

    # Embedding normalization (key insight from paper)
    normalize_transaction_emb: bool = True

    def __post_init__(self):
        if self.transformer is None:
            self.transformer = TransformerConfig()
        if self.dcnv2 is None:
            self.dcnv2 = DCNv2Config()
        # Ensure dimensions match
        self.transaction_emb_dim = self.transformer.d_model
        self.feature_emb_dim = self.dcnv2.output_dim


class NuFormer(nn.Module):
    """nuFormer: Joint Fusion of Transaction Transformer + DCNv2.

    Architecture:
        Transaction Sequence -> Transformer -> user_embedding (d_model)
                                                    |
                                                LayerNorm
                                                    |
        Tabular Features -> DCNv2 -> feature_embedding (128)
                                                    |
                                [user_emb || feature_emb] -> MLP -> Prediction
    """

    def __init__(self, config: NuFormerConfig):
        super().__init__()
        self.config = config

        # Transaction transformer
        self.transformer = TransactionTransformer(config.transformer)

        # DCNv2 for tabular features
        self.dcnv2 = DCNv2(config.dcnv2)

        # Transaction embedding normalization (key paper insight)
        if config.normalize_transaction_emb:
            self.emb_norm = nn.LayerNorm(config.transaction_emb_dim)
        else:
            self.emb_norm = nn.Identity()

        # Fusion MLP: [transaction_emb || feature_emb] -> prediction
        fusion_input_dim = config.transaction_emb_dim + config.feature_emb_dim
        self.fusion_mlp = nn.Sequential(
            nn.Linear(fusion_input_dim, config.fusion_hidden_dim),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim, config.fusion_hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(config.fusion_dropout),
            nn.Linear(config.fusion_hidden_dim // 2, config.num_classes),
        )

        # Initialize fusion MLP
        for module in self.fusion_mlp:
            if isinstance(module, nn.Linear):
                nn.init.xavier_uniform_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(
        self,
        input_ids: torch.Tensor,
        tabular_features: torch.Tensor,
        attention_mask: torch.Tensor = None,
    ) -> dict[str, torch.Tensor]:
        """Forward pass for joint fusion.

        Args:
            input_ids: (B, T) token IDs for transaction sequence
            tabular_features: (B, n_features) tabular feature values
            attention_mask: (B, T) binary attention mask

        Returns:
            dict with 'logits' (B, num_classes) and intermediate embeddings
        """
        # Get transaction embedding (final token)
        transaction_emb = self.transformer.get_user_embedding(input_ids, attention_mask)
        transaction_emb = self.emb_norm(transaction_emb)  # (B, d_model)

        # Get feature embedding from DCNv2
        feature_emb = self.dcnv2(tabular_features)  # (B, feature_emb_dim)

        # Concatenate and predict
        fused = torch.cat([transaction_emb, feature_emb], dim=-1)
        logits = self.fusion_mlp(fused)

        return {
            "logits": logits,
            "transaction_embedding": transaction_emb,
            "feature_embedding": feature_emb,
        }

    def num_parameters(self) -> dict[str, int]:
        """Count parameters by component."""
        return {
            "transformer": sum(p.numel() for p in self.transformer.parameters()),
            "dcnv2": sum(p.numel() for p in self.dcnv2.parameters()),
            "fusion_mlp": sum(p.numel() for p in self.fusion_mlp.parameters()),
            "emb_norm": sum(p.numel() for p in self.emb_norm.parameters()),
            "total": sum(p.numel() for p in self.parameters()),
        }


class FineTuneHead(nn.Module):
    """Simple fine-tuning head for classification (no tabular features).

    Used for fine-tuning the transformer alone before joint fusion.
    """

    def __init__(self, d_model: int = 1024, num_classes: int = 2, dropout: float = 0.1):
        super().__init__()
        self.norm = nn.LayerNorm(d_model)
        self.head = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, num_classes),
        )

    def forward(self, embedding: torch.Tensor) -> torch.Tensor:
        """Predict from user embedding.

        Args:
            embedding: (B, d_model) user embedding from transformer

        Returns:
            (B, num_classes) logits
        """
        return self.head(self.norm(embedding))
