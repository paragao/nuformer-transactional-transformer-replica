"""DCNv2: Deep & Cross Network v2 for tabular feature processing.

Implements the cross network from Wang et al. (2021):
"DCN V2: Improved Deep & Cross Network and Practical Lessons for
Web-Scale Learning to Rank Systems"

Used in nuFormer for processing tabular features before fusion
with transformer embeddings.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn


@dataclass
class DCNv2Config:
    """Configuration for DCNv2."""

    input_dim: int = 291  # Number of tabular features
    cross_layers: int = 3
    deep_layers: list[int] = None  # Default: [512, 256]
    output_dim: int = 128  # Projection dimension for fusion
    dropout: float = 0.1
    weight_decay: float = 0.01  # Important for regularization per paper

    # PLR embedding (Gorishniy et al. 2022) — applied before cross/deep networks
    use_plr: bool = True
    plr_dim: int = 8  # Embedding dim per feature (4 sin + 4 cos)
    plr_frequencies: int = 4  # Number of frequency pairs

    def __post_init__(self):
        if self.deep_layers is None:
            self.deep_layers = [512, 256]


class CrossLayer(nn.Module):
    """Single cross layer from DCNv2.

    x_{l+1} = x_0 * (W_l * x_l + b_l) + x_l
    """

    def __init__(self, dim: int, dropout: float = 0.1):
        super().__init__()
        self.weight = nn.Linear(dim, dim, bias=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x0: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        """Cross layer forward.

        Args:
            x0: Original input (B, D)
            x: Current layer input (B, D)

        Returns:
            x_{l+1} = x0 * (W * x + b) + x
        """
        cross = x0 * self.dropout(self.weight(x))
        return cross + x


class DCNv2(nn.Module):
    """Deep & Cross Network v2 with optional PLR embeddings.

    Combines cross network (explicit feature interactions) with
    deep network (implicit interactions) for tabular features.

    Architecture:
        Input -> [PLR Embedding] -> [CrossNetwork || DeepNetwork] -> Concat -> OutputProjection

    When use_plr=True, raw numerical features are first embedded via
    periodic linear representations (learned sin/cos frequencies per feature),
    then projected back to input_dim before feeding into cross/deep networks.
    """

    def __init__(self, config: DCNv2Config):
        super().__init__()
        self.config = config

        # PLR embedding (optional, applied before cross/deep)
        if config.use_plr:
            from .numerical_embeddings import NumericalFeatureEmbedder
            self.plr_embedder = NumericalFeatureEmbedder(
                n_numerical_features=config.input_dim,
                embedding_dim_per_feature=config.plr_dim,
                n_frequencies=config.plr_frequencies,
            )
            # Project PLR output (input_dim * plr_dim) back to input_dim
            plr_output_dim = config.input_dim * config.plr_dim
            self.plr_proj = nn.Linear(plr_output_dim, config.input_dim)
            self.input_norm = nn.LayerNorm(config.input_dim)
        else:
            self.plr_embedder = None
            self.plr_proj = None
            self.input_norm = nn.BatchNorm1d(config.input_dim)

        # Cross network
        self.cross_layers = nn.ModuleList([
            CrossLayer(config.input_dim, dropout=config.dropout)
            for _ in range(config.cross_layers)
        ])

        # Deep network
        deep_dims = [config.input_dim] + config.deep_layers
        deep_modules = []
        for i in range(len(deep_dims) - 1):
            deep_modules.extend([
                nn.Linear(deep_dims[i], deep_dims[i + 1]),
                nn.ReLU(),
                nn.Dropout(config.dropout),
            ])
        self.deep_network = nn.Sequential(*deep_modules)

        # Output projection
        concat_dim = config.input_dim + config.deep_layers[-1]
        self.output_proj = nn.Sequential(
            nn.Linear(concat_dim, config.output_dim),
            nn.ReLU(),
            nn.Dropout(config.dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward pass.

        Args:
            x: Tabular features (B, input_dim)

        Returns:
            Feature embedding (B, output_dim)
        """
        # PLR embedding: (B, 291) -> embed -> (B, 291*8) -> project -> (B, 291)
        if self.plr_embedder is not None:
            x = self.plr_embedder(x)
            x = self.plr_proj(x)

        # Normalize
        x = self.input_norm(x)

        # Cross network
        x0 = x
        x_cross = x
        for layer in self.cross_layers:
            x_cross = layer(x0, x_cross)

        # Deep network
        x_deep = self.deep_network(x)

        # Concatenate and project
        combined = torch.cat([x_cross, x_deep], dim=-1)
        return self.output_proj(combined)
