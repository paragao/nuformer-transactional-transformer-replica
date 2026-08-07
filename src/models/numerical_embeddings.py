"""Numerical Embeddings using Periodic Activations.

Implements the approach from Gorishniy et al. (2022):
"On Embeddings for Numerical Features in Tabular Deep Learning"

Key insight: Embedding numerical features with learned periodic functions
(sin/cos at different frequencies) significantly improves DNN performance
on tabular data, enabling parity with gradient-boosted trees.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn


class NumericalEmbedding(nn.Module):
    """Embed a single numerical feature using periodic activations.

    For each numerical feature x, produces an embedding:
        e(x) = [sin(2*pi*f_1*x + phi_1), cos(2*pi*f_1*x + phi_1),
                sin(2*pi*f_2*x + phi_2), cos(2*pi*f_2*x + phi_2),
                ..., linear(x)]

    Where f_i are learned frequencies and phi_i are learned phases.
    """

    def __init__(self, embedding_dim: int = 16, n_frequencies: int = 8):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.n_frequencies = n_frequencies

        # Learnable frequencies and phases
        self.frequencies = nn.Parameter(torch.randn(n_frequencies) * 0.1)
        self.phases = nn.Parameter(torch.zeros(n_frequencies))

        # Linear component
        linear_dim = embedding_dim - 2 * n_frequencies
        assert linear_dim >= 0, "embedding_dim must be >= 2 * n_frequencies"
        if linear_dim > 0:
            self.linear = nn.Linear(1, linear_dim)
        else:
            self.linear = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed a single numerical feature.

        Args:
            x: (B,) or (B, 1) numerical values

        Returns:
            (B, embedding_dim) embedded values
        """
        if x.dim() == 1:
            x = x.unsqueeze(-1)  # (B, 1)

        # Periodic components: 2*pi*f*x + phi
        # x: (B, 1), frequencies: (n_freq,) -> (B, n_freq)
        angles = 2 * math.pi * x * self.frequencies.unsqueeze(0) + self.phases.unsqueeze(0)

        # Sin and cos: (B, 2*n_freq)
        periodic = torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)

        if self.linear is not None:
            linear_out = self.linear(x)
            return torch.cat([periodic, linear_out], dim=-1)
        else:
            return periodic


class NumericalFeatureEmbedder(nn.Module):
    """Embed all numerical features in a tabular dataset.

    Each numerical feature gets its own NumericalEmbedding, then
    all embeddings are concatenated.
    """

    def __init__(
        self,
        n_numerical_features: int,
        embedding_dim_per_feature: int = 16,
        n_frequencies: int = 8,
    ):
        super().__init__()
        self.n_features = n_numerical_features
        self.embedding_dim = embedding_dim_per_feature
        self.output_dim = n_numerical_features * embedding_dim_per_feature

        self.embeddings = nn.ModuleList([
            NumericalEmbedding(embedding_dim_per_feature, n_frequencies)
            for _ in range(n_numerical_features)
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed all numerical features.

        Args:
            x: (B, n_features) numerical feature values

        Returns:
            (B, n_features * embedding_dim) concatenated embeddings
        """
        embedded = [self.embeddings[i](x[:, i]) for i in range(self.n_features)]
        return torch.cat(embedded, dim=-1)


class CategoricalFeatureEmbedder(nn.Module):
    """Embed categorical features using trainable lookup tables.

    Each categorical feature gets its own embedding table.
    """

    def __init__(
        self,
        n_categories_per_feature: list[int],
        embedding_dim: int = 16,
    ):
        super().__init__()
        self.n_features = len(n_categories_per_feature)
        self.embedding_dim = embedding_dim
        self.output_dim = self.n_features * embedding_dim

        self.embeddings = nn.ModuleList([
            nn.Embedding(n_cats + 1, embedding_dim)  # +1 for unknown
            for n_cats in n_categories_per_feature
        ])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Embed all categorical features.

        Args:
            x: (B, n_features) integer category indices

        Returns:
            (B, n_features * embedding_dim) concatenated embeddings
        """
        embedded = [self.embeddings[i](x[:, i].long()) for i in range(self.n_features)]
        return torch.cat(embedded, dim=-1)


class TabularFeatureEncoder(nn.Module):
    """Complete tabular feature encoder.

    Combines numerical embeddings (periodic activations) and
    categorical embeddings (lookup tables), then projects to a
    fixed-size representation suitable for DCNv2 input.
    """

    def __init__(
        self,
        n_numerical: int = 280,
        n_categorical: int = 11,
        n_categories_per_feature: list[int] = None,
        num_embedding_dim: int = 8,
        cat_embedding_dim: int = 8,
        output_dim: int = 291,
    ):
        super().__init__()

        if n_categories_per_feature is None:
            n_categories_per_feature = [10] * n_categorical  # default 10 categories each

        # Numerical embeddings
        self.numerical_embedder = NumericalFeatureEmbedder(
            n_numerical_features=n_numerical,
            embedding_dim_per_feature=num_embedding_dim,
            n_frequencies=4,
        )

        # Categorical embeddings
        self.categorical_embedder = CategoricalFeatureEmbedder(
            n_categories_per_feature=n_categories_per_feature,
            embedding_dim=cat_embedding_dim,
        )

        # Project to output_dim
        total_emb_dim = self.numerical_embedder.output_dim + self.categorical_embedder.output_dim
        self.projection = nn.Sequential(
            nn.Linear(total_emb_dim, output_dim),
            nn.ReLU(),
        )

    def forward(self, numerical: torch.Tensor, categorical: torch.Tensor) -> torch.Tensor:
        """Encode tabular features.

        Args:
            numerical: (B, n_numerical) float features
            categorical: (B, n_categorical) integer features

        Returns:
            (B, output_dim) encoded features
        """
        num_emb = self.numerical_embedder(numerical)
        cat_emb = self.categorical_embedder(categorical)
        combined = torch.cat([num_emb, cat_emb], dim=-1)
        return self.projection(combined)
