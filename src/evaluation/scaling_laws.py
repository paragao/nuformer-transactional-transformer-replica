"""Scaling laws analysis for nuFormer.

Implements analysis of:
1. Data scaling: How does performance change with training data size?
2. Context length scaling: Effect of sequence length on downstream metrics
3. Model size scaling: 24M vs 330M parameter efficiency
4. Compute-optimal analysis: Chinchilla-style tokens/params ratio

The paper shows:
- Log-linear improvement with data size (5M -> 100M rows)
- Diminishing returns beyond 2048 context length
- 330M significantly better than 24M at all data scales
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


# ---------------------------------------------------------------------------
# Scaling law data structures
# ---------------------------------------------------------------------------


@dataclass
class ScalingPoint:
    """A single data point in a scaling curve."""

    x_value: float  # independent variable (data size, context len, etc.)
    metric_value: float  # observed metric (AUC, loss, etc.)
    metric_name: str = "auc_roc"
    compute_flops: float = 0.0  # estimated training FLOPs
    wall_time_hours: float = 0.0


@dataclass
class ScalingCurve:
    """A scaling curve: metric vs. some scaling variable."""

    name: str
    x_label: str  # "Training Rows", "Context Length", etc.
    y_label: str  # "AUC-ROC", "Val Loss", etc.
    points: list[ScalingPoint] = field(default_factory=list)

    def x_values(self) -> np.ndarray:
        return np.array([p.x_value for p in self.points])

    def y_values(self) -> np.ndarray:
        return np.array([p.metric_value for p in self.points])

    def fit_power_law(self) -> dict[str, float]:
        """Fit y = a * x^b (power law) in log-log space.

        Returns dict with 'a', 'b', 'r_squared'.
        """
        x = np.log(self.x_values())
        y = np.log(self.y_values())

        # Linear fit in log-log space
        coeffs = np.polyfit(x, y, 1)
        b = coeffs[0]
        a = np.exp(coeffs[1])

        # R-squared
        y_pred = coeffs[0] * x + coeffs[1]
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"a": a, "b": b, "r_squared": r_squared}

    def fit_log_linear(self) -> dict[str, float]:
        """Fit y = a * log(x) + b (log-linear, common for AUC vs data).

        Returns dict with 'a', 'b', 'r_squared'.
        """
        x = np.log(self.x_values())
        y = self.y_values()

        coeffs = np.polyfit(x, y, 1)
        a = coeffs[0]
        b = coeffs[1]

        y_pred = a * x + b
        ss_res = np.sum((y - y_pred) ** 2)
        ss_tot = np.sum((y - y.mean()) ** 2)
        r_squared = 1 - ss_res / ss_tot if ss_tot > 0 else 0.0

        return {"a": a, "b": b, "r_squared": r_squared}

    def extrapolate_power_law(self, target_x: float) -> float:
        """Predict metric at target_x using fitted power law."""
        params = self.fit_power_law()
        return params["a"] * (target_x ** params["b"])

    def extrapolate_log_linear(self, target_x: float) -> float:
        """Predict metric at target_x using log-linear fit."""
        params = self.fit_log_linear()
        return params["a"] * np.log(target_x) + params["b"]


# ---------------------------------------------------------------------------
# Compute estimation
# ---------------------------------------------------------------------------


def estimate_training_flops(
    model_params: int,
    n_tokens: int,
    n_epochs: int = 1,
) -> float:
    """Estimate training FLOPs using the Kaplan et al. approximation.

    FLOPs ~ 6 * N * D (forward + backward, per token)
    Where N = model params, D = total tokens processed.
    """
    total_tokens = n_tokens * n_epochs
    return 6 * model_params * total_tokens


def compute_optimal_tokens(model_params: int, compute_budget_flops: float) -> int:
    """Chinchilla-optimal tokens given compute budget.

    Optimal ratio: D/N ~ 20 (Hoffmann et al., 2022)
    But for fine-tuning tasks the ratio is typically lower.
    """
    # D_optimal = C / (6 * N) from C = 6*N*D
    return int(compute_budget_flops / (6 * model_params))


def tokens_per_gpu_hour(
    model_params: int,
    gpu_type: str = "H200",
    dtype: str = "bfloat16",
) -> float:
    """Estimate tokens/second for a given GPU.

    Based on empirical measurements for causal LMs.
    """
    # Rough estimates (tok/s per GPU, batch saturated)
    throughput_map = {
        # (params, gpu) -> tok/s approximate
        ("24M", "H200"): 500_000,
        ("330M", "H200"): 80_000,
        ("330M", "A100"): 50_000,
        ("24M", "A100"): 300_000,
    }

    # Find closest match
    size_key = "24M" if model_params < 100_000_000 else "330M"
    key = (size_key, gpu_type)

    tps = throughput_map.get(key, 50_000)
    return tps * 3600  # tokens per hour


# ---------------------------------------------------------------------------
# Analysis helpers
# ---------------------------------------------------------------------------


def analyze_data_scaling(
    data_sizes: list[int],
    aucs: list[float],
) -> dict:
    """Analyze data scaling behavior.

    Returns:
        dict with scaling curve, fit parameters, and extrapolation
    """
    curve = ScalingCurve(
        name="Data Scaling",
        x_label="Training Rows",
        y_label="AUC-ROC",
    )
    for size, auc in zip(data_sizes, aucs):
        curve.points.append(ScalingPoint(x_value=size, metric_value=auc))

    log_linear = curve.fit_log_linear()

    return {
        "curve": curve,
        "fit": log_linear,
        "extrapolation_200M": curve.extrapolate_log_linear(200_000_000),
        "extrapolation_500M": curve.extrapolate_log_linear(500_000_000),
    }


def analyze_context_scaling(
    context_lengths: list[int],
    aucs: list[float],
) -> dict:
    """Analyze context length scaling behavior."""
    curve = ScalingCurve(
        name="Context Length Scaling",
        x_label="Context Length (tokens)",
        y_label="AUC-ROC",
    )
    for ctx, auc in zip(context_lengths, aucs):
        curve.points.append(ScalingPoint(x_value=ctx, metric_value=auc))

    return {
        "curve": curve,
        "fit": curve.fit_log_linear(),
        "improvement_512_to_4096": (aucs[-1] - aucs[0]) / aucs[0] * 100
        if len(aucs) >= 2 else 0.0,
    }


def compute_efficiency_frontier(
    experiments: list[dict],  # each has 'flops', 'auc', 'name'
) -> list[dict]:
    """Identify Pareto-optimal experiments on the compute-performance frontier.

    An experiment is on the frontier if no other experiment achieves
    better performance with less compute.
    """
    # Sort by compute
    sorted_exps = sorted(experiments, key=lambda x: x["flops"])

    frontier = []
    best_auc = -1.0

    for exp in sorted_exps:
        if exp["auc"] > best_auc:
            frontier.append(exp)
            best_auc = exp["auc"]

    return frontier
