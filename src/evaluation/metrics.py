"""Evaluation metrics for nuFormer.

Implements the metrics used in the paper:
- AUC-ROC (primary metric, reported as relative % vs baseline)
- Average Precision (AP / PR-AUC)
- Accuracy, F1
- Calibration (Brier score, reliability diagram data)
- Lift curves (for business impact measurement)

All metrics operate on (labels, probabilities) pairs and support
both single-run and bootstrap confidence interval estimation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


@dataclass
class EvalResult:
    """Container for evaluation results."""

    auc_roc: float = 0.0
    average_precision: float = 0.0
    accuracy: float = 0.0
    f1: float = 0.0
    brier_score: float = 0.0
    log_loss: float = 0.0
    n_samples: int = 0
    positive_rate: float = 0.0

    # Optional: confidence intervals (from bootstrap)
    auc_ci_low: Optional[float] = None
    auc_ci_high: Optional[float] = None

    # Lift at various thresholds
    lift_at_10pct: float = 0.0  # lift in top 10% of scores
    lift_at_20pct: float = 0.0

    def to_dict(self) -> dict[str, float]:
        """Convert to flat dict for logging."""
        d = {
            "auc_roc": self.auc_roc,
            "average_precision": self.average_precision,
            "accuracy": self.accuracy,
            "f1": self.f1,
            "brier_score": self.brier_score,
            "log_loss": self.log_loss,
            "n_samples": self.n_samples,
            "positive_rate": self.positive_rate,
            "lift_at_10pct": self.lift_at_10pct,
            "lift_at_20pct": self.lift_at_20pct,
        }
        if self.auc_ci_low is not None:
            d["auc_ci_low"] = self.auc_ci_low
            d["auc_ci_high"] = self.auc_ci_high
        return d

    def relative_auc(self, baseline_auc: float) -> float:
        """Compute relative AUC improvement (paper's primary metric)."""
        if baseline_auc == 0:
            return 0.0
        return (self.auc_roc - baseline_auc) / baseline_auc * 100


# ---------------------------------------------------------------------------
# Core metrics computation
# ---------------------------------------------------------------------------


def compute_metrics(
    labels: np.ndarray,
    probabilities: np.ndarray,
    threshold: float = 0.5,
) -> EvalResult:
    """Compute all evaluation metrics.

    Args:
        labels: (N,) binary ground truth labels
        probabilities: (N,) predicted probability of positive class
        threshold: classification threshold for accuracy/F1

    Returns:
        EvalResult with all metrics populated
    """
    from sklearn.metrics import (
        roc_auc_score,
        average_precision_score,
        accuracy_score,
        f1_score,
        brier_score_loss,
        log_loss,
    )

    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities).ravel()

    # Binary predictions
    predictions = (probabilities >= threshold).astype(int)

    # Core metrics
    result = EvalResult(
        auc_roc=roc_auc_score(labels, probabilities),
        average_precision=average_precision_score(labels, probabilities),
        accuracy=accuracy_score(labels, predictions),
        f1=f1_score(labels, predictions, zero_division=0),
        brier_score=brier_score_loss(labels, probabilities),
        log_loss=log_loss(labels, probabilities),
        n_samples=len(labels),
        positive_rate=labels.mean(),
    )

    # Lift computation
    result.lift_at_10pct = _compute_lift(labels, probabilities, top_pct=0.10)
    result.lift_at_20pct = _compute_lift(labels, probabilities, top_pct=0.20)

    return result


def compute_metrics_with_ci(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bootstrap: int = 1000,
    ci_level: float = 0.95,
    seed: int = 42,
) -> EvalResult:
    """Compute metrics with bootstrap confidence intervals on AUC.

    Args:
        labels: (N,) binary labels
        probabilities: (N,) predicted probabilities
        n_bootstrap: number of bootstrap resamples
        ci_level: confidence level (default 95%)
        seed: random seed for reproducibility

    Returns:
        EvalResult with CI bounds on AUC
    """
    from sklearn.metrics import roc_auc_score

    # Base metrics
    result = compute_metrics(labels, probabilities)

    # Bootstrap AUC
    rng = np.random.default_rng(seed)
    n = len(labels)
    auc_samples = []

    for _ in range(n_bootstrap):
        idx = rng.integers(0, n, size=n)
        boot_labels = labels[idx]
        boot_probs = probabilities[idx]

        # Need both classes in bootstrap sample
        if len(np.unique(boot_labels)) < 2:
            continue
        auc_samples.append(roc_auc_score(boot_labels, boot_probs))

    auc_samples = np.array(auc_samples)
    alpha = (1 - ci_level) / 2
    result.auc_ci_low = float(np.percentile(auc_samples, 100 * alpha))
    result.auc_ci_high = float(np.percentile(auc_samples, 100 * (1 - alpha)))

    return result


# ---------------------------------------------------------------------------
# Lift and calibration utilities
# ---------------------------------------------------------------------------


def _compute_lift(labels: np.ndarray, probabilities: np.ndarray, top_pct: float) -> float:
    """Compute lift: positive rate in top-scoring segment vs overall."""
    n_top = max(1, int(len(labels) * top_pct))
    top_idx = np.argsort(probabilities)[-n_top:]
    top_positive_rate = labels[top_idx].mean()
    overall_positive_rate = labels.mean()

    if overall_positive_rate == 0:
        return 0.0
    return top_positive_rate / overall_positive_rate


def calibration_curve(
    labels: np.ndarray,
    probabilities: np.ndarray,
    n_bins: int = 10,
) -> dict[str, np.ndarray]:
    """Compute calibration curve (reliability diagram data).

    Returns:
        dict with 'bin_centers', 'true_freq', 'mean_predicted', 'bin_counts'
    """
    labels = np.asarray(labels).ravel()
    probabilities = np.asarray(probabilities).ravel()

    bin_edges = np.linspace(0, 1, n_bins + 1)
    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

    true_freq = np.zeros(n_bins)
    mean_predicted = np.zeros(n_bins)
    bin_counts = np.zeros(n_bins, dtype=int)

    for i in range(n_bins):
        mask = (probabilities >= bin_edges[i]) & (probabilities < bin_edges[i + 1])
        if i == n_bins - 1:  # include right edge in last bin
            mask = (probabilities >= bin_edges[i]) & (probabilities <= bin_edges[i + 1])

        bin_counts[i] = mask.sum()
        if bin_counts[i] > 0:
            true_freq[i] = labels[mask].mean()
            mean_predicted[i] = probabilities[mask].mean()

    return {
        "bin_centers": bin_centers,
        "true_freq": true_freq,
        "mean_predicted": mean_predicted,
        "bin_counts": bin_counts,
    }


def compute_ks_statistic(labels: np.ndarray, probabilities: np.ndarray) -> float:
    """Kolmogorov-Smirnov statistic: max separation between CDFs.

    Commonly used in credit scoring to measure discriminatory power.
    """
    pos_probs = probabilities[labels == 1]
    neg_probs = probabilities[labels == 0]

    # Sort all unique thresholds
    thresholds = np.sort(np.unique(probabilities))

    max_ks = 0.0
    for t in thresholds:
        tpr = (pos_probs >= t).mean()
        fpr = (neg_probs >= t).mean()
        ks = abs(tpr - fpr)
        max_ks = max(max_ks, ks)

    return max_ks
