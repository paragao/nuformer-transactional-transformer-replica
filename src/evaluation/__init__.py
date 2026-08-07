"""Evaluation framework for nuFormer.

Modules:
- metrics: AUC, AP, calibration, lift, KS statistic
- ablation: Paper Table 2 replication + scaling experiments
- scaling_laws: Data/context/model size scaling analysis
- tracking: MLFlow integration with SageMaker backend
"""

from .metrics import (
    EvalResult,
    compute_metrics,
    compute_metrics_with_ci,
    calibration_curve,
    compute_ks_statistic,
)
from .ablation import AblationConfig, AblationSuite
from .scaling_laws import (
    ScalingCurve,
    ScalingPoint,
    estimate_training_flops,
    analyze_data_scaling,
    analyze_context_scaling,
)
from .tracking import ExperimentTracker, TrackingConfig, log_training_state

__all__ = [
    "EvalResult",
    "compute_metrics",
    "compute_metrics_with_ci",
    "calibration_curve",
    "compute_ks_statistic",
    "AblationConfig",
    "AblationSuite",
    "ScalingCurve",
    "ScalingPoint",
    "estimate_training_flops",
    "analyze_data_scaling",
    "analyze_context_scaling",
    "ExperimentTracker",
    "TrackingConfig",
    "log_training_state",
]
