"""Tests for evaluation framework."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np


def test_metrics_basic():
    """Test basic metrics computation."""
    from src.evaluation.metrics import compute_metrics, EvalResult

    # Create deterministic test data
    rng = np.random.default_rng(42)
    n = 1000
    labels = rng.integers(0, 2, size=n)

    # Probabilities correlated with labels (AUC should be > 0.5)
    probabilities = labels * 0.6 + rng.uniform(0, 0.4, size=n)
    probabilities = np.clip(probabilities, 0, 1)

    result = compute_metrics(labels, probabilities)

    assert isinstance(result, EvalResult)
    assert 0.5 < result.auc_roc <= 1.0, f"AUC should be > 0.5, got {result.auc_roc}"
    assert 0 < result.average_precision <= 1.0
    assert 0 < result.accuracy <= 1.0
    assert 0 <= result.brier_score <= 1.0
    assert result.n_samples == n
    assert 0 < result.positive_rate < 1.0
    assert result.lift_at_10pct > 1.0, "Top decile should have lift > 1"

    print(f"  test_metrics_basic PASSED (AUC={result.auc_roc:.4f}, "
          f"AP={result.average_precision:.4f}, Lift@10%={result.lift_at_10pct:.2f}x)")


def test_metrics_perfect():
    """Test with perfect predictions."""
    from src.evaluation.metrics import compute_metrics

    labels = np.array([0, 0, 0, 1, 1, 1])
    probs = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])

    result = compute_metrics(labels, probs)
    assert result.auc_roc == 1.0
    assert result.accuracy == 1.0
    print("  test_metrics_perfect PASSED")


def test_metrics_with_ci():
    """Test bootstrap confidence intervals."""
    from src.evaluation.metrics import compute_metrics_with_ci

    rng = np.random.default_rng(123)
    n = 500
    labels = rng.integers(0, 2, size=n)
    probs = labels * 0.5 + rng.uniform(0, 0.5, size=n)
    probs = np.clip(probs, 0, 1)

    result = compute_metrics_with_ci(labels, probs, n_bootstrap=200, seed=42)

    assert result.auc_ci_low is not None
    assert result.auc_ci_high is not None
    assert result.auc_ci_low <= result.auc_roc <= result.auc_ci_high
    assert result.auc_ci_high - result.auc_ci_low < 0.15  # CI should be reasonable

    print(f"  test_metrics_with_ci PASSED (AUC={result.auc_roc:.4f} "
          f"[{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}])")


def test_relative_auc():
    """Test relative AUC computation."""
    from src.evaluation.metrics import EvalResult

    result = EvalResult(auc_roc=0.8125)
    baseline_auc = 0.8000

    rel = result.relative_auc(baseline_auc)
    assert abs(rel - 1.5625) < 0.01, f"Expected ~1.56%, got {rel}"
    print(f"  test_relative_auc PASSED ({rel:.4f}%)")


def test_calibration():
    """Test calibration curve computation."""
    from src.evaluation.metrics import calibration_curve

    rng = np.random.default_rng(42)
    n = 1000
    # Well-calibrated model: P(Y=1|p) ~ p
    probs = rng.uniform(0, 1, size=n)
    labels = (rng.uniform(0, 1, size=n) < probs).astype(int)

    cal = calibration_curve(labels, probs, n_bins=5)

    assert "bin_centers" in cal
    assert "true_freq" in cal
    assert len(cal["bin_centers"]) == 5
    assert cal["bin_counts"].sum() == n

    # For a well-calibrated model, true_freq should ~ bin_centers
    nonzero = cal["bin_counts"] > 0
    max_diff = np.max(np.abs(cal["true_freq"][nonzero] - cal["bin_centers"][nonzero]))
    assert max_diff < 0.15, f"Calibration error too high: {max_diff}"

    print(f"  test_calibration PASSED (max cal error: {max_diff:.3f})")


def test_ks_statistic():
    """Test KS statistic computation."""
    from src.evaluation.metrics import compute_ks_statistic

    rng = np.random.default_rng(42)
    n = 500
    labels = np.concatenate([np.zeros(n), np.ones(n)])
    # Good separation
    probs = np.concatenate([rng.uniform(0, 0.5, n), rng.uniform(0.5, 1, n)])

    ks = compute_ks_statistic(labels, probs)
    assert 0.5 < ks <= 1.0, f"KS should be high for good separation, got {ks}"
    print(f"  test_ks_statistic PASSED (KS={ks:.4f})")


def test_ablation_suite():
    """Test ablation suite creation and summary."""
    from src.evaluation.ablation import AblationSuite, AblationConfig
    from src.evaluation.metrics import EvalResult

    suite = AblationSuite.paper_table2()
    assert len(suite.experiments) == 5

    # Set fake results
    suite.set_baseline(0.80)
    for i, exp in enumerate(suite.experiments):
        exp.result = EvalResult(
            auc_roc=0.80 + i * 0.003,
            average_precision=0.30 + i * 0.01,
            accuracy=0.85,
            f1=0.60,
            brier_score=0.15,
            log_loss=0.40,
            n_samples=10000,
            positive_rate=0.15,
            lift_at_10pct=2.0 + i * 0.2,
            lift_at_20pct=1.5 + i * 0.1,
        )

    table = suite.summary_table()
    assert "lightgbm_baseline" in table
    assert "joint_nuformer" in table
    assert "%" in table  # relative AUC should show %

    print(f"  test_ablation_suite PASSED")
    print(f"    Table:\n{table}")


def test_ablation_save_load(tmp_path="/tmp/test_ablation.json"):
    """Test ablation suite serialization."""
    from src.evaluation.ablation import AblationSuite
    from src.evaluation.metrics import EvalResult

    suite = AblationSuite.paper_table2()
    suite.set_baseline(0.80)
    suite.experiments[0].result = EvalResult(
        auc_roc=0.80, average_precision=0.30, accuracy=0.85,
        f1=0.60, brier_score=0.15, log_loss=0.40,
        n_samples=1000, positive_rate=0.15,
        lift_at_10pct=2.0, lift_at_20pct=1.5,
    )

    suite.save(tmp_path)
    loaded = AblationSuite.load(tmp_path)

    assert loaded.baseline_auc == 0.80
    assert len(loaded.experiments) == 5
    assert loaded.experiments[0].result.auc_roc == 0.80

    # Cleanup
    Path(tmp_path).unlink(missing_ok=True)
    print("  test_ablation_save_load PASSED")


def test_scaling_laws():
    """Test scaling law analysis."""
    from src.evaluation.scaling_laws import (
        analyze_data_scaling,
        analyze_context_scaling,
        estimate_training_flops,
        ScalingCurve,
        ScalingPoint,
    )

    # Data scaling (log-linear pattern like paper)
    data_sizes = [5_000_000, 20_000_000, 40_000_000, 100_000_000]
    aucs = [0.790, 0.802, 0.808, 0.815]

    result = analyze_data_scaling(data_sizes, aucs)
    assert "curve" in result
    assert "fit" in result
    assert result["fit"]["r_squared"] > 0.9, "Log-linear fit should be good"

    # Extrapolation
    pred_200m = result["extrapolation_200M"]
    assert pred_200m > aucs[-1], "More data should extrapolate higher"

    print(f"  test_scaling_laws PASSED (R²={result['fit']['r_squared']:.4f}, "
          f"pred@200M={pred_200m:.4f})")

    # Context scaling
    ctx_lengths = [512, 1024, 2048, 4096]
    ctx_aucs = [0.795, 0.805, 0.812, 0.815]

    ctx_result = analyze_context_scaling(ctx_lengths, ctx_aucs)
    assert ctx_result["improvement_512_to_4096"] > 0

    print(f"    Context improvement 512->4096: "
          f"{ctx_result['improvement_512_to_4096']:.2f}%")

    # FLOPs estimation
    flops = estimate_training_flops(
        model_params=330_000_000,
        n_tokens=50_000 * 12_288 * 2048,  # steps * batch * seq_len
    )
    assert flops > 1e18  # Should be > 1 exaFLOP
    print(f"    Estimated training FLOPs: {flops:.2e}")


def test_tracking_disabled():
    """Test that tracker works gracefully when MLFlow is unavailable."""
    from src.evaluation.tracking import ExperimentTracker, TrackingConfig

    # Use a bogus URI so it fails to connect
    config = TrackingConfig(tracking_uri="http://nonexistent:5000")
    tracker = ExperimentTracker(config=config)

    # These should all be no-ops without error
    tracker.start_run("test_run")
    tracker.log_params({"lr": 0.001})
    tracker.log_metrics({"loss": 0.5}, step=1)
    tracker.end_run()

    print("  test_tracking_disabled PASSED (graceful fallback)")


if __name__ == "__main__":
    print("Running evaluation tests...\n")
    test_metrics_basic()
    test_metrics_perfect()
    test_metrics_with_ci()
    test_relative_auc()
    test_calibration()
    test_ks_statistic()
    test_ablation_suite()
    test_ablation_save_load()
    test_scaling_laws()
    test_tracking_disabled()
    print("\nAll evaluation tests PASSED!")
