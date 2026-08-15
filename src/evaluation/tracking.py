"""MLFlow integration for nuFormer experiment tracking.

Wraps MLFlow API for consistent logging across all training stages.
Connects to SageMaker-managed MLFlow tracking server.

Usage:
    tracker = ExperimentTracker("nuformer-pretrain")
    tracker.start_run("pretrain_330M_ctx2048")
    tracker.log_params({...})
    tracker.log_metrics({"loss": 0.5, "ppl": 12.3}, step=100)
    tracker.end_run()
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional


@dataclass
class TrackingConfig:
    """MLFlow tracking configuration."""

    tracking_uri: str = "arn:aws:sagemaker:ap-southeast-3:159553542841:mlflow-tracking-server/paragao-mlflow-tracker"
    experiment_name: str = "nuformer-replication"
    aws_profile: str = "compute-sa-team-Administrator"
    region: str = "ap-southeast-3"


class ExperimentTracker:
    """MLFlow experiment tracker with SageMaker backend.

    Handles:
    - Run lifecycle (start, log, end)
    - Parameter logging (model config, training config)
    - Metric logging (loss, AUC, perplexity, etc.)
    - Artifact logging (checkpoints, plots, configs)
    - Automatic tagging (git hash, node count, GPU type)
    """

    def __init__(
        self,
        experiment_name: str = "nuformer-replication",
        config: Optional[TrackingConfig] = None,
    ):
        self.config = config or TrackingConfig(experiment_name=experiment_name)
        self._mlflow = None
        self._run = None
        self._enabled = False

        self._setup()

    def _setup(self):
        """Initialize MLFlow connection."""
        try:
            import mlflow

            # Set AWS credentials
            os.environ.setdefault("AWS_PROFILE", self.config.aws_profile)
            os.environ.setdefault("AWS_DEFAULT_REGION", self.config.region)

            mlflow.set_tracking_uri(self.config.tracking_uri)
            mlflow.set_experiment(self.config.experiment_name)
            self._mlflow = mlflow
            self._enabled = True
            print(f"MLFlow connected: {self.config.experiment_name}")
        except Exception as e:
            print(f"MLFlow not available (continuing without tracking): {e}")
            self._enabled = False

    @property
    def enabled(self) -> bool:
        return self._enabled

    def start_run(self, run_name: str, tags: Optional[dict[str, str]] = None):
        """Start a new MLFlow run."""
        if not self._enabled:
            return

        all_tags = {
            "framework": "pytorch",
            "model": "nuformer",
        }

        # Auto-detect environment
        if "SLURM_JOB_ID" in os.environ:
            all_tags["slurm_job_id"] = os.environ["SLURM_JOB_ID"]
            all_tags["slurm_nodes"] = os.environ.get("SLURM_NNODES", "?")

        # Git hash
        try:
            import subprocess
            git_hash = subprocess.check_output(
                ["git", "rev-parse", "--short", "HEAD"],
                stderr=subprocess.DEVNULL,
            ).decode().strip()
            all_tags["git_hash"] = git_hash
        except Exception:
            pass

        if tags:
            all_tags.update(tags)

        self._run = self._mlflow.start_run(run_name=run_name, tags=all_tags)
        print(f"MLFlow run started: {run_name} (id={self._run.info.run_id[:8]})")

    def log_params(self, params: dict[str, Any]):
        """Log parameters (called once per run)."""
        if not self._enabled or self._run is None:
            return

        # Flatten nested dicts
        flat = self._flatten_dict(params)
        # MLFlow has 500 char limit on param values
        for k, v in flat.items():
            str_v = str(v)[:500]
            try:
                self._mlflow.log_param(k, str_v)
            except Exception:
                pass  # ignore duplicate param errors on resume

    def log_metrics(self, metrics: dict[str, float], step: Optional[int] = None):
        """Log metrics (called repeatedly during training)."""
        if not self._enabled or self._run is None:
            return

        self._mlflow.log_metrics(metrics, step=step)

    def log_metric(self, key: str, value: float, step: Optional[int] = None):
        """Log a single metric."""
        if not self._enabled or self._run is None:
            return

        self._mlflow.log_metric(key, value, step=step)

    def log_artifact(self, local_path: str, artifact_path: Optional[str] = None):
        """Log a file as an artifact."""
        if not self._enabled or self._run is None:
            return

        if Path(local_path).exists():
            self._mlflow.log_artifact(local_path, artifact_path)

    def log_text(self, text: str, filename: str):
        """Log text content as an artifact."""
        if not self._enabled or self._run is None:
            return

        # Write to temp file then log
        import tempfile
        with tempfile.NamedTemporaryFile(mode="w", suffix=f"_{filename}", delete=False) as f:
            f.write(text)
            f.flush()
            self._mlflow.log_artifact(f.name)

    def log_eval_result(self, result: "EvalResult", prefix: str = "eval"):
        """Log an EvalResult object as metrics."""
        if not self._enabled or self._run is None:
            return

        metrics = {f"{prefix}/{k}": v for k, v in result.to_dict().items()
                   if isinstance(v, (int, float))}
        self._mlflow.log_metrics(metrics)

    def end_run(self, status: str = "FINISHED"):
        """End the current run."""
        if not self._enabled or self._run is None:
            return

        self._mlflow.end_run(status=status)
        self._run = None

    @staticmethod
    def _flatten_dict(d: dict, parent_key: str = "", sep: str = ".") -> dict:
        """Flatten nested dict with dot-separated keys."""
        items = []
        for k, v in d.items():
            new_key = f"{parent_key}{sep}{k}" if parent_key else k
            if isinstance(v, dict):
                items.extend(ExperimentTracker._flatten_dict(v, new_key, sep).items())
            else:
                items.append((new_key, v))
        return dict(items)


# ---------------------------------------------------------------------------
# Convenience: log training state
# ---------------------------------------------------------------------------


def log_training_state(
    tracker: ExperimentTracker,
    step: int,
    loss: float,
    lr: float,
    grad_norm: float,
    tokens_per_sec: float = 0.0,
    extra: Optional[dict[str, float]] = None,
):
    """Log standard training metrics at a given step."""
    metrics = {
        "train/loss": loss,
        "train/lr": lr,
        "train/grad_norm": grad_norm,
    }
    if tokens_per_sec > 0:
        metrics["train/tokens_per_sec"] = tokens_per_sec
        metrics["train/mfu"] = 0.0  # placeholder for model FLOPs utilization

    if extra:
        metrics.update(extra)

    tracker.log_metrics(metrics, step=step)
