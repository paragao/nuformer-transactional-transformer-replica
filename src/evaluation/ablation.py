"""Ablation study framework for nuFormer.

Reproduces the paper's Table 2 ablation:
- LightGBM baseline (features only)
- Transformer only (fine-tuned embeddings)
- Late Fusion (LightGBM + embeddings as feature)
- Joint Fusion without numerical embeddings
- Joint Fusion (full nuFormer)

Also supports:
- Context length ablation (512, 1024, 2048, 4096)
- Data scaling (5M, 20M, 40M, 100M rows)
- Model size ablation (24M vs 330M)
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import numpy as np

from .metrics import EvalResult, compute_metrics, compute_metrics_with_ci


# ---------------------------------------------------------------------------
# Ablation configurations
# ---------------------------------------------------------------------------


@dataclass
class AblationConfig:
    """Configuration for a single ablation experiment."""

    name: str
    description: str

    # Model variant
    use_transformer: bool = True
    use_dcnv2: bool = True
    use_numerical_embeddings: bool = True
    use_lora: bool = True
    fusion_mode: str = "joint"  # "joint", "late", "transformer_only", "features_only"

    # Context / data
    context_length: int = 2048
    train_rows: int = 20_000_000
    model_size: str = "330M"  # "24M" or "330M"

    # LoRA
    lora_rank: int = 16

    # Results (filled after evaluation)
    result: Optional[EvalResult] = None
    train_time_hours: float = 0.0


@dataclass
class AblationSuite:
    """Collection of ablation experiments matching Table 2 of the paper."""

    experiments: list[AblationConfig] = field(default_factory=list)
    baseline_auc: float = 0.0  # LightGBM baseline for relative comparison

    @classmethod
    def paper_table2(cls) -> "AblationSuite":
        """Create ablation suite matching paper's Table 2."""
        suite = cls()
        suite.experiments = [
            AblationConfig(
                name="lightgbm_baseline",
                description="LightGBM on 291 tabular features (baseline)",
                use_transformer=False,
                use_dcnv2=False,
                use_numerical_embeddings=False,
                use_lora=False,
                fusion_mode="features_only",
            ),
            AblationConfig(
                name="transformer_only",
                description="Fine-tuned transformer (no tabular features)",
                use_transformer=True,
                use_dcnv2=False,
                use_numerical_embeddings=False,
                use_lora=True,
                fusion_mode="transformer_only",
            ),
            AblationConfig(
                name="late_fusion",
                description="LightGBM + transformer embeddings as extra feature",
                use_transformer=True,
                use_dcnv2=False,
                use_numerical_embeddings=False,
                use_lora=True,
                fusion_mode="late",
            ),
            AblationConfig(
                name="joint_no_num_emb",
                description="Joint fusion DCNv2 without numerical embeddings",
                use_transformer=True,
                use_dcnv2=True,
                use_numerical_embeddings=False,
                use_lora=True,
                fusion_mode="joint",
            ),
            AblationConfig(
                name="joint_nuformer",
                description="Full nuFormer (joint fusion + numerical embeddings)",
                use_transformer=True,
                use_dcnv2=True,
                use_numerical_embeddings=True,
                use_lora=True,
                fusion_mode="joint",
            ),
        ]
        return suite

    @classmethod
    def context_length_ablation(cls) -> "AblationSuite":
        """Create context length ablation (paper Figure 3)."""
        suite = cls()
        for ctx_len in [512, 1024, 2048, 4096]:
            suite.experiments.append(AblationConfig(
                name=f"ctx_{ctx_len}",
                description=f"nuFormer with context length {ctx_len}",
                context_length=ctx_len,
                fusion_mode="joint",
            ))
        return suite

    @classmethod
    def data_scaling_ablation(cls) -> "AblationSuite":
        """Create data scaling ablation (paper Figure 4)."""
        suite = cls()
        for n_rows in [5_000_000, 20_000_000, 40_000_000, 100_000_000]:
            suite.experiments.append(AblationConfig(
                name=f"data_{n_rows // 1_000_000}M",
                description=f"nuFormer trained on {n_rows // 1_000_000}M rows",
                train_rows=n_rows,
                fusion_mode="joint",
            ))
        return suite

    @classmethod
    def model_size_ablation(cls) -> "AblationSuite":
        """Create model size ablation."""
        suite = cls()
        for size in ["24M", "330M"]:
            suite.experiments.append(AblationConfig(
                name=f"model_{size}",
                description=f"nuFormer {size} parameters",
                model_size=size,
                fusion_mode="joint",
            ))
        return suite

    def set_baseline(self, auc: float):
        """Set the baseline AUC for relative comparison."""
        self.baseline_auc = auc

    def summary_table(self) -> str:
        """Generate markdown summary table (like paper's Table 2)."""
        lines = [
            "| Configuration | AUC-ROC | Relative AUC | AP | Lift@10% |",
            "|---------------|---------|--------------|-----|----------|",
        ]
        for exp in self.experiments:
            if exp.result is None:
                lines.append(f"| {exp.name} | - | - | - | - |")
                continue

            rel_auc = exp.result.relative_auc(self.baseline_auc) if self.baseline_auc else 0.0
            ci_str = ""
            if exp.result.auc_ci_low is not None:
                ci_str = f" [{exp.result.auc_ci_low:.4f}, {exp.result.auc_ci_high:.4f}]"

            lines.append(
                f"| {exp.name} | {exp.result.auc_roc:.4f}{ci_str} | "
                f"{rel_auc:+.2f}% | {exp.result.average_precision:.4f} | "
                f"{exp.result.lift_at_10pct:.2f}x |"
            )

        return "\n".join(lines)

    def save(self, path: str):
        """Save results to JSON."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        data = {
            "baseline_auc": self.baseline_auc,
            "experiments": [],
        }
        for exp in self.experiments:
            exp_data = {
                "name": exp.name,
                "description": exp.description,
                "config": {
                    "use_transformer": exp.use_transformer,
                    "use_dcnv2": exp.use_dcnv2,
                    "use_numerical_embeddings": exp.use_numerical_embeddings,
                    "fusion_mode": exp.fusion_mode,
                    "context_length": exp.context_length,
                    "train_rows": exp.train_rows,
                    "model_size": exp.model_size,
                    "lora_rank": exp.lora_rank,
                },
                "train_time_hours": exp.train_time_hours,
            }
            if exp.result is not None:
                exp_data["metrics"] = exp.result.to_dict()
            data["experiments"].append(exp_data)

        with open(path, "w") as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> "AblationSuite":
        """Load results from JSON."""
        with open(path) as f:
            data = json.load(f)

        suite = cls(baseline_auc=data.get("baseline_auc", 0.0))
        for exp_data in data["experiments"]:
            cfg = exp_data["config"]
            exp = AblationConfig(
                name=exp_data["name"],
                description=exp_data["description"],
                use_transformer=cfg["use_transformer"],
                use_dcnv2=cfg["use_dcnv2"],
                use_numerical_embeddings=cfg["use_numerical_embeddings"],
                fusion_mode=cfg["fusion_mode"],
                context_length=cfg["context_length"],
                train_rows=cfg["train_rows"],
                model_size=cfg["model_size"],
                lora_rank=cfg["lora_rank"],
                train_time_hours=exp_data.get("train_time_hours", 0.0),
            )
            if "metrics" in exp_data:
                m = exp_data["metrics"]
                exp.result = EvalResult(
                    auc_roc=m["auc_roc"],
                    average_precision=m["average_precision"],
                    accuracy=m["accuracy"],
                    f1=m["f1"],
                    brier_score=m["brier_score"],
                    log_loss=m["log_loss"],
                    n_samples=m["n_samples"],
                    positive_rate=m["positive_rate"],
                    lift_at_10pct=m["lift_at_10pct"],
                    lift_at_20pct=m["lift_at_20pct"],
                    auc_ci_low=m.get("auc_ci_low"),
                    auc_ci_high=m.get("auc_ci_high"),
                )
            suite.experiments.append(exp)
        return suite
