"""Run evaluation on a trained nuFormer checkpoint.

Loads a model checkpoint, runs inference on test set, computes all
metrics, and optionally logs to MLFlow.

Usage:
    python scripts/evaluate.py --checkpoint ckpt/joint_fusion/best.pt --test-data data/processed/test
    python scripts/evaluate.py --checkpoint ckpt/finetune/best.pt --mode transformer_only
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


def load_model(checkpoint_path: str, mode: str = "joint"):
    """Load model from checkpoint.

    Args:
        checkpoint_path: path to .pt checkpoint
        mode: 'joint' (nuFormer), 'transformer_only' (finetune head)

    Returns:
        (model, device)
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)

    if mode == "joint":
        from src.models.nuformer import NuFormer, NuFormerConfig
        from src.models.transformer import TransformerConfig
        from src.models.dcnv2 import DCNv2Config

        # Reconstruct config (or load from checkpoint if stored)
        config = ckpt.get("config", None)
        if config is None:
            # Default 330M config
            transformer_config = TransformerConfig(
                vocab_size=24078, d_model=1024, n_layers=24,
                n_heads=16, d_ff=4096, max_seq_len=2048,
            )
            dcnv2_config = DCNv2Config(
                input_dim=291, cross_layers=3,
                deep_layers=[512, 256], output_dim=128,
            )
            nuformer_config = NuFormerConfig(
                transformer=transformer_config, dcnv2=dcnv2_config,
            )
        else:
            nuformer_config = config

        model = NuFormer(nuformer_config)
        model.load_state_dict(ckpt["model"], strict=False)

    elif mode == "transformer_only":
        from src.models.transformer import TransactionTransformer, TransformerConfig
        from src.models.nuformer import FineTuneHead

        transformer_config = TransformerConfig(
            vocab_size=24078, d_model=1024, n_layers=24,
            n_heads=16, d_ff=4096, max_seq_len=2048,
        )
        model = TransactionTransformer(transformer_config)
        # Load LoRA weights if present
        if "lora" in ckpt:
            model.load_state_dict(ckpt["lora"], strict=False)
        head = FineTuneHead(d_model=1024, num_classes=2)
        if "head" in ckpt:
            head.load_state_dict(ckpt["head"])
        model = (model, head)  # tuple

    model_obj = model if not isinstance(model, tuple) else model[0]
    model_obj.to(device)
    model_obj.eval()
    if isinstance(model, tuple):
        model[1].to(device)
        model[1].eval()

    return model, device


@torch.no_grad()
def run_inference(
    model,
    device: torch.device,
    test_sequences: np.ndarray,
    test_features: np.ndarray | None = None,
    batch_size: int = 64,
    mode: str = "joint",
) -> np.ndarray:
    """Run inference and return predicted probabilities.

    Returns:
        (N,) array of predicted probabilities for positive class
    """
    all_probs = []
    n = len(test_sequences)

    for i in range(0, n, batch_size):
        batch_seq = torch.from_numpy(test_sequences[i:i+batch_size]).long().to(device)
        attention_mask = (batch_seq != 74).long()

        if mode == "joint" and test_features is not None:
            batch_feat = torch.from_numpy(test_features[i:i+batch_size]).float().to(device)
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                output = model(batch_seq, batch_feat, attention_mask)
            logits = output["logits"]
        elif mode == "transformer_only":
            transformer, head = model
            with torch.amp.autocast("cuda", dtype=torch.bfloat16, enabled=torch.cuda.is_available()):
                user_emb = transformer.get_user_embedding(batch_seq, attention_mask)
                logits = head(user_emb)
        else:
            raise ValueError(f"Unknown mode: {mode}")

        probs = F.softmax(logits, dim=-1)[:, 1]
        all_probs.append(probs.cpu().numpy())

    return np.concatenate(all_probs)


def main():
    parser = argparse.ArgumentParser(description="nuFormer Evaluation")
    parser.add_argument("--checkpoint", required=True, help="Model checkpoint path")
    parser.add_argument("--mode", choices=["joint", "transformer_only"], default="joint")
    parser.add_argument("--test-sequences", default="data/processed/test_sequences.npy")
    parser.add_argument("--test-labels", default="data/processed/test_labels.npy")
    parser.add_argument("--test-features", default="data/processed/test_features.npy")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--output", default="results/eval_results.json")
    parser.add_argument("--baseline-auc", type=float, default=0.0,
                        help="Baseline AUC for relative comparison")
    parser.add_argument("--bootstrap", action="store_true", help="Compute CI via bootstrap")
    args = parser.parse_args()

    print(f"Evaluating: {args.checkpoint}")
    print(f"Mode: {args.mode}")

    # Load model
    model, device = load_model(args.checkpoint, args.mode)

    # Load test data
    test_seq_path = Path(args.test_sequences)
    test_labels_path = Path(args.test_labels)

    if not test_seq_path.exists():
        print(f"ERROR: Test sequences not found at {test_seq_path}")
        print("Generating dummy test data for pipeline validation...")

        rng = np.random.default_rng(42)
        test_sequences = rng.integers(0, 24078, (200, 128))
        test_labels = rng.integers(0, 2, 200)
        test_features = rng.randn(200, 291).astype(np.float32) if args.mode == "joint" else None
    else:
        test_sequences = np.load(str(test_seq_path), mmap_mode="r")
        test_labels = np.load(str(test_labels_path))
        test_features = None
        if args.mode == "joint" and Path(args.test_features).exists():
            test_features = np.load(args.test_features, mmap_mode="r")

    # Run inference
    t0 = time.time()
    probabilities = run_inference(
        model, device, test_sequences, test_features,
        batch_size=args.batch_size, mode=args.mode,
    )
    inference_time = time.time() - t0
    print(f"Inference time: {inference_time:.1f}s ({len(test_labels)/inference_time:.0f} samples/s)")

    # Compute metrics
    from src.evaluation.metrics import compute_metrics, compute_metrics_with_ci

    if args.bootstrap:
        result = compute_metrics_with_ci(test_labels, probabilities)
    else:
        result = compute_metrics(test_labels, probabilities)

    # Print results
    print(f"\n{'='*50}")
    print(f"  Evaluation Results")
    print(f"{'='*50}")
    print(f"  AUC-ROC:       {result.auc_roc:.4f}", end="")
    if result.auc_ci_low is not None:
        print(f"  [{result.auc_ci_low:.4f}, {result.auc_ci_high:.4f}]")
    else:
        print()
    print(f"  AP:            {result.average_precision:.4f}")
    print(f"  Accuracy:      {result.accuracy:.4f}")
    print(f"  F1:            {result.f1:.4f}")
    print(f"  Brier Score:   {result.brier_score:.4f}")
    print(f"  Log Loss:      {result.log_loss:.4f}")
    print(f"  Lift@10%:      {result.lift_at_10pct:.2f}x")
    print(f"  Lift@20%:      {result.lift_at_20pct:.2f}x")
    print(f"  N samples:     {result.n_samples:,}")
    print(f"  Positive rate: {result.positive_rate:.3f}")

    if args.baseline_auc > 0:
        rel_auc = result.relative_auc(args.baseline_auc)
        print(f"\n  Relative AUC vs baseline ({args.baseline_auc:.4f}): {rel_auc:+.2f}%")

    print(f"{'='*50}")

    # Save results
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    results = {
        "checkpoint": args.checkpoint,
        "mode": args.mode,
        "inference_time_s": inference_time,
        "metrics": result.to_dict(),
    }
    if args.baseline_auc > 0:
        results["relative_auc_pct"] = result.relative_auc(args.baseline_auc)

    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved: {output_path}")


if __name__ == "__main__":
    main()
