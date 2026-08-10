"""Joint Fusion training: end-to-end nuFormer.

Trains Transformer (LoRA) + DCNv2 + Fusion MLP jointly on
transaction sequences + tabular features for classification.

This is the final stage that produces the full nuFormer model,
combining learned transaction embeddings with tabular feature
interactions via DCNv2.

Usage:
    python -m src.training.joint_fusion --pretrain-ckpt ckpt/pretrain/final.pt
    torchrun --nproc-per-node=8 -m src.training.joint_fusion --pretrain-ckpt ...
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.training.pretrain import PretrainConfig

# Allow torch.load to unpickle checkpoints saved from __main__
sys.modules["__main__"].PretrainConfig = PretrainConfig


@dataclass
class JointFusionConfig:
    """Joint fusion training configuration."""

    # Data
    train_sequences_path: str = "data/processed/train_sequences.npy"
    train_labels_path: str = "data/processed/train_labels.npy"
    train_features_path: str = "data/processed/train_features.npy"
    val_sequences_path: str = "data/processed/val_sequences.npy"
    val_labels_path: str = "data/processed/val_labels.npy"
    val_features_path: str = "data/processed/val_features.npy"
    max_seq_len: int = 2048

    # Pre-trained model
    pretrain_checkpoint: str = "ckpt/pretrain/final.pt"

    # Transformer
    vocab_size: int = 24078
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 4096

    # DCNv2
    num_tabular_features: int = 291
    dcnv2_cross_layers: int = 3
    dcnv2_deep_dims: list = None
    dcnv2_output_dim: int = 128
    dcnv2_dropout: float = 0.1

    # Fusion
    fusion_hidden_dim: int = 256
    fusion_dropout: float = 0.3
    num_classes: int = 2

    # LoRA
    lora_rank: int = 16
    lora_alpha: float = 32.0
    freeze_transformer: bool = False  # If True, skip LoRA and freeze all transformer params

    # Optimization (different LRs per component)
    batch_size: int = 32
    gradient_accumulation_steps: int = 4
    max_steps: int = 10_000
    warmup_steps: int = 500
    transformer_lr: float = 5e-6
    dcnv2_lr: float = 1e-4
    fusion_lr: float = 5e-5
    min_lr_ratio: float = 0.1
    weight_decay: float = 0.05
    dcnv2_weight_decay: float = 0.2  # stronger for cross layers
    grad_clip: float = 1.0
    label_smoothing: float = 0.1

    # Precision
    dtype: str = "bfloat16"

    # Checkpointing & logging
    checkpoint_dir: str = "ckpt/joint_fusion"
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 2000

    def __post_init__(self):
        if self.dcnv2_deep_dims is None:
            self.dcnv2_deep_dims = [512, 256]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class JointFusionDataset(Dataset):
    """Dataset for joint fusion: sequences + tabular features + labels."""

    def __init__(
        self,
        sequences_path: str,
        labels_path: str,
        features_path: str,
        max_seq_len: int = 2048,
    ):
        import numpy as np

        self.sequences = np.load(sequences_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.features = np.load(features_path, mmap_mode="r")
        self.max_seq_len = max_seq_len
        self.pad_token_id = 74

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx].copy()).long()[:self.max_seq_len]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        features = torch.from_numpy(self.features[idx].copy()).float()
        attention_mask = (seq != self.pad_token_id).long()

        return {
            "input_ids": seq,
            "attention_mask": attention_mask,
            "tabular_features": features,
            "labels": label,
        }


class DummyJointDataset(Dataset):
    """Dummy dataset for pipeline validation."""

    def __init__(self, size: int = 500, seq_len: int = 128,
                 n_features: int = 291, vocab_size: int = 24078):
        self.size = size
        self.seq_len = seq_len
        self.n_features = n_features
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "tabular_features": torch.randn(self.n_features),
            "labels": torch.randint(0, 2, ()),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class JointFusionTrainer:
    """End-to-end joint fusion training of nuFormer.

    Component-specific learning rates (from paper):
    - Transformer (LoRA): low LR (5e-5) to preserve pre-trained knowledge
    - DCNv2 + embeddings: higher LR (1e-3) with stronger weight decay
    - Fusion MLP: moderate LR (5e-4)
    """

    def __init__(self, config: JointFusionConfig):
        self.config = config
        self.step = 0
        self.best_auc = 0.0

        self._setup_distributed()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    def _setup_distributed(self):
        """Initialize distributed context."""
        self.distributed = "RANK" in os.environ and torch.distributed.is_available()
        if self.distributed and not torch.distributed.is_initialized():
            torch.distributed.init_process_group(backend="nccl")
            self.rank = torch.distributed.get_rank()
            self.world_size = torch.distributed.get_world_size()
            self.local_rank = int(os.environ.get("LOCAL_RANK", 0))
        else:
            self.rank = 0
            self.world_size = 1
            self.local_rank = 0

        if torch.cuda.is_available():
            self.device = torch.device(f"cuda:{self.local_rank}")
            torch.cuda.set_device(self.device)
        else:
            self.device = torch.device("cpu")

        self.dtype = getattr(torch, self.config.dtype)

    def _setup_model(self):
        """Build nuFormer: pre-trained transformer + LoRA + DCNv2 + fusion."""
        from src.models.transformer import TransactionTransformer, TransformerConfig
        from src.models.nuformer import NuFormer, NuFormerConfig
        from src.models.dcnv2 import DCNv2Config
        from src.models.lora import apply_lora

        # Transformer config
        transformer_config = TransformerConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            dropout=0.0,
        )

        # DCNv2 config
        dcnv2_config = DCNv2Config(
            input_dim=self.config.num_tabular_features,
            cross_layers=self.config.dcnv2_cross_layers,
            deep_layers=self.config.dcnv2_deep_dims,
            output_dim=self.config.dcnv2_output_dim,
            dropout=self.config.dcnv2_dropout,
        )

        # nuFormer config
        nuformer_config = NuFormerConfig(
            transformer=transformer_config,
            dcnv2=dcnv2_config,
            fusion_hidden_dim=self.config.fusion_hidden_dim,
            fusion_dropout=self.config.fusion_dropout,
            num_classes=self.config.num_classes,
        )

        self.model = NuFormer(nuformer_config)

        # Load pre-trained transformer weights
        ckpt_path = Path(self.config.pretrain_checkpoint)
        if ckpt_path.exists():
            if self.rank == 0:
                print(f"Loading pre-trained transformer: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.model.transformer.load_state_dict(ckpt["model"], strict=False)
        else:
            if self.rank == 0:
                print("WARNING: No pre-trained checkpoint, training from scratch")

        # Apply LoRA to transformer (freeze base, add adapters) — or freeze entirely
        if self.config.freeze_transformer:
            # Freeze all transformer params (use as fixed feature extractor)
            for p in self.model.transformer.parameters():
                p.requires_grad = False
            if self.rank == 0:
                print("Transformer FROZEN (feature extractor only)")
        else:
            self.model.transformer = apply_lora(
                self.model.transformer,
                rank=self.config.lora_rank,
                alpha=self.config.lora_alpha,
            )

        self.model.to(self.device)

        # DDP wrapping
        if self.distributed:
            from torch.nn.parallel import DistributedDataParallel as DDP
            self.model = DDP(self.model, device_ids=[self.local_rank])

        if self.rank == 0:
            raw_model = self.model.module if self.distributed else self.model
            params = raw_model.num_parameters()
            trainable = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
            total = sum(p.numel() for p in self.model.parameters())
            print(f"Parameters: {params}")
            print(f"Trainable: {trainable:,} / {total:,} ({trainable/total*100:.1f}%)")
            if self.distributed:
                print(f"DDP enabled ({self.world_size} GPUs)")

    def _setup_data(self):
        """Setup data loaders."""
        seq_path = Path(self.config.train_sequences_path)
        if (seq_path.exists() and Path(self.config.train_labels_path).exists()
                and Path(self.config.train_features_path).exists()):
            train_dataset = JointFusionDataset(
                self.config.train_sequences_path,
                self.config.train_labels_path,
                self.config.train_features_path,
                self.config.max_seq_len,
            )
        else:
            if self.rank == 0:
                print("WARNING: Using dummy joint fusion data")
            train_dataset = DummyJointDataset(
                size=500, seq_len=128, n_features=self.config.num_tabular_features,
                vocab_size=self.config.vocab_size,
            )

        train_sampler = None
        shuffle = True
        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            train_sampler = DistributedSampler(train_dataset, shuffle=True)
            shuffle = False

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=shuffle, sampler=train_sampler,
            num_workers=4, pin_memory=True, drop_last=True,
        )

        val_seq = Path(self.config.val_sequences_path)
        if (val_seq.exists() and Path(self.config.val_labels_path).exists()
                and Path(self.config.val_features_path).exists()):
            val_dataset = JointFusionDataset(
                self.config.val_sequences_path,
                self.config.val_labels_path,
                self.config.val_features_path,
                self.config.max_seq_len,
            )
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, num_workers=2
            )
        else:
            self.val_loader = None

    def _setup_optimizer(self):
        """Component-specific optimizer with different LRs and weight decays."""
        raw_model = self.model.module if self.distributed else self.model

        # Categorize parameters by component
        dcnv2_params = list(raw_model.dcnv2.parameters())
        fusion_params = list(raw_model.fusion_mlp.parameters()) + list(raw_model.emb_norm.parameters())

        param_groups = []

        if not self.config.freeze_transformer:
            lora_params = [p for n, p in raw_model.transformer.named_parameters() if p.requires_grad]
            param_groups.append({
                "params": lora_params,
                "lr": self.config.transformer_lr,
                "weight_decay": self.config.weight_decay,
            })

        param_groups.extend([
            {
                "params": dcnv2_params,
                "lr": self.config.dcnv2_lr,
                "weight_decay": self.config.dcnv2_weight_decay,
            },
            {
                "params": fusion_params,
                "lr": self.config.fusion_lr,
                "weight_decay": self.config.weight_decay,
            },
        ])

        self.optimizer = torch.optim.AdamW(param_groups)

        if self.rank == 0:
            n_groups = len(param_groups)
            if self.config.freeze_transformer:
                print(f"Optimizer: {n_groups} param groups "
                      f"(dcnv2={self.config.dcnv2_lr}, fusion={self.config.fusion_lr})")
            else:
                print(f"Optimizer: {n_groups} param groups "
                      f"(transformer={self.config.transformer_lr}, "
                      f"dcnv2={self.config.dcnv2_lr}, "
                      f"fusion={self.config.fusion_lr})")

    def _get_lr_scale(self, step: int) -> float:
        """Cosine decay factor (applied to all groups)."""
        if step < self.config.warmup_steps:
            return step / self.config.warmup_steps
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.config.min_lr_ratio + coeff * (1.0 - self.config.min_lr_ratio)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate: loss, accuracy, AUC."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        all_probs = []
        all_labels = []
        total_loss = 0.0
        total = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            features = batch["tabular_features"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
                output = self.model(input_ids, features, attention_mask)
                loss = F.cross_entropy(output["logits"], labels)

            total_loss += loss.item() * labels.size(0)
            total += labels.size(0)

            probs = F.softmax(output["logits"], dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        self.model.train()

        metrics = {"val_loss": total_loss / max(1, total)}

        try:
            from sklearn.metrics import roc_auc_score, average_precision_score
            if len(set(all_labels)) > 1:
                metrics["val_auc"] = roc_auc_score(all_labels, all_probs)
                metrics["val_ap"] = average_precision_score(all_labels, all_probs)
        except ImportError:
            pass

        return metrics

    def train(self):
        """Main joint fusion training loop."""
        cfg = self.config

        if self.rank == 0:
            effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps * self.world_size
            print(f"\n{'='*60}")
            print(f"  nuFormer Joint Fusion Training")
            print(f"{'='*60}")
            print(f"  Max steps:        {cfg.max_steps:,}")
            print(f"  Effective batch:  {effective_batch}")
            print(f"  Tabular features: {cfg.num_tabular_features}")
            print(f"  DCNv2 layers:     {cfg.dcnv2_cross_layers}")
            print(f"{'='*60}\n")

        self.model.train()
        train_iter = iter(self.train_loader)
        running_loss = 0.0
        t0 = time.time()

        while self.step < cfg.max_steps:
            # Update LR (scale applied to base LRs)
            lr_scale = self._get_lr_scale(self.step)
            if cfg.freeze_transformer:
                base_lrs = [cfg.dcnv2_lr, cfg.fusion_lr]
            else:
                base_lrs = [cfg.transformer_lr, cfg.dcnv2_lr, cfg.fusion_lr]
            for pg, base_lr in zip(self.optimizer.param_groups, base_lrs):
                pg["lr"] = base_lr * lr_scale

            # Gradient accumulation
            for _ in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                features = batch["tabular_features"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
                    output = self.model(input_ids, features, attention_mask)
                    loss = F.cross_entropy(
                        output["logits"], labels,
                        label_smoothing=cfg.label_smoothing,
                    )
                    loss = loss / cfg.gradient_accumulation_steps

                loss.backward()
                running_loss += loss.item() * cfg.gradient_accumulation_steps

            # Optimizer step
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), cfg.grad_clip)
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            # Log
            if self.step % cfg.log_interval == 0 and self.rank == 0:
                avg_loss = running_loss / cfg.log_interval
                elapsed = time.time() - t0
                print(f"  step {self.step:5d}/{cfg.max_steps} | "
                      f"loss {avg_loss:.4f} | lr_scale {lr_scale:.3f} | {elapsed:.1f}s")
                running_loss = 0.0
                t0 = time.time()

            # Eval
            if self.step % cfg.eval_interval == 0:
                metrics = self.evaluate()
                if self.rank == 0 and metrics:
                    m_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"  [EVAL] step {self.step}: {m_str}")

                    if metrics.get("val_auc", 0) > self.best_auc:
                        self.best_auc = metrics["val_auc"]
                        self._save(f"{cfg.checkpoint_dir}/best.pt")

            # Periodic save
            if self.step % cfg.save_interval == 0:
                self._save(f"{cfg.checkpoint_dir}/step_{self.step:06d}.pt")

        self._save(f"{cfg.checkpoint_dir}/final.pt")
        if self.rank == 0:
            print(f"\n  Joint Fusion complete. Best AUC: {self.best_auc:.4f}")

    def _save(self, path: str):
        """Save full model checkpoint."""
        if self.rank != 0:
            return
        raw_model = self.model.module if self.distributed else self.model
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model": raw_model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "step": self.step,
            "best_auc": self.best_auc,
            "config": self.config,
        }, path)
        print(f"  Saved: {path}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="nuFormer Joint Fusion Training")
    parser.add_argument("--pretrain-ckpt", default="ckpt/pretrain/final.pt")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--transformer-lr", type=float, default=5e-6)
    parser.add_argument("--dcnv2-lr", type=float, default=1e-4)
    parser.add_argument("--fusion-lr", type=float, default=5e-5)
    parser.add_argument("--freeze-transformer", action="store_true",
                        help="Freeze transformer (feature extractor only)")
    parser.add_argument("--checkpoint-dir", type=str, default="ckpt/joint_fusion")
    args = parser.parse_args()

    config = JointFusionConfig(
        pretrain_checkpoint=args.pretrain_ckpt,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        transformer_lr=args.transformer_lr,
        dcnv2_lr=args.dcnv2_lr,
        fusion_lr=args.fusion_lr,
        freeze_transformer=args.freeze_transformer,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = JointFusionTrainer(config)
    trainer.train()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
