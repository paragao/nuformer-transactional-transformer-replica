"""Fine-tuning: LoRA adaptation for downstream classification.

Loads a pre-trained transformer checkpoint, applies LoRA to attention
projections (~1% trainable params), and trains a classification head
for credit card activation prediction.

Usage:
    python -m src.training.finetune --pretrain-ckpt ckpt/pretrain/final.pt
    torchrun --nproc-per-node=8 -m src.training.finetune --pretrain-ckpt ...
"""

from __future__ import annotations

import math
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import sys
import pickle
import io

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from src.training.pretrain import PretrainConfig

# Ensure __main__ has PretrainConfig so torch.load can unpickle checkpoints
# saved when pretrain.py ran as __main__
sys.modules["__main__"].PretrainConfig = PretrainConfig


@dataclass
class FinetuneConfig:
    """Fine-tuning configuration."""

    # Data
    train_data_path: str = "data/processed_300k/train_sequences.npy"
    train_labels_path: str = "data/processed_300k/train_labels.npy"
    val_data_path: str = "data/processed_300k/val_sequences.npy"
    val_labels_path: str = "data/processed_300k/val_labels.npy"
    max_seq_len: int = 2048

    # Pre-trained model
    pretrain_checkpoint: str = "ckpt/pretrain/final.pt"

    # Model
    vocab_size: int = 24078
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 4096
    num_classes: int = 2

    # LoRA
    lora_rank: int = 16
    lora_alpha: float = 32.0
    lora_target_modules: list = None

    # Optimization
    batch_size: int = 64
    gradient_accumulation_steps: int = 2
    max_steps: int = 5_000
    warmup_steps: int = 300
    learning_rate: float = 2e-5
    head_lr_multiplier: float = 2.0
    min_lr: float = 1e-6
    weight_decay: float = 0.05
    grad_clip: float = 1.0

    # Precision
    dtype: str = "bfloat16"

    # Checkpointing & logging
    checkpoint_dir: str = "ckpt/finetune"
    log_interval: int = 10
    eval_interval: int = 200
    save_interval: int = 1000

    def __post_init__(self):
        if self.lora_target_modules is None:
            self.lora_target_modules = ["qkv_proj", "out_proj"]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class FinetuneDataset(Dataset):
    """Dataset for classification fine-tuning: sequences + labels."""

    def __init__(self, sequences_path: str, labels_path: str, max_seq_len: int = 2048):
        import numpy as np

        self.sequences = np.load(sequences_path, mmap_mode="r")
        self.labels = np.load(labels_path, mmap_mode="r")
        self.max_seq_len = max_seq_len
        self.pad_token_id = 74

    def __len__(self) -> int:
        return len(self.sequences)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = torch.from_numpy(self.sequences[idx].copy()).long()[:self.max_seq_len]
        label = torch.tensor(int(self.labels[idx]), dtype=torch.long)
        attention_mask = (seq != self.pad_token_id).long()
        return {"input_ids": seq, "attention_mask": attention_mask, "labels": label}


class DummyFinetuneDataset(Dataset):
    """Dummy dataset for pipeline validation."""

    def __init__(self, size: int = 500, seq_len: int = 128, vocab_size: int = 24078):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        return {
            "input_ids": torch.randint(0, self.vocab_size, (self.seq_len,)),
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
            "labels": torch.randint(0, 2, ()),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class FineTuner:
    """Fine-tuning loop with LoRA + classification head."""

    def __init__(self, config: FinetuneConfig):
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
        """Load pre-trained transformer, apply LoRA, add classification head."""
        from src.models.transformer import TransactionTransformer, TransformerConfig
        from src.models.lora import apply_lora, get_lora_state_dict
        from src.models.nuformer import FineTuneHead

        # Build transformer
        model_config = TransformerConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            max_seq_len=self.config.max_seq_len,
            dropout=0.0,  # no dropout during fine-tuning (LoRA acts as regularizer)
        )
        self.transformer = TransactionTransformer(model_config)

        # Load pre-trained weights
        ckpt_path = Path(self.config.pretrain_checkpoint)
        if ckpt_path.exists():
            if self.rank == 0:
                print(f"Loading pre-trained: {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            self.transformer.load_state_dict(ckpt["model"], strict=False)
        else:
            if self.rank == 0:
                print(f"WARNING: No checkpoint at {ckpt_path}, using random init")

        # Apply LoRA (freezes all params, adds trainable LoRA adapters)
        self.transformer = apply_lora(
            self.transformer,
            rank=self.config.lora_rank,
            alpha=self.config.lora_alpha,
            target_modules=self.config.lora_target_modules,
        )

        # Classification head (fully trainable)
        self.head = FineTuneHead(
            d_model=self.config.d_model,
            num_classes=self.config.num_classes,
        )

        self.transformer.to(self.device)
        self.head.to(self.device)

        if self.rank == 0:
            lora_params = sum(
                p.numel() for p in self.transformer.parameters() if p.requires_grad
            )
            head_params = sum(p.numel() for p in self.head.parameters())
            total = sum(p.numel() for p in self.transformer.parameters()) + head_params
            print(f"Trainable: LoRA={lora_params:,} + Head={head_params:,} / "
                  f"Total={total:,} ({(lora_params + head_params) / total * 100:.2f}%)")

    def _setup_data(self):
        """Setup data loaders."""
        train_path = Path(self.config.train_data_path)
        if train_path.exists() and Path(self.config.train_labels_path).exists():
            train_dataset = FinetuneDataset(
                self.config.train_data_path,
                self.config.train_labels_path,
                self.config.max_seq_len,
            )
        else:
            if self.rank == 0:
                print("WARNING: Using dummy fine-tuning data")
            train_dataset = DummyFinetuneDataset(
                size=500, seq_len=128, vocab_size=self.config.vocab_size
            )

        self.train_loader = DataLoader(
            train_dataset, batch_size=self.config.batch_size,
            shuffle=True, num_workers=4, pin_memory=True, drop_last=True,
        )

        val_path = Path(self.config.val_data_path)
        if val_path.exists() and Path(self.config.val_labels_path).exists():
            val_dataset = FinetuneDataset(
                self.config.val_data_path,
                self.config.val_labels_path,
                self.config.max_seq_len,
            )
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size, num_workers=2
            )
        else:
            self.val_loader = None

    def _setup_optimizer(self):
        """Separate LR for LoRA params vs head params."""
        lora_params = [p for p in self.transformer.parameters() if p.requires_grad]
        head_params = list(self.head.parameters())

        self.optimizer = torch.optim.AdamW([
            {"params": lora_params, "lr": self.config.learning_rate},
            {"params": head_params, "lr": self.config.learning_rate * self.config.head_lr_multiplier},
        ], weight_decay=self.config.weight_decay)

    def _get_lr(self, step: int) -> float:
        """Cosine schedule with warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        coeff = 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Evaluate: loss, accuracy, AUC."""
        if self.val_loader is None:
            return {}

        self.transformer.eval()
        self.head.eval()

        all_probs = []
        all_labels = []
        total_loss = 0.0
        correct = 0
        total = 0

        for batch in self.val_loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = batch["labels"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
                user_emb = self.transformer.get_user_embedding(input_ids, attention_mask)
                logits = self.head(user_emb)
                loss = F.cross_entropy(logits, labels)

            total_loss += loss.item() * labels.size(0)
            preds = logits.argmax(dim=-1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

            probs = F.softmax(logits, dim=-1)[:, 1]
            all_probs.extend(probs.cpu().tolist())
            all_labels.extend(labels.cpu().tolist())

        self.transformer.train()
        self.head.train()

        metrics = {
            "val_loss": total_loss / max(1, total),
            "val_acc": correct / max(1, total),
        }

        # AUC if sklearn available
        try:
            from sklearn.metrics import roc_auc_score
            if len(set(all_labels)) > 1:
                metrics["val_auc"] = roc_auc_score(all_labels, all_probs)
        except ImportError:
            pass

        return metrics

    def train(self):
        """Main fine-tuning loop."""
        cfg = self.config

        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"  nuFormer Fine-tuning (LoRA r={cfg.lora_rank})")
            print(f"{'='*60}")
            print(f"  Max steps:    {cfg.max_steps:,}")
            print(f"  Batch/GPU:    {cfg.batch_size}")
            print(f"  LR (LoRA):    {cfg.learning_rate}")
            print(f"  LR (Head):    {cfg.learning_rate * cfg.head_lr_multiplier}")
            print(f"{'='*60}\n")

        self.transformer.train()
        self.head.train()
        train_iter = iter(self.train_loader)
        running_loss = 0.0
        t0 = time.time()

        while self.step < cfg.max_steps:
            lr = self._get_lr(self.step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr * (cfg.head_lr_multiplier if pg is self.optimizer.param_groups[1] else 1.0)

            for _ in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                input_ids = batch["input_ids"].to(self.device, non_blocking=True)
                attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)
                labels = batch["labels"].to(self.device, non_blocking=True)

                with torch.amp.autocast("cuda", dtype=self.dtype, enabled=torch.cuda.is_available()):
                    user_emb = self.transformer.get_user_embedding(input_ids, attention_mask)
                    logits = self.head(user_emb)
                    loss = F.cross_entropy(logits, labels)
                    loss = loss / cfg.gradient_accumulation_steps

                loss.backward()
                running_loss += loss.item() * cfg.gradient_accumulation_steps

            torch.nn.utils.clip_grad_norm_(
                list(self.transformer.parameters()) + list(self.head.parameters()),
                cfg.grad_clip,
            )
            self.optimizer.step()
            self.optimizer.zero_grad(set_to_none=True)
            self.step += 1

            # Log
            if self.step % cfg.log_interval == 0 and self.rank == 0:
                avg_loss = running_loss / cfg.log_interval
                elapsed = time.time() - t0
                print(f"  step {self.step:5d}/{cfg.max_steps} | "
                      f"loss {avg_loss:.4f} | lr {lr:.2e} | {elapsed:.1f}s")
                running_loss = 0.0
                t0 = time.time()

            # Eval
            if self.step % cfg.eval_interval == 0:
                metrics = self.evaluate()
                if self.rank == 0 and metrics:
                    m_str = " | ".join(f"{k}={v:.4f}" for k, v in metrics.items())
                    print(f"  [EVAL] step {self.step}: {m_str}")

                    # Save best by AUC
                    if metrics.get("val_auc", 0) > self.best_auc:
                        self.best_auc = metrics["val_auc"]
                        self._save(f"{cfg.checkpoint_dir}/best.pt")

            # Periodic save
            if self.step % cfg.save_interval == 0:
                self._save(f"{cfg.checkpoint_dir}/step_{self.step:06d}.pt")

        self._save(f"{cfg.checkpoint_dir}/final.pt")
        if self.rank == 0:
            print(f"\n  Fine-tuning complete. Best AUC: {self.best_auc:.4f}")

    def _save(self, path: str):
        """Save LoRA + head checkpoint."""
        if self.rank != 0:
            return
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        from src.models.lora import get_lora_state_dict
        state = {
            "lora": get_lora_state_dict(self.transformer),
            "head": self.head.state_dict(),
            "step": self.step,
            "best_auc": self.best_auc,
        }
        torch.save(state, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    import argparse

    parser = argparse.ArgumentParser(description="nuFormer LoRA Fine-tuning")
    parser.add_argument("--pretrain-ckpt", default="ckpt/pretrain/final.pt")
    parser.add_argument("--max-steps", type=int, default=10_000)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--weight-decay", type=float, default=0.05)
    parser.add_argument("--head-lr-multiplier", type=float, default=2.0)
    parser.add_argument("--lora-rank", type=int, default=16)
    parser.add_argument("--checkpoint-dir", type=str, default="ckpt/finetune")
    args = parser.parse_args()

    config = FinetuneConfig(
        pretrain_checkpoint=args.pretrain_ckpt,
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        head_lr_multiplier=args.head_lr_multiplier,
        lora_rank=args.lora_rank,
        checkpoint_dir=args.checkpoint_dir,
    )

    trainer = FineTuner(config)
    trainer.train()

    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
