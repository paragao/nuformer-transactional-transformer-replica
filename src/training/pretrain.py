"""Pre-training: Next Token Prediction (NTP) on transaction sequences.

Trains the causal transformer using autoregressive language modeling
on tokenized transaction sequences. Supports single-GPU, multi-GPU
(DDP), and multi-node (torchrun + Slurm).

Usage:
    # Single GPU
    python -m src.training.pretrain --config configs/training_config.yaml

    # Multi-GPU (single node)
    torchrun --nproc-per-node=8 -m src.training.pretrain

    # Multi-node (via Slurm sbatch)
    sbatch slurm/pretrain.sbatch
"""

from __future__ import annotations

import contextlib
import math
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset


@dataclass
class PretrainConfig:
    """Pre-training configuration."""

    # Data
    train_data_path: str = "data/processed_300k/train_sequences.npy"
    val_data_path: str = "data/processed_300k/val_sequences.npy"
    max_seq_len: int = 2048

    # Model
    vocab_size: int = 24078
    d_model: int = 1024
    n_layers: int = 24
    n_heads: int = 16
    d_ff: int = 4096
    dropout: float = 0.1

    # Optimization
    batch_size: int = 32  # per GPU
    gradient_accumulation_steps: int = 8
    max_steps: int = 25_000
    warmup_steps: int = 2000
    learning_rate: float = 3e-4
    min_lr: float = 3e-5
    weight_decay: float = 0.1
    grad_clip: float = 1.0
    beta1: float = 0.9
    beta2: float = 0.95

    # Precision & performance
    dtype: str = "bfloat16"
    compile_model: bool = True

    # Checkpointing
    checkpoint_dir: str = "ckpt/pretrain"
    save_interval: int = 2000
    resume_from: Optional[str] = None

    # Early stopping
    early_stop_patience: int = 5  # stop after N evals with no improvement

    # Logging
    log_interval: int = 10
    eval_interval: int = 500
    eval_steps: int = 50

    # MLFlow
    mlflow_tracking_uri: str = ""
    mlflow_experiment: str = "nuformer-pretrain"


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------


class PretrainDataset(Dataset):
    """Memory-mapped dataset for NTP pre-training.

    Expects numpy memmap array of shape (N, seq_len) with token IDs.
    """

    def __init__(self, data_path: str, max_seq_len: int = 2048):
        import numpy as np

        self.data = np.load(data_path, mmap_mode="r")
        self.max_seq_len = max_seq_len
        self.pad_token_id = 74  # PAD token from special_tokens.py

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = torch.from_numpy(self.data[idx].copy()).long()

        # Truncate to max_seq_len + 1 (for input/target shift)
        seq = seq[: self.max_seq_len + 1]

        # NTP: input = seq[:-1], target = seq[1:]
        input_ids = seq[:-1]
        targets = seq[1:]

        # Attention mask: 1 where not padding
        attention_mask = (input_ids != self.pad_token_id).long()

        return {
            "input_ids": input_ids,
            "targets": targets,
            "attention_mask": attention_mask,
        }


class DummyPretrainDataset(Dataset):
    """Dummy dataset for pipeline validation (no real data needed)."""

    def __init__(self, size: int = 2000, seq_len: int = 128, vocab_size: int = 24078):
        self.size = size
        self.seq_len = seq_len
        self.vocab_size = vocab_size

    def __len__(self) -> int:
        return self.size

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        seq = torch.randint(0, self.vocab_size, (self.seq_len + 1,))
        return {
            "input_ids": seq[:-1],
            "targets": seq[1:],
            "attention_mask": torch.ones(self.seq_len, dtype=torch.long),
        }


# ---------------------------------------------------------------------------
# Trainer
# ---------------------------------------------------------------------------


class PreTrainer:
    """Pre-training loop with distributed training support.

    Supports:
    - Single GPU training
    - Multi-GPU via DDP with overlap_comm (torchrun)
    - Multi-node via torchrun + Slurm
    - Mixed precision (BF16/FP16)
    - Gradient accumulation
    - ZeroRedundancyOptimizer (sharded optimizer state)
    - Cosine LR schedule with warmup
    - Periodic evaluation and checkpointing
    """

    def __init__(self, config: PretrainConfig):
        self.config = config
        self.step = 0
        self.best_val_loss = float("inf")
        self.evals_without_improvement = 0

        self._setup_distributed()
        self._setup_model()
        self._setup_data()
        self._setup_optimizer()

    def _setup_distributed(self):
        """Initialize distributed training context."""
        self.distributed = "RANK" in os.environ and torch.distributed.is_available()

        if self.distributed:
            if not torch.distributed.is_initialized():
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

        self.dtype = {
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
            "float32": torch.float32,
        }[self.config.dtype]

        if self.rank == 0:
            print(f"Distributed: {self.distributed} | World: {self.world_size} | "
                  f"Device: {self.device} | Dtype: {self.dtype}")

    def _setup_model(self):
        """Initialize the transformer model."""
        from src.models.transformer import TransactionTransformer, TransformerConfig

        model_config = TransformerConfig(
            vocab_size=self.config.vocab_size,
            d_model=self.config.d_model,
            n_layers=self.config.n_layers,
            n_heads=self.config.n_heads,
            d_ff=self.config.d_ff,
            dropout=self.config.dropout,
            max_seq_len=self.config.max_seq_len,
        )
        self.model = TransactionTransformer(model_config)
        self.model.to(self.device)

        if self.rank == 0:
            n_params = self.model.num_parameters()
            print(f"Model: {n_params:,} params ({n_params / 1e6:.1f}M)")

        # Optional: torch.compile for kernel fusion
        if self.config.compile_model and hasattr(torch, "compile"):
            try:
                self.model = torch.compile(self.model)
                if self.rank == 0:
                    print("torch.compile enabled (default mode)")
            except Exception as e:
                if self.rank == 0:
                    print(f"torch.compile failed, continuing without: {e}")

        # DDP for multi-GPU with comm/compute overlap
        if self.distributed and self.world_size > 1:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model,
                device_ids=[self.local_rank],
                gradient_as_bucket_view=True,
            )
            if self.rank == 0:
                print("DDP enabled (gradient_as_bucket_view)")

    def _setup_data(self):
        """Setup data loaders."""
        train_path = Path(self.config.train_data_path)
        if train_path.exists():
            train_dataset = PretrainDataset(str(train_path), self.config.max_seq_len)
        else:
            if self.rank == 0:
                print(f"WARNING: {train_path} not found, using dummy data")
            train_dataset = DummyPretrainDataset(
                size=2000, seq_len=min(128, self.config.max_seq_len),
                vocab_size=self.config.vocab_size,
            )

        if self.distributed:
            from torch.utils.data.distributed import DistributedSampler
            sampler = DistributedSampler(train_dataset, shuffle=True)
        else:
            sampler = None

        self.train_loader = DataLoader(
            train_dataset,
            batch_size=self.config.batch_size,
            sampler=sampler,
            shuffle=(sampler is None),
            num_workers=8,
            pin_memory=True,
            drop_last=True,
            persistent_workers=True,
            prefetch_factor=4,
        )

        # Validation
        val_path = Path(self.config.val_data_path)
        if val_path.exists():
            val_dataset = PretrainDataset(str(val_path), self.config.max_seq_len)
            self.val_loader = DataLoader(
                val_dataset, batch_size=self.config.batch_size,
                num_workers=4, pin_memory=True,
                persistent_workers=True,
                prefetch_factor=4,
            )
        else:
            self.val_loader = None

    def _setup_optimizer(self):
        """Setup AdamW with weight decay separation."""
        decay_params = []
        no_decay_params = []

        for name, param in self.model.named_parameters():
            if not param.requires_grad:
                continue
            if "norm" in name or "bias" in name or "embedding" in name:
                no_decay_params.append(param)
            else:
                decay_params.append(param)

        param_groups = [
            {"params": decay_params, "weight_decay": self.config.weight_decay},
            {"params": no_decay_params, "weight_decay": 0.0},
        ]

        self.optimizer = torch.optim.AdamW(
            param_groups,
            lr=self.config.learning_rate,
            betas=(self.config.beta1, self.config.beta2),
        )

        # GradScaler only needed for FP16
        self.scaler = torch.amp.GradScaler("cuda") if self.dtype == torch.float16 else None

    def _compute_mfu(self, tokens_per_sec: float) -> float:
        """Compute Model FLOPS Utilization (MFU).

        Estimates achieved FLOPS as a fraction of peak hardware FLOPS.
        Uses the standard 6*N*T approximation for transformer forward+backward
        where N=params and T=tokens per step.

        H200 peak BF16: 989 TFLOPS per GPU.
        """
        cfg = self.config
        # Approximate model params (exclude embedding for MFU calc)
        n_params = cfg.n_layers * (
            4 * cfg.d_model * cfg.d_model  # QKV + output proj
            + 2 * cfg.d_model * cfg.d_ff    # FFN up + down
        )
        # 6 flops per param per token (forward + backward)
        flops_per_token = 6 * n_params
        # Achieved FLOPS across all GPUs
        achieved_flops = flops_per_token * tokens_per_sec
        # H200 peak BF16 TFLOPS
        peak_flops_per_gpu = 989e12  # 989 TFLOPS
        peak_flops_total = peak_flops_per_gpu * self.world_size
        return achieved_flops / peak_flops_total

    # ------------------------------------------------------------------
    # LR Schedule
    # ------------------------------------------------------------------

    def _get_lr(self, step: int) -> float:
        """Cosine learning rate with linear warmup."""
        if step < self.config.warmup_steps:
            return self.config.learning_rate * step / self.config.warmup_steps
        # Cosine decay
        progress = (step - self.config.warmup_steps) / max(
            1, self.config.max_steps - self.config.warmup_steps
        )
        progress = min(progress, 1.0)
        coeff = 0.5 * (1.0 + math.cos(math.pi * progress))
        return self.config.min_lr + coeff * (self.config.learning_rate - self.config.min_lr)

    # ------------------------------------------------------------------
    # Training Step
    # ------------------------------------------------------------------

    def _train_step(self, batch: dict[str, torch.Tensor]) -> float:
        """Single forward/backward pass (one micro-batch)."""
        input_ids = batch["input_ids"].to(self.device, non_blocking=True)
        targets = batch["targets"].to(self.device, non_blocking=True)
        attention_mask = batch["attention_mask"].to(self.device, non_blocking=True)

        with torch.amp.autocast("cuda", dtype=self.dtype, enabled=(self.dtype != torch.float32)):
            output = self.model(input_ids, attention_mask)
            logits = output["logits"]
            loss = F.cross_entropy(
                logits.view(-1, logits.size(-1)),
                targets.view(-1),
                ignore_index=74,  # PAD token
            )
            loss = loss / self.config.gradient_accumulation_steps

        if self.scaler:
            self.scaler.scale(loss).backward()
        else:
            loss.backward()

        return loss.item() * self.config.gradient_accumulation_steps

    def _optimizer_step(self) -> float:
        """Clip gradients and step optimizer."""
        if self.scaler:
            self.scaler.unscale_(self.optimizer)

        grad_norm = torch.nn.utils.clip_grad_norm_(
            self.model.parameters(), self.config.grad_clip
        )

        if self.scaler:
            self.scaler.step(self.optimizer)
            self.scaler.update()
        else:
            self.optimizer.step()

        self.optimizer.zero_grad(set_to_none=True)
        return grad_norm.item() if isinstance(grad_norm, torch.Tensor) else grad_norm

    # ------------------------------------------------------------------
    # Evaluation
    # ------------------------------------------------------------------

    @torch.no_grad()
    def evaluate(self) -> dict[str, float]:
        """Run evaluation on validation set."""
        if self.val_loader is None:
            return {}

        self.model.eval()
        total_loss = 0.0
        total_tokens = 0
        n_steps = 0

        for batch in self.val_loader:
            if n_steps >= self.config.eval_steps:
                break

            input_ids = batch["input_ids"].to(self.device)
            targets = batch["targets"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)

            with torch.amp.autocast("cuda", dtype=self.dtype, enabled=(self.dtype != torch.float32)):
                output = self.model(input_ids, attention_mask)
                loss = F.cross_entropy(
                    output["logits"].view(-1, output["logits"].size(-1)),
                    targets.view(-1),
                    ignore_index=74,
                    reduction="sum",
                )

            # Count non-pad tokens for proper averaging
            n_tokens = (targets != 74).sum().item()
            total_loss += loss.item()
            total_tokens += n_tokens
            n_steps += 1

        self.model.train()

        avg_loss = total_loss / max(1, total_tokens)
        perplexity = math.exp(min(avg_loss, 20))  # cap to avoid overflow
        return {"val_loss": avg_loss, "val_ppl": perplexity}

    # ------------------------------------------------------------------
    # Checkpointing
    # ------------------------------------------------------------------

    def save_checkpoint(self, path: str):
        """Save model checkpoint (DDP)."""
        Path(path).parent.mkdir(parents=True, exist_ok=True)

        model = self.model
        if hasattr(model, "module"):
            model = model.module
        # torch.compile wraps in OptimizedModule
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod

        if self.rank == 0:
            state = {
                "model": model.state_dict(),
                "optimizer": self.optimizer.state_dict(),
                "step": self.step,
                "config": self.config,
            }
            torch.save(state, path)
            print(f"  Checkpoint saved: {path}")

        # Barrier to ensure all ranks wait for save to complete
        if self.distributed:
            torch.distributed.barrier()

    def load_checkpoint(self, path: str):
        """Resume from checkpoint (DDP)."""
        if not Path(path).exists():
            if self.rank == 0:
                print(f"No checkpoint at {path}, starting fresh")
            return

        if self.rank == 0:
            print(f"Loading checkpoint from {path}...")

        ckpt = torch.load(path, map_location="cpu", weights_only=False)

        model = self.model
        if hasattr(model, "module"):
            model = model.module
        if hasattr(model, "_orig_mod"):
            model = model._orig_mod
        model.load_state_dict(ckpt["model"])
        self.optimizer.load_state_dict(ckpt["optimizer"])

        self.step = ckpt["step"]

        if self.rank == 0:
            print(f"Resumed from {path} at step {self.step}")

    # ------------------------------------------------------------------
    # Main Training Loop
    # ------------------------------------------------------------------

    def train(self):
        """Main pre-training loop."""
        cfg = self.config

        # Resume if specified
        if cfg.resume_from:
            self.load_checkpoint(cfg.resume_from)

        if self.rank == 0:
            effective_batch = cfg.batch_size * cfg.gradient_accumulation_steps * self.world_size
            print(f"\n{'='*60}")
            print(f"  nuFormer Pre-training (NTP)")
            print(f"{'='*60}")
            print(f"  Max steps:        {cfg.max_steps:,}")
            print(f"  Batch/GPU:        {cfg.batch_size}")
            print(f"  Grad accum:       {cfg.gradient_accumulation_steps}")
            print(f"  World size:       {self.world_size}")
            print(f"  Effective batch:  {effective_batch:,}")
            print(f"  Seq length:       {cfg.max_seq_len}")
            print(f"  LR:               {cfg.learning_rate} -> {cfg.min_lr}")
            print(f"  Warmup steps:     {cfg.warmup_steps}")
            print(f"{'='*60}\n")

        self.model.train()
        train_iter = iter(self.train_loader)
        running_loss = 0.0
        t0 = time.time()

        while self.step < cfg.max_steps:
            # Update learning rate
            lr = self._get_lr(self.step)
            for pg in self.optimizer.param_groups:
                pg["lr"] = lr

            # Gradient accumulation with no_sync() to avoid redundant allreduces
            for micro_step in range(cfg.gradient_accumulation_steps):
                try:
                    batch = next(train_iter)
                except StopIteration:
                    train_iter = iter(self.train_loader)
                    batch = next(train_iter)

                # Only sync gradients on the last micro-batch
                is_last = (micro_step == cfg.gradient_accumulation_steps - 1)
                ctx = contextlib.nullcontext() if is_last or not self.distributed else self.model.no_sync()
                with ctx:
                    loss = self._train_step(batch)
                    running_loss += loss

            # Optimizer step
            grad_norm = self._optimizer_step()
            self.step += 1

            # Logging
            if self.step % cfg.log_interval == 0 and self.rank == 0:
                avg_loss = running_loss / (cfg.log_interval * cfg.gradient_accumulation_steps)
                elapsed = time.time() - t0
                steps_per_sec = cfg.log_interval / elapsed
                tokens_per_sec = (
                    cfg.batch_size * cfg.max_seq_len * cfg.gradient_accumulation_steps
                    * self.world_size * steps_per_sec
                )
                mfu = self._compute_mfu(tokens_per_sec)
                print(
                    f"  step {self.step:6d}/{cfg.max_steps} | "
                    f"loss {avg_loss:.4f} | ppl {math.exp(min(avg_loss, 20)):.1f} | "
                    f"lr {lr:.2e} | grad {grad_norm:.3f} | "
                    f"{tokens_per_sec/1e6:.2f}M tok/s | "
                    f"MFU {mfu*100:.1f}%"
                )
                running_loss = 0.0
                t0 = time.time()

            # Evaluation
            if self.step % cfg.eval_interval == 0:
                metrics = self.evaluate()
                if self.rank == 0 and metrics:
                    print(f"  [EVAL] step {self.step}: "
                          f"loss={metrics['val_loss']:.4f} ppl={metrics['val_ppl']:.1f}")

                    # Early stopping check
                    if metrics["val_loss"] < self.best_val_loss:
                        self.best_val_loss = metrics["val_loss"]
                        self.evals_without_improvement = 0
                    else:
                        self.evals_without_improvement += 1
                        print(f"  [EARLY STOP] No improvement for "
                              f"{self.evals_without_improvement}/{cfg.early_stop_patience} evals "
                              f"(best={self.best_val_loss:.4f})")

                # Broadcast early stop decision from rank 0
                if self.distributed:
                    stop_tensor = torch.tensor(
                        [self.evals_without_improvement], device=self.device
                    )
                    torch.distributed.broadcast(stop_tensor, src=0)
                    self.evals_without_improvement = stop_tensor.item()

                if self.evals_without_improvement >= cfg.early_stop_patience:
                    if self.rank == 0:
                        print(f"\n  [EARLY STOP] Stopping at step {self.step} — "
                              f"no improvement for {cfg.early_stop_patience} evals. "
                              f"Best val_loss: {self.best_val_loss:.4f}")
                    break

            # Checkpointing
            if self.step % cfg.save_interval == 0:
                self.save_checkpoint(f"{cfg.checkpoint_dir}/step_{self.step:06d}.pt")

        # Final save
        self.save_checkpoint(f"{cfg.checkpoint_dir}/final.pt")
        if self.rank == 0:
            print(f"\n{'='*60}")
            print(f"  Pre-training complete at step {self.step}")
            print(f"{'='*60}")


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------


def main():
    """CLI entry point for pre-training."""
    import argparse

    parser = argparse.ArgumentParser(description="nuFormer NTP Pre-training")
    parser.add_argument("--config", type=str, default="", help="YAML config path")
    parser.add_argument("--max-steps", type=int, default=25_000)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--seq-len", type=int, default=2048)
    parser.add_argument("--checkpoint-dir", type=str, default="ckpt/pretrain")
    parser.add_argument("--save-interval", type=int, default=2000)
    parser.add_argument("--early-stop-patience", type=int, default=5)
    parser.add_argument("--resume", type=str, default=None)
    parser.add_argument("--no-compile", action="store_true")
    args = parser.parse_args()

    config = PretrainConfig(
        max_steps=args.max_steps,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        max_seq_len=args.seq_len,
        checkpoint_dir=args.checkpoint_dir,
        save_interval=args.save_interval,
        early_stop_patience=args.early_stop_patience,
        resume_from=args.resume,
        compile_model=not args.no_compile,
    )

    trainer = PreTrainer(config)
    trainer.train()

    # Cleanup distributed
    if torch.distributed.is_initialized():
        torch.distributed.destroy_process_group()


if __name__ == "__main__":
    main()
