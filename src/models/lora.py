"""LoRA: Low-Rank Adaptation for fine-tuning.

Implements LoRA (Hu et al., 2021) for parameter-efficient fine-tuning
of the pre-trained transformer. Only adapts attention Q/K/V projections.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """Linear layer with LoRA adaptation.

    W' = W + alpha/r * B @ A
    Where A: (in, r), B: (r, out), r << min(in, out)
    """

    def __init__(self, original: nn.Linear, rank: int = 16, alpha: float = 32.0):
        super().__init__()
        self.original = original
        self.rank = rank
        self.alpha = alpha
        self.scaling = alpha / rank

        in_features = original.in_features
        out_features = original.out_features

        # Freeze original weights
        original.weight.requires_grad = False
        if original.bias is not None:
            original.bias.requires_grad = False

        # LoRA matrices
        self.lora_A = nn.Parameter(torch.zeros(in_features, rank))
        self.lora_B = nn.Parameter(torch.zeros(rank, out_features))

        # Initialize A with random normal, B with zeros (so initial output = original)
        nn.init.kaiming_uniform_(self.lora_A, a=5**0.5)
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Forward with LoRA adaptation."""
        # Original forward
        out = self.original(x)
        # LoRA delta
        lora_out = (x @ self.lora_A @ self.lora_B) * self.scaling
        return out + lora_out

    def merge(self) -> nn.Linear:
        """Merge LoRA weights into original linear layer (for inference)."""
        merged = nn.Linear(
            self.original.in_features,
            self.original.out_features,
            bias=self.original.bias is not None,
        )
        merged.weight.data = self.original.weight.data + (self.lora_A @ self.lora_B).T * self.scaling
        if self.original.bias is not None:
            merged.bias.data = self.original.bias.data
        return merged


def apply_lora(model: nn.Module, rank: int = 16, alpha: float = 32.0, target_modules: list[str] = None) -> nn.Module:
    """Apply LoRA to target modules in a model.

    Args:
        model: The model to adapt
        rank: LoRA rank
        alpha: LoRA scaling factor
        target_modules: List of module name patterns to adapt
                       Default: ['qkv_proj'] (attention projections)

    Returns:
        Model with LoRA-adapted layers (original weights frozen)
    """
    if target_modules is None:
        target_modules = ["qkv_proj", "out_proj"]

    # Freeze all parameters first
    for param in model.parameters():
        param.requires_grad = False

    # Apply LoRA to target modules
    lora_params = 0
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            # Check if this module matches any target
            if any(target in name for target in target_modules):
                # Replace with LoRA version
                parent_name = ".".join(name.split(".")[:-1])
                child_name = name.split(".")[-1]
                parent = model.get_submodule(parent_name) if parent_name else model
                lora_layer = LoRALinear(module, rank=rank, alpha=alpha)
                setattr(parent, child_name, lora_layer)
                lora_params += lora_layer.lora_A.numel() + lora_layer.lora_B.numel()

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    print(f"LoRA applied: {lora_params:,} LoRA params, "
          f"{trainable_params:,} trainable / {total_params:,} total "
          f"({trainable_params / total_params * 100:.2f}%)")

    return model


def get_lora_state_dict(model: nn.Module) -> dict[str, torch.Tensor]:
    """Extract only LoRA parameters from model state dict."""
    return {
        name: param
        for name, param in model.state_dict().items()
        if "lora_A" in name or "lora_B" in name
    }
