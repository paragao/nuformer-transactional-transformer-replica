"""Training pipelines for nuFormer.

Three stages:
1. pretrain   - Next Token Prediction (NTP) on transaction sequences
2. finetune   - LoRA fine-tuning + classification head
3. joint_fusion - End-to-end Transformer + DCNv2 + Fusion MLP
"""

from .pretrain import PreTrainer, PretrainConfig
from .finetune import FineTuner, FinetuneConfig
from .joint_fusion import JointFusionTrainer, JointFusionConfig

__all__ = [
    "PreTrainer",
    "PretrainConfig",
    "FineTuner",
    "FinetuneConfig",
    "JointFusionTrainer",
    "JointFusionConfig",
]
