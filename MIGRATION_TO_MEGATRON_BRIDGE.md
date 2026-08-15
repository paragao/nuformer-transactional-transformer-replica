# Migration to Megatron-Core: Assessment & Plan

## 1. Motivation

### Why Consider Megatron-Core?

| Benefit | Relevance to nuFormer (329M, single-node) |
|---------|-------------------------------------------|
| Tensor Parallelism (TP) | Low — model fits on 1 GPU |
| Pipeline Parallelism (PP) | Low — same reason |
| FP8 via TransformerEngine | Medium — ~1.5x throughput gain |
| Sequence Parallelism (SP) | Medium — useful if scaling context to 4096+ |
| Context Parallelism (CP) | Low — not at 100K+ seq length |
| Distributed Checkpointing (DCP) | Low — single-node, small model |
| Multi-node scaling | High — required if scaling to 100M+ real users |
| NeMo 2.0 ecosystem (Curator, PEFT) | Medium — ready-made LoRA, data pipelines |

**Bottom line**: The primary reasons to migrate are (1) multi-node scaling when moving to real
production data at Nubank scale, and (2) FP8 throughput gains via TransformerEngine. At 329M
parameters on 8x H200 single-node with synthetic data, the current PyTorch-native stack is
sufficient.

## 2. Current Architecture vs Megatron Target

| Component | Current (PyTorch Native) | Megatron-Core Target |
|-----------|--------------------------|----------------------|
| Transformer | `CausalTransformer(nn.Module)` | `megatron.core.models.gpt.GPTModel` |
| Attention | flash-attn manual integration | TransformerEngine fused attention |
| Positional encoding | NoPE (custom) | `position_embedding_type="none"` |
| LoRA | Custom `lora.py` | NeMo PEFT `LoRAAdapter` mixin |
| Precision | BF16 via `torch.amp` | FP8/BF16 via TransformerEngine |
| Distribution | DDP (single-node 8 GPU) | FSDP2 / TP / PP (multi-node) |
| DCNv2 + PLR | Custom `dcnv2.py` + `numerical_embeddings.py` | **Stays as plain PyTorch** |
| Fusion MLP | Custom `nuformer.py` | **Stays as plain PyTorch** |
| Data pipeline | numpy `.npy` + custom `Dataset` | Megatron indexed binary (`.bin`+`.idx`) |
| Checkpointing | `torch.save` / `torch.load` | Megatron `save_checkpoint` + DCP |
| Training loop | Custom loop in `joint_fusion.py` | Megatron `pretrain()` with callbacks |
| Optimizer | Per-group AdamW (3 LR schedules) | Megatron optimizer (needs customization) |

## 3. Migration Strategy (Hybrid Approach)

### Option A: Hybrid Architecture (Recommended)

Wrap only the transformer tower in Megatron-Core. Keep DCNv2+PLR and the fusion MLP as plain
PyTorch modules. This is the pragmatic approach because:

- The DCNv2 branch is small (~1M params) — no benefit from TP/PP
- PLR embeddings are a simple batched matrix op — no Megatron abstraction needed
- The fusion MLP is trivial (3 linear layers)
- Megatron's GPTModel already supports NoPE + FlashAttention natively

```python
class NuFormerMegatron(MegatronModule):
    def __init__(self, config, transformer_config):
        super().__init__(config)
        # Megatron-managed transformer (329M params)
        self.transformer = GPTModel(
            config=transformer_config,
            transformer_layer_spec=get_gpt_layer_with_transformer_engine_spec(),
            position_embedding_type="none",
        )
        # Plain PyTorch — unchanged from current implementation
        self.dcnv2 = DCNv2WithPLR(
            num_features=291, plr_dim=16, num_frequencies=8,
            cross_layers=5, deep_layers=[1024, 512, 256], output_dim=128
        )
        self.fusion_mlp = FusionMLP(input_dim=1152, hidden_dim=256, num_classes=2)

    def forward(self, tokens, tabular_features):
        # Transformer branch (Megatron handles TP/SP internally)
        txn_emb = self.transformer(tokens)[:, -1, :]  # last-token
        # DCNv2 branch (plain PyTorch)
        feat_emb = self.dcnv2(tabular_features)
        # Fusion
        fused = torch.cat([txn_emb, feat_emb], dim=-1)
        return self.fusion_mlp(fused)
```

### Option B: Full Megatron Spec (Not recommended)

Write a custom `ModuleSpec` that registers DCNv2 as a Megatron submodule. This would require:
- Conforming to Megatron's `TransformerLayer` spec interface
- Handling DCNv2's non-standard forward pass within Megatron's pipeline
- No practical benefit since DCNv2 doesn't need parallelism

**Verdict**: Option A. Only the transformer benefits from Megatron's parallelism and kernels.

## 4. Component Breakdown

### 4.1 Transformer Tower

**Effort**: 2-3 days | **Risk**: Low

The causal transformer maps directly to Megatron's `GPTModel`:

```python
from megatron.core.transformer.transformer_config import TransformerConfig
from megatron.core.models.gpt import GPTModel
from megatron.core.transformer.spec_utils import ModuleSpec

transformer_config = TransformerConfig(
    num_layers=24,
    hidden_size=1024,
    num_attention_heads=16,
    ffn_hidden_size=4096,
    hidden_dropout=0.0,
    attention_dropout=0.0,
    fp16=False,
    bf16=True,
    fp8=True,  # Enable FP8 via TransformerEngine
    position_embedding_type="none",  # NoPE
    use_flash_attention=True,
)
```

Key changes:
- Remove custom `transformer.py` — replaced by Megatron's GPTModel
- FlashAttention is handled internally by TransformerEngine
- NoPE is a config option (`position_embedding_type="none"`)
- FP8 is enabled with a single flag (TransformerEngine handles scaling)
- Vocab size (24,078) and max seq length (2048) set in model config

### 4.2 DCNv2 + PLR Tower (Hybrid Wrapper)

**Effort**: 1-2 days | **Risk**: Medium (optimizer group handling)

The DCNv2 and PLR modules stay as-is (`src/models/dcnv2.py`, `src/models/numerical_embeddings.py`).
The only integration point is ensuring Megatron's optimizer handles these parameters correctly.

Key considerations:
- DCNv2+PLR params (~1M) must use a separate LR (3e-4 or 5e-4) from the transformer (5e-6)
- Megatron's `OptimizerConfig` typically uses a single LR schedule
- Solution: register DCNv2+PLR params in a separate param group using Megatron's
  `get_megatron_optimizer()` with `param_groups` override

```python
# Param group separation for multi-LR training
param_groups = [
    {"params": transformer_params, "lr": 5e-6, "name": "transformer"},
    {"params": dcnv2_plr_params, "lr": 5e-4, "name": "dcnv2_plr"},
    {"params": fusion_mlp_params, "lr": 5e-5, "name": "fusion"},
]
```

No changes to `VectorizedPLREmbedder` or `DCNv2` forward pass. The `NumericalFeatureEmbedder`
alias remains for backward compatibility.

### 4.3 Data Pipeline

**Effort**: 2-3 days | **Risk**: Medium (custom side-features)

**Current**: numpy `.npy` files → custom `JointFusionDataset` → DataLoader + DistributedSampler.

**Megatron expects**: indexed binary datasets (`.bin` + `.idx`) created by `preprocess_data.py`.

The challenge is that Megatron's data pipeline has no concept of "side features" — it only handles
token sequences. Options:

**Option 1: Dual-loader (Recommended)**
- Convert tokenized sequences to Megatron indexed format for the transformer branch
- Keep tabular features in numpy/memmap, loaded by a parallel PyTorch DataLoader
- Synchronize batches by user ID

**Option 2: Custom MegatronDataset subclass**
- Subclass `GPTDataset` to return `(tokens, tabular_features, labels)` tuples
- Override `__getitem__` to load both modalities
- More tightly integrated but requires understanding Megatron's dataset internals

**Conversion script** (for Option 1):
```bash
python tools/preprocess_data.py \
    --input /fsx/paragao/nuformer/data/train_sequences.jsonl \
    --tokenizer-type GPT2BPETokenizer \
    --vocab-file /fsx/paragao/nuformer/tokenizer/vocab.json \
    --merges-file /fsx/paragao/nuformer/tokenizer/merges.txt \
    --output-prefix /fsx/paragao/nuformer/data/megatron/train \
    --workers 32
```

Note: Our BPE tokenizer (24K vocab) would need to be wrapped in Megatron's tokenizer interface.

### 4.4 Training Loop

**Effort**: 3-4 days | **Risk**: High (multi-LR, early stop, eval)

**Current**: custom training loop with manual DDP, gradient accumulation, periodic eval, early
stopping (patience=5).

**Megatron**: uses `pretrain()` entry point with callback functions:
- `forward_step_func(batch, model)` — runs the model and computes loss
- `train_valid_test_datasets_provider(train_val_test_num_samples)` — returns datasets

Key challenges:

1. **Multi-LR optimizer**: Megatron's default is a single LR schedule. Our joint fusion uses 3
   rates (transformer=5e-6, DCNv2=5e-4, fusion=5e-5). Requires custom `OptimizerConfig` or
   param group overrides in `get_megatron_optimizer()`.

2. **Early stopping**: Not built into Megatron's training loop. Must add a callback/hook that
   monitors val AUC and triggers `sys.exit()` or sets a flag to break the training loop.

3. **Custom metrics**: Megatron tracks loss/LR by default. AUC, AP, and per-class accuracy
   require custom logging in the `forward_step_func` or a validation hook.

4. **Label smoothing**: Must be applied in `forward_step_func` — Megatron's default cross-entropy
   doesn't support it.

```python
def forward_step_func(batch, model):
    tokens, tabular_features, labels = batch
    logits = model(tokens, tabular_features)
    loss = label_smoothed_cross_entropy(logits, labels, smoothing=0.05)
    return loss, {"logits": logits, "labels": labels}

def train_valid_test_datasets_provider(train_val_test_num_samples):
    train_ds = NuFormerMegatronDataset("train", ...)
    val_ds = NuFormerMegatronDataset("val", ...)
    return train_ds, val_ds, None
```

### 4.5 Checkpointing

**Effort**: 1 day | **Risk**: Low

With the hybrid approach:
- Transformer weights: saved/loaded via Megatron's `save_checkpoint()`/`load_checkpoint()`
- DCNv2 + PLR + Fusion MLP: saved alongside as standard `torch.save` state dict
- Pre-trained checkpoint loading: need a conversion script to map our current `final.pt`
  (PyTorch state dict) into Megatron's checkpoint format

```python
# Conversion from current format to Megatron
def convert_pretrained_to_megatron(pytorch_ckpt_path, megatron_ckpt_dir):
    state = torch.load(pytorch_ckpt_path)
    # Map key names: "layers.0.attn.qkv.weight" -> Megatron's naming convention
    megatron_state = remap_keys(state["transformer"])
    save_megatron_checkpoint(megatron_state, megatron_ckpt_dir)
    # Save DCNv2 separately
    torch.save(state["dcnv2"], f"{megatron_ckpt_dir}/dcnv2_state.pt")
```

For multi-node, Megatron's Distributed Checkpointing (DCP) handles shard-level save/load
automatically — no changes needed beyond the initial conversion.

### 4.6 LoRA / PEFT

**Effort**: 1 day | **Risk**: Low

**Current**: Custom `lora.py` with manual weight injection into Q/K/V projections.

**Megatron/NeMo**: NeMo 2.0 provides `LoRAAdapter` as a mixin that wraps any Megatron model:

```python
from nemo.collections.nlp.parts.peft import LoRAAdapterConfig

lora_config = LoRAAdapterConfig(
    target_modules=["attention.linear_qkv"],  # Q/K/V in Megatron naming
    dim=32,  # rank
    alpha=64,
    dropout=0.0,
)
model = NuFormerMegatron(config, transformer_config)
model.add_adapter(lora_config)  # Injects LoRA into transformer layers
```

Benefits over custom implementation:
- Automatic handling of LoRA weight merging for inference
- Compatible with Megatron's distributed checkpointing
- Supports more targets (gate projections, MLP layers) without code changes
- NeMo's PEFT utilities handle adapter save/load separately from base model

## 5. Effort Estimate

| Component | Effort | Risk | Notes |
|-----------|--------|------|-------|
| Transformer → GPTModel | 2-3 days | Low | Well-documented path |
| LoRA → NeMo PEFT | 1 day | Low | Drop-in replacement |
| DCNv2 hybrid wrapper | 1-2 days | Medium | Optimizer group handling |
| Data pipeline conversion | 2-3 days | Medium | Custom side-features loader |
| Training loop adaptation | 3-4 days | High | Multi-LR, early stop, eval hooks |
| Checkpoint conversion | 1 day | Low | One-time key remapping script |
| Testing & validation | 2-3 days | Medium | Must reproduce AUC=0.8938 baseline |
| **Total** | **~2 weeks** | | |

### Phased Approach (Lighter Alternative)

If full migration is too heavy, a lighter Phase 0 can capture most of the FP8 benefit:

| Phase | Scope | Effort | Gain |
|-------|-------|--------|------|
| 0 | Swap flash-attn for TransformerEngine attention (FP8) in current PyTorch loop | 2-3 days | ~1.3x throughput |
| 1 | Wrap transformer in Megatron GPTModel, keep everything else | 1 week | FP8 + TP ready |
| 2 | Full Megatron training loop, DCP, NeMo PEFT | 1 week | Multi-node ready |

## 6. Risk Assessment

| Risk | Impact | Mitigation |
|------|--------|------------|
| Multi-LR not natively supported | Training divergence or suboptimal convergence | Test with single-LR first; add param groups incrementally |
| Megatron error messages are opaque | Debugging takes 3-5x longer than PyTorch | Keep fallback to current codebase; use Megatron's `--debug` flag |
| AUC regression after migration | Wasted effort if we can't match 0.8938 | Run A/B: same data, same hyperparams, compare metrics at step 1000 |
| Data pipeline mismatch | Subtle bugs in tokenized sequence ordering | Validate first 100 batches match between old and new loaders |
| NeMo/Megatron version coupling | Breaking changes between NGC container versions | Pin to `nemo:26.04`, test upgrades separately |
| Custom early stopping | Training runs longer than needed, wastes GPU hours | Implement as a simple eval hook that writes a sentinel file |

## 7. Decision Criteria

### Migrate NOW if:

- Scaling model to 1B+ parameters (TP becomes necessary)
- Moving to multi-node training (real data at 100M+ users)
- FP8 throughput is critical (halving training time from 4h to ~2.5h matters)
- Integrating with NeMo ecosystem for production serving (TensorRT-LLM export)

### Stay on PyTorch Native if:

- Continuing with synthetic data experiments (300K users, single-node)
- Rapid prototyping of new architectural variants (DCNv2 modifications, new fusion strategies)
- Team is small and Megatron expertise is limited
- Training time (4h) is already acceptable

### Current recommendation: **Stay on PyTorch Native**

The project is a replication study on synthetic data. The architecture validated successfully
(AUC 0.8941). Migration makes sense only when/if we move to real production data at scale, which
would require multi-node training and where the 2-week investment pays off in operational
efficiency.

## 8. Prerequisites (Already Met)

All infrastructure prerequisites are already in place:

| Prerequisite | Status | Location |
|-------------|--------|----------|
| Megatron-Core installed | In container | `nvcr.io/nvidia/nemo:26.04` |
| TransformerEngine installed | In container | TE >=1.13 with FP8 support |
| NeMo 2.0 framework | In container | Full PEFT, data, training utilities |
| H200 GPUs (sm_90) | Available | p5en.48xlarge, 8x H200 141GB |
| EFA networking | Configured | 16 EFA interfaces, 3200 Gbps |
| Multi-node Slurm | Operational | 16 nodes in cluster |
| Enroot/Pyxis | Configured | Container orchestration ready |
| FSx storage | Mounted | `/fsx` with sufficient IOPS |

No additional infrastructure work is needed. The migration is purely a software refactoring
exercise within the existing container and cluster setup.
