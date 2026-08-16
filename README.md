# nuFormer — Transactional Transformer for Financial Classification

Replication of ["Your Spending Needs Attention: Modeling Financial Habits with Transformers"](https://arxiv.org/abs/2507.23267)
(Braithwaite, Cavalcanti, McEver et al., Nubank, July 2025) on synthetic financial
transaction data using AWS ParallelCluster with p5en.48xlarge instances (8x NVIDIA H200 141GB).

## Results

### Best Performance (Joint Fusion v2)

| Phase | Key Metric | Best Value | At Step | Early Stopped |
|-------|-----------|-----------|---------|---------------|
| Pre-training (NTP) | Perplexity | **1.6** (val_loss=0.4817) | 3,500 | Yes (step 6,000) |
| Fine-tuning (LoRA only) | Val AUC | **0.8296** | 3,600 | Cancelled at 4,000 |
| Joint Fusion v1 (PLR+DCNv2) | Val AUC | 0.8938 | 3,600 | Yes (step 4,600) |
| Joint Fusion v2 (Enhanced PLR+DCNv2) | Val AUC | **0.8941** | 3,000 | Yes (step 4,000) |

### AUC Improvement Breakdown

| Model | Val AUC | Delta vs Previous |
|-------|---------|-------------------|
| LoRA fine-tuning (transformer only) | 0.8296 | baseline |
| + PLR(dim=8) + DCNv2(3 cross) — v1 | 0.8938 | +0.0642 |
| + PLR(dim=16) + DCNv2(5 cross) — v2 | **0.8941** | **+0.0645** |

### v1 vs v2 Comparison

| Metric | v1 | v2 | Delta |
|--------|-----|-----|-------|
| **Best Val AUC** | 0.8938 | **0.8941** | +0.0003 |
| **Best Val AP** | 0.6319 | **0.6337** | +0.0018 |
| **Best Val Loss** | 0.3345 | **0.3131** | -0.0214 |
| Best step (AUC) | 3,600 | 3,000 | -600 steps |
| Early stop step | 4,600 | 4,000 | -600 steps |
| Training time | ~4h 11min | ~3h 40min | ~30min faster |

**Config differences (v1 -> v2):**

| Parameter | v1 | v2 |
|-----------|----|----|
| PLR dim per feature | 8 | 16 |
| PLR frequencies | 4 | 8 |
| DCNv2 cross layers | 3 | 5 |
| DCNv2 deep layers | [512, 256] | [1024, 512, 256] |
| DCNv2+PLR LR | 3e-4 | 5e-4 |
| LoRA rank | 16 | 32 |
| Label smoothing | 0.1 | 0.05 |

**Key findings:**
- V2 achieves marginally higher AUC (+0.0003) but significantly better calibration (lower val_loss)
- Both models plateau in the 0.893-0.894 AUC range, suggesting a data-level ceiling
- V2 converges faster (600 fewer steps to best AUC) due to higher capacity + aggressive LR
- The reduced label smoothing in v2 improves loss without hurting AUC
- Further gains likely require more training data or architectural innovations beyond PLR+DCNv2

### Joint Fusion v1 Training Curve

| Step | Val Loss | Val AUC | Val AP |
|------|----------|---------|--------|
| 200 | 0.3568 | 0.8651 | 0.5694 |
| 400 | 0.3415 | 0.8837 | 0.6061 |
| 600 | 0.3480 | 0.8878 | 0.6134 |
| 800 | 0.3554 | 0.8895 | 0.6183 |
| 1000 | 0.3546 | 0.8912 | 0.6206 |
| 1200 | 0.3531 | 0.8917 | 0.6223 |
| 1400 | 0.3483 | 0.8924 | 0.6236 |
| 1600 | 0.3659 | 0.8926 | 0.6227 |
| 1800 | 0.3416 | 0.8929 | 0.6266 |
| 2000 | 0.3448 | 0.8928 | 0.6243 |
| 2200 | 0.3392 | 0.8930 | 0.6315 |
| 2400 | 0.3497 | 0.8935 | 0.6284 |
| 2600 | 0.3437 | 0.8937 | 0.6308 |
| 2800 | 0.3460 | 0.8935 | 0.6310 |
| 3000 | 0.3371 | 0.8936 | 0.6311 |
| 3200 | 0.3485 | 0.8934 | 0.6303 |
| 3400 | 0.3345 | 0.8936 | 0.6311 |
| **3600** | **0.3449** | **0.8938** | **0.6309** |
| 3800 | 0.3427 | 0.8932 | 0.6301 |
| 4000 | 0.3419 | 0.8935 | 0.6314 |
| 4200 | 0.3455 | 0.8937 | 0.6319 |
| 4400 | 0.3508 | 0.8934 | 0.6287 |
| 4600 | 0.3447 | 0.8934 | 0.6299 |

Early stopped at step 4,600 (patience=5 evals). Training time: ~4h 11min on 8x H200.

### Joint Fusion v2 Training Curve

| Step | Val Loss | Val AUC | Val AP |
|------|----------|---------|--------|
| 200 | 0.3321 | 0.8745 | 0.5910 |
| 400 | 0.3344 | 0.8853 | 0.6030 |
| 600 | 0.3474 | 0.8901 | 0.6172 |
| 800 | 0.3153 | 0.8908 | 0.6193 |
| 1000 | 0.3262 | 0.8912 | 0.6220 |
| 1200 | 0.3242 | 0.8925 | 0.6249 |
| 1400 | 0.3158 | 0.8925 | 0.6282 |
| 1600 | 0.3215 | 0.8923 | 0.6279 |
| 1800 | 0.3144 | 0.8928 | 0.6284 |
| 2000 | 0.3234 | 0.8933 | 0.6294 |
| 2200 | 0.3177 | 0.8930 | 0.6290 |
| 2400 | 0.3142 | 0.8934 | 0.6298 |
| 2600 | 0.3173 | 0.8938 | 0.6299 |
| 2800 | 0.3199 | 0.8937 | 0.6294 |
| **3000** | **0.3131** | **0.8941** | **0.6324** |
| 3200 | 0.3204 | 0.8935 | 0.6298 |
| 3400 | 0.3150 | 0.8936 | 0.6315 |
| 3600 | 0.3191 | 0.8938 | 0.6313 |
| 3800 | 0.3144 | 0.8936 | 0.6306 |
| 4000 | 0.3164 | 0.8939 | 0.6337 |

Early stopped at step 4,000 (patience=5 evals). Training time: ~3h 40min on 8x H200.

---

## Architecture

```
Transaction Sequence (B, T)
        |
        v
┌──────────────────────────────────┐
│  Causal Transformer (NoPE)       │
│  24 layers, d=1024, 16 heads     │
│  FlashAttention, LoRA r=16-32   │
│  329M params (2.4-4.0M trainable)|
└───────────────┬──────────────────┘
                |
        last-token embedding (B, 1024)
                |                          Tabular Features (B, 291)
                |                                   |
                |                                   v
                |                    ┌──────────────────────────────┐
                |                    │  PLR Embeddings              │
                |                    │  291 features x 8/16-dim     │
                |                    │  Learned frequencies/phases  │
                |                    └──────────────┬───────────────┘
                |                                   |
                |                           (B, 291*d_plr)
                |                                   |
                |                                   v
                |                    ┌──────────────────────────────┐
                |                    │  Linear Projection           │
                |                    │  (291*d_plr) -> 291          │
                |                    └──────────────┬───────────────┘
                |                                   |
                |                                   v
                |                    ┌──────────────────────────────┐
                |                    │  DCNv2 (Deep & Cross Net v2) │
                |                    │  3-5 cross layers + MLP      │
                |                    │  [512, 256] deep layers      │
                |                    │  -> 128-dim output           │
                |                    └──────────────┬───────────────┘
                |                                   |
                v                                   v
        ┌───────────────────────────────────────────────────┐
        │              Concatenation                         │
        │         [txn_emb(1024) || feat_emb(128)]          │
        └───────────────────────┬───────────────────────────┘
                                |
                                v
                ┌───────────────────────────────┐
                │  Fusion MLP                   │
                │  1152 -> 256 -> 256 -> 2      │
                │  GELU + Dropout(0.3)          │
                └───────────────┬───────────────┘
                                |
                                v
                        logits (B, 2)
```

**Total Parameters**: ~330M (3.97-4.5M trainable depending on config; see v1/v2 tables below)

---

## How to Reproduce

### Prerequisites

- AWS ParallelCluster with Slurm + Enroot/Pyxis
- Instance: `p5en.48xlarge` (8x H200 141GB, EFA networking)
- Storage: FSx for Lustre mounted at `/fsx`
- Container: `nuformer-efa-26.04.sqsh` (PyTorch 2.6+, FlashAttention, sklearn)

### Step 1: Build Container

```bash
sbatch slurm/build_container.sbatch
```

Builds the Enroot squashfs image from `Dockerfile.nuformer-efa`.

### Step 2: Generate Synthetic Data

```bash
sbatch slurm/datagen.sbatch
```

Generates 300K synthetic financial users with:
- Transaction sequences (tokenized, variable length up to 2048)
- 291 tabular features (280 numerical + 11 one-hot categorical)
- Binary fraud labels (~5% positive rate)

### Step 3: Train Tokenizer

```bash
sbatch slurm/train_tokenizer.sbatch
```

Trains a BPE tokenizer (24K vocab) on transaction descriptions and saves to `tokenizer/tokenizer.json`.
This tokenizer is used by all downstream scripts (processing, inference).

> **Note**: A pre-trained tokenizer is included in the repo at `tokenizer/tokenizer.json`.
> You only need to re-run this step if the transaction corpus changes.

### Step 4: Process Data

```bash
sbatch slurm/process_data.sbatch
```

Converts raw data to numpy arrays (`train_sequences.npy`, `train_features.npy`, `train_labels.npy`, etc.) with train/val split.
Supports loading the saved tokenizer via `--tokenizer tokenizer/tokenizer.json`.

### Step 5: Pre-training (Self-Supervised)

```bash
sbatch slurm/pretrain.sbatch
```

- **Task**: Next Token Prediction (autoregressive LM)
- **Config**: 8x H200, batch=6144 (effective), 25K max steps
- **Result**: Converges to PPL=1.6 (val_loss=0.4817), early stops at step 6,000
- **Output**: `ckpt/pretrain/final.pt`

### Step 6: Fine-tuning (Optional Baseline)

```bash
sbatch slurm/finetune.sbatch
```

- **Task**: Binary classification with LoRA on transformer + linear head
- **Config**: LoRA r=16, LR=2e-5, 5K max steps
- **Result**: AUC=0.8296 (plateaus by step 3,600)
- **Output**: `ckpt/finetune/best.pt`

This establishes a transformer-only baseline. Joint Fusion significantly surpasses this.

### Step 7: Joint Fusion Training

```bash
sbatch slurm/joint_fusion.sbatch
```

- **Task**: End-to-end training with PLR embeddings + DCNv2 + Transformer (LoRA)
- **Config**: PLR dim=8, 3 cross layers, DCNv2 LR=3e-4, batch=64/GPU, early stop patience=5
- **Result**: AUC=0.8938 (surpasses 0.89 target by step 800)
- **Output**: `ckpt/joint_fusion/best.pt`

For the enhanced v2 configuration (AUC=0.8941):

```bash
sbatch slurm/joint_fusion_v2.sbatch
```

### Step 8: Evaluation

```bash
python scripts/evaluate.py --checkpoint ckpt/joint_fusion/best.pt
```

### Step 9: Batch Inference

Run inference on raw (non-tokenized) transaction data using a trained checkpoint.
The pipeline handles tokenization on-the-fly — no pre-processing required.

```bash
sbatch slurm/inference.sbatch
```

Or run directly (e.g., for testing on a smaller dataset):

```bash
python scripts/batch_inference.py \
    --checkpoint ckpt/joint_fusion_v2/best.pt \
    --tokenizer tokenizer/tokenizer.json \
    --transactions data/raw_300k/transactions.parquet \
    --features data/raw_300k/tabular_features.parquet \
    --output predictions.parquet \
    --batch-size 128 \
    --threshold 0.5
```

**Output format** (parquet):

| Column | Description |
|--------|-------------|
| `user_id` | User identifier |
| `fraud_probability` | Model output (0.0 to 1.0) |
| `predicted_label` | Binary decision (1 if probability >= threshold) |
| `confidence` | How confident the model is in its prediction |

**Interpreting results:**

- The training data has a ~15% positive base rate. A well-behaved model should predict
  positive rates in a similar range on similar data.
- The default threshold of 0.5 is arbitrary. Adjust it based on your precision/recall
  tradeoff requirements (lower threshold = more positives, higher recall, lower precision).
- Use `scripts/evaluate.py` with ground truth labels to measure actual model performance
  (AUC, AP, calibration).

**Quick test with the Slurm job:**

You can override any parameter via environment variables without editing the sbatch file:

```bash
# Use a different checkpoint
CHECKPOINT=/fsx/paragao/nuformer/ckpt/joint_fusion/best.pt sbatch slurm/inference.sbatch

# Point to different input data
TRANSACTIONS=/fsx/paragao/nuformer/data/test/transactions.parquet \
FEATURES=/fsx/paragao/nuformer/data/test/tabular_features.parquet \
sbatch slurm/inference.sbatch
```

**Expected input data format:**

`transactions.parquet` must contain columns: `user_id`, `timestamp`, `amount`, `description`

`tabular_features.parquet` must contain a `user_id` column plus 291 numeric feature columns
(matching the training schema).

**Performance notes:**

On 1x H200, scoring 300K users (~1,600 transactions/user) takes approximately 90 minutes total,
with tokenization dominating (~60% of runtime). GPU inference itself runs at ~1,000+ users/s.
For faster runs, pre-tokenize data using `scripts/process_data.py` and use the training
evaluation pipeline directly.

---

## Training Configurations

### Pre-training

| Parameter | Value |
|-----------|-------|
| Architecture | 24L, d=1024, 16 heads, d_ff=4096 |
| Positional encoding | None (NoPE) |
| Attention | FlashAttention-2 |
| Vocab size | 24,078 |
| Max seq length | 2,048 |
| Batch size (effective) | 6,144 |
| Learning rate | 3e-4 (cosine decay) |
| Warmup steps | 500 |
| Precision | BF16 |
| Parameters | 329M |

### Fine-tuning (LoRA)

| Parameter | Value |
|-----------|-------|
| LoRA rank | 16 |
| LoRA alpha | 32 |
| LoRA targets | Q, K, V projections |
| Trainable params | 2.89M (0.88%) |
| Learning rate | 2e-5 |
| Batch size | 32/GPU |
| Label smoothing | 0.1 |

### Joint Fusion v1

| Parameter | Value |
|-----------|-------|
| PLR embedding dim | 8 per feature |
| PLR frequencies | 4 (sin/cos pairs) |
| DCNv2 cross layers | 3 |
| DCNv2 deep layers | [512, 256] |
| DCNv2 output dim | 128 |
| Fusion MLP | 1152->256->256->2 |
| Transformer LR | 5e-6 |
| DCNv2+PLR LR | 3e-4 |
| Fusion LR | 5e-5 |
| Batch size | 64/GPU (effective 1024) |
| Grad accumulation | 2 |
| Label smoothing | 0.1 |
| Early stop patience | 5 evals |
| Trainable params | 3.97M (1.2%) |

### Joint Fusion v2

| Parameter | Value |
|-----------|-------|
| PLR embedding dim | 16 per feature |
| PLR frequencies | 8 (sin/cos pairs) |
| DCNv2 cross layers | 5 |
| DCNv2 deep layers | [1024, 512, 256] |
| DCNv2 output dim | 128 |
| Fusion MLP | 1152->256->256->2 |
| Transformer LR | 5e-6 |
| DCNv2+PLR LR | 5e-4 |
| Fusion LR | 5e-5 |
| LoRA rank | 32 |
| Batch size | 64/GPU (effective 1024) |
| Grad accumulation | 2 |
| Label smoothing | 0.05 |
| Early stop patience | 5 evals |

---

## Cluster Setup

| Component | Details |
|-----------|---------|
| Instance | `p5en.48xlarge` |
| GPUs | 8x NVIDIA H200 (141GB HBM3e each) |
| Interconnect | EFA (Elastic Fabric Adapter) |
| NCCL | aws-ofi-nccl plugin |
| Storage | FSx for Lustre (`/fsx`) |
| Scheduler | Slurm (AWS ParallelCluster) |
| Container | Enroot/Pyxis with squashfs images |
| Region | ap-southeast-3 |
| MLFlow | SageMaker MLflow tracking server |

---

## Key Files

| File | Description |
|------|-------------|
| `src/models/nuformer.py` | Full NuFormer model (Transformer + DCNv2 + Fusion MLP) |
| `src/models/dcnv2.py` | DCNv2 with integrated PLR embeddings |
| `src/models/numerical_embeddings.py` | PLR periodic activations (NumericalFeatureEmbedder) |
| `src/models/transformer.py` | Causal Transformer with NoPE + FlashAttention |
| `src/models/lora.py` | LoRA adapter implementation |
| `src/training/pretrain.py` | Self-supervised pre-training (NTP) |
| `src/training/finetune.py` | LoRA fine-tuning for classification |
| `src/training/joint_fusion.py` | End-to-end joint fusion training |
| `src/synthetic_data/` | Synthetic transaction + feature generation |
| `src/tokenization/` | Transaction tokenizer (BPE + special tokens) |
| `slurm/pretrain.sbatch` | Pre-training job script |
| `slurm/finetune.sbatch` | Fine-tuning job script |
| `slurm/joint_fusion.sbatch` | Joint Fusion v1 job script |
| `slurm/joint_fusion_v2.sbatch` | Joint Fusion v2 job script (enhanced config) |
| `scripts/generate_data.py` | Data generation entrypoint |
| `scripts/process_data.py` | Data processing pipeline |
| `scripts/evaluate.py` | Model evaluation script |
| `scripts/batch_inference.py` | End-to-end inference (raw data → predictions) |
| `scripts/train_tokenizer.py` | Standalone BPE tokenizer training |
| `tokenizer/tokenizer.json` | Pre-trained BPE tokenizer (251 tokens + 78 special) |
| `slurm/inference.sbatch` | Batch inference job script (1x GPU) |
| `logs/` | Training summaries for all phases |
| `FINDINGS.md` | Paper analysis and implementation notes |

---

## References

- Braithwaite et al., "Your Spending Needs Attention: Modeling Financial Habits with Transformers", arXiv:2507.23267, 2025
- Gorishniy et al., "On Embeddings for Numerical Features in Tabular Deep Learning", NeurIPS 2022
- Wang et al., "DCN V2: Improved Deep & Cross Network", WWW 2021
- Hu et al., "LoRA: Low-Rank Adaptation of Large Language Models", ICLR 2022

---

## License

See [LICENSE](LICENSE).
