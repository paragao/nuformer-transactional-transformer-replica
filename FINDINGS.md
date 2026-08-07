# nuFormer Replication: Paper Findings & Implementation Plan

## Paper: "Your Spending Needs Attention: Modeling Financial Habits with Transformers"

- **Authors**: D. T. Braithwaite, Misael Cavalcanti, R. Austin McEver, et al. (Nubank)
- **arXiv**: [2507.23267v1](https://arxiv.org/abs/2507.23267) (July 31, 2025)
- **Model Name**: **nuFormer**

---

## 1. Paper Summary

### Key Contributions

1. **Transaction Tokenization**: Modified "text-is-all-you-need" approach using special tokens for numerical/categorical features + BPE for text descriptions
2. **Causal Transformer with NoPE + FlashAttention**: Enables long context lengths (up to 4096 tokens) on single GPU without positional embeddings
3. **Self-Supervised Pre-training**: Next Token Prediction (NTP) on massive transaction data (100B+ transactions, 100M+ members)
4. **LoRA Fine-tuning**: Prevents overfitting/catastrophic forgetting during task-specific adaptation
5. **Joint Fusion (nuFormer)**: End-to-end training combining transformer embeddings with tabular features via DCNv2
6. **Production Results**: +1.25% relative AUC improvement over LightGBM baseline (3x typical model launch impact), 4.4% churn reduction

### Production Impact at Nubank

- Deployed across multiple downstream tasks (credit, retention, recommendations)
- 3x the typical model improvement seen at production launches
- 4.4% reduction in customer churn
- Serves 100M+ members

---

## 2. Technical Architecture

### 2.1 Transaction Tokenization (Section 3.1)

Each transaction `t = {(amt, v_amt), (date, v_date), (desc, v_desc)}` is tokenized as:

| Feature | Tokenization | Vocabulary Size |
|---------|-------------|-----------------|
| Amount Sign (inflow/outflow) | Special token | 2 |
| Amount Bucket (quantized, log-scale) | Special token | 21 |
| Month | Special token | 12 |
| Day of month | Special token | 31 |
| Day of week | Special token | 7 |
| Description | BPE tokenization | ~16K |

**Transaction encoding**: `tau(t) = [sign_token, amt_token, month_token, day_token, weekday_token] + BPE(desc)`

**User sequence**: Transactions concatenated with separator tokens, truncated to context length (512-4096).

### 2.2 Model Architecture

**Pre-training:**
- Causal transformer (GPT-style, decoder-only)
- **NoPE**: No positional embeddings (causal attention mask provides implicit position)
- **FlashAttention**: Memory-efficient attention for long contexts
- **Loss**: Next Token Prediction (NTP) - standard autoregressive LM loss

**Fine-tuning:**
- Final token embedding -> MLP prediction head -> binary classification
- **LoRA** (Low-Rank Adaptation) on attention Q/K/V projections
- Cross-entropy loss for credit card activation prediction

**Joint Fusion (nuFormer):**
```
Tabular Features (291 dims)
    -> Numerical Embeddings (periodic activations, learned frequencies)
    -> Categorical Embeddings (trainable lookup tables)
    -> DCNv2 (cross layers with weight decay + dropout)
    -> Low-dim feature embedding (128d)
                    |
                    v
[feature_emb(128) || transaction_emb(1024)] -> MLP -> Prediction

Transaction Sequence -> Causal Transformer -> [CLS/final token] -> LayerNorm -> transaction_emb(1024)
```

Key design decisions:
- DCNv2 processes tabular features and projects to low-dimensional embedding
- Concatenation (not addition) of feature and transaction embeddings
- **Regularization**: Weight decay + dropout on DCNv2 cross layers
- **Normalization**: LayerNorm on transaction embeddings before concatenation
- End-to-end training: Transformer + DCNv2 + MLP trained jointly

### 2.3 Key Hyperparameters

| Component | Values |
|-----------|--------|
| Model sizes | 24M, **330M** parameters |
| Context lengths | 512, 1024, 2048, **4096** |
| Pre-train data | 20M rows |
| Fine-tune data | 5M, 20M, 40M, 100M rows |
| Tabular features | 291 (numerical + categorical) |
| Training rows (recsys experiment) | 203M |
| Test rows | 2M |
| LoRA rank | 16 (inferred from standard practice) |

### 2.4 Key Results (from paper)

| Configuration | Relative AUC vs LightGBM Baseline |
|---------------|-----------------------------------|
| LightGBM (features only) | 0% (baseline) |
| Late Fusion (LightGBM + finetuned embeddings) | +0.97% |
| **Joint Fusion (nuFormer)** | **+1.25%** |
| DCNv2 alone (no numerical embeddings) | -0.40% (worse) |
| DCNv2 + numerical embeddings (no transformer) | ~0% (parity) |
| DCNv2 + numerical embeddings + transformer (joint) | **+1.25%** |

---

## 3. NVIDIA Tools Investigation

### 3.1 NeMo Safe Synthesizer (for Tabular Features)

**Verdict: HIGHLY RELEVANT**

NeMo Safe Synthesizer is an LLM-based tabular data synthesizer that:
- Fine-tunes a language model (SmolLM3-3B) on your tabular data
- Learns patterns, correlations, and statistical properties
- Generates new synthetic records preserving data utility
- Supports time-series mode for ordered transaction sequences
- Has built-in evaluation reports comparing synthetic vs. original distributions

**Our use case**: Generate realistic 291-dim tabular feature vectors with inter-feature correlations that would be hard to create manually.

**Pipeline**:
1. Create seed dataset (1K users) with hand-crafted correlations
2. Configure Safe Synthesizer in time-series mode (acct_id grouping, txn_index ordering)
3. Fine-tune SmolLM3-3B on seed tabular data
4. Generate 100K+ synthetic user feature rows
5. Validate with built-in KS-test, correlation preservation metrics

### 3.2 NeMo Curator (for Quality Control)

**Verdict: USEFUL for description deduplication**

NeMo Curator provides:
- Semantic deduplication using embeddings (all-MiniLM-L6-v2)
- K-means clustering + pairwise cosine similarity within clusters
- 90% similarity threshold for near-duplicate removal
- Quality filtering (rule-based + model-based)

**Our use case**: Ensure diversity in synthetic transaction descriptions.

**Pipeline**:
```
Generate descriptions -> Rule-based filters -> Semantic dedup (Curator) -> Quality scoring
```

### 3.3 PersonaLedger (Reference Dataset)

The PersonaLedger paper (arXiv:2601.03149) provides:
- 24.7M synthetic transactions from 22K+ personas
- Rule-grounded LLM generation (Llama-3.3-70B)
- Realistic amounts: mean $66.24, std $184.46 (power-law)
- Merchant categories: 74,623 unique merchant names
- Temporal patterns: ~50 txns/month average
- Sign-log transformation: `sign(amount) * log(1 + |amount|)`

---

## 4. Framework Decision: PyTorch Native

### Why PyTorch Native (not Megatron-Core)

| Criterion | PyTorch Native | Megatron-Core |
|-----------|---------------|---------------|
| **330M on H200** | Fits on 1-2 GPUs easily | Designed for much larger models |
| **Development speed** | Fast iteration, standard debugging | Steep learning curve |
| **FlashAttention** | Native via flash-attn package | Built-in + TE kernels |
| **FSDP2** | Native in torch 2.5+ | Has own FSDP (nvFSDP) |
| **torch.compile** | Full support | Limited/not needed |
| **HF interop** | Native | Requires Megatron-Bridge conversion |
| **Overkill factor** | Right-sized | TP/PP/EP not needed at 330M |
| **Custom architecture** | Easy (standard nn.Module) | Must conform to Megatron specs |

**Decision**: PyTorch Native with:
- `torch.compile(mode="max-autotune")` for kernel fusion
- `flash-attn>=2.6` for FlashAttention-3 on H200
- `transformer-engine>=1.13` for FP8 support
- FSDP2 with hybrid sharding for multi-node
- DCP for distributed checkpointing

---

## 5. Infrastructure

### Cluster
- **Access**: `ssh p5en.smml.aiml.aws.dev`
- **Nodes**: 16 total (Slurm), **6 available** for this project
- **GPU**: p5en.48xlarge = 8x H200 (141GB HBM3e) per node = **48 H200s total**
- **Network**: 16 EFA interfaces/node, 3200 Gbps
- **Container**: Enroot + Pyxis on Slurm

### Custom Container
- **Base**: `nvcr.io/nvidia/nemo:26.04`
- **EFA upgrade**: EFA 1.48.0, GDRCopy v2.5.2, NCCL v2.30.4-1
- **Key**: `--disable-build-ngc` flag for full EFA install on NGC base
- **NVCC gencode**: sm_80, sm_86, sm_89, sm_90, sm_100, sm_103

### MLFlow Tracking
- **Server**: `arn:aws:sagemaker:ap-southeast-3:159553542841:mlflow-tracking-server/paragao-mlflow-tracker`
- **AWS Profile**: `compute-sa-team-Administrator`
- **Region**: `ap-southeast-3`

---

## 6. Synthetic Data Design

### 6.1 Transaction Distributions

| Category | Amount Distribution | Frequency | Temporal Pattern |
|----------|---------------------|-----------|------------------|
| Salary | Normal(mu=4500, sigma=1500) | 2x/month (1st, 15th) | Fixed dates |
| Rent | Normal(mu=1500, sigma=500) | 1x/month (1st) | Fixed date |
| Groceries | LogNormal(mu=4.17, sigma=0.5) ~ $65 | 2-4x/week | Weekday bias |
| Dining | LogNormal(mu=3.56, sigma=0.6) ~ $35 | 3-8x/week | Weekend spike |
| Coffee | Normal(mu=5.5, sigma=1.5) | Daily (weekday) | 6-11 AM |
| Transport | LogNormal(mu=3.22, sigma=0.7) ~ $25 | Daily (weekday) | Commute hours |
| Shopping | LogNormal(mu=4.32, sigma=0.8) ~ $75 | 1-2x/week | Weekend |
| Utilities | Normal(mu=150, sigma=50) | 1x/month (15th) | Fixed |
| Subscriptions | Normal(mu=15, sigma=8) | Monthly (variable day) | Recurring |
| Healthcare | LogNormal(mu=4.79, sigma=1.0) ~ $120 | Sporadic | - |
| Entertainment | LogNormal(mu=3.22, sigma=0.7) ~ $25 | 1-2x/week | Weekend/evening |
| Transfers/PIX | LogNormal(mu=6.21, sigma=1.2) ~ $500 | Variable | - |

### 6.2 Seasonal Multipliers
- Nov: 1.15x | Dec: 1.30x | Jan: 0.85x | Feb: 0.95x | Mar-Oct: 1.0x

### 6.3 Label: Credit Card Activation
- Binary: did user activate a credit card within 6 months?
- Positive rate: ~15% (class imbalanced)
- Signal: spending patterns, transaction diversity, income stability

---

## 7. Implementation Timeline

| Phase | Branch | Duration | Deliverable |
|-------|--------|----------|-------------|
| 0 | `main` | Day 1 | Repo, FINDINGS.md, Dockerfile, .gitignore |
| 1 | `phase-1-data-generation` | Days 1-3 | Synthetic data generator + validation set |
| 2 | `phase-2-tokenization` | Days 3-4 | Tokenization pipeline + memmap datasets |
| 3 | `phase-3-models` | Days 4-6 | 330M Transformer, DCNv2, nuFormer |
| 4 | `phase-4-training` | Days 6-12 | Pretrain/Finetune/JointFusion + Slurm |
| 5 | `phase-5-evaluation` | Days 12-14 | Metrics, ablation, MLFlow, scaling laws |
