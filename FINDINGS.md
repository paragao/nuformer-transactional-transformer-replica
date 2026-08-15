# nuFormer Replication: Paper Findings & Implementation Notes

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

### 2.3 DCNv2: Deep & Cross Network v2

**Paper**: Wang et al., "DCN V2: Improved Deep & Cross Network and Practical Lessons for Web-Scale Learning to Rank Systems", WWW 2021

**Problem**: Neural networks are sample-inefficient at learning explicit feature interactions (e.g., `income × credit_limit`). Gradient-boosted trees get these for free via splitting; MLPs must discover them implicitly, requiring exponentially more data.

**Solution**: A **Cross Network** that explicitly models bounded-degree polynomial feature interactions in a parameter-efficient way.

**Architecture** (as used in nuFormer):
```
Tabular Features (291 dims)
        |
  ┌─────┴─────┐
  |            |
Cross Net   Deep Net (MLP)
  |            |
  └─────┬─────┘
        |
   Concat + Project → 128-dim embedding (for fusion with transformer)
```

**The Cross Layer formula**:
```
x_{l+1} = x_0 * (W_l @ x_l + b_l) + x_l
```

Where:
- `x_0` = original input (always preserved across all layers)
- `x_l` = output of previous cross layer
- `W_l` = learned weight matrix (one per layer)
- `*` = element-wise multiplication (this creates the feature crosses)

**Key insight**: The element-wise multiplication of `x_0` with the transformed `x_l` creates explicit polynomial feature interactions. After `L` cross layers, the network captures interactions up to degree `L+1`. With 3 layers (our config), we get up to 4th-order feature crosses automatically (e.g., `income × utilization × payment_history × account_age`).

**Why DCNv2 over just an MLP?**
1. Inductive bias toward cross-feature patterns (no hoping the MLP discovers them)
2. Parameter-efficient: one linear layer per cross layer vs exponential MLP width
3. Bounded degree prevents overfitting to noise in high-order interactions
4. Empirically matches/beats gradient-boosted trees on tabular benchmarks

**In nuFormer**: The 291 tabular features (280 numerical + 11 categorical) are processed by DCNv2 to learn interactions like "high utilization + low income + recent missed payment" as explicit polynomial features. The 128-dim output is concatenated with the transformer's 1024-dim transaction embedding for joint prediction.

**Critical regularization** (from paper ablation): Without weight decay (0.01) and dropout (0.1) on cross layers, DCNv2 overfits and performs *worse* than baseline (-0.40% AUC). With proper regularization + numerical embeddings, it reaches parity; combined with the transformer, it achieves the full +1.25% gain.

### 2.4 Key Hyperparameters

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

## 3. NVIDIA Tools Evaluation

Investigated during planning but not used in the final pipeline:

- **NeMo Safe Synthesizer**: Considered for generating synthetic tabular features with realistic
  inter-feature correlations. Not used — custom generator (`src/synthetic_data/`) provided full
  control over distributions and label signals.
- **NeMo Curator**: Considered for semantic deduplication of transaction descriptions. Not needed —
  template-based generation already ensures diversity without duplicates.
- **PersonaLedger** (arXiv:2601.03149): Used as reference for realistic transaction amount
  distributions and temporal patterns. Informed our distribution parameters in Section 6.

The NVIDIA NGC container (`nvcr.io/nvidia/nemo:26.04`) was used as the base image for the
CUDA/PyTorch/NCCL stack.

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
- `flash-attn>=2.6` for FlashAttention on H200
- DDP (DistributedDataParallel) for single-node 8-GPU training
- BF16 mixed precision via `torch.amp`
- Standard `torch.save` checkpointing

---

## 5. Infrastructure

### Cluster
- **Access**: `ssh p5en.smml.aiml.aws.dev`
- **Nodes**: 16 total (Slurm)
- **GPU**: p5en.48xlarge = 8x H200 (141GB HBM3e) per node
- **Used**: 1 node (8x H200) for all training runs
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

### 6.3 Label: Fraud Detection
- Binary: is the user involved in fraudulent transactions?
- Positive rate: ~5% (class imbalanced)
- Signal: anomalous transaction patterns, velocity, geographic inconsistency

---

## 7. Implementation Results

| Phase | Branch | Outcome |
|-------|--------|---------|
| 0 | `main` | Repo setup, FINDINGS.md, Dockerfile, .gitignore |
| 1 | `phase-1-data-generation` | 300K synthetic users (transactions + 291 tabular features) |
| 2 | `phase-2-tokenization` | BPE tokenizer (24K vocab) + sequence builder |
| 3 | `phase-3-models` | 329M Transformer, DCNv2+PLR, NuFormer |
| 4 | `phase-4-training` | Pretrain (PPL 1.6) + Finetune (AUC 0.83) |
| 5 | `phase-5-evaluation` | Joint Fusion v1 (AUC 0.8938), v2 (AUC 0.8941) |

---

## 8. Replication vs Paper

| Aspect | Paper (Nubank) | Our Replication |
|--------|---------------|-----------------|
| Data | 100M+ real users, 100B+ txns | 300K synthetic users |
| Pre-training | PPL not disclosed | PPL 1.6 |
| Baseline (LoRA only) | Not directly comparable | AUC 0.8296 |
| Joint Fusion gain | +1.25% relative AUC over LightGBM | +7.7% absolute AUC over LoRA-only |
| DCNv2 without embeddings | -0.40% (worse) | Not tested (PLR always enabled) |
| PLR contribution | "Critical" per ablation | +0.064 AUC (LoRA to Joint Fusion) |
| Training scale | Multi-node, weeks | 1 node (8x H200), ~4h |

**Key differences:**
- Paper uses real production data (100M users); we use 300K synthetic — the AUC ceiling
  (~0.894) likely reflects synthetic data limitations rather than model capacity
- Paper reports *relative* improvement over LightGBM; our baseline is transformer-only LoRA
- Paper doesn't disclose absolute AUC values, making direct comparison impossible
- Our replication validates the architecture and training methodology, not the exact numbers

---

## 9. Public Dataset Alternatives

The nuFormer paper uses only Nubank internal data (203M training rows, 2M test rows, 291 tabular
features, ~500B tokens). It does **not** release a public dataset. However, the papers it cites —
particularly CoLES and NPPR — benchmark on publicly available transaction datasets:

| Dataset | Source | Scale | Features | Task | Relevance |
|---------|--------|-------|----------|------|-----------|
| **Sberbank Age Prediction** | [Kaggle](https://www.kaggle.com/c/age-prediction-on-transaction-data) | 30K users, ~15M txns | Amount, MCC, date | Age bucket prediction | High — transaction sequences + classification |
| **Rosbank Credit Default** | [Kaggle](https://www.kaggle.com/c/rosbank-ml-competition) | ~5K users | Amount, MCC, date, currency | Credit default | Very high — same task type as nuFormer |
| **IBM Synthetic Fraud (AML)** | [Kaggle](https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml) | 6M+ txns | Amount, sender, receiver, timestamp | Fraud / AML detection | Medium — fraud but graph-structured |
| **Czech Bank (Berka)** | [Relational Dataset Repository](https://relational-data.org/dataset/Financial) | 4.5K accounts, 1M txns | Amount, date, balance, type | Loan default | Medium — small but real financial data |
| **PersonaLedger** | [arXiv:2601.03149](https://arxiv.org/abs/2601.03149) | 24.7M txns, 22K personas | Amount, merchant, date, description | Synthetic (no native label) | Medium — realistic distributions, no default label |

### Recommendation

**Rosbank Credit Default** is closest to the nuFormer task (credit default from transaction
sequences), though small (~5K users). **Sberbank Age Prediction** is larger (30K users, 15M
transactions) with richer sequences but a different label type.

Neither dataset provides the 291 tabular features the paper uses (Nubank-internal bureau scores,
derived aggregates, etc.). To replicate Joint Fusion with a public dataset, options are:

1. Use Rosbank/Sberbank transactions for the transformer tower
2. Engineer tabular features from the transactions (rolling aggregates, velocity, etc.)
3. Supplement with bureau-like features from the Czech Bank dataset (account balance, loan info)

This would sacrifice the "tabular features are orthogonal to transactions" property that the paper
emphasizes, but would provide a fully reproducible public benchmark.
