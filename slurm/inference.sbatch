#!/bin/bash
#SBATCH --job-name=nuformer-inference
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00
#SBATCH --account=smml
#SBATCH --output=/fsx/paragao/logs/inference_%j.out
#SBATCH --error=/fsx/paragao/logs/inference_%j.err

# =============================================================================
# nuFormer Batch Inference (Single GPU)
#
# Runs inference on raw transaction data using the trained Joint Fusion model.
# Accepts non-tokenized transactions + tabular features, tokenizes on-the-fly,
# and outputs per-user fraud probabilities.
#
# Run: sbatch slurm/inference.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/paragao/containers/nuformer-efa-26.04.sqsh"
WORKSPACE="/fsx/paragao/nuformer"
CHECKPOINT="${CHECKPOINT:-$WORKSPACE/ckpt/joint_fusion_v2/best.pt}"
TOKENIZER="$WORKSPACE/tokenizer/tokenizer.json"
TRANSACTIONS="${TRANSACTIONS:-$WORKSPACE/data/raw_300k/transactions.parquet}"
FEATURES="${FEATURES:-$WORKSPACE/data/raw_300k/tabular_features.parquet}"
OUTPUT="${OUTPUT:-$WORKSPACE/predictions/batch_$(date +%Y%m%d_%H%M%S).parquet}"

echo "=============================================="
echo "  nuFormer Batch Inference"
echo "=============================================="
echo "  Job ID:        $SLURM_JOB_ID"
echo "  Checkpoint:    $CHECKPOINT"
echo "  Tokenizer:     $TOKENIZER"
echo "  Transactions:  $TRANSACTIONS"
echo "  Features:      $FEATURES"
echo "  Output:        $OUTPUT"
echo "=============================================="

mkdir -p /fsx/paragao/logs
mkdir -p "$(dirname "$OUTPUT")"

/opt/slurm/bin/srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx,$HOME:$HOME" \
    bash -c "
        export PYTHONPATH=$WORKSPACE
        export PYTHONUNBUFFERED=1
        cd $WORKSPACE && \
        python scripts/batch_inference.py \
            --checkpoint $CHECKPOINT \
            --tokenizer $TOKENIZER \
            --transactions $TRANSACTIONS \
            --features $FEATURES \
            --output $OUTPUT \
            --batch-size 128
    "

echo "=== Inference Complete ==="
echo "Output: $OUTPUT"
