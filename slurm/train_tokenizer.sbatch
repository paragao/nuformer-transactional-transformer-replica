#!/bin/bash
#SBATCH --job-name=nuformer-tokenizer
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:10:00
#SBATCH --account=smml
#SBATCH --output=/fsx/paragao/logs/tokenizer_%j.out
#SBATCH --error=/fsx/paragao/logs/tokenizer_%j.err

# =============================================================================
# Train BPE Tokenizer for nuFormer
#
# CPU-only job. Reads transaction descriptions, trains BPE (24K vocab),
# and saves to tokenizer/ directory within the workspace.
#
# Run: sbatch slurm/train_tokenizer.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/paragao/containers/nuformer-efa-26.04.sqsh"
WORKSPACE="/fsx/paragao/nuformer"
INPUT="$WORKSPACE/data/raw/dev/transactions.parquet"
OUTPUT="$WORKSPACE/tokenizer"
SEQUENCES="$WORKSPACE/data/processed_300k/train_sequences.npy"

echo "=============================================="
echo "  BPE Tokenizer Training"
echo "=============================================="
echo "  Job ID:  $SLURM_JOB_ID"
echo "  Input:   $INPUT"
echo "  Output:  $OUTPUT"
echo "=============================================="

mkdir -p /fsx/paragao/logs

/opt/slurm/bin/srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx,$HOME:$HOME" \
    bash -c "
        export PYTHONPATH=$WORKSPACE
        cd $WORKSPACE && \
        python scripts/train_tokenizer.py \
            --input $INPUT \
            --output $OUTPUT \
            --vocab-size 24000 \
            --verify-against $SEQUENCES
    "

echo "=== Tokenizer Training Complete ==="
