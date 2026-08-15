#!/bin/bash
#SBATCH --job-name=datagen-300k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/fsx/paragao/logs/datagen_300k_%j.out
#SBATCH --error=/fsx/paragao/logs/datagen_300k_%j.err

# =============================================================================
# Generate 300K user synthetic dataset
# Runs inside the nuformer-efa container (has polars for fast parquet output)
#
# Run: sbatch slurm/datagen.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/paragao/containers/nuformer-efa-26.04.sqsh"
WORKSPACE="/fsx/paragao/nuformer"

echo "=============================================="
echo "  Data Generation: 300K users"
echo "=============================================="
echo "  Node:    $(hostname)"
echo "  Started: $(date -u)"
echo "=============================================="

/opt/slurm/bin/srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx" \
    bash -c "
        cd $WORKSPACE && \
        PYTHONPATH=. python3 -u scripts/generate_data.py \
            --num-users 300000 \
            --output-dir data/raw_300k \
            --seed 42
    "

echo ""
echo "=== Data Generation Complete ==="
echo "  Finished: $(date -u)"
ls -lh $WORKSPACE/data/raw_300k/
