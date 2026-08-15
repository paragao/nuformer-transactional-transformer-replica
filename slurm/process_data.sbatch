#!/bin/bash
#SBATCH --job-name=process-300k
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=/fsx/paragao/logs/process_data_300k_%j.out
#SBATCH --error=/fsx/paragao/logs/process_data_300k_%j.err

# =============================================================================
# Process 300K raw data into numpy arrays for training
#
# Input:  data/raw_300k/ (parquet files)
# Output: data/processed_300k/ (numpy arrays: sequences, labels, features)
#
# Run: sbatch slurm/process_data.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/paragao/containers/nuformer-efa-26.04.sqsh"
WORKSPACE="/fsx/paragao/nuformer"

echo "=============================================="
echo "  Data Processing: 300K users -> numpy"
echo "=============================================="
echo "  Node:    $(hostname)"
echo "  Input:   $WORKSPACE/data/raw_300k/"
echo "  Output:  $WORKSPACE/data/processed_300k/"
echo "  Started: $(date -u)"
echo "=============================================="

/opt/slurm/bin/srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx" \
    bash -c "
        cd $WORKSPACE && \
        PYTHONPATH=. python3 -u scripts/process_data.py \
            --input-dir data/raw_300k \
            --output-dir data/processed_300k \
            --max-seq-len 2048
    "

echo ""
echo "=== Data Processing Complete ==="
echo "  Finished: $(date -u)"
ls -lh $WORKSPACE/data/processed_300k/
