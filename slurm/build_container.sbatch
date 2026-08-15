#!/bin/bash
#SBATCH --job-name=build-nuformer-efa
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=02:00:00
#SBATCH --exclusive
#SBATCH --output=/fsx/paragao/logs/build_container_%j.out
#SBATCH --error=/fsx/paragao/logs/build_container_%j.err

# =============================================================================
# Build nuformer-efa:26.04 container from Dockerfile and convert to enroot sqsh
#
# Run: sbatch slurm/build_container.sbatch
# =============================================================================

set -euo pipefail

WORKSPACE="/fsx/paragao/nuformer"
CONTAINER_OUT="/fsx/paragao/containers/nuformer-efa-26.04.sqsh"

cd "$WORKSPACE"

echo "=============================================="
echo "  Building nuformer-efa:26.04"
echo "=============================================="
echo "  Node:       $(hostname)"
echo "  Dockerfile: $WORKSPACE/Dockerfile.nuformer-efa"
echo "  Output:     $CONTAINER_OUT"
echo "  Started:    $(date -u)"
echo "=============================================="

# Step 1: Docker build
echo ""
echo "[1/3] Docker build..."
time docker build -t nuformer-efa:26.04 -f Dockerfile.nuformer-efa .

echo ""
echo "[2/3] Converting to enroot sqsh..."
rm -f "$CONTAINER_OUT"
time enroot import -o "$CONTAINER_OUT" dockerd://nuformer-efa:26.04

echo ""
echo "[3/3] Verification..."
echo "--- PyTorch + CUDA ---"
enroot start --rw -m /fsx:/fsx "$CONTAINER_OUT" \
  bash -c 'python3 -c "import torch; print(f\"PyTorch {torch.__version__}, CUDA {torch.version.cuda}\")"'

echo "--- EFA + NCCL libs ---"
enroot start --rw -m /fsx:/fsx "$CONTAINER_OUT" \
  bash -c 'ls /opt/amazon/efa/lib/libfabric.so /opt/nccl/build/lib/libnccl.so && echo "EFA + NCCL: OK"'

echo "--- OFI NCCL plugin ---"
enroot start --rw -m /fsx:/fsx "$CONTAINER_OUT" \
  bash -c 'find /opt/amazon -name "*nccl-ofi*" -o -name "*nccl-net*" | head -5'

echo "--- Python packages ---"
enroot start --rw -m /fsx:/fsx "$CONTAINER_OUT" \
  bash -c 'python3 -c "import polars, sklearn; print(f\"polars={polars.__version__}, sklearn={sklearn.__version__}\")"'

echo ""
echo "=============================================="
echo "  BUILD COMPLETE: $CONTAINER_OUT"
echo "  Size: $(du -h $CONTAINER_OUT | cut -f1)"
echo "  Finished: $(date -u)"
echo "=============================================="
