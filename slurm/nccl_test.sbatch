#!/bin/bash
#SBATCH --job-name=nccl-test
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=00:15:00
#SBATCH --exclusive
#SBATCH --output=logs/nccl_test_%j.log

# =============================================================================
# NCCL / EFA Connectivity Test
#
# Validates inter-node GPU communication before launching full training.
# Expected: >400 GB/s bus bandwidth on H200 with EFA.
#
# Run: sbatch slurm/nccl_test.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/containers/nuformer-efa.sqsh"

export FI_PROVIDER=efa
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
export NCCL_P2P_NET_CHUNKSIZE=2097156
export NCCL_SOCKET_IFNAME=^docker,lo,veth

export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500

echo "=============================================="
echo "  NCCL / EFA Connectivity Test"
echo "=============================================="
echo "  Nodes:  $SLURM_NNODES"
echo "  GPUs:   $((SLURM_NNODES * 8))"
echo "  Master: $MASTER_ADDR"
echo "=============================================="

mkdir -p logs

srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx" \
    bash -c '
        echo "--- Node: $(hostname) ---"
        echo "EFA interfaces:"
        fi_info -p efa 2>/dev/null | head -5 || echo "  EFA not found!"
        echo ""

        # Use NCCL test binary if available
        if command -v all_reduce_perf &>/dev/null; then
            echo "Running all_reduce_perf..."
            all_reduce_perf -b 8 -e 4G -f 2 -g 1
        else
            echo "Running PyTorch NCCL all-reduce test..."
            torchrun \
                --nnodes='"$SLURM_NNODES"' \
                --nproc-per-node=8 \
                --rdzv_id='"$SLURM_JOB_ID"' \
                --rdzv_backend=c10d \
                --rdzv_endpoint='"$MASTER_ADDR"':'"$MASTER_PORT"' \
                -c "
import torch
import torch.distributed as dist
import time
import os

dist.init_process_group(backend=\"nccl\")
rank = dist.get_rank()
world_size = dist.get_world_size()
local_rank = int(os.environ.get(\"LOCAL_RANK\", 0))
device = torch.device(f\"cuda:{local_rank}\")
torch.cuda.set_device(device)

# Warmup
tensor = torch.ones(1024*1024*64, device=device)  # 256MB
for _ in range(5):
    dist.all_reduce(tensor)
torch.cuda.synchronize()

# Benchmark
sizes_mb = [1, 16, 64, 256, 1024]
for size_mb in sizes_mb:
    n_elements = size_mb * 1024 * 1024 // 4  # float32
    tensor = torch.ones(n_elements, device=device)
    torch.cuda.synchronize()

    t0 = time.time()
    n_iters = 20
    for _ in range(n_iters):
        dist.all_reduce(tensor)
    torch.cuda.synchronize()
    elapsed = time.time() - t0

    data_bytes = n_elements * 4 * n_iters
    algbw = data_bytes / elapsed / 1e9
    busbw = algbw * 2 * (world_size - 1) / world_size

    if rank == 0:
        print(f\"  {size_mb:4d} MB | algbw={algbw:.1f} GB/s | busbw={busbw:.1f} GB/s\")

if rank == 0:
    print(f\"\nWorld size: {world_size}\")
    if busbw > 100:
        print(\"EFA connectivity: GOOD\")
    elif busbw > 10:
        print(\"EFA connectivity: DEGRADED (check NIC binding)\")
    else:
        print(\"EFA connectivity: FAILED\")

dist.destroy_process_group()
"
        fi
    '

echo "=== NCCL Test Complete ==="
