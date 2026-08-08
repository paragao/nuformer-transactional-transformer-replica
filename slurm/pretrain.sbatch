#!/bin/bash
#SBATCH --job-name=nuformer-pretrain
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=7-00:00:00
#SBATCH --exclusive
#SBATCH --account=smml
#SBATCH --output=/fsx/paragao/logs/pretrain_%j.out
#SBATCH --error=/fsx/paragao/logs/pretrain_%j.err

# =============================================================================
# nuFormer Pre-training (NTP) on 2x p5en.48xlarge (16x H200)
#
# Expected throughput: ~25M tokens/step * steps/s
# Expected duration: ~12-18h for 50K steps
#
# Run: sbatch slurm/pretrain.sbatch
# =============================================================================

set -euo pipefail

# --- Configuration ---
CONTAINER_IMAGE="/fsx/paragao/nemo-25.11.01-efa-fixed.sqsh"
WORKSPACE="/fsx/paragao/nuformer"
MAX_STEPS=50000
BATCH_SIZE=16
SEQ_LEN=2048
LR="3e-4"

# --- EFA / NCCL Environment ---
export FI_PROVIDER=efa
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
export NCCL_P2P_NET_CHUNKSIZE=2097156
export NCCL_SOCKET_IFNAME=^docker,lo,veth
export CUDA_DEVICE_MAX_CONNECTIONS=1
export TORCH_NCCL_ASYNC_ERROR_HANDLING=1

# --- MLFlow ---
export AWS_PROFILE=compute-sa-team-Administrator
export AWS_DEFAULT_REGION=ap-southeast-3
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:ap-southeast-3:159553542841:mlflow-tracking-server/paragao-mlflow-tracker"

# --- Distributed Setup ---
export MASTER_ADDR=$(/opt/slurm/bin/scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
WORLD_SIZE=$((SLURM_NNODES * 8))

echo "=============================================="
echo "  nuFormer Pre-training"
echo "=============================================="
echo "  Job ID:          $SLURM_JOB_ID"
echo "  Nodes:           $SLURM_NNODES"
echo "  GPUs/node:       8"
echo "  Total GPUs:      $WORLD_SIZE"
echo "  Master:          $MASTER_ADDR:$MASTER_PORT"
echo "  Batch/GPU:       $BATCH_SIZE"
echo "  Grad accum:      16"
echo "  Effective batch: $((BATCH_SIZE * 16 * WORLD_SIZE))"
echo "  Seq length:      $SEQ_LEN"
echo "  Max steps:       $MAX_STEPS"
echo "=============================================="

mkdir -p logs

# --- Launch ---
/opt/slurm/bin/srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx,$HOME:$HOME" \
    --export=ALL \
    bash -c "
        export FI_PROVIDER=efa
        export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
        export NCCL_P2P_NET_CHUNKSIZE=2097156
        export NCCL_SOCKET_IFNAME=^docker,lo,veth
        export CUDA_DEVICE_MAX_CONNECTIONS=1
        export TORCH_NCCL_ASYNC_ERROR_HANDLING=1
        export NCCL_DEBUG=INFO
        export PYTHONPATH=$WORKSPACE
        export TORCHELASTIC_ERROR_FILE=/fsx/paragao/logs/elastic_error_\${SLURM_PROCID}.json
        cd $WORKSPACE && \
        torchrun \
            --nnodes=$SLURM_NNODES \
            --nproc-per-node=8 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            -m src.training.pretrain \
                --max-steps=$MAX_STEPS \
                --batch-size=$BATCH_SIZE \
                --seq-len=$SEQ_LEN \
                --lr=$LR \
                --no-compile \
                --checkpoint-dir=$WORKSPACE/ckpt/pretrain
    "

echo "=== Pre-training Complete ==="
