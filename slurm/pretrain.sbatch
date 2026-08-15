#!/bin/bash
#SBATCH --job-name=nuformer-pretrain
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=1-12:00:00
#SBATCH --exclusive
#SBATCH --account=smml
#SBATCH --nodelist=p5en-dy-gpu-[6-7,9]
#SBATCH --output=/fsx/paragao/logs/pretrain_%j.out
#SBATCH --error=/fsx/paragao/logs/pretrain_%j.err

# =============================================================================
# nuFormer Pre-training (NTP) on 4x p5en.48xlarge (32x H200)
#
# Clean end-to-end run:
#   - 25K steps max, checkpoint every 2000
#   - Early stopping: stop if no val_loss improvement for 5 evals
#   - 300K user dataset (3x previous)
#   - Outputs tokens/sec and MFU
#
# Run: sbatch slurm/pretrain.sbatch
# =============================================================================

set -euo pipefail

# --- Configuration ---
CONTAINER_IMAGE="/fsx/paragao/containers/nuformer-efa-26.04.sqsh"
WORKSPACE="/fsx/paragao/nuformer"
MAX_STEPS=25000
SAVE_INTERVAL=2000
EARLY_STOP_PATIENCE=5
BATCH_SIZE=32
SEQ_LEN=2048
LR="3e-4"

# --- EFA / NCCL Environment ---
export FI_PROVIDER=efa
export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
export NCCL_NET_PLUGIN=/opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so
export NCCL_SOCKET_IFNAME=^docker,lo,veth

# --- PyTorch Memory ---
export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"

# --- MLFlow ---
export AWS_PROFILE=compute-sa-team-Administrator
export AWS_DEFAULT_REGION=ap-southeast-3
export MLFLOW_TRACKING_URI="arn:aws:sagemaker:ap-southeast-3:159553542841:mlflow-tracking-server/paragao-mlflow-tracker"

# --- Distributed Setup ---
export MASTER_ADDR=$(/opt/slurm/bin/scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
WORLD_SIZE=$((SLURM_NNODES * 8))

echo "=============================================="
echo "  nuFormer Pre-training (Clean E2E)"
echo "=============================================="
echo "  Job ID:          $SLURM_JOB_ID"
echo "  Nodes:           $SLURM_NNODES"
echo "  GPUs/node:       8"
echo "  Total GPUs:      $WORLD_SIZE"
echo "  Master:          $MASTER_ADDR:$MASTER_PORT"
echo "  Batch/GPU:       $BATCH_SIZE"
echo "  Grad accum:      8"
echo "  Effective batch: $((BATCH_SIZE * 8 * WORLD_SIZE))"
echo "  Seq length:      $SEQ_LEN"
echo "  Max steps:       $MAX_STEPS"
echo "  Save interval:   $SAVE_INTERVAL"
echo "  Early stop:      $EARLY_STOP_PATIENCE evals"
echo "=============================================="

mkdir -p /fsx/paragao/logs

# --- Launch ---
/opt/slurm/bin/srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx,$HOME:$HOME" \
    --export=ALL \
    bash -c "
        export FI_PROVIDER=efa
        export NCCL_TUNER_PLUGIN=/opt/amazon/ofi-nccl/lib/libnccl-ofi-tuner.so
        export NCCL_NET_PLUGIN=/opt/amazon/aws-ofi-nccl/lib/libnccl-net-ofi.so
        export NCCL_SOCKET_IFNAME=^docker,lo,veth
        export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
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
                --save-interval=$SAVE_INTERVAL \
                --early-stop-patience=$EARLY_STOP_PATIENCE \
                --checkpoint-dir=$WORKSPACE/ckpt/pretrain
    "

echo "=== Pre-training Complete ==="
