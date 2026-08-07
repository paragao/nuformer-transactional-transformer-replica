#!/bin/bash
#SBATCH --job-name=nuformer-finetune
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=06:00:00
#SBATCH --exclusive
#SBATCH --output=logs/finetune_%j.log
#SBATCH --error=logs/finetune_%j.err

# =============================================================================
# nuFormer Fine-tuning (LoRA) on 6x p5en.48xlarge (48x H200)
#
# Loads pre-trained checkpoint, applies LoRA (~1% trainable),
# trains classification head for credit card activation.
#
# Run: sbatch slurm/finetune.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/containers/nuformer-efa.sqsh"
WORKSPACE="/fsx/nuformer"
PRETRAIN_CKPT="$WORKSPACE/ckpt/pretrain/final.pt"
MAX_STEPS=10000
BATCH_SIZE=32
LR="1e-4"
LORA_RANK=16

# --- EFA / NCCL ---
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

# --- Distributed ---
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
WORLD_SIZE=$((SLURM_NNODES * 8))

echo "=============================================="
echo "  nuFormer Fine-tuning (LoRA r=$LORA_RANK)"
echo "=============================================="
echo "  Job ID:     $SLURM_JOB_ID"
echo "  GPUs:       $WORLD_SIZE"
echo "  Checkpoint: $PRETRAIN_CKPT"
echo "  Max steps:  $MAX_STEPS"
echo "=============================================="

mkdir -p logs

srun \
    --container-image="$CONTAINER_IMAGE" \
    --container-mounts="/fsx:/fsx,$HOME:$HOME" \
    bash -c "
        cd $WORKSPACE && \
        torchrun \
            --nnodes=$SLURM_NNODES \
            --nproc-per-node=8 \
            --rdzv_id=$SLURM_JOB_ID \
            --rdzv_backend=c10d \
            --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
            -m src.training.finetune \
                --pretrain-ckpt=$PRETRAIN_CKPT \
                --max-steps=$MAX_STEPS \
                --batch-size=$BATCH_SIZE \
                --lr=$LR \
                --lora-rank=$LORA_RANK \
                --checkpoint-dir=$WORKSPACE/ckpt/finetune
    "

echo "=== Fine-tuning Complete ==="
