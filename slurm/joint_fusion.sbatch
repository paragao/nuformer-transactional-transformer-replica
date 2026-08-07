#!/bin/bash
#SBATCH --job-name=nuformer-joint
#SBATCH --nodes=6
#SBATCH --ntasks-per-node=8
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=12
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --output=logs/joint_fusion_%j.log
#SBATCH --error=logs/joint_fusion_%j.err

# =============================================================================
# nuFormer Joint Fusion Training on 6x p5en.48xlarge (48x H200)
#
# End-to-end: Transformer (LoRA) + DCNv2 + Fusion MLP
# This is the final model configuration from the paper.
#
# Run: sbatch slurm/joint_fusion.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/containers/nuformer-efa.sqsh"
WORKSPACE="/fsx/nuformer"
PRETRAIN_CKPT="$WORKSPACE/ckpt/pretrain/final.pt"
MAX_STEPS=20000
BATCH_SIZE=32

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
echo "  nuFormer Joint Fusion"
echo "=============================================="
echo "  Job ID:     $SLURM_JOB_ID"
echo "  GPUs:       $WORLD_SIZE"
echo "  Checkpoint: $PRETRAIN_CKPT"
echo "  Max steps:  $MAX_STEPS"
echo "  Components: Transformer(LoRA) + DCNv2(3 cross) + FusionMLP"
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
            -m src.training.joint_fusion \
                --pretrain-ckpt=$PRETRAIN_CKPT \
                --max-steps=$MAX_STEPS \
                --batch-size=$BATCH_SIZE \
                --checkpoint-dir=$WORKSPACE/ckpt/joint_fusion
    "

echo "=== Joint Fusion Complete ==="
