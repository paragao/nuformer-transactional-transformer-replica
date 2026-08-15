#!/bin/bash
#SBATCH --job-name=nuformer-jf-v2
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --gres=gpu:8
#SBATCH --cpus-per-task=48
#SBATCH --mem=0
#SBATCH --time=12:00:00
#SBATCH --exclusive
#SBATCH --account=smml
#SBATCH --exclude=p5en-dy-gpu-1
#SBATCH --output=/fsx/paragao/logs/joint_fusion_v2_%j.out
#SBATCH --error=/fsx/paragao/logs/joint_fusion_v2_%j.err

# =============================================================================
# nuFormer Joint Fusion v2 — Enhanced PLR + Deeper DCNv2
#
# Changes vs v1:
#   - PLR dim: 8 -> 16 (richer per-feature embeddings)
#   - PLR frequencies: 4 -> 8 (more periodic components)
#   - DCNv2 cross layers: 3 -> 5 (higher-order feature interactions)
#   - DCNv2 deep layers: [512, 256] -> [1024, 512, 256]
#   - DCNv2 LR: 3e-4 -> 5e-4 (PLR benefits from higher LR)
#   - LoRA rank: 16 -> 32 (more adaptation capacity)
#   - Label smoothing: 0.1 -> 0.05 (less output regularization)
#
# Target: AUC > 0.92 (vs v1 best ~0.8935)
#
# Run: sbatch slurm/joint_fusion_v2.sbatch
# =============================================================================

set -euo pipefail

CONTAINER_IMAGE="/fsx/paragao/containers/nuformer-efa-26.04.sqsh"
WORKSPACE="/fsx/paragao/nuformer"
PRETRAIN_CKPT="$WORKSPACE/ckpt/pretrain/final.pt"
MAX_STEPS=8000
BATCH_SIZE=64

# --- v2 hyperparameters ---
PLR_DIM=16
PLR_FREQ=8
CROSS_LAYERS=5
DEEP_LAYERS="1024,512,256"
DCNv2_LR=5e-4
LORA_RANK=32
LABEL_SMOOTHING=0.05

# --- EFA / NCCL ---
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

# --- Distributed ---
export MASTER_ADDR=$(/opt/slurm/bin/scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=29500
WORLD_SIZE=$((SLURM_NNODES * 8))

echo "=============================================="
echo "  nuFormer Joint Fusion v2 (Enhanced PLR)"
echo "=============================================="
echo "  Job ID:       $SLURM_JOB_ID"
echo "  Nodes:        $SLURM_NNODES"
echo "  GPUs:         $WORLD_SIZE"
echo "  Pretrain:     $PRETRAIN_CKPT"
echo "  Max steps:    $MAX_STEPS"
echo "  Batch/GPU:    $BATCH_SIZE"
echo "  PLR:          dim=$PLR_DIM, freq=$PLR_FREQ"
echo "  DCNv2:        cross=$CROSS_LAYERS, deep=[$DEEP_LAYERS]"
echo "  DCNv2 LR:     $DCNv2_LR"
echo "  LoRA rank:    $LORA_RANK"
echo "  Label smooth: $LABEL_SMOOTHING"
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
        export NCCL_DEBUG=INFO
        export PYTORCH_CUDA_ALLOC_CONF='expandable_segments:True'
        export PYTHONPATH=$WORKSPACE
        export TORCHELASTIC_ERROR_FILE=/fsx/paragao/logs/elastic_error_jfv2_\${SLURM_PROCID}.json
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
                --dcnv2-lr=$DCNv2_LR \
                --lora-rank=$LORA_RANK \
                --label-smoothing=$LABEL_SMOOTHING \
                --plr-dim=$PLR_DIM \
                --plr-frequencies=$PLR_FREQ \
                --cross-layers=$CROSS_LAYERS \
                --deep-layers=$DEEP_LAYERS \
                --checkpoint-dir=$WORKSPACE/ckpt/joint_fusion_v2
    "

echo "=== Joint Fusion v2 Complete ==="
