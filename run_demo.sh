#!/bin/bash

# set -x -e

eval "$(conda shell.bash hook)"
conda activate llm

# Verify the environment is activated
if [ $? -eq 0 ]; then
    echo "Conda environment 'llm' activated successfully."
else
    echo "Failed to activate Conda environment 'llm'. Please check if it exists."
    exit 1
fi

accelerate env

cd /home/yingqi/repo/HMT-pytorch

export NCCL_DEBUG=INFO
export AMD_SERIALIZE_KERNEL=3
export PYTORCH_ROCM_ARCH=gfx90a
export HSA_OVERRIDE_GFX_VERSION=9.0.0
export HIP_VISIBLE_DEVICES=0,1
export ROCM_PATH=/opt/rocm

# disable NCCL P2P and IB for debugging purpose 
# export NCCL_IB_DISABLE=1
# # Disable NCCL P2P communication
# export NCCL_P2P_DISABLE=1


# Training setup
export GPUS_PER_NODE=1
# so processes know who to talk to
export MASTER_ADDR=$(scontrol show hostnames $SLURM_JOB_NODELIST | head -n 1)
export MASTER_PORT=6000
export NNODES=$SLURM_NNODES
export NODE_RANK=$SLURM_PROCID 
export WORLD_SIZE=$(($GPUS_PER_NODE*$NNODES))

echo $SLURM_PROCID

# accelerate launch \
#     --multi_gpu \
#     --num_machines $NNODES \
#     --num_processes $WORLD_SIZE \
#     --main_process_ip "$MASTER_ADDR" \
#     --main_process_port $MASTER_PORT \
#     --num_processes $WORLD_SIZE \
#     --machine_rank $SLURM_PROCID \
#     --role $SLURMD_NODENAME: \
#     --max_restarts 0 \
#     --tee 3 \
#     tests/test_accelerate.py

export HF_HOME="/home/yingqi/scratch/c01/cache"

echo "See Cache"
ls $HF_HOME

accelerate launch --debug --num_processes 2 --num_machines 1 --machine_rank 0 /home/yingqi/repo/HMT-pytorch/hmt_src/main.py \
    --task_subset=wikitext-103-raw-v1 \
    --use_lora \
    --learning_rate=1e-4 \
    --model_name=facebook/opt-350m \
    --lr_decay \
    --lr_decay_gamma=0.6 \
    --training_step=600 \
    --num_sensory=32 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=2 \
    --test_length=30000 

