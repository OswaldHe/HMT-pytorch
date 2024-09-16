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

cd /home/yingqi/repo/HMT-pytorch

export NCCL_DEBUG=INFO
# export AMD_SERIALIZE_KERNEL=3
# export PYTORCH_ROCM_ARCH=gfx90a
# export HSA_OVERRIDE_GFX_VERSION=9.0.0
# export HIP_VISIBLE_DEVICES=0,1
# export ROCM_PATH=/opt/rocm

# disable NCCL P2P and IB for debugging purpose 
# export NCCL_IB_DISABLE=1
# # Disable NCCL P2P communication
# export NCCL_P2P_DISABLE=1

export TORCH_DISTRIBUTED_DEBUG=INFO

echo "See Cache"
ls -a $HF_HOME
echo "find accelerate config"
ls -a $HF_HOME/accelerate

accelerate env

accelerate launch /home/yingqi/repo/HMT-pytorch/hmt_src/main.py \
    --learning_rate=1e-4 \
    --model_name=facebook/opt-350m \
    --task_subset=wikitext-103-raw-v1 \
    --use_lora \
    --lr_decay \
    --lr_decay_gamma=0.6 \
    --training_step=600 \
    --num_sensory=32 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=2 \
    --test_length=30000 \
    --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt 