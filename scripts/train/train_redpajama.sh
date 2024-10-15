#!/bin/bash

# Check if the first argument is provided
if [ -z "$1" ]; then
    echo "Error: HMT-pytorch path is not provided."
    echo "Usage: $0 <path_to_HMT-pytorch>"
    echo "Example: $0 /home/user/repo/HMT-pytorch"
    exit 1
fi

# Assign the first argument to a variable
HMT_PYTORCH_PATH="$1"

# Check if the directory exists
if [ ! -d "$HMT_PYTORCH_PATH" ]; then
    echo "Error: The provided path '$HMT_PYTORCH_PATH' does not exist or is not a directory."
    exit 1
fi

# Change to the HMT-pytorch directory
cd "$HMT_PYTORCH_PATH"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate envaccelerate launch $HMT_PYTORCH_PATH/train_redpajama.py \
    --learning_rate=1e-4 \
    --model_name=facebook/opt-350m \
    --task_name=togethercomputer/RedPajama-Data-V2 \
    --task_subset=sample \
    --use_lora \
    --lr_decay \
    --lr_decay_gamma=0.6 \
    --training_step=100 \
    --num_sensory=32 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=2 \
    --test_length=30000 \
    --save_dir=checkpoints/rp_opt-350m \
    --save_interval=10 \
    --dir_logs=${HMT_PYTORCH_PATH}/tensorboard_logs \
    --token_file=huggingface_token.txt \
    --validation_interval=10 \
    --validation_steps=10 \
    --curriculum \
    --recache_splits=test