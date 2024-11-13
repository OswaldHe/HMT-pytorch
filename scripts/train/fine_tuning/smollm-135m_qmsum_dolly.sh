#!/bin/bash

# IMPORTANT: Please set the CHECKPOINT and HMT_PYTORCH_PATH variables to the path to the checkpoint you want to use.
export CHECKPOINT=/home/yingqi/repo/HMT-pytorch/models/smollm/model_weights_0_lv_2.pth
export HMT_PYTORCH_PATH=/home/yingqi/repo/HMT-pytorch

# Check if CHECKPOINT is set
if [ -z "$CHECKPOINT" ]; then
    echo "Error: Please provide a path to the checkpoint. "
    exit 1
fi

# Check if the directory exists
if [ ! -d "$HMT_PYTORCH_PATH" ]; then
    echo "Error: The provided path '$HMT_PYTORCH_PATH' does not exist or is not a directory."
    exit 1
fi

# Change to the HMT-pytorch directory
cd "$HMT_PYTORCH_PATH"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env # Manually remove the cache dir if necessary. It is used to force recaching. 


# Uncomment to disable wandb tracking
# export WANDB_MODE=offline

# first try 1e-3 then 1e-5

accelerate launch $HMT_PYTORCH_PATH/hmt_tools/training/fine_tunning.py \
    --learning_rate=1e-3 \
    --model_name=HuggingFaceTB/SmolLM-135M-Instruct \
    --task_name=dolly_sum \
    --task_subset=sample \
    --training_step=1000 \
    --epochs=2 \
    --num_sensory=32 \
    --segment_length=1024 \
    --shuffle_train \
    --shuffle \
    --bptt_depth=6 \
    --num_seg_save=8 \
    --batch_size=1 \
    --mem_recall_hidden_dim=1536 \
    --save_dir=/home/yingqi/scratch/checkpoints/fine_tuning/smollm-135m/dolly_sum \
    --save_interval=200 \
    --token_file=huggingface_token.txt \
    --validation_interval=40 \
    --validation_steps=50 \
    --wandb_run=dolly_fine_tuning_smollm_135m \
    --wandb_project=qa_fine_tuning \
    --max_context_length=40000 \
    --is_qa_task \
    --rouge \
    --it \
    --with_text \
    --load_from_ckpt="${CHECKPOINT}"

accelerate launch $HMT_PYTORCH_PATH/hmt_tools/training/fine_tunning.py \
    --learning_rate=1e-5 \
    --model_name=HuggingFaceTB/SmolLM-135M-Instruct \
    --task_name=dolly_sum \
    --task_subset=sample \
    --training_step=1000 \
    --epochs=2 \
    --num_sensory=32 \
    --segment_length=1024 \
    --shuffle_train \
    --shuffle \
    --bptt_depth=6 \
    --num_seg_save=8 \
    --batch_size=1 \
    --mem_recall_hidden_dim=1536 \
    --save_dir=/home/yingqi/scratch/checkpoints/fine_tuning/smollm-135m/dolly_sum \
    --save_interval=200 \
    --token_file=huggingface_token.txt \
    --validation_interval=40 \
    --validation_steps=50 \
    --wandb_run=dolly_fine_tuning_smollm_135m \
    --wandb_project=qa_fine_tuning \
    --max_context_length=40000 \
    --is_qa_task \
    --rouge \
    --it \
    --with_text \
    --load_from_ckpt="${CHECKPOINT}"
