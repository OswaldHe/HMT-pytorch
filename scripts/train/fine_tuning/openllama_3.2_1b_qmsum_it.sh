#!/bin/bash

# IMPORTANT: Please set the CHECKPOINT and HMT_PYTORCH_PATH variables to the path to the checkpoint you want to use.
export CHECKPOINT=/home/yingqi/scratch/hmt_pretrained/llama-3.2-1b-instruct/model_weights_800.pth
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

accelerate launch $HMT_PYTORCH_PATH/hmt_tools/training/fine_tunning.py \
    --learning_rate=1e-5 \
    --model_name=meta-llama/Llama-3.2-1B-Instruct \
    --task_name=qmsum \
    --task_subset=sample \
    --training_step=1000 \
    --num_sensory=32 \
    --segment_length=512 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=1 \
    --save_dir=/home/yingqi/scratch/checkpoints/fine_tuning/openllama_3.2_1b_qmsum_it \
    --save_interval=40 \
    --token_file=huggingface_token.txt \
    --validation_interval=40 \
    --validation_steps=50 \
    --wandb_run=qmsum_fine_tuning \
    --wandb_project=rebuttle_finetuning \
    --max_context_length=10000 \
    --epochs=2 \
    --is_qa_task \
    --max_new_tokens=512 \
    --rouge \
    --it \
    --load_from_ckpt="${CHECKPOINT}"