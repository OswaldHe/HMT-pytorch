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

accelerate env# Manually remove the cache dir if necessary. It is used to force recaching. 



# Uncomment to disable wandb tracking
export WANDB_MODE=offline

accelerate launch $HMT_PYTORCH_PATH/tools/training/fine_tunning.py \
    --learning_rate=1e-5 \
    --model_name=HuggingFaceTB/SmolLM-135M \
    --task_name=ioeddk/qmsum \
    --task_subset=sample \
    --training_step=1000 \
    --epochs=5 \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=1 \
    --mem_recall_hidden_dim=1536 \
    --save_dir=checkpoints/fine_tuning/smollm-135m/qmsum \
    --save_interval=20 \
    --token_file=huggingface_token.txt \
    --validation_interval=40 \
    --validation_steps=30 \
    --wandb_run=qmsum_fine_tuning \
    --wandb_project=qa_fine_tuning \
    --max_context_length=20000 \
    --is_qa_task \
    --rouge \
    --load_from_ckpt=hmt_pretrained/smollm-135m/model_weights_500_lv_3.pth 