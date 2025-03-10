#!/bin/bash

# IMPORTANT: Please set the CHECKPOINT and HMT_PYTORCH_PATH variables to the path to the checkpoint and HMT-pytorch repository you want to use.
export CHECKPOINT=
export HMT_PYTORCH_PATH=

# Check if CHECKPOINT is set
if [ -z "$CHECKPOINT" ]; then
    echo "Error: Please provide a path to the checkpoint. "
    exit 1
fi

# Optionally, you can print the checkpoint path for verification
echo "Using checkpoint: $CHECKPOINT"

# Check if the directory exists
if [ ! -d "$HMT_PYTORCH_PATH" ]; then
    echo "Error: The provided path '$HMT_PYTORCH_PATH' does not exist or is not a directory."
    exit 1
fi

# Change to the HMT-pytorch directory
cd "$HMT_PYTORCH_PATH"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO


# Uncomment to disable wandb tracking
export WANDB_MODE=offline


python tools/evaluation/generate.py \
            --learning_rate=1e-4 \
            --model_name=Qwen/Qwen2.5-0.5B \
            --task_name=qmsum \
            --task_subset=sample \
            --training_step=100 \
            --num_sensory=32 \
            --segment_length=512 \
            --bptt_depth=6 \
            --train_set_split=2 \
            --num_seg_save=8 \
            --batch_size=2 \
            --test_length=10000 \
            --save_dir=checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=huggingface_token.txt \
            --validation_interval=10 \
            --validation_step=10 \
            --test_step=100 \
            --curriculum \
            --curriculum_segs=2,4,6,8 \
            --mem_recall_hidden_dim=4864 \
                        --wandb_project=wandb_pretrained_evaluation \
            --wandb_run="generate_${checkpoint}_testlen${test_length}" \
            --max_context_length=40000 \
            --load_from_ckpt=${CHECKPOINT}      
