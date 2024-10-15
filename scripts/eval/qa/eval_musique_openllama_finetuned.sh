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

accelerate env

accelerate launch  ${HMT_PYTORCH_PATH}/tools/evaluation/eval.py \
    --learning_rate=1e-4 \
    --model_name=openlm-research/open_llama_3b_v2 \
    --task_name=musique \
    --task_subset=sample \
    --training_step=100 \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=1 \
    --test_length=3000 \
    --test_step=200 \
    --save_dir=checkpoints/opt-350m/qmsum \
    --save_interval=10 \
    --max_new_tokens=32 \
    --token_file=huggingface_token.txt \
    --validation_interval=10 \
    --curriculum \
    --use_lora \
    --curriculum_segs=2,3,4,6,8 \
    --wandb_run=baseline \
    --wandb_project=qa_fine_tuning_evaluation \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --load_from_ckpt="hmt_pretrained/openllama_3b_v2/model_weights_0_lv_2.pth"

accelerate launch  ${HMT_PYTORCH_PATH}/tools/evaluation/eval.py \
    --learning_rate=1e-4 \
    --model_name=openlm-research/open_llama_3b_v2 \
    --task_name=musique \
    --task_subset=sample \
    --training_step=100 \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --max_new_tokens=32 \
    --batch_size=1 \
    --use_lora \
    --test_length=3000 \
    --test_step=200 \
    --save_dir=checkpoints/opt-350m/qmsum \
    --save_interval=10 \
    --token_file=huggingface_token.txt \
    --validation_interval=10 \
    --curriculum \
    --curriculum_segs=2,3,4,6,8 \
    --wandb_run=fine_tuned \
    --wandb_project=qa_fine_tuning_evaluation \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --load_from_ckpt="hmt_pretrained/openllama_3b_v2/openllama-musique/model_weights_1301.pth"