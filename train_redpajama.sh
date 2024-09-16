#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate llm

if [ "$(hostname)" = "vastlab" ]; then export HF_HOME="/home/yingqi/scratch/head";
elif [ "$(hostname)" = "amx1" ]; then export HF_HOME="/home/yingqi/scratch/c00/cache";
elif [ "$(hostname)" = "amx2" ]; then export HF_HOME="/home/yingqi/scratch/c01/cache";
elif [ "$(hostname)" = "amx3" ]; then export HF_HOME="/home/yingqi/scratch/c02/cache"; fi

# Verify the environment is activated
if [ $? -eq 0 ]; then
    echo "Conda environment 'llm' activated successfully."
else
    echo "Failed to activate Conda environment 'llm'. Please check if it exists."
    exit 1
fi

cd /home/yingqi/repo/HMT-pytorch

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env

echo HF_HOME=$HF_HOME

accelerate launch /home/yingqi/repo/HMT-pytorch/train_redpajama.py \
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
    --save_dir=/home/yingqi/scratch/c00/checkpoints/rp_opt-350m \
    --save_interval=10 \
    --dir_logs=/home/yingqi/repo/HMT-pytorch/tensorboard_logs \
    --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
    --validation_interval=10 \
    --validation_steps=10 \
    --curriculum \
    --wandb_entity=yic033-ucsd

# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T