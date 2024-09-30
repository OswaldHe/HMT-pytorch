#!/bin/bash

eval "$(conda shell.bash hook)"
conda activate llm

if [ "$(hostname)" = "vastlab" ]; then export HF_HOME="/home/yingqi/scratch/head";
elif [ "$(hostname)" = "c00" ]; then export HF_HOME="/home/yingqi/scratch/hf_home";
elif [ "$(hostname)" = "c01" ]; then export HF_HOME="/home/yingqi/scratch/c01/cache";
elif [ "$(hostname)" = "c02" ]; then export HF_HOME="/home/yingqi/scratch/c02/cache"; fi

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

# Manually remove the cache dir if necessary. It is used to force recaching. 
# rm -rf /home/yingqi/scratch/c00/cache/tokenized
# rm -rf /home/yingqi/scratch/c00/cache/grouped

# Uncomment to disable wandb tracking
# export WANDB_MODE=offline

accelerate launch /home/yingqi/repo/HMT-pytorch/tools/training/fine_tunning.py \
    --learning_rate=1e-5 \
    --model_name=facebook/opt-350m \
    --task_name=musique \
    --task_subset=sample \
    --training_step=1000 \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=1 \
    --save_dir=/home/yingqi/scratch/c00/checkpoints/fine_tuning/opt-350m/musique \
    --save_interval=20 \
    --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
    --validation_interval=40 \
    --validation_steps=30 \
    --wandb_entity=yic033-ucsd \
    --wandb_run=musique_fine_tuning \
    --wandb_project=qa_fine_tuning \
    --max_context_length=16000 \
    --is_qa_task \
    --load_from_ckpt="/home/yingqi/scratch/c00/hmt_pretrained/opt-350m/model_weights_0_lv_2.pth"

# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T