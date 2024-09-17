#!/bin/bash

#SBATCH --job-name=train_hmt_redpajama
#SBATCH --output=job.out
#SBATCH --partition=mi2508x
#SBATCH -t 12:00:00

eval "$(conda shell.bash hook)"
conda activate hmt

# Verify the environment is activated
if [ $? -eq 0 ]; then
    echo "Conda environment 'llm' activated successfully."
else
    echo "Failed to activate Conda environment 'llm'. Please check if it exists."
    exit 1
fi

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

cd /home1/oswaldhe/HMT-pytorch/script/HMT-pytorch
accelerate env

accelerate launch train_redpajama.py \
    --learning_rate=1e-5 \
    --model_name=facebook/opt-350m \
    --task_name=togethercomputer/RedPajama-Data-V2 \
    --task_subset=sample \
    --training_step=800 \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=2 \
    --test_length=10000 \
    --save_dir=/work1/zifanhe/oswaldhe/rp_opt-350m \
    --save_interval=100 \
    --dir_logs=/work1/zifanhe/oswaldhe/tensorboard_logs \
    --token_file=huggingface_token.txt \
    --validation_interval=10 \
    --validation_steps=10 \
    --curriculum


# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T

