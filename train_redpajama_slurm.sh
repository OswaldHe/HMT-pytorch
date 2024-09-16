#!/bin/bash

#SBATCH --job-name=train_redpajama
#SBATCH --output=/home/yingqi/repo/HMT-pytorch/train_redpajama.out
#SBATCH --partition=batch
#SBATCH --gpus=mi210:2
#SBATCH --ntasks=1
#SBATCH --mem=100G
#SBATCH --cpus-per-task=10
#SBATCH --qos=lowest
#SBATCH --nodes=1
#SBATCH --nodelist=c01

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
export TORCH_DISTRIBUTED_DEBUG=INFO
export HF_HOME=/home/yingqi/scratch/c00/cache

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
    --validation_interval=1 \
    --validation_steps=10 \
    --curriculum


# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T