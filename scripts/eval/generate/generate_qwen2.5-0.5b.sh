#!/bin/bash

eval "$(conda shell.bash hook)"

if [ "$(hostname)" = "c03" ];
then
    conda activate llm-nv
else 
    conda activate llm
fi

if [ "$(hostname)" = "vastlab" ]; then export HF_HOME="/home/yingqi/scratch/head";
elif [ "$(hostname)" = "c00" ]; then export HF_HOME="/home/yingqi/scratch/hf_home";
elif [ "$(hostname)" = "c01" ]; then export HF_HOME="/home/yingqi/scratch/c01/cache";
elif [ "$(hostname)" = "c02" ]; then export HF_HOME="/home/yingqi/scratch/c02/cache"; 
elif [ "$(hostname)" = "c03" ]; then export HF_HOME="/home/yingqi/scratch/c03/cache"; fi

# Verify the environment is activated
if [ $? -eq 0 ]; then
    echo "Conda environment 'llm' activated successfully."
else
    echo "Failed to activate Conda environment 'llm'. Please check if it exists."
    exit 1
fi

export PYTHONPATH=/home/yingqi/repo/HMT-pytorch:$PYTHONPATH

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env

echo HF_HOME=$HF_HOME

export WEIGHT_BASE=/home/yingqi/scratch/c00/hmt_pretrained/qwen2.5-0.5b

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
            --save_dir=/home/yingqi/scratch/c00/checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
            --validation_interval=10 \
            --validation_step=10 \
            --test_step=100 \
            --curriculum \
            --curriculum_segs=2,4,6,8 \
            --mem_recall_hidden_dim=4864 \
            --wandb_entity=yic033-ucsd \
            --wandb_project=wandb_pretrained_evaluation \
            --wandb_run="generate_${checkpoint}_testlen${test_length}" \
            --max_context_length=40000 \
            --load_from_ckpt="/home/yingqi/repo/HMT-pytorch/tmp/model_weights_0_lv_2.pth"       


# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T