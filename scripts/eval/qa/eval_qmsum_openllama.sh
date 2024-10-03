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

cd /home/yingqi/repo/HMT-pytorch

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env

echo HF_HOME=$HF_HOME

export WEIGHT_BASE=~/scratch/c00/hmt_pretrained/opt-350m

checkpoints=(
    model_weights_0_lv_2.pth
)

# export WANDB_MODE=offline

# FIXME: We need to modify the num_seg_save so that num_seg_save * 990 is at least as long as 
# the length of the lables, or the number of tokens in the labels will be capped at this length. 
# If the actual label length is loneger, then the size mismatch error will occur for cross-entropy loss.


accelerate launch /home/yingqi/repo/HMT-pytorch/tools/evaluation/eval.py \
    --learning_rate=1e-4 \
    --model_name=openlm-research/open_llama_3b_v2 \
    --task_name=qmsum \
    --task_subset=sample \
    --training_step=100 \
    --use_lora \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=1 \
    --test_length=3000 \
    --test_step=200 \
    --save_dir=/home/yingqi/scratch/checkpoints/opt-350m/qmsum \
    --save_interval=10 \
    --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
    --validation_interval=10 \
    --curriculum \
    --max_new_tokens=512 \
    --curriculum_segs=2,3,4,6,8 \
    --wandb_entity=yic033-ucsd \
    --wandb_run=baseline \
    --wandb_project=qa_fine_tuning_evaluation \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --load_from_ckpt="/home/yingqi/scratch/hmt_pretrained/openllama_3b_v2/model_weights_0_lv_2.pth"

accelerate launch /home/yingqi/repo/HMT-pytorch/tools/evaluation/eval.py \
    --learning_rate=1e-4 \
    --model_name=openlm-research/open_llama_3b_v2 \
    --task_name=qmsum \
    --task_subset=sample \
    --training_step=100 \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --use_lora \
    --train_set_split=2 \
    --num_seg_save=8 \
    --max_new_tokens=512 \
    --batch_size=1 \
    --test_length=3000 \
    --test_step=200 \
    --save_dir=/home/yingqi/scratch/checkpoints/opt-350m/qmsum \
    --save_interval=10 \
    --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
    --validation_interval=10 \
    --curriculum \
    --curriculum_segs=2,3,4,6,8 \
    --wandb_entity=yic033-ucsd \
    --wandb_run=fine_tuned \
    --wandb_project=qa_fine_tuning_evaluation \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --load_from_ckpt="/home/yingqi/scratch/hmt_pretrained/openllama_3b_v2/openllama-qmsum/model_weights_71.pth"


# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T