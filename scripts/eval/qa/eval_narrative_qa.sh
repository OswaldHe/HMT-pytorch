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

test_lengths=(3000)

# FIXME: We need to modify the num_seg_save so that num_seg_save * 990 is at least as long as 
# the length of the lables, or the number of tokens in the labels will be capped at this length. 
# If the actual label length is loneger, then the size mismatch error will occur for cross-entropy loss.


for test_length in "${test_lengths[@]}"; do
    # remove the dataset cache
    rm -rf /home/yingqi/scratch/c00/cache/grouped
    rm -rf /home/yingqi/scratch/c00/cache/tokenized
    for checkpoint in "${checkpoints[@]}"; do
        accelerate launch tools/evaluation/eval.py \
            --learning_rate=1e-4 \
            --model_name=facebook/opt-350m \
            --task_name=deepmind/narrativeqa \
            --task_subset=sample \
            --training_step=100 \
            --num_sensory=32 \
            --segment_length=1024 \
            --bptt_depth=6 \
            --train_set_split=2 \
            --num_seg_save=8 \
            --batch_size=1 \
            --test_length=${test_length} \
            --test_step=200 \
            --save_dir=/home/yingqi/scratch/c00/checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
            --validation_interval=10 \
            --validation_steps=10 \
            --curriculum \
            --curriculum_segs=2,3,4,6,8 \
            --wandb_entity=yic033-ucsd \
            --wandb_run=evaluate_narrative_qa_baseline \
            --wandb_project=qa_fine_tuning \
            --load_from_ckpt="/home/yingqi/scratch/c00/checkpoints/fine_tuning/opt-350m/narrative_qa/model_weights_580.pth" \
            --is_qa_task    
    done
done

# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T