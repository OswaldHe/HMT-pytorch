#!/bin/bash

eval "$(conda shell.bash hook)"

if [ "$(hostname)" = "c03" ];
then
    conda activate llm-nv
else 
    conda activate llm
fi

if [ "$(hostname)" = "vastlab" ]; then export HF_HOME="/home/yingqi/scratch/head";
elif [ "$(hostname)" = "c00" ]; then export HF_HOME="/home/yingqi/scratch/c00/cache";
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

export WEIGHT_BASE=/home/yingqi/repo/hmt_pretrained/opt-350m

checkpoints=(
    model_weights_0_lv_0.pth
    model_weights_0_lv_1.pth
    model_weights_0_lv_2.pth
    model_weights_0_lv_3.pth
    model_weights_0_lv_4.pth
    model_weights_700_lv_0.pth
    model_weights_700_lv_1.pth
    model_weights_700_lv_2.pth
    model_weights_700_lv_3.pth
    model_weights_700_lv_4.pth
)

checkpoints_step2=(
    model_weights_0_lv_1_step2.pth
    model_weights_0_lv_2_step2.pth
    model_weights_0_lv_3_step2.pth
    model_weights_0_lv_4_step2.pth
    model_weights_700_lv_1_step2.pth
    model_weights_700_lv_2_step2.pth
    model_weights_700_lv_3_step2.pth
    model_weights_700_lv_4_step2.pth
)

test_lengths=(3000 10000 60000)
 
for test_length in "${test_lengths[@]}"; do
    # remove the dataset cache
    rm -rf /home/yingqi/scratch/c00/cache/grouped
    rm -rf /home/yingqi/scratch/c00/cache/tokenized
    for checkpoint in "${checkpoints[@]}"; do
        accelerate launch /home/yingqi/repo/HMT-pytorch/tools/eval.py \
            --learning_rate=1e-4 \
            --model_name=facebook/opt-350m \
            --task_name=togethercomputer/RedPajama-Data-V2 \
            --task_subset=sample \
            --training_step=100 \
            --num_sensory=32 \
            --segment_length=1024 \
            --bptt_depth=6 \
            --train_set_split=2 \
            --num_seg_save=8 \
            --batch_size=2 \
            --test_length=${test_length} \
            --save_dir=/home/yingqi/scratch/c00/checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
            --validation_interval=10 \
            --validation_steps=10 \
            --curriculum \
            --curriculum_segs=2,3,4,6,8 \
            --wandb_entity=yic033-ucsd \
            --wandb_run="evaluate_${checkpoint}_testlen${test_length}" \
            --load_from_ckpt="${WEIGHT_BASE}/${checkpoint}"        
    done
done

for test_length in "${test_lengths[@]}"; do
    # remove the dataset cache
    rm -rf /home/yingqi/scratch/c00/cache/grouped
    rm -rf /home/yingqi/scratch/c00/cache/tokenized
    for checkpoint in "${checkpoints_step2[@]}"; do
        accelerate launch /home/yingqi/repo/HMT-pytorch/tools/eval.py \
            --learning_rate=1e-4 \
            --model_name=facebook/opt-350m \
            --task_name=togethercomputer/RedPajama-Data-V2 \
            --task_subset=sample \
            --training_step=100 \
            --num_sensory=32 \
            --segment_length=1024 \
            --bptt_depth=6 \
            --train_set_split=2 \
            --num_seg_save=8 \
            --batch_size=2 \
            --test_length=${test_length} \
            --save_dir=/home/yingqi/scratch/c00/checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
            --validation_interval=10 \
            --validation_steps=10 \
            --curriculum \
            --curriculum_segs=4,6,8,10 \
            --wandb_entity=yic033-ucsd \
            --wandb_run="evaluate_${checkpoint}_testlen${test_length}" \
            --load_from_ckpt="${WEIGHT_BASE}/${checkpoint}"        
    done
done

# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T