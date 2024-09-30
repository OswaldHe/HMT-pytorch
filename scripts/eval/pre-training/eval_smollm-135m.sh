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

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env

echo HF_HOME=$HF_HOME

export WEIGHT_BASE=/home/yingqi/scratch/c00/hmt_pretrained/smollm-135m

checkpoints=(
    model_weights_0_lv_1.pth  
    model_weights_0_lv_2.pth 
    model_weights_0_lv_3.pth 
    model_weights_500_lv_3.pth
)


test_lengths=(3000 10000 60000)
 
for test_length in "${test_lengths[@]}"; do
    # remove the dataset cache
    rm -rf /home/yingqi/scratch/c00/cache/grouped/togethercomputer
    rm -rf /home/yingqi/scratch/c00/cache/tokenized/togethercomputer
    for checkpoint in "${checkpoints[@]}"; do
        accelerate launch tools/evaluation/eval.py \
            --learning_rate=1e-4 \
            --model_name=HuggingFaceTB/SmolLM-135M \
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
            --mem_recall_hidden_dim=1536 \
            --save_interval=10 \
            --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
            --validation_interval=10 \
            --validation_step=10 \
            --curriculum \
            --curriculum_segs=2,4,6,8 \
            --wandb_entity=yic033-ucsd \
            --wandb_project=wandb_pretrained_evaluation \
            --wandb_run="evaluate_${checkpoint}_testlen${test_length}" \
            --load_from_ckpt="${WEIGHT_BASE}/${checkpoint}"        
    done
done

# Available model names:
# meta-llama/Llama-2-7b-hf
# facebook/opt-350m

# Valid task_subset: sample, sample-10B, sample-100B, sample-1T