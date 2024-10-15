#!/bin/bash

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate envexport WEIGHT_BASE=hmt_pretrained/qwen2.5-0.5b

# Uncomment to disable wandb tracking
# export WANDB_MODE=offline

checkpoints=(
    model_weights_0_lv_1.pth
    model_weights_0_lv_2.pth
    model_weights_0_lv_3.pth
)

test_lengths=(3000 10000 60000)
 
for test_length in "${test_lengths[@]}"; do
    # remove the dataset cache

    for checkpoint in "${checkpoints[@]}"; do
        accelerate launch tools/evaluation/eval.py \
            --learning_rate=1e-4 \
            --model_name=Qwen/Qwen2.5-0.5B \
            --task_name=togethercomputer/RedPajama-Data-V2 \
            --task_subset=sample \
            --training_step=100 \
            --num_sensory=32 \
            --segment_length=512 \
            --bptt_depth=6 \
            --train_set_split=2 \
            --num_seg_save=8 \
            --batch_size=2 \
            --test_length=${test_length} \
            --save_dir=checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=huggingface_token.txt \
            --validation_interval=10 \
            --validation_step=10 \
            --test_step=100 \
            --curriculum \
            --curriculum_segs=2,4,6,8 \
            --mem_recall_hidden_dim=4864 \
                        --wandb_project=wandb_pretrained_evaluation \
            --wandb_run="evaluate_${checkpoint}_testlen${test_length}" \
            --load_from_ckpt="${WEIGHT_BASE}/${checkpoint}"        
    done
done