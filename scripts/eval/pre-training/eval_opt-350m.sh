#!/bin/bash


# IMPORTANT: Please set the CHECKPOINT_BASE and HMT_PYTORCH_PATH variables to the path to the checkpoint you want to use.
export CHECKPOINT_BASE=
export HMT_PYTORCH_PATH=

# Check if CHECKPOINT_BASE is set
if [ -z "$CHECKPOINT_BASE" ]; then
    echo "Error: Please provide a path to the checkpoint base. "
    exit 1
fi

# Optionally, you can print the checkpoint path for verification
echo "Using checkpoint base: $CHECKPOINT_BASE"

# Check if the directory exists
if [ ! -d "$HMT_PYTORCH_PATH" ]; then
    echo "Error: The provided path '$HMT_PYTORCH_PATH' does not exist or is not a directory."
    exit 1
fi

# Change to the HMT-pytorch directory
cd "$HMT_PYTORCH_PATH"


export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate envexport WEIGHT_BASE=hmt_pretrained/opt-350m

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

    for checkpoint in "${checkpoints[@]}"; do
        accelerate launch ${HMT_PYTORCH_PATH}/hmt_tools/evaluation/eval.py \
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
            --save_dir=checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=huggingface_token.txt \
            --validation_interval=10 \
            --validation_step=10 \
            --curriculum \
            --curriculum_segs=2,3,4,6,8 \
            --wandb_project=wandb_pretrained_evaluation \
                        --wandb_run="evaluate_${checkpoint}_testlen${test_length}" \
            --load_from_ckpt="${CHECKPOINT_BASE}/${checkpoint}"        
    done
done

for test_length in "${test_lengths[@]}"; do
    # remove the dataset cache

    for checkpoint in "${checkpoints_step2[@]}"; do
        accelerate launch ${HMT_PYTORCH_PATH}/hmt_tools/eval.py \
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
            --save_dir=checkpoints/rp_opt-350m \
            --save_interval=10 \
            --token_file=huggingface_token.txt \
            --validation_interval=10 \
            --validation_steps=10 \
            --curriculum \
            --curriculum_segs=4,6,8,10 \
                        --wandb_run="evaluate_${checkpoint}_testlen${test_length}" \
            --load_from_ckpt="${CHECKPOINT_BASE}/${checkpoint}"        
    done
done