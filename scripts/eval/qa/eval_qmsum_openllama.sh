#!/bin/bash

# IMPORTANT: Please set the ZEROSHOT_CHECKPOINT and FINETUNED_CHECKPOINT and HMT_PYTORCH_PATH variables to the path to the checkpoint you want to use.
export ZEROSHOT_CHECKPOINT=/home/yingqi/scratch/hmt_pretrained/openllama_3b_v2/model_weights_0_lv_2.pth
# export FINETUNED_CHECKPOINT=/home/yingqi/scratch/hmt_pretrained/openllama_3b_v2/openllama-qmsum/model_weights_71.pth
export HMT_PYTORCH_PATH=/home/yingqi/repo/HMT-pytorch

# Check if ZEROSHOT_CHECKPOINT is set
if [ -z "$ZEROSHOT_CHECKPOINT" ]; then
    echo "Error: Please provide a path to the zeroshot checkpoint. "
    exit 1
fi

# Check if FINETUNED_CHECKPOINT is set
# if [ -z "$FINETUNED_CHECKPOINT" ]; then
#     echo "Error: Please provide a path to the finetuned checkpoint. "
#     exit 1
# fi

# Optionally, you can print the checkpoint path for verification
echo "Using zeroshot checkpoint: $ZEROSHOT_CHECKPOINT"
echo "Using finetuned checkpoint: $FINETUNED_CHECKPOINT"

# Check if the directory exists
if [ ! -d "$HMT_PYTORCH_PATH" ]; then
    echo "Error: The provided path '$HMT_PYTORCH_PATH' does not exist or is not a directory."
    exit 1
fi

# Change to the HMT-pytorch directory
cd "$HMT_PYTORCH_PATH"

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

# export WANDB_MODE=offline
accelerate launch --main_process_port=$MAIN_PROCESS_PORT ${HMT_PYTORCH_PATH}/hmt_tools/evaluation/eval.py \
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
    --save_dir=checkpoints/opt-350m/qmsum \
    --save_interval=10 \
    --token_file=huggingface_token.txt \
    --validation_interval=10 \
    --curriculum \
    --max_new_tokens=512 \
    --curriculum_segs=2,3,4,6,8 \
    --wandb_project=re-evaluation \
    --wandb_entity=yic033-ucsd \
    --wandb_run=zeroshot_new_prompt \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --temperature=0.8 \
    --do_sample \
    --num_beams=5 \
    --save_generated_texts=qmsum_openllama_zeroshot.csv \
    --load_from_ckpt="${ZEROSHOT_CHECKPOINT}"

# accelerate launch --main_process_port=$MAIN_PROCESS_PORT ${HMT_PYTORCH_PATH}/hmt_tools/evaluation/eval.py \
#     --learning_rate=1e-4 \
#     --model_name=openlm-research/open_llama_3b_v2 \
#     --task_name=qmsum \
#     --task_subset=sample \
#     --training_step=100 \
#     --num_sensory=32 \
#     --segment_length=1024 \
#     --bptt_depth=6 \
#     --use_lora \
#     --train_set_split=2 \
#     --num_seg_save=8 \
#     --max_new_tokens=512 \
#     --batch_size=1 \
#     --test_length=3000 \
#     --test_step=200 \
#     --save_dir=checkpoints/opt-350m/qmsum \
#     --save_interval=10 \
#     --token_file=huggingface_token.txt \
#     --validation_interval=10 \
#     --curriculum \
#     --curriculum_segs=2,3,4,6,8 \
#     --wandb_project=re-evaluation \
#     --wandb_entity=yic033-ucsd \
#     --wandb_run=finetuned \
#     --rouge \
#     --is_qa_task \
#     --max_context_length=16000000 \
#     --save_generated_texts=qmsum_openllama_finetuned.csv \
#     --load_from_ckpt="${FINETUNED_CHECKPOINT}"
