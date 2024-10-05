#!/bin/bash

eval "$(conda shell.bash hook)"

if [ "$(hostname)" = "c03" ];
then
    conda activate llm-nv
else 
    conda activate llm
fi

# Verify the environment is activated
if [ $? -eq 0 ]; then
    echo "Conda environment 'llm' activated successfully."
else
    echo "Failed to activate Conda environment 'llm'. Please check if it exists."
    exit 1
fi


if [ "$(hostname)" = "vastlab" ]; then export HF_HOME="/home/yingqi/scratch/head";
elif [ "$(hostname)" = "c00" ]; then export HF_HOME="/home/yingqi/scratch/hf_home";
elif [ "$(hostname)" = "c01" ]; then export HF_HOME="/home/yingqi/scratch/c01/cache";
elif [ "$(hostname)" = "c02" ]; then export HF_HOME="/home/yingqi/scratch/c02/cache"; 
elif [ "$(hostname)" = "c03" ]; then export HF_HOME="/home/yingqi/scratch/c03/cache"; fi


cd /home/yingqi/repo/HMT-pytorch

export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env

# Configure the task. Define the task specific parameters here, and the configuration of the models remain the change when evaluating different tasks. 
export TASK_NAME=narrativeqa
export MAX_NEW_TOKENS=128

# >>> SmolLM-135M >>>
accelerate launch --main_process_port=29501 /home/yingqi/repo/HMT-pytorch/tools/evaluation/eval.py \
    --learning_rate=1e-4 \
    --model_name=HuggingFaceTB/SmolLM-135M \
    --task_name=${TASK_NAME} \
    --task_subset=sample \
    --training_step=100 \
    --num_sensory=32 \
    --segment_length=1024 \
    --bptt_depth=6 \
    --train_set_split=2 \
    --num_seg_save=8 \
    --batch_size=1 \
    --test_length=3000 \
    --mem_recall_hidden_dim=1536 \
    --test_step=200 \
    --save_dir=/home/yingqi/scratch/checkpoints/opt-350m/qmsum \
    --save_interval=10 \
    --token_file=/home/yingqi/repo/HMT-pytorch/huggingface_token.txt \
    --validation_interval=10 \
    --curriculum \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --curriculum_segs=2,3,4,6,8 \
    --wandb_entity=yic033-ucsd \
    --wandb_run=baseline \
    --wandb_project=qa_fine_tuning_evaluation \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --load_from_ckpt="/home/yingqi/scratch/hmt_pretrained/smollm-135m/model_weights_0_lv_2.pth"
# <<< SmolLM-135M <<<

# >>> OPT-350M >>>
accelerate launch --main_process_port=29501 /home/yingqi/repo/HMT-pytorch/tools/evaluation/eval.py \
    --learning_rate=1e-4 \
    --model_name=facebook/opt-350m \
    --task_name=${TASK_NAME} \
    --task_subset=sample \
    --training_step=100 \
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
    --curriculum_segs=2,3,4,6,8 \
    --wandb_entity=yic033-ucsd \
    --wandb_run=baseline \
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --wandb_project=qa_fine_tuning_evaluation \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --load_from_ckpt="/home/yingqi/scratch/hmt_pretrained/opt-350m/model_weights_0_lv_2_step2.pth"
# <<< OPT-350M <<<

# >>> OpenLLaMA >>>
accelerate launch --main_process_port=29501 /home/yingqi/repo/HMT-pytorch/tools/evaluation/eval.py \
    --learning_rate=1e-4 \
    --model_name=openlm-research/open_llama_3b_v2 \
    --task_name=${TASK_NAME} \
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
    --max_new_tokens=${MAX_NEW_TOKENS} \
    --curriculum_segs=2,3,4,6,8 \
    --wandb_entity=yic033-ucsd \
    --wandb_run=baseline \
    --wandb_project=qa_fine_tuning_evaluation \
    --rouge \
    --is_qa_task \
    --max_context_length=16000000 \
    --load_from_ckpt="/home/yingqi/scratch/hmt_pretrained/openllama_3b_v2/model_weights_0_lv_2.pth"
# <<< OpenLLaMA <<<