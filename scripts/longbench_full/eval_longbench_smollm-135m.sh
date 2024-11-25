#!/bin/bash

# IMPORTANT: Please set the ZEROSHOT_CHECKPOINT and FINETUNED_CHECKPOINT and HMT_PYTORCH_PATH variables to the path to the checkpoint you want to use.
export ZEROSHOT_CHECKPOINT=/home/yingqi/scratch/hmt_pretrained/smollm-135m/model_weights_0_lv_2.pth
export HMT_PYTORCH_PATH=/home/yingqi/repo/HMT-pytorch

# Optionally, you can print the checkpoint path for verification
echo "Using zeroshot checkpoint: $ZEROSHOT_CHECKPOINT"

# Check if the directory exists
if [ ! -d "$HMT_PYTORCH_PATH" ]; then
    echo "Error: The provided path '$HMT_PYTORCH_PATH' does not exist or is not a directory."
    exit 1
fi

# Change to the HMT-pytorch directory
cd "$HMT_PYTORCH_PATH"

# Define list of task names
# QA_TASK_NAMES=("multifieldqa_en" "narrativeqa" "qasper" "hotpotqa" "2wikimqa" "musique" "dureader")
QA_TASK_NAMES=("triviaqa")

# Iterate through tasks
for task in "${QA_TASK_NAMES[@]}"; do
    echo "Evaluating QA task: $task"
    accelerate launch --main_process_port=$MAIN_PROCESS_PORT ${HMT_PYTORCH_PATH}/hmt_tools/evaluation/eval_longbench.py \
        --learning_rate=1e-4 \
        --model_name=HuggingFaceTB/SmolLM-135M \
        --task_name=THUDM/LongBench \
        --task_subset=$task \
        --num_sensory=32 \
        --segment_length=1024 \
        --bptt_depth=6 \
        --train_set_split=2 \
        --num_seg_save=8 \
        --batch_size=1 \
        --test_length=3000 \
        --mem_recall_hidden_dim=1536 \
        --test_step=200 \
        --token_file=huggingface_token.txt \
        --wandb_project=rebuttle_evaluation \
        --wandb_entity=yic033-ucsd \
        --wandb_run=longbench_smollm-135m_zeroshot_${task} \
        --rouge \
        --is_qa_task \
        --temperature=1.0 \
        --num_beams=1 \
        --max_context_length=16000000 \
        --save_generated_texts=benchmark_results/longbench_smollm-135m_zeroshot_${task}.csv \
        --load_from_ckpt="${ZEROSHOT_CHECKPOINT}"
done

# DATASETS=("gov_report" "qmsum" "multi_news" "vcsum" "trec" "triviaqa" "samsum" "lsht" 
#             "passage_count" "passage_retrieval_en" "passage_retrieval_zh" "lcc" "repobench-p")
DATASETS=("multi_news")

# Iterate through tasks
for task in "${DATASETS[@]}"; do
    echo "Evaluating Task: $task"
    accelerate launch --main_process_port=$MAIN_PROCESS_PORT ${HMT_PYTORCH_PATH}/hmt_tools/evaluation/eval_longbench.py \
        --learning_rate=1e-4 \
        --model_name=HuggingFaceTB/SmolLM-135M \
        --task_name=THUDM/LongBench \
        --task_subset=$task \
        --num_sensory=32 \
        --segment_length=1024 \
        --bptt_depth=6 \
        --train_set_split=2 \
        --num_seg_save=8 \
        --batch_size=1 \
        --test_length=3000 \
        --mem_recall_hidden_dim=1536 \
        --test_step=200 \
        --token_file=huggingface_token.txt \
        --wandb_project=rebuttle_evaluation \
        --wandb_entity=yic033-ucsd \
        --wandb_run=longbench_smollm-135m_zeroshot_${task} \
        --rouge \
        --max_context_length=16000000 \
        --temperature=1.0 \
        --num_beams=1 \
        --save_generated_texts=benchmark_results/longbench_smollm-135m_zeroshot_${task}.csv \
        --load_from_ckpt="${ZEROSHOT_CHECKPOINT}"
done
