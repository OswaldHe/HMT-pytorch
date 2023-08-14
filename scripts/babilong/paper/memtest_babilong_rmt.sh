#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=encoder
BACKBONE_CLS=transformers:BertForSequenceClassification
TASK_NAME=babilong
METRIC=exact_match

ITERS=50
TBS=1

TGT_LEN=4000
MODEL_INPUT_SIZE=4000
INPUT_SEQ_LEN=4000

MAX_N_SEGMENTSS=(1)

for N in 1
do

for MODEL_NAME in bert-base-cased 
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MEMORY_SIZE=0
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
BS=1

for SEGMENT_ORDERING in regular
do

SCHEDULER=linear

for LR in 1e-05
do
MODEL_CLS=modeling_rmt.experimental:RMTEncoderCPUOffload
MODEL_CLS=modeling_rmt:RMTEncoderForSequenceClassification
BPTT=-1

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
horovodrun --gloo -np $NP python run_finetuning_babilong_random_rmt_memtest.py \
        --model_path ../runs/memtest/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-{$MAX_N_SEGMENTS}seg_mem${MEMORY_SIZE}_bptt-${BPTT}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_memtest/run_$N \
        --tokenizer $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_cfg /home/jovyan/rmt/t5-experiments/bert_configs/bert_base_uncased_4k.json \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $MODEL_INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --segment_ordering $SEGMENT_ORDERING \
        --bptt_depth $BPTT \
        --batch_size $BS --gradient_accumulation_steps 1 \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 1 \
        --data_n_workers 2 \
        --log_interval 1 --valid_interval 1 \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
        
done
done
done
done
done
echo "done"
