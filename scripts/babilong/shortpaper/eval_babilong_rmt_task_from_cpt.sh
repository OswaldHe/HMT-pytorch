#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=encoder
BACKBONE_CLS=transformers:AutoModelForSequenceClassification
TASK_NAME=babilong
METRIC=exact_match

ITERS=3000
TBS=128

TGT_LEN=512


MODEL_INPUT_SIZE=512
MAX_N_SEGMENTSS=(1 2 3 4 5 6 7 8 9 10 11 12 13 14 15)
BSS=(64 64 32 32 16 16 16 8 8 8 4 4 4 4 4 2 2 2 2 2 1 1 1 1)

for N in 1
do

for MODEL_NAME in bert-base-cased 
do

for (( j=0; j<${#MAX_N_SEGMENTSS[@]}; j++ ))
do
MEMORY_SIZE=10
MAX_N_SEGMENTS=${MAX_N_SEGMENTSS[j]} 
INPUT_SEQ_LEN=$((499*MAX_N_SEGMENTS))
BS=${BSS[j]}

for SEGMENT_ORDERING in regular
do

SCHEDULER=linear

for LR in 1e-05
do

MODEL_CLS=modeling_rmt:RMTEncoderForSequenceClassification

for SOURCE_N_SEGMENTS in 1 2 3 4 5 6 7 8
do

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MODEL_CLS N_SEG MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MODEL_CLS $MAX_N_SEGMENTS $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
horovodrun --gloo -np $NP python run_finetuning_babilong_rmt.py \
        --model_path ../runs/curriculum_task/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}-{$MAX_N_SEGMENTS}seg_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_${SEGMENT_ORDERING}_${SOURCE_N_SEGMENTS}-${MAX_N_SEGMENTS}seg_eval/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_cpt ../runs/curriculum_task/babilong/bert-base-cased/lr1e-05_linear_adamw_wd1e-03_$((499*SOURCE_N_SEGMENTS))-512-{$SOURCE_N_SEGMENTS}seg_mem10_bs32_iters3000_regular_from_cpt_$((SOURCE_N_SEGMENTS-1))-${SOURCE_N_SEGMENTS}/run_$N/ \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size $MODEL_INPUT_SIZE \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --max_n_segments $MAX_N_SEGMENTS \
        --segment_ordering $SEGMENT_ORDERING \
        --validate_only \
        --bptt_depth -1 \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/5)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/100)) --valid_interval $(($ITERS/10)) \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 5 \
        --early_stopping_patience 15 \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
        
done
done
done
done
done
done
echo "done"
