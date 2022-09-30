#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..


CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-base
MODEL_TYPE=encoder-decoder
MODEL_CLS=modeling_rmt_enc_dec:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:T5ForConditionalGeneration
TASK_NAME=quality 
METRIC=exact_match


PREFIX=""

TGT_LEN=1024

SCHEDULER=linear

ITERS=3200
TBS=8
BS=2


INPUT_SEQ_LENS=(1002 )
MEMORY_SIZES=(10 )

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]} 

for LR in 2e-04 
do

for SEGMENT_ORDERING in repeat_first regular 
do

for N in 1 2
do

echo RUNNING: TASK_NAME SRC_LEN MODEL_NAME MEMORY_SIZE INPUT_SEQ_LEN LR N
echo RUNNING: $TASK_NAME $SRC_LEN $MODEL_NAME $MEMORY_SIZE $INPUT_SEQ_LEN $LR $N
horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/debug/${TASK_NAME}/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}_mem${MEMORY_SIZE}_bs${TBS}_iters${ITERS}_sl_${SEGMENT_ORDERING}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --backbone_cls $BACKBONE_CLS \
        --use_generate_on_valid \
        --input_prefix "$PREFIX" \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size 512 \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --segment_ordering $SEGMENT_ORDERING \
        --sum_loss \
        --save_best \
        --bptt_depth -1 \
        --backbone_trainable \
        --target_seq_len $TGT_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/20)) --valid_interval $(($ITERS/20)) \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42))
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"

