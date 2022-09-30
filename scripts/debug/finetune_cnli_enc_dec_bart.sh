#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=encoder-decoder
MODEL_NAME=facebook/bart-base
MODEL_TYPE=encoder-decoder
MODEL_CLS=modeling_rmt_enc_dec:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:BartForConditionalGeneration
TASK_NAME=contract_nli

TGT_LEN=1024

METRIC=exact_match
SCHEDULER=linear
ITERS=4500
TBS=32
BS=16

INPUT_SEQ_LENS=(512 500)
MEMORY_SIZES=(0 0)

for LR in 1e-05 2e-04
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]} 

for N in 1
do

for SCHEDULER in constant_with_warmup 
do

echo N, MODEL_NAME, MODEL_ATTR, MEMORY_SIZE, INPUT_SEQ_LEN, SCHEDULER
echo $N, $MODEL_NAME, $MODEL_ATTR, $MEMORY_SIZE, $INPUT_SEQ_LEN, $SCHEDULER

horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/debug/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}_mem${MEMORY_SIZE}_sum_loss/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --model_attr 'model' \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size 512 \
        --target_seq_len 1024 \
        --use_generate_on_valid \
        --num_mem_tokens $MEMORY_SIZE \
        --segment_ordering regular \
        --bptt_depth -1 \
        --backbone_trainable \
        --sum_loss \
        --batch_size 1 --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters 9000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 0 \
        --log_interval $(($ITERS/40)) --valid_interval $(($ITERS/40)) \
        --optimize_metric $METRIC --optimize_mode max \
        --show_valid_examples 10 \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"