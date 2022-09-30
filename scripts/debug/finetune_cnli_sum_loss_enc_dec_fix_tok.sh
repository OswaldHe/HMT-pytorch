#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_TYPE=encoder-decoder
MODEL_NAME=t5-base
MODEL_CLS=modeling_rmt_enc_dec_fix_tok:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:T5ForConditionalGeneration
TASK_NAME=contract_nli
METRIC=exact_match

INPUT_SEQ_LENS=(1002 1503)
MEMORY_SIZES=(10 10)

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]} 

for N in 1 2
do

SCHEDULER=linear 

for LR in 1e-03
do

for SEGMENT_ORDERING in regular repeat_first
do

echo N, MODEL_NAME, MODEL_ATTR, MEMORY_SIZE, INPUT_SEQ_LEN, SCHEDULER
echo $N, $MODEL_NAME, $MODEL_ATTR, $MEMORY_SIZE, $INPUT_SEQ_LEN, $SCHEDULER

horovodrun --gloo -np $NP python run_finetuning_scrolls_fix_tok.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/fix_tok/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}_mem${MEMORY_SIZE}_sum_loss_rep_first/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --backbone_cls $BACKBONE_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --input_size 512 \
        --target_seq_len 1024 \
        --use_generate_on_valid \
        --num_mem_tokens $MEMORY_SIZE \
        --segment_ordering $SEGMENT_ORDERING \
        --bptt_depth -1 \
        --backbone_trainable \
        --sum_loss \
        --batch_size 1 --gradient_accumulation_steps 8 \
        --iters 3000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 0 \
        --log_interval 50 --valid_interval 250 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"