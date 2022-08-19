#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-base
MODEL_TYPE=encoder-decoder
MODEL_CLS=modeling_rmt_enc_dec:RMTEncoderDecoderForConditionalGeneration
BACKBONE_CLS=transformers:T5ForConditionalGeneration
TASK_NAME=quality 

INPUT_SEQ_LENS=(1002 1503)
MEMORY_SIZES=(10 10)

TGT_LEN=512

METRIC=exact_match

for SCHEDULER in linear constant_with_warmup
do

for N in 1 2 
do

for (( j=0; j<${#MEMORY_SIZES[@]}; j++ ))
do
MEMORY_SIZE=${MEMORY_SIZES[j]}
INPUT_SEQ_LEN=${INPUT_SEQ_LENS[j]}

LR=1e-05

echo N, MODEL_NAME, MEMORY_SIZE, INPUT_SEQ_LEN, SCHEDULER
echo $N, $MODEL_NAME, $MEMORY_SIZE, $INPUT_SEQ_LEN, $SCHEDULER

horovodrun --gloo -np $NP python run_finetuning_scrolls_rmt.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/debug/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}-${TGT_LEN}_mem${MEMORY_SIZE}_sum_loss/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --backbone_cls $BACKBONE_CLS \
        --use_generate_on_valid \
        --backbone_trainable \
        --sum_loss \
        --input_seq_len $INPUT_SEQ_LEN \
        --target_seq_len $TGT_LEN \
        --num_mem_tokens $MEMORY_SIZE \
        --input_size 512 \
        --bptt_depth -1 \
        --batch_size  --gradient_accumulation_steps 2 \
        --iters 2000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 200 \
        --data_n_workers 0 \
        --log_interval 100 --valid_interval 200 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"
