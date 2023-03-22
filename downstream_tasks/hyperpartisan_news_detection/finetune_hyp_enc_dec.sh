#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetun_hyp_enc_dec.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(facebook/bart-base t5-base)
MODEL_TYPE=encoder-decoder
MODEL_CLSS=(Bart T5)

TBS=32 # total batch size
BS=8 # batch size per gpu, * grad_acc_steps
WD=1e-03

DATA_PATH=/home/kuratov/data/hyperpartisan_news_detection

for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_CLS=${MODEL_CLSS[i]}
for SRC_LEN in 256 512 1024
do
for LR in 1e-04 5e-05 1e-05
do
for SCHEDULER in linear constant_with_warmup
do
for N in 1 2 3
do
MODEL_PATH=./runs/finetune/hyperpartisan_news_detection/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd${WD}_${SRC_LEN}/run_$N
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.hyperpartisan_news_detection.run_finetuning_hyp \
        --data_path ${DATA_PATH}/train.jsonl \
        --valid_data_path ${DATA_PATH}/dev.jsonl \
        --test_data_path ${DATA_PATH}/test.jsonl \
        --model_path $MODEL_PATH \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:${MODEL_CLS}ForConditionalGeneration \
        --use_generate_on_valid \
        --show_valid_examples 5 \
        --input_seq_len $SRC_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --save_best --iters 1000 \
        --optimizer AdamW  --weight_decay $WD \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max --early_stopping_patience 10 \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
done
echo "run_finetuning_hyp.py done"
echo "done"
