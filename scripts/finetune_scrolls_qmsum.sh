#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(facebook/bart-base t5-base)
MODEL_TYPE=encoder-decoder
MODEL_CLSS=(Bart T5)
TASK_NAME=qmsum
PREFIXES=("" "summarize: ")

# SCROLLS TASKS (train/valid/test)
# gov_report (17457/972/973) 10ep ~5500 iters with bs 32
# summ_screen_fd (3673/338/337) 10ep ~1150 iters with bs 32
# qmsum (1257/272/281) 20ep ~800 iters with bs 32
# narrative_qa (55003/5878/10306) 2ep ~3500 iters with bs 32
# qasper (2567/1726/1399) 20ep ~1605 iters with bs 32
# quality (2523/2086/2128) 20ep ~1600 iters with bs 32
# contract_nli (7191/1037/2091) 20ep ~4500 iters with bs 32

# BART 256 512 1024
# LED 1024 4096 16384
# SRC_LEN=256
# BART 1024
# LED 1024
# 
TGT_LEN=1024

METRIC=rouge/geometric_mean
SCHEDULER=linear

for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_CLS=${MODEL_CLSS[i]}
PREFIX=${PREFIXES[i]}
for SRC_LEN in 256 512 1024
do
for LR in 2e-04 1e-04 5e-05 2e-05 1e-05
do
for N in 1 2 3
do
horovodrun --gloo -np $NP python run_finetuning_scrolls.py \
        --task_name $TASK_NAME \
        --model_path ./runs/finetune/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${SRC_LEN}-${TGT_LEN}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:${MODEL_CLS}ForConditionalGeneration \
        --use_generate_on_valid \
        --input_prefix "$PREFIX" \
        --input_seq_len $SRC_LEN \
        --target_seq_len $TGT_LEN \
        --batch_size 4 --gradient_accumulation_steps 4 \
        --iters 1000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"
