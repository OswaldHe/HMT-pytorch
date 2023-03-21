#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetun_hyp_enc_dec.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(bert-base-cased roberta-base microsoft/deberta-v3-base)
MODEL_TYPE=encoder
MODEL_CLSS=(Bert Roberta DebertaV2)
TASK_NAME=contract_nli

# SCROLLS TASKS (train/valid/test)
# gov_report (17457/972/973) 10ep ~5500 iters with bs 32
# summ_screen_fd (3673/338/337) 10ep ~1150 iters with bs 32
# qmsum (1257/272/281) 20ep ~800 iters with bs 32
# narrative_qa (55003/5878/10306) 2ep ~3500 iters with bs 32
# qasper (2567/1726/1399) 20ep ~1605 iters with bs 32
# quality (2523/2086/2128) 20ep ~1600 iters with bs 32
# contract_nli (7191/1037/2091) 20ep ~4500 iters with bs 32

TBS=32 # total batch size
BS=8 # batch size per gpu, * grad_acc_steps
WD=1e-03

TGT_LEN=1024
METRIC=exact_match
ITERS=9000
PATIENCE=15

for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_CLS=${MODEL_CLSS[i]}
for SRC_LEN in 512 256
do
for LR in 1e-04 5e-05 2e-05
do
for SCHEDULER in linear constant_with_warmup
do
for N in 1 2 3
do
MODEL_PATH=./runs/finetune/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd${WD}_len${SRC_LEN}_it${ITERS}/run_$N
echo $MODEL_PATH
horovodrun --gloo -np $NP python -m downstream_tasks.scrolls.run_finetuning_scrolls \
        --task_name $TASK_NAME \
        --model_path $MODEL_PATH \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:${MODEL_CLS}ForSequenceClassification \
        --input_seq_len $SRC_LEN \
        --target_seq_len $TGT_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay $WD \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/40)) --valid_interval $(($ITERS/40)) \
        --optimize_metric $METRIC --optimize_mode max --early_stopping_patience $PATIENCE \
        --save_best \
        --seed $(($N+42))
# find $MODEL_PATH | grep .pth | xargs -l rm -rf
done
done
done
done
done
echo "run_finetuning_scrolls.py done"
echo "done"
