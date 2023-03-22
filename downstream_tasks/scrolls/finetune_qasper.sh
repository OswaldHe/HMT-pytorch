#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./finetun_hyp_enc_dec.sh
set -e
cd ../..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAMES=(facebook/bart-base t5-base)
MODEL_TYPE=encoder-decoder
MODEL_CLSS=(Bart T5)
TASK_NAME=qasper

# SCROLLS TASKS (train/valid/test)
# gov_report (17457/972/973) 10ep ~5500 iters with bs 32
# summ_screen_fd (3673/338/337) 10ep ~1150 iters with bs 32
# qmsum (1257/272/281) 20ep ~800 iters with bs 32
# narrative_qa (55003/5878/10306) 2ep ~3500 iters with bs 32
# qasper (2567/1726/1399) 20ep ~1605 iters with bs 32
# quality (2523/2086/2128) 20ep ~1600 iters with bs 32
# contract_nli (7191/1037/2091) 20ep ~4500 iters with bs 32

TBS=32 # total batch size
BS=16 # batch size per gpu, * grad_acc_steps
WD=1e-03

TGT_LEN=1024
METRIC=f1
ITERS=3200

for (( i=0; i<${#MODEL_NAMES[@]}; i++ ))
do
MODEL_NAME=${MODEL_NAMES[i]}
MODEL_CLS=${MODEL_CLSS[i]}
for SRC_LEN in 256 512 1024
do
for LR in 2e-04 1e-04 5e-05 2e-05
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
        --model_cls transformers:${MODEL_CLS}ForConditionalGeneration \
        --use_generate_on_valid \
        --show_valid_examples 5 \
        --input_seq_len $SRC_LEN \
        --target_seq_len $TGT_LEN \
        --batch_size $BS --gradient_accumulation_steps $(($TBS/($BS*$NP))) \
        --iters $ITERS \
        --optimizer AdamW  --weight_decay $WD \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps $(($ITERS/10)) \
        --data_n_workers 2 \
        --log_interval $(($ITERS/40)) --valid_interval $(($ITERS/40)) \
        --optimize_metric $METRIC --optimize_mode max --early_stopping_patience 15 \
        --save_best \
        --seed $(($N+42))
done
done
done
done
done
echo "run_finetuning_scrolls.py done"
echo "done"
