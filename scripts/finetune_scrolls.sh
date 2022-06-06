#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=t5-base
MODEL_TYPE=encoder-decoder
MODEL_CLS=transformers:T5ForConditionalGeneration
TASK_NAME=contract_nli

for LR in 3e-04 1e-04 5e-05
do
for SCHEDULER in constant_with_warmup cosine
do
for N in 1 2
do
horovodrun --gloo -np $NP python run_finetuning_scrolls.py \
        --task_name $TASK_NAME \
        --model_path ./runs/finetune/$MODEL_NAME/$TASK_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --use_generate_on_valid \
        --input_seq_len 512 \
        --target_seq_len 4 \
        --batch_size 4 --gradient_accumulation_steps 2 \
        --save_best --iters 5000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 2 \
        --log_interval 100 --valid_interval 100 \
        --optimize_metric exact_match --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
echo "run_bert_pretraining.py done"
echo "done"
