#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1
#todo: large
MODEL_NAMES=(facebook/bart-base t5-base)
MODEL_TYPE=encoder-decoder
MODEL_CLSS=(Bart T5)

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
horovodrun --gloo -np $NP python run_finetuning_hyp.py \
        --data_path /home/jovyan/data/hyperpartisan_news_detection/train.jsonl \
        --valid_data_path /home/jovyan/data/hyperpartisan_news_detection/dev.jsonl \
        --test_data_path /home/jovyan/data/hyperpartisan_news_detection/test.jsonl \
        --model_path ./runs/finetune/hyperpartisan_news_detection/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${SRC_LEN}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls transformers:${MODEL_CLS}ForConditionalGeneration \
        --use_generate_on_valid \
        --input_seq_len $SRC_LEN \
        --batch_size 8 --gradient_accumulation_steps 2 \
        --save_best --iters 1000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr ${LR} --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --log_interval 50 --valid_interval 50 \
        --optimize_metric f1 --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"
