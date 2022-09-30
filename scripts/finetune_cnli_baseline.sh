#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

CUBLAS_WORKSPACE_CONFIG=:4096:2
CUDA_LAUNCH_BLOCKING=1

MODEL_NAME=bert-base-cased
MODEL_ATTR=bert
MODEL_CLS=transformers:BertForSequenceClassification
MODEL_TYPE=encoder
TASK_NAME=contract_nli
METRIC=exact_match

INPUT_SEQ_LEN=1024

for N in 1
do


LR=1e-05
SCHEDULER=linear

echo N, MODEL_NAME MODEL_ATTR, MEMORY_SIZE, INPUT_SEQ_LEN
echo $N, $MODEL_NAME $MODEL_ATTR, $MEMORY_SIZE, $INPUT_SEQ_LEN

horovodrun --gloo -np $NP python run_finetuning_scrolls.py \
        --task_name $TASK_NAME \
        --model_path ../runs/finetune/debug/$TASK_NAME/$MODEL_NAME/lr${LR}_${SCHEDULER}_adamw_wd1e-03_${INPUT_SEQ_LEN}_mem${MEMORY_SIZE}/run_$N \
        --from_pretrained $MODEL_NAME \
        --model_type $MODEL_TYPE \
        --model_cls $MODEL_CLS \
        --input_seq_len $INPUT_SEQ_LEN \
        --target_seq_len 2 \
        --save_best \
        --batch_size 8 --gradient_accumulation_steps 2 \
        --iters 5000 \
        --optimizer AdamW  --weight_decay 0.001 \
        --lr $LR --lr_scheduler $SCHEDULER --num_warmup_steps 100 \
        --data_n_workers 0 \
        --log_interval 100 --valid_interval 200 \
        --optimize_metric $METRIC --optimize_mode max \
        --seed $(($N+42)) \
        --clip_grad_value 5.0
done
done
done
done
done
echo "run_bert_pretraining.py done"
echo "done"