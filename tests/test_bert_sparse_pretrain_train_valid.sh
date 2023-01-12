#!/usr/bin/env bash
# CUDA_VISIBLE_DEVICES=1,2 NP=2 ./test_bert_sparse_pretrain_train_valid.sh
set -e
cd ..

horovodrun --gloo -np $NP python run_bert_pretraining.py \
--data_path ./data/toy_wiki/train_text_sentence --valid_data_path ./data/toy_wiki/valid_text_sentence \
--tokenizer ./vocabs/bert-base-uncased/ --model_cfg ./bert_configs/bert_base_uncased-4L_preln_sparse_rope.json \
--model_cls modeling_bert:BertForPreTraining \
--model_path ./tests/runs/test_bert_pretrain \
--batch_size 16 --gradient_accumulation_steps 2 \
--data_n_epochs 1 --save_best --iters 100 --log_interval 25 --save_interval 25 --valid_interval 50 \
--lr 3e-04 --lr_scheduler constant_with_warmup --num_warmup_steps 50 \
--fp16 --apex_opt_lvl O2
echo "run_bert_pretraining.py done"
echo "cleaning..."
rm -rf ./tests/runs/test_bert_pretrain
echo "done"
