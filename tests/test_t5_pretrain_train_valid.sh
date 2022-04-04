#!/usr/bin/env bash
set -e
cd ..

horovodrun --gloo -np $NP python run_t5_pretraining.py \
--data_path ./data/toy_wiki/train/ --valid_data_path ./data/toy_wiki/valid/ \
--model_cfg ./t5configs/t5-micro.json \
--model_path ./tests/runs/test_t5_pretrain \
--batch_size 16 --gradient_accumulation_steps 2 \
--lr 3e-04 --save_best --iters 100 --log_interval 25 --save_interval 50 --valid_interval 50
echo "run_t5_pretraining.py done"
echo "cleaning..."
rm -rf ./tests/runs/test_t5_pretrain
echo "done"
