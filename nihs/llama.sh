#!/bin/bash

TASKS="qa1,qa2"
LENGTHS="0k,1k,2k,4k,8k"
MODEL="meta-llama/Llama-3.2-1B-Instruct"

python nihs_eval.py --config=configs/llama-3.2-1b-instruct.yaml --results_folder=./results --tasks=$TASKS --length=$LENGTHS
python visualize.py --results_folder=./results --model_name=$MODEL --prompt_name='instruction_yes_examples_yes_post_prompt_yes_chat_template_yes' --tasks=$TASKS --lengths=$LENGTHS --plot_name='llama-3.2-1b-instruct.pdf'
