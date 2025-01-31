export NCCL_DEBUG=INFO
export TORCH_DISTRIBUTED_DEBUG=INFO

accelerate env 

# Uncomment to disable wandb tracking
export WANDB_MODE=offline

accelerate launch hmt_src/main.py \
    --task_name=eda_qa \
    --use_lora \
    --learning_rate=1e-4 \
    --lr_decay \
    --lr_decay_gamma=0.7 \
    --model_name=meta-llama/Llama-3.1-8B-Instruct \
    --token_file "your_huggingface_token" \
    --training_step=2500 \
    --mem_recall_context=400 \
    --num_sensory=32 \
    --segment_length=512 \
    --bptt_depth=11 \
    --num_seg_save=11 \
    --load_from_ckpt "llama3.1-8b-instruct-eda-qa-finetune" \
    --inference_only \
    --batch_size=1 \
    --eval_step=10 \
    --test_step=10 \
    --shuffle \
    --wandb_run=llama3-8b-openroad-qa \
    --wandb_project=openroad_hmt \
    --wandb_entity=your_wandb_entity \