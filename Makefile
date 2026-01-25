DATASET ?= longalign
VARIANT ?= base

MODEL ?= meta-llama/Llama-3.2-1B-Instruct
CKPT_ROOT ?= /work1/jasoncong/jameszhang23/HMT-OPT-pytorch/checkpoints
PROMPT ?= /home1/jameszhang23/HMT-pytorch/hmt_src/prompt.txt

LR ?= 1e-4
LR_DECAY_GAMMA ?= 0.6
NUM_EPOCHS ?= 1
TRAIN_STEP ?= 500
EVAL_STEP ?= 100
TEST_STEP ?= 100

TASK_ARGS :=
MODEL_ARGS :=
TRAIN_ARGS :=
EVAL_ARGS :=
GEN_ARGS :=
SAVE_CKPT :=

BATCH ?= 2
BPTT_DEP ?= 8

SEG_LEN ?= 1024
MEM_REC_SIZE ?= 1
MEM_REC_WIN ?= 1024
NUM_SENSORY ?= 32
TEST_MAX_LEN ?= 65536
MAX_NEW_TOKEN ?= 16384



ifeq ($(DATASET),wikitext)
	TASK_ARGS := --task_subset=wikitext-2-v1
endif

ifeq ($(DATASET),fineweb)
	TASK_ARGS := --task_name=HuggingFaceFW/fineweb --task_subset=sample-10BT
endif

ifeq ($(DATASET),longalign)
	TASK_ARGS := --task_name=zai-org/LongAlign-10k
endif

ifeq ($(VARIANT),base)
	MODEL_ARGS := --recurrent_type=memory_only --baseline_only --segment_length=$(SEG_LEN) 
	TRAIN_ARGS := --batch_size=1 --test_max_context_length=$(TEST_MAX_LEN) --learning_rate=$(LR) --lr_decay --lr_decay_gamma=$(LR_DECAY_GAMMA) --num_epochs=$(NUM_EPOCHS)
	EVAL_ARGS := --batch_size=$(BATCH) --test_max_context_length=$(TEST_MAX_LEN)
	GEN_ARGS := --chat --generate_prompt="$(PROMPT)" --max_new_tokens=$(MAX_NEW_TOKEN)
	SAVE_CKPT := $(CKPT_ROOT)/$(MODEL)-$(VARIANT)-$(DATASET)-seg_$(SEG_LEN)/
endif

ifneq (,$(filter $(VARIANT),mem_recall summary_memory))
	MODEL_ARGS := --recurrent_type=$(VARIANT) --segment_length=$(SEG_LEN) --mem_recall_size=$(MEM_REC_SIZE) --mem_window_size=$(MEM_REC_WIN) --num_sensory=$(NUM_SENSORY)
	TRAIN_ARGS := --batch_size=$(BATCH) --bptt_depth=$(BPTT_DEP) --test_max_context_length=$(TEST_MAX_LEN) --learning_rate=$(LR) --lr_decay --lr_decay_gamma=$(LR_DECAY_GAMMA) --num_epochs=$(NUM_EPOCHS)
	EVAL_ARGS := --batch_size=$(BATCH) --bptt_depth=$(BPTT_DEP) --test_max_context_length=$(TEST_MAX_LEN)
	GEN_ARGS := --chat --generate_prompt="$(PROMPT)" --max_new_tokens=$(MAX_NEW_TOKEN)
	SAVE_CKPT := $(CKPT_ROOT)/$(MODEL)-$(VARIANT)-$(DATASET)-seg_$(SEG_LEN)-bptt_$(BPTT_DEP)-recall_$(MEM_REC_SIZE)/
endif



.PHONY: train eval generate

train:
	accelerate launch hmt_src/main.py \
		$(TASK_ARGS) \
		--model_name=$(MODEL) \
		$(MODEL_ARGS) \
		--training_step=$(TRAIN_STEP) \
		--eval_step=$(EVAL_STEP) \
		--test_step=$(TEST_STEP) \
		$(TRAIN_ARGS) \
		--save_ckpt="$(SAVE_CKPT)"

eval:
	accelerate launch hmt_src/main.py \
		$(TASK_ARGS) \
		--model_name=$(MODEL) \
		$(MODEL_ARGS) \
		--load_from_ckpt="$(SAVE_CKPT)" \
		--eval_step=$(EVAL_STEP) \
		--test_step=$(TEST_STEP) \
		$(EVAL_ARGS) \
		--evaluate_only

generate:
	accelerate launch hmt_src/main.py \
		--model_name=$(MODEL) \
		$(MODEL_ARGS) \
		--load_from_ckpt="$(SAVE_CKPT)" \
		$(GEN_ARGS) \
		--generate_only
