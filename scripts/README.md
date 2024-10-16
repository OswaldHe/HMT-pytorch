# Usage
This folder contains the HMT training and evaluation scripts. The training folder contains the script to pre-train the HMT extra parameters on the RedPajama dataset and to fine tuning the models on down stream tasks. The evaluation folder contains the scripts to evaluate the models' zero-shot performance and the fine-tuning performance for the down stream tasks. 

***On the top of each script, there are some variables that need to be set before running the script. Mostly they are asking for the path to the HMT-pytorch repository and the path to the checkpoints. In some cases the path to the base folder of checkpoints is required, because the script evaluates all the checkpoints in the folder.***

# Training
## Pre-training Extra Parameters on RedPajama
`train_redpajama.sh` is the script to pre-train the HMT extra parameters on the RedPajama dataset. 

## Fine-tuning on Down Stream Tasks
The scripts in the `train/fine-tuning` folder are the scripts to fine-tuning the models on the down stream tasks. They have the format of `<model>_<task>.sh`, where `<model>` is the name of the model and `<task>` is the name of the task. Since they start fine tuning based on the pre-trained models, the pre-trained checkpoints are required, and you must provide them to the `CHECKPOINT` variable on the top of the script. 

# Evaluation
The scripts in the `eval` folder are the scripts to evaluate the models' performance.
## Evaluating the Pre-training Checkpoints
`pre-training`: each script in this folder evaluates a group of checkpoints that is generated during the HMT parameter pre-training process, thus the `CHECKPOINT_BASE` variable is required to be set to the path of the folder that contains the pre-trained checkpoints. Additionally, a list named `checkpoints` could be found explicting defining the specific checkpoints in the base to evaluate. 

## Evaluating the Fine-tuned Checkpoints
2. `qa`: each script in this folder contains two calls to the evaluation, the first is the evaluation based on the non-fine-tuned model checkpoint (zero-shot), and the second is the evaluation based on the fine-tuned model checkpoint. Thus, in addition to `HMT_PYTORCH_PATH`, two more variables are required to be set, `ZEROSHOT_CHECKPOINT` and `FINETUNED_CHECKPOINT`, to the path of the zeroshot and finetuned checkpoints, respectively. 
