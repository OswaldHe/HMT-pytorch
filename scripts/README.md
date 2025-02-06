# Usage
This folder contains the HMT training and evaluation scripts. The training folder contains the script to pre-train the HMT extra parameters on the RedPajama dataset and to fine tuning the models on down stream tasks. The evaluation folder contains the scripts to evaluate the models' zero-shot performance and the fine-tuning performance for the down stream tasks. 

***On the top of each script, there are some variables that need to be set before running the script. Mostly they are asking for the path to the HMT-pytorch repository and the path to the checkpoints. In some cases the path to the base folder of checkpoints is required, because the script evaluates all the checkpoints in the folder.***

# Training
## Pre-training Extra Parameters on RedPajama
`train_redpajama.sh` is the script to pre-train the HMT extra parameters on the RedPajama dataset. 

## Fine-tuning on Down Stream Tasks
The scripts in the `train/fine-tuning` folder are the scripts to fine-tuning the models on the down stream tasks. They have the format of `<model>_<task>.sh`, where `<model>` is the name of the model and `<task>` is the name of the task. Since they start fine tuning based on the pre-trained models, the pre-trained checkpoints are required, and you must provide them to the `CHECKPOINT` variable on the top of the script. 
