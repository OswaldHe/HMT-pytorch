import pandas as pd
import os

QA_TASKS = ["multi_news"]
METRIC = 'rouge_l'
# NON_QA_TASKS = ["gov_report", "qmsum", "multi_news", "vcsum", "trec", "triviaqa", "samsum", "lsht", 
#             "passage_count", "passage_retrieval_en", "passage_retrieval_zh", "lcc", "repobench-p"]

# ALL_TASKS = QA_TASKS + NON_QA_TASKS

RESULTS_DIR = "benchmark_results"
models = ["smollm-135m", "opt-350m", "openllama_3b_v2"]

def load_results(model_name):
    results = {}
    for task_name in QA_TASKS:
        data_file_name = f"{RESULTS_DIR}/longbench_{model_name}_zeroshot_{task_name}.csv"
        if os.path.exists(data_file_name):
            data = pd.read_csv(data_file_name)
            results[task_name] = data
        else:
            print(f"File {data_file_name} does not exist")
    return results

all_model_results = {}
for model_name in models:
    results = load_results(model_name)
    all_model_results[model_name] = results

for model_name, results in all_model_results.items():
    for task_name, data in results.items():
        if METRIC in data.columns:
            print(f"{model_name} {task_name}: {data[METRIC].mean()}")
        else:
                print(f"{task_name} does not contain {METRIC}")

