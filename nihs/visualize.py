import seaborn as sns
import matplotlib
import matplotlib.pylab as plt
from matplotlib.colors import LinearSegmentedColormap
import os

import pandas as pd
import numpy as np

from babilong.metrics import compare_answers, TASK_LABELS
import argparse

def visualize_accuracy(results_folder, model_name, prompt_name, tasks, lengths):
    accuracy = np.zeros((len(tasks), len(lengths)))
    for j, task in enumerate(tasks):
        for i, ctx_length in enumerate(lengths):
            fname = f'./{results_folder}/{model_name}/{task}_{ctx_length}_{prompt_name}.csv'
            if not os.path.isfile(fname):
                print(f'No such file: {fname}')
                continue

            df = pd.read_csv(fname)

            if df['output'].dtype != object:
                df['output'] = df['output'].astype(str)
            df['output'] = df['output'].fillna('')


            df['correct'] = df.apply(lambda row: compare_answers(row['target'], row['output'],
                                                                row['question'], TASK_LABELS[task]
                                                                ), axis=1)
            score = df['correct'].sum()
            accuracy[j, i] = 100 * score / len(df) if len(df) > 0 else 0

    # Set large font sizes for better visibility in the PDF
    matplotlib.rc('font', size=14)

    # Create a colormap for the heatmap
    cmap = LinearSegmentedColormap.from_list('ryg', ["red", "yellow", "green"], N=256)

    # Create the heatmap
    fig, ax = plt.subplots(figsize=(5, 3.5))  # Adjust the size as necessary
    sns.heatmap(accuracy, cmap=cmap, vmin=0, vmax=100, annot=True, fmt=".0f",
                linewidths=.5, xticklabels=lengths, yticklabels=tasks, ax=ax)
    ax.set_title(f'Performance of {model_name} \n on BABILong \n')
    ax.set_xlabel('Context size')
    ax.set_ylabel('Tasks')

    # Save the figure to a PDF
    plt.savefig('all_tasks_performance.pdf', bbox_inches='tight')
    # plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--results_folder', type=str, default='./babilong_evals')
    parser.add_argument('--model_name', type=str, default='meta-llama/Llama-3.2-1B-Instruct')
    parser.add_argument('--prompt_name', type=str, default='instruction_yes_examples_yes_post_prompt_yes_chat_template_yes')
    parser.add_argument('--tasks', type=str, default='qa1,qa2')
    parser.add_argument('--lengths', type=str, default='0k,1k')

    args = parser.parse_args()

    tasks = args.tasks.split(',')
    lengths = args.lengths.split(',')

    visualize_accuracy(args.results_folder, args.model_name, args.prompt_name, tasks, lengths)
