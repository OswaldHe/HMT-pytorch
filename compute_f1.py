import pandas as pd



# Load the CSV file
df = pd.read_csv('/home/yingqi/repo/HMT-pytorch/benchmark_results/longbench_openllama_3b_v2_zeroshot_multi_news.csv')

# Extract decoded text and answer columns
decoded_texts = df['decoded_text'].tolist()
answers = df['answer'].tolist()

print(decoded_texts[0])
print(answers[0])