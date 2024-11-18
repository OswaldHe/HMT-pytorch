import json

with open('/home/yingqi/repo/HMT-pytorch/nihs/train_sets/llama_3.2_1b/qa1/0k.json', 'r') as file:
    data = json.load(file)

print(data[0])
print(len(data))