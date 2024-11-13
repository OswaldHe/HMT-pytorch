from typing import List
from itertools import chain

def interleaving_sample(examples, context_len):
    interleave = {}
    for k in examples.keys():
        interleave[k] = []
        for i in range(0, len(examples[k]), 2):
            first = examples[k][i]
            if i+1 >= len(examples[k]):
                interleave[k].append(first)
                break
            second = examples[k][i+1]

            res = []
            j = 0
            while j < len(first) and j < len(second):
                res.extend(first[j:j+context_len]) 
                res.extend(second[j:j+context_len])
                j+=context_len
            if j < len(first):
                res.extend(first[j:])
            if j < len(second):
                res.extend(second[j:])
            interleave[k].append(res)

    return interleave

def dilated_sample(examples, insert_len, period, insert_str, tokenizer):
    res = {}
    tok = tokenizer(insert_str)['input_ids'][1]
    attn_mask = tokenizer(insert_str)['attention_mask'][1]
    for k in examples.keys():
        res[k] = []
        for sample in examples[k]:
            ans = []
            i = 0
            while i < len(sample):
                ans.extend(sample[i:i+period])
                if k == 'input_ids':
                    ans.extend(insert_len * [tok]) #padding token [double space]
                else:
                    ans.extend(insert_len * [attn_mask])
                i+=period
            res[k].append(ans)

    return res

def group_texts(examples, block_size, history_size=None, with_answer=False, **kwargs):
    if kwargs.get('interleave_dataset', False):
        interleave_len = kwargs.get('interleave_len', None)
        assert interleave_len is not None, "Interleave Length must be provided if interleave dataset. "
        examples = interleaving_sample(examples, interleave_len)
    elif kwargs.get('dilate_dataset', False):
        dilate_len = kwargs.get('dilate_len', None)
        dilate_str = kwargs.get('dilate_str', None)
        assert dilate_len is not None and dilate_str is not None, "Dilate Length and Dilate String must be provided if dilate dataset. "
        tokenizer = kwargs.get('tokenizer', None)
        assert tokenizer is not None, "Tokenizer must be provided if dilate dataset. "
        examples = dilated_sample(examples, dilate_len, dilate_len, dilate_str, tokenizer=tokenizer)
    concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    if history_size is None:
        result = {
            k: [t[i : i + block_size] for i in range(0, total_length, block_size)]
            for k, t in concatenated_examples.items()
        }
    else:
        result = {
            k: [t[max({0, i - history_size}) : i + block_size] for i in range(0, total_length, history_size)]
            for k, t in concatenated_examples.items()
        }
    result["labels"] = result["input_ids"].copy()
    if with_answer: result["answer"] = result["answer"].copy()
    return result

def group_texts_qa(examples, with_answer=False, with_text=False):
    result = {k: v for k, v in examples.items()}
    result["labels"] = result["input_ids"].copy()
    result["mask_size"] = result["mask_size"].copy()  # qa tasks should not be grouped
    if with_answer: result["answer"] = result["answer"].copy()
    if with_text: result["text"] = result["text"].copy()
    return result

# Function to load or create grouped dataset
def group_dataset(dataset, split, history_size, block_size, levels: List[int]=None, is_qa_task=False, with_answer=False, **kwargs):
    if levels is not None:
        grouped_datasets = []
        for i,n_segs in enumerate(levels):
            num_data_per_level = len(dataset) // len(levels)
            data_subset = dataset.select(range(i * num_data_per_level, (i + 1) * num_data_per_level))
            curr_n_segments = n_segs
            curr_history_size = (curr_n_segments - 1) * block_size

            grouped_dataset = data_subset.map(
                lambda x: group_texts(x, curr_history_size, block_size, with_answer=with_answer),
                batched=True,
                desc=f"Grouping {split} in chunks of {block_size}, {n_segs} segments, " + (f" and history {history_size}" if split == 'train' else ""),
                num_proc=8
            )
            grouped_datasets.append(grouped_dataset)
        return grouped_datasets
    else:
        if is_qa_task:
            grouped_dataset = dataset.map(
                lambda x: group_texts_qa(x, with_answer=with_answer, with_text=kwargs.get('with_text', False)),
                batched=True,
                desc=f"Grouping {split} in chunks of {block_size}" + (f" and no history size for this QA task" if split == 'train' else ""),
                remove_columns=None,
                num_proc=8
            )
        else:
            grouped_dataset = dataset.map(
                lambda x: group_texts(x, history_size, block_size, with_answer=with_answer, with_text=kwargs.get('with_text', False)),
                batched=True,
                desc=f"Grouping {split} in chunks of {block_size}" + (f" and history {history_size}" if split == 'train' else ""),
                num_proc=16
            )
        return grouped_dataset
