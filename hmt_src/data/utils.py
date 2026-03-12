import random
from functools import partial

from datasets import Dataset
from datasets.iterable_dataset import IterableDataset


def gen_from_iterable_dataset(iterable_ds):
    """Yield elements from an iterable dataset."""
    yield from iterable_ds


def _materialize_if_iterable(ds):
    if ds is None:
        return None
    if isinstance(ds, IterableDataset):
        return Dataset.from_generator(
            partial(gen_from_iterable_dataset, ds), features=ds.features
        )
    return ds


def _take_first_n(ds, n: int, shuffle: bool = False, seed: int = 0):
    if ds is None:
        return None
    # IterableDataset exposes take/skip helpers
    if hasattr(ds, "take"):
        dataset = ds.shuffle(seed=seed) if shuffle and hasattr(ds, "shuffle") else ds
        return dataset.take(n)

    # Regular Dataset fallback
    if not hasattr(ds, "select"):
        raise ValueError("Dataset object must support take or select operations.")

    indices = list(range(len(ds)))
    if shuffle:
        random.Random(seed).shuffle(indices)
    indices = indices[:n]
    return ds.select(indices)


def apply_train_set_split(train_ds, valid_ds, test_ds, args, base_dataset=None):
    """Apply the optional --train_set_split slicing consistently across datasets."""
    if args.train_set_split is None:
        return train_ds, valid_ds, test_ds

    n = int(args.train_set_split)
    if base_dataset is not None:
        train_subset = base_dataset.take(n)
        valid_subset = base_dataset.skip(n).take(n)
        test_subset = base_dataset.skip(2 * n).take(n)
    else:
        train_subset = _take_first_n(
            train_ds, n, shuffle=getattr(args, "shuffle_train", False), seed=args.seed
        )
        valid_subset = _take_first_n(valid_ds, n)
        test_subset = _take_first_n(test_ds, n)

    return (
        _materialize_if_iterable(train_subset),
        _materialize_if_iterable(valid_subset),
        _materialize_if_iterable(test_subset),
    )


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

def dilated_sample(examples, insert_len, period, insert_str):
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