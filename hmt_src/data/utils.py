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
