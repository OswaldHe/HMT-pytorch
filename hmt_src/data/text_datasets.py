from itertools import chain

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader


def create_text_dataloaders(
    train_ds,
    valid_ds,
    test_ds,
    tokenizer,
    batch_size,
    args,
    block_size,
    history_size,
    group_test=True,
):
    """Tokenize datasets and build autoregressive dataloaders."""
    reference_ds = valid_ds or train_ds or test_ds
    if reference_ds is None:
        raise ValueError("At least one dataset split must be provided.")

    column_names = reference_ds.column_names
    text_column_name = "text" if "text" in column_names else column_names[0]

    def group_texts(examples, current_block_size, history_size=None):
        concatenated_examples = {k: list(chain(*examples[k])) for k in examples.keys()}
        total_length = len(concatenated_examples[list(examples.keys())[0]])

        if history_size is None:
            result = {
                k: [
                    t[i : i + current_block_size] for i in range(0, total_length, current_block_size)
                ]
                for k, t in concatenated_examples.items()
            }
        else:
            result = {
                k: [
                    t[i : i + history_size] for i in range(0, total_length, current_block_size)
                ]
                for k, t in concatenated_examples.items()
            }
        result["labels"] = result["input_ids"].copy()
        return result

    def tokenize_function(examples):
        return tokenizer(examples[text_column_name])

    if train_ds is not None:
        train_ds_tok = train_ds.map(
            tokenize_function,
            batched=True,
            batch_size=4,
            remove_columns=column_names,
            desc="Running tokenizer on training dataset",
            num_proc=8,
        )
    else:
        train_ds_tok = None

    valid_ds_tok = valid_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on valid dataset",
        num_proc=2,
    )

    test_ds_tok = test_ds.map(
        tokenize_function,
        batched=True,
        remove_columns=column_names,
        desc="Running tokenizer on test dataset",
        num_proc=2,
    )

    if train_ds_tok is not None:
        train_dataset = train_ds_tok.map(
            lambda x: group_texts(x, block_size, history_size),
            batched=True,
            desc=f"Grouping train in chunks of {block_size} and history {history_size}",
        )
    else:
        train_dataset = None

    valid_dataset = valid_ds_tok.map(
        lambda x: group_texts(x, block_size, history_size),
        batched=True,
        desc=f"Grouping valid in chunks of {block_size} and history {history_size}",
    )

    if group_test:
        test_dataset = test_ds_tok.map(
            lambda x: group_texts(x, block_size, args.test_max_context_length),
            batched=True,
            desc=f"Grouping test in chunks of {block_size} and set max context length {args.test_max_context_length}",
        )
    else:
        test_dataset = test_ds_tok

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id

    def collate_fn(batch):
        input_ids = [torch.tensor(b["input_ids"][::-1]) for b in batch]
        labels = [torch.tensor(b["labels"][::-1]) for b in batch]
        attention_mask = [torch.tensor(b["attention_mask"][::-1]) for b in batch]
        input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
        labels = pad_sequence(labels, padding_value=-100).T.flip(1)
        attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

        collated = {
            "input_ids": input_ids,
            "labels": labels,
            "attention_mask": attention_mask,
        }

        if input_ids.shape[1] != block_size:
            labels_mask = torch.ones_like(input_ids, dtype=bool)
            labels_mask[:, :-block_size] = False
            collated["labels_mask"] = labels_mask

        return collated

    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    if train_dataset is not None:
        train_rnd_generator = torch.Generator()
        train_rnd_generator.manual_seed(args.seed)
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            collate_fn=collate_fn,
            shuffle=args.shuffle,
            drop_last=False,
            generator=train_rnd_generator,
            pin_memory=True,
        )
    else:
        train_dataloader = valid_dataloader

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        shuffle=False,
        drop_last=True,
        pin_memory=True,
    )

    return train_dataloader, valid_dataloader, test_dataloader
