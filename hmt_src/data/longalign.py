import logging
import copy
from typing import List, Dict, Optional
import torch
import datasets
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pad_sequence

from hmt_src.data.utils import apply_train_set_split

logger = logging.getLogger(__name__)


def longalign_truncate_user_content(
    messages: List[Dict],
    tokenizer,
    max_len: int,
    margin: int = 64,
) -> Optional[List[Dict]]:
    """
    Truncate ONLY the user message BEFORE applying the chat template.
    This keeps the chat template structure clean and intact.

    messages: [
        {"role": "user", "content": "..."},
        {"role": "assistant", "content": "..."}
    ]

    Strategy:
    1. Estimate the token length of the assistant answer.
    2. Reserve a margin for system/template overhead.
    3. Give the rest of the token budget to the user content.
    4. Truncate user content from the LEFT (keep the ending containing the question).

    If the max_len is too small to fit answer + margin, return None (caller may drop sample).
    """
    msgs = copy.deepcopy(messages)

    # ---- 1. Estimate answer token length
    answer_text = msgs[1]["content"]
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False
    ).input_ids + [tokenizer.eos_token_id]
    answer_len = len(answer_ids)

    # ---- 2. Compute budget for user content
    budget_for_user = max_len - answer_len - margin
    if budget_for_user <= 0:
        # Not enough space even for answer + template overhead
        return None

    # ---- 3. Tokenize user content and truncate from the left if needed
    user_text = msgs[0]["content"]
    user_ids = tokenizer(user_text, add_special_tokens=False).input_ids

    if len(user_ids) > budget_for_user:
        # Keep only the last N tokens
        user_ids = user_ids[-budget_for_user:]

        # Convert back to text (good enough for Chinese or English)
        user_text_trunc = tokenizer.decode(
            user_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False,
        )
        msgs[0]["content"] = user_text_trunc

    return msgs


def longalign_tokenize_function(
    messages: List[Dict],
    tokenizer,
    max_len: int = None,
):
    """
    Full preprocessing pipeline for a single-turn LongAlign sample.

    Steps:
    1. Truncate user content before applying chat template.
    2. Build prompt with chat template (only containing the user message).
    3. Append assistant tokens.
    4. Create labels: mask prompt part as IGNORE_INDEX, keep answer tokens.
    5. Final safety truncation (if still too long).
    6. Pad to max_len.

    Returns a dict with tensors (input_ids, labels, attention_mask).
    If the sample cannot fit into max_len, returns None.
    """
    IGNORE_INDEX = -100

    assert len(messages) == 2
    assert messages[0]["role"] == "user"
    assert messages[1]["role"] == "assistant"

    if max_len is not None:
        messages = longalign_truncate_user_content(messages, tokenizer, max_len)
        if messages is None:
            return None

    # ---- 2. Build prompt (user-only) using chat template or a fallback
    if getattr(tokenizer, "chat_template", None):
        prompt_text = tokenizer.apply_chat_template(
            messages[:-1],               # only the user message
            tokenize=False,
            add_generation_prompt=True,  # add the assistant header
        )
    else:
        # Minimal fallback template: "User: ...\nAssistant:"
        user_text = messages[0]["content"]
        prompt_text = f"User: {user_text}\nAssistant:"
    prompt_ids = tokenizer(prompt_text, add_special_tokens=False).input_ids

    # ---- 3. Tokenize assistant answer
    answer_text = messages[1]["content"]
    answer_ids = tokenizer(
        answer_text,
        add_special_tokens=False
    ).input_ids + [tokenizer.eos_token_id]

    input_ids = prompt_ids + answer_ids
    labels = [IGNORE_INDEX] * len(prompt_ids) + answer_ids[:]

    # ---- 4. Safety truncation: prioritize removing from the prompt side
    if max_len is not None and len(input_ids) > max_len:
        # keep the *last* max_len tokens
        input_ids = input_ids[-max_len:]
        labels    = labels[-max_len:]   # aligned
    
    attention_mask = [1] * len(input_ids)

    return {
        "input_ids": input_ids,
        "labels": labels,
        "attention_mask": attention_mask,
    }



def load_longalign_dataloaders(
    args, tokenizer, batch_size, block_size, max_len, include_train=True
):
    if args.streaming:
        base_ds = datasets.load_dataset(
            args.task_name, args.task_subset, split="train", streaming=True
        )
        train_ds = base_ds if include_train else None
        valid_ds = test_ds = base_ds
    else:
        train_ds = None
        if include_train:
            train_ds = datasets.load_dataset(
                args.task_name, args.task_subset, split="train[:80%]"
            )
        valid_ds = datasets.load_dataset(
            args.task_name, args.task_subset, split="train[80%:90%]"
        )
        test_ds = datasets.load_dataset(
            args.task_name, args.task_subset, split="train[90%:]"
        )
        base_ds = None

    train_ds, valid_ds, test_ds = apply_train_set_split(
        train_ds,
        valid_ds,
        test_ds,
        args,
        base_dataset=base_ds if include_train else None,
    )

    id_pad_value = tokenizer.pad_token_id if tokenizer.pad_token_id is not None else tokenizer.eos_token_id
    
    def collate_fn(batch):
        input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
        labels = [torch.tensor(b['labels'][::-1]) for b in batch]
        attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]

        input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
        labels = pad_sequence(labels, padding_value=-100).T.flip(1)
        attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)
        
        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask}
        return collated

    reference_ds = train_ds or valid_ds or test_ds
    column_names = reference_ds.column_names

    train_tok = None
    if train_ds is not None:
        train_tok = train_ds.map(
            lambda x: longalign_tokenize_function(x["messages"], tokenizer, max_len),
            batched=False,
            remove_columns=column_names,
            desc='tokenize LongAlign training dataset',
            num_proc=8
        )

    valid_tok = valid_ds.map(
        lambda x: longalign_tokenize_function(x["messages"], tokenizer, max_len),
        batched=False,
        remove_columns=column_names,
        desc='tokenize LongAlign valid dataset',
        num_proc=8
    )

    if args.test_max_context_length is not None:
        test_tok = test_ds.map(
            lambda x: longalign_tokenize_function(x["messages"], tokenizer, args.test_max_context_length),
            batched=False,
            remove_columns=column_names,
            desc='tokenize LongAlign test dataset',
            num_proc=8
        )
    else:
        test_tok = test_ds.map(
            lambda x: longalign_tokenize_function(x["messages"], tokenizer),
            batched=False,
            remove_columns=column_names,
            desc='tokenize LongAlign test dataset',
            num_proc=8
        )

    generator = torch.Generator()
    generator.manual_seed(args.seed)
    train_dataloader = None
    if train_tok is not None:
        train_dataloader = DataLoader(train_tok, batch_size=batch_size, collate_fn=collate_fn,
                                        shuffle=True, drop_last=False, generator=generator, pin_memory=True)

    valid_dataloader = DataLoader(valid_tok, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=False, drop_last=False, pin_memory=True)

    test_dataloader = DataLoader(test_tok, batch_size=batch_size, collate_fn=collate_fn,
                                    shuffle=False, drop_last=False, pin_memory=True)

    logger.info("Prepared LongAlign dataloaders")
    return train_dataloader, valid_dataloader, test_dataloader

    
    
