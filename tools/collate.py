import torch
from torch.nn.utils.rnn import pad_sequence

def collate_fn(batch, id_pad_value, is_qa_task, block_size, batch_size):
    input_ids = [torch.tensor(b['input_ids'][::-1]) for b in batch]
    labels = [torch.tensor(b['labels'][::-1]) for b in batch]
    attention_mask = [torch.tensor(b['attention_mask'][::-1]) for b in batch]
    input_ids = pad_sequence(input_ids, padding_value=id_pad_value).T.flip(1)
    labels = pad_sequence(labels, padding_value=-100).T.flip(1)
    attention_mask = pad_sequence(attention_mask, padding_value=0).T.flip(1)

    if is_qa_task:
        assert batch_size == 1, "QA Tasks currently only support batch_size = 1 and batches can't be collated"

    if is_qa_task:
        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask,
                    'mask_size': torch.tensor(batch[0]['mask_size']) }
    else:
        collated = {'input_ids': input_ids,
                    'labels': labels,
                    'attention_mask': attention_mask}

    if input_ids.shape[1] != block_size:
        labels_mask = torch.ones_like(input_ids, dtype=bool)
        labels_mask[:, :-block_size] = False
        collated['labels_mask'] = labels_mask

    return collated