import logging

from hmt_src.data.openroad_qa_preprocess import OpenROAD

logger = logging.getLogger(__name__)


def load_eda_qa_dataloaders(
    args,
    tokenizer,
    batch_size,
    block_size,
    max_len,
    mode,
    neg_sample,
    include_train=True,
):
    if max_len is None:
        max_len = args.bptt_depth * block_size
    train_dataloader, valid_dataloader = OpenROAD(
        tokenizer,
        batch_size=batch_size,
        max_len=max_len,
        mode=mode,
        neg_sample=neg_sample,
        include_train=include_train,
    )
    test_dataloader = valid_dataloader
    logger.info("Prepared OpenROAD QA dataloaders")
    return train_dataloader, valid_dataloader, test_dataloader
