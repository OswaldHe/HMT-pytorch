from .chatqa_long_sft import load_chatqa_long_sft_dataloaders
from .eda_corpus import load_eda_corpus_dataloaders
from .eda_qa import load_eda_qa_dataloaders
from .fineweb import load_fineweb_dataloaders
from .generic_text import load_generic_text_dataloaders
from .longbench import load_longbench_dataloaders
from .pile_arxiv import load_pile_arxiv_dataloaders
from .pubmed_qa import load_pubmedqa_dataloaders
from .longalign import load_longalign_dataloaders


def generate_dataloaders(args, tokenizer, batch_size, block_size, history_size, include_train=True):
    """Select the appropriate dataloader factory based on the configured task."""
    eda_qa_max_len = args.bptt_depth * block_size
    eda_qa_mode = getattr(args, "eda_qa_mode", "hard")
    eda_qa_neg_sample = getattr(args, "eda_qa_neg_sample", 12)
    eda_corpus_path = args.task_subset
    loader_map = {
        'pubmed_qa': lambda: load_pubmedqa_dataloaders(
            args, tokenizer, batch_size, include_train=include_train
        ),
        'nvidia/ChatQA2-Long-SFT-data': lambda: load_chatqa_long_sft_dataloaders(
            args, tokenizer, batch_size, include_train=include_train
        ),
        'eda_qa': lambda: load_eda_qa_dataloaders(
            args,
            tokenizer,
            batch_size,
            block_size,
            eda_qa_max_len,
            eda_qa_mode,
            eda_qa_neg_sample,
            include_train=include_train,
        ),
        'ioeddk/qmsum': lambda: load_longbench_dataloaders(
            args, tokenizer, batch_size, block_size, history_size, include_train=include_train
        ),
        'HuggingFaceFW/fineweb': lambda: load_fineweb_dataloaders(
            args, tokenizer, batch_size, block_size, history_size, include_train=include_train
        ),
        'suolyer/pile_arxiv': lambda: load_pile_arxiv_dataloaders(
            args, tokenizer, batch_size, block_size, history_size, include_train=include_train
        ),
        'eda_corpus': lambda: load_eda_corpus_dataloaders(
            args,
            tokenizer,
            batch_size,
            block_size,
            history_size,
            eda_corpus_path,
            include_train=include_train,
        ),
        'zai-org/LongAlign-10k': lambda: load_longalign_dataloaders(
            args, tokenizer, batch_size, block_size, history_size, include_train=include_train
        ),
    }

    loader_fn = loader_map.get(
        args.task_name,
        lambda: load_generic_text_dataloaders(
            args, tokenizer, batch_size, block_size, history_size, include_train=include_train
        ),
    )

    return loader_fn()
