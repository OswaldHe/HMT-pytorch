from .chatqa_long_sft import load_chatqa_long_sft_dataloaders
from .eda_corpus import load_eda_corpus_dataloaders
from .eda_qa import load_eda_qa_dataloaders
from .fineweb import load_fineweb_dataloaders
from .generic_text import load_generic_text_dataloaders
from .pile_arxiv import load_pile_arxiv_dataloaders
from .pubmed_qa import load_pubmedqa_dataloaders


def generate_dataloaders(args, tokenizer, batch_size, block_size, history_size):
    """Select the appropriate dataloader factory based on the configured task."""
    loader_map = {
        'pubmed_qa': lambda: load_pubmedqa_dataloaders(args, tokenizer, batch_size),
        'nvidia/ChatQA2-Long-SFT-data': lambda: load_chatqa_long_sft_dataloaders(args, tokenizer, batch_size),
        'eda_qa': lambda: load_eda_qa_dataloaders(args, tokenizer, batch_size, block_size),
        'HuggingFaceFW/fineweb': lambda: load_fineweb_dataloaders(args, tokenizer, batch_size, block_size, history_size),
        'suolyer/pile_arxiv': lambda: load_pile_arxiv_dataloaders(args, tokenizer, batch_size, block_size, history_size),
        'eda_corpus': lambda: load_eda_corpus_dataloaders(args, tokenizer, batch_size, block_size, history_size),
    }

    loader_fn = loader_map.get(
        args.task_name,
        lambda: load_generic_text_dataloaders(
            args, tokenizer, batch_size, block_size, history_size
        ),
    )

    return loader_fn()
