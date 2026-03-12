import logging
from typing import Any, Dict, List, Optional, Tuple

from datasets import Dataset, DatasetDict, load_dataset
from datasets.exceptions import NonMatchingSplitsSizesError


logger = logging.getLogger(__name__)

SENTENCE_KEYS = [
    "sentences",
    "sents",
    "sentence",
    "document",
    "text",
    "article",
    "paragraphs",
]

BOUNDARY_KEYS = [
    "boundaries",
    "segment_boundaries",
    "topic_boundaries",
    "labels",
    "label",
    "targets",
]

SEGMENT_ID_KEYS = [
    "segment_id",
    "segment_ids",
    "section_id",
    "section_ids",
    "topic_id",
    "topic_ids",
]

SECTION_KEYS = [
    "sections",
    "section_sentences",
    "section_texts",
]


def _safe_to_str(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.strip()
    return str(value).strip()


def _to_sentence_list(value: Any) -> List[str]:
    if value is None:
        return []

    if isinstance(value, str):
        if "\n" in value:
            return [line.strip() for line in value.split("\n") if line.strip()]
        text = value.strip()
        return [text] if text else []

    if not isinstance(value, list):
        text = _safe_to_str(value)
        return [text] if text else []

    if not value:
        return []

    if all(isinstance(item, str) for item in value):
        return [item.strip() for item in value if item and item.strip()]

    sentences: List[str] = []
    for item in value:
        if isinstance(item, list):
            sentences.extend(_to_sentence_list(item))
        else:
            text = _safe_to_str(item)
            if text:
                sentences.append(text)
    return sentences


def _normalize_boundaries(raw_values: Any, num_sentences: int) -> Optional[List[int]]:
    if raw_values is None:
        return None
    if not isinstance(raw_values, list):
        return None

    values: List[int] = []
    for value in raw_values:
        if isinstance(value, list):
            values.extend([1 if bool(v) else 0 for v in value])
        else:
            values.append(1 if bool(value) else 0)

    if len(values) == num_sentences:
        values[0] = 0
        return values
    if len(values) == num_sentences - 1:
        return [0] + values
    return None


def _boundaries_from_segment_ids(segment_ids: Any, num_sentences: int) -> Optional[List[int]]:
    if not isinstance(segment_ids, list) or len(segment_ids) != num_sentences:
        return None

    boundaries = [0]
    for prev_id, cur_id in zip(segment_ids[:-1], segment_ids[1:]):
        boundaries.append(1 if cur_id != prev_id else 0)
    return boundaries


def _extract_from_sections(value: Any) -> Tuple[List[str], Optional[List[int]]]:
    if not isinstance(value, list):
        return [], None

    sentences: List[str] = []
    boundaries: List[int] = []

    for section in value:
        section_sents = _to_sentence_list(section)
        if not section_sents:
            continue
        for idx, sentence in enumerate(section_sents):
            sentences.append(sentence)
            boundaries.append(1 if idx == 0 else 0)

    if not sentences:
        return [], None

    boundaries[0] = 0
    return sentences, boundaries


def extract_sentences_and_boundaries(example: Dict[str, Any]) -> Tuple[List[str], List[int]]:
    section_sentences: List[str] = []
    section_boundaries: Optional[List[int]] = None
    for key in SECTION_KEYS:
        if key in example:
            section_sentences, section_boundaries = _extract_from_sections(example[key])
            if section_sentences:
                break

    sentences: List[str] = []
    for key in SENTENCE_KEYS:
        if key in example:
            candidate = _to_sentence_list(example[key])
            if len(candidate) >= 2:
                sentences = candidate
                break

    if not sentences and section_sentences:
        sentences = section_sentences

    if len(sentences) < 2:
        return [], []

    boundaries: Optional[List[int]] = None
    for key in BOUNDARY_KEYS:
        if key in example:
            boundaries = _normalize_boundaries(example[key], len(sentences))
            if boundaries is not None:
                break

    if boundaries is None:
        for key in SEGMENT_ID_KEYS:
            if key in example:
                boundaries = _boundaries_from_segment_ids(example[key], len(sentences))
                if boundaries is not None:
                    break

    if boundaries is None and section_boundaries is not None and len(section_boundaries) == len(sentences):
        boundaries = section_boundaries

    if boundaries is None:
        boundaries = [0] * len(sentences)

    boundaries[0] = 0
    return sentences, boundaries


def build_sentence_pair_examples(batch: Dict[str, List[Any]], max_pairs_per_doc: Optional[int]) -> Dict[str, List[Any]]:
    output = {"text_a": [], "text_b": [], "labels": []}
    batch_size = len(next(iter(batch.values()))) if batch else 0

    for idx in range(batch_size):
        example = {key: values[idx] for key, values in batch.items()}
        sentences, boundaries = extract_sentences_and_boundaries(example)
        if len(sentences) < 2 or len(boundaries) != len(sentences):
            continue

        doc_pairs = zip(sentences[:-1], sentences[1:], boundaries[1:])
        for pair_idx, (left, right, boundary) in enumerate(doc_pairs):
            if max_pairs_per_doc is not None and pair_idx >= max_pairs_per_doc:
                break
            if not left or not right:
                continue
            output["text_a"].append(left)
            output["text_b"].append(right)
            output["labels"].append(int(boundary))

    return output


def _get_split(dataset_dict: DatasetDict, split_names: List[str]) -> Optional[Dataset]:
    for name in split_names:
        if name in dataset_dict:
            return dataset_dict[name]
    return None


def resolve_train_valid_test_splits(dataset_dict: DatasetDict, seed: int) -> Tuple[Dataset, Dataset, Dataset]:
    train_ds = _get_split(dataset_dict, ["train"])
    valid_ds = _get_split(dataset_dict, ["validation", "valid", "dev"])
    test_ds = _get_split(dataset_dict, ["test", "eval", "evaluation"])

    if train_ds is None:
        first_split_name = list(dataset_dict.keys())[0]
        logger.warning("No train split detected. Falling back to split '%s'.", first_split_name)
        train_ds = dataset_dict[first_split_name]

    if valid_ds is None and test_ds is None:
        split = train_ds.train_test_split(test_size=0.2, seed=seed)
        train_ds = split["train"]
        valid_test = split["test"].train_test_split(test_size=0.5, seed=seed)
        valid_ds = valid_test["train"]
        test_ds = valid_test["test"]
        logger.warning("Validation/test split not found. Created 80/10/10 split from train.")
    elif valid_ds is None:
        split = train_ds.train_test_split(test_size=0.1, seed=seed)
        train_ds = split["train"]
        valid_ds = split["test"]
        logger.warning("Validation split not found. Created validation split from train.")
    elif test_ds is None:
        split = valid_ds.train_test_split(test_size=0.5, seed=seed)
        valid_ds = split["train"]
        test_ds = split["test"]
        logger.warning("Test split not found. Split validation into validation/test.")

    return train_ds, valid_ds, test_ds


def maybe_limit_split(split: Dataset, max_docs: Optional[int]) -> Dataset:
    if max_docs is None:
        return split
    limit = min(max_docs, len(split))
    return split.select(range(limit))


def prepare_pair_dataset(
    split: Dataset,
    tokenizer,
    split_name: str,
    max_length: int,
    max_docs: Optional[int] = None,
    max_pairs: Optional[int] = None,
    max_pairs_per_doc: Optional[int] = None,
) -> Dataset:
    split = maybe_limit_split(split, max_docs)
    if len(split) == 0:
        raise ValueError(f"Split '{split_name}' is empty.")

    pair_dataset = split.map(
        lambda batch: build_sentence_pair_examples(batch, max_pairs_per_doc=max_pairs_per_doc),
        batched=True,
        remove_columns=split.column_names,
        desc=f"Building sentence pairs for {split_name}",
    )

    if len(pair_dataset) == 0:
        raise ValueError(f"No training pairs were created for split '{split_name}'.")

    if max_pairs is not None:
        pair_dataset = pair_dataset.select(range(min(max_pairs, len(pair_dataset))))

    def tokenize_batch(batch: Dict[str, List[str]]) -> Dict[str, Any]:
        return tokenizer(
            batch["text_a"],
            batch["text_b"],
            truncation=True,
            max_length=max_length,
        )

    tokenized = pair_dataset.map(
        tokenize_batch,
        batched=True,
        desc=f"Tokenizing {split_name}",
    )
    tokenized = tokenized.remove_columns(["text_a", "text_b"])
    return tokenized


def load_segmentation_pair_datasets(args, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    dataset_name = getattr(args, "dataset_name", "TankNee/wiki-727k")
    dataset_config = getattr(args, "dataset_config", None)
    dataset_kwargs = {
        "num_proc": args.num_proc,
        "trust_remote_code": args.trust_remote_code,
    }
    if args.drop_titles:
        if dataset_name == "saeedabc/wiki727k":
            dataset_kwargs["drop_titles"] = True
        else:
            logger.warning("--drop_titles is only supported for saeedabc/wiki727k and will be ignored for %s.", dataset_name)
    if args.cache_dir:
        dataset_kwargs["cache_dir"] = args.cache_dir

    dataset_dict = _load_segmentation_dataset(dataset_name, dataset_kwargs, dataset_config)
    train_ds, valid_ds, test_ds = resolve_train_valid_test_splits(dataset_dict, seed=args.seed)

    train_pairs = prepare_pair_dataset(
        train_ds,
        tokenizer=tokenizer,
        split_name="train",
        max_length=args.max_length,
        max_docs=args.max_train_docs,
        max_pairs=args.max_train_pairs,
        max_pairs_per_doc=args.max_pairs_per_doc,
    )
    valid_pairs = prepare_pair_dataset(
        valid_ds,
        tokenizer=tokenizer,
        split_name="validation",
        max_length=args.max_length,
        max_docs=args.max_valid_docs,
        max_pairs=args.max_valid_pairs,
        max_pairs_per_doc=args.max_pairs_per_doc,
    )
    test_pairs = prepare_pair_dataset(
        test_ds,
        tokenizer=tokenizer,
        split_name="test",
        max_length=args.max_length,
        max_docs=args.max_test_docs,
        max_pairs=args.max_test_pairs,
        max_pairs_per_doc=args.max_pairs_per_doc,
    )

    logger.info("Prepared pair datasets | train=%d valid=%d test=%d", len(train_pairs), len(valid_pairs), len(test_pairs))
    return train_pairs, valid_pairs, test_pairs


def _load_segmentation_dataset(
    dataset_name: str,
    dataset_kwargs: Dict[str, Any],
    dataset_config: Optional[str] = None,
) -> DatasetDict:
    def _load():
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, **dataset_kwargs)
        return load_dataset(dataset_name, **dataset_kwargs)

    try:
        return _load()
    except (UnicodeDecodeError, NonMatchingSplitsSizesError) as err:
        retry_kwargs = dict(dataset_kwargs)
        retry_kwargs["download_mode"] = "force_redownload"
        retry_kwargs["verification_mode"] = "no_checks"
        logger.warning(
            "Initial load failed for %s (%s). Retrying with force_redownload + verification_mode='no_checks'.",
            dataset_name,
            type(err).__name__,
        )
        if dataset_config:
            return load_dataset(dataset_name, dataset_config, **retry_kwargs)
        return load_dataset(dataset_name, **retry_kwargs)


def load_wiki727k_pair_datasets(args, tokenizer) -> Tuple[Dataset, Dataset, Dataset]:
    # Backward-compatible alias.
    return load_segmentation_pair_datasets(args, tokenizer)
