import json
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModelForSequenceClassification
from transformers.modeling_outputs import SequenceClassifierOutput


class LSTMSegmenter(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        pad_token_id: int,
        embedding_dim: int = 256,
        hidden_dim: int = 256,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.pad_token_id = pad_token_id
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=pad_token_id)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_dim * 2, 2)

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        token_type_ids: Optional[torch.Tensor] = None,
    ) -> SequenceClassifierOutput:
        del token_type_ids
        embeddings = self.embedding(input_ids)

        if attention_mask is None:
            lengths = input_ids.ne(self.pad_token_id).sum(dim=1).clamp_min(1).cpu()
        else:
            lengths = attention_mask.sum(dim=1).clamp_min(1).cpu()

        packed = nn.utils.rnn.pack_padded_sequence(
            embeddings,
            lengths,
            batch_first=True,
            enforce_sorted=False,
        )
        _, (hidden, _) = self.lstm(packed)
        pooled = torch.cat([hidden[-2], hidden[-1]], dim=-1)
        pooled = self.dropout(pooled)
        logits = self.classifier(pooled)

        loss = None
        if labels is not None:
            loss = F.cross_entropy(logits, labels)
        return SequenceClassifierOutput(loss=loss, logits=logits)


def create_model(args, tokenizer) -> Tuple[nn.Module, Dict]:
    if args.model_type == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            num_labels=2,
            cache_dir=args.cache_dir,
        )
        return model, {}

    pad_token_id = tokenizer.pad_token_id
    if pad_token_id is None:
        raise ValueError("Tokenizer must have pad token for LSTM model.")

    lstm_config = {
        "vocab_size": len(tokenizer),
        "pad_token_id": pad_token_id,
        "embedding_dim": args.lstm_embedding_dim,
        "hidden_dim": args.lstm_hidden_dim,
        "num_layers": args.lstm_num_layers,
        "dropout": args.lstm_dropout,
    }
    model = LSTMSegmenter(**lstm_config)
    return model, {"lstm_config": lstm_config}


def save_checkpoint(model, tokenizer, output_dir: str, model_type: str, metadata: Optional[Dict] = None):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    tokenizer.save_pretrained(output_path.as_posix())

    if model_type == "bert":
        model.save_pretrained(output_path.as_posix())
    else:
        torch.save(model.state_dict(), output_path / "pytorch_model.bin")
        lstm_config = (metadata or {}).get("lstm_config", {})
        with open(output_path / "lstm_config.json", "w", encoding="utf-8") as fout:
            json.dump(lstm_config, fout, indent=2)

    if metadata:
        with open(output_path / "metadata.json", "w", encoding="utf-8") as fout:
            json.dump(metadata, fout, indent=2)


def load_checkpoint(checkpoint_dir: str, model_type: str, args, tokenizer, device: torch.device) -> nn.Module:
    checkpoint_path = Path(checkpoint_dir)
    if model_type == "bert":
        model = AutoModelForSequenceClassification.from_pretrained(checkpoint_path.as_posix())
        return model.to(device)

    config_path = checkpoint_path / "lstm_config.json"
    weights_path = checkpoint_path / "pytorch_model.bin"
    if not config_path.exists() or not weights_path.exists():
        raise FileNotFoundError(f"Missing LSTM checkpoint files in {checkpoint_dir}.")

    with open(config_path, "r", encoding="utf-8") as fin:
        lstm_config = json.load(fin)

    if "vocab_size" not in lstm_config:
        lstm_config["vocab_size"] = len(tokenizer)
    if "pad_token_id" not in lstm_config:
        if tokenizer.pad_token_id is None:
            raise ValueError("Tokenizer must define pad token id for loading LSTM checkpoint.")
        lstm_config["pad_token_id"] = tokenizer.pad_token_id

    model = LSTMSegmenter(**lstm_config)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    return model.to(device)

