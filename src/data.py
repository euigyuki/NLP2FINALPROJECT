"""Data loading and preprocessing for SpartQA-YN."""

from dataclasses import dataclass
from typing import Dict, List

from datasets import DatasetDict, load_dataset
from transformers import PreTrainedTokenizerBase

LABELS: List[str] = ["Yes", "No", "DK"]
LABEL2ID: Dict[str, int] = {label: i for i, label in enumerate(LABELS)}
ID2LABEL: Dict[int, str] = {i: label for i, label in enumerate(LABELS)}


@dataclass(frozen=True)
class DataConfig:
    dataset_name: str = "metaeval/spartqa-yn"
    max_length: int = 256
    tokenizer_name: str = "roberta-base"


def load_raw() -> DatasetDict:
    return load_dataset("metaeval/spartqa-yn")


def encode_label(example: Dict) -> Dict:
    return {"label": LABEL2ID[example["answer"]]}


def tokenize_fn(tokenizer: PreTrainedTokenizerBase, max_length: int):
    def _apply(batch: Dict) -> Dict:
        return tokenizer(
            batch["story"],
            batch["question"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )
    return _apply


def prepare(config: DataConfig, tokenizer: PreTrainedTokenizerBase) -> DatasetDict:
    raw = load_raw()
    encoded = raw.map(encode_label)
    tokenized = encoded.map(
        tokenize_fn(tokenizer, config.max_length),
        batched=True,
        remove_columns=["story", "question", "answer"],
    )
    return tokenized
