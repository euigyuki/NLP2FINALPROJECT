"""Fine-tune RoBERTa-base on SpartQA-YN.

Run: python train.py
"""

import os
import numpy as np
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    DataCollatorWithPadding,
    Trainer,
    TrainingArguments,
)
from sklearn.metrics import accuracy_score, f1_score

from data import DataConfig, ID2LABEL, LABEL2ID, LABELS, prepare


OUTPUT_DIR = "../outputs/roberta-base-spartqa-yn"
SEED = 42


def compute_metrics(eval_pred):
    """Compute accuracy and macro-F1."""
    logits, labels = eval_pred
    preds = np.argmax(logits, axis=-1)
    return {
        "accuracy": accuracy_score(labels, preds),
        "macro_f1": f1_score(labels, preds, average="macro"),
    }


def main():
    config = DataConfig()
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer_name)
    datasets = prepare(config, tokenizer)

    model = AutoModelForSequenceClassification.from_pretrained(
        config.tokenizer_name,
        num_labels=len(LABELS),
        id2label=ID2LABEL,
        label2id=LABEL2ID,
    )

    training_args = TrainingArguments(
        output_dir=OUTPUT_DIR,
        num_train_epochs=3,
        per_device_train_batch_size=16,
        per_device_eval_batch_size=32,
        learning_rate=2e-5,
        weight_decay=0.01,
        warmup_steps=500,
        eval_strategy="epoch",
        save_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="macro_f1",
        greater_is_better=True,
        save_total_limit=2,
        logging_steps=100,
        seed=SEED,
        report_to="none",
        fp16=True,
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=datasets["train"],
        eval_dataset=datasets["validation"],
        processing_class=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer),
        compute_metrics=compute_metrics,
    )

    trainer.train()
    test_metrics = trainer.evaluate(datasets["test"], metric_key_prefix="test")
    print("Test metrics:", test_metrics)

    trainer.save_model(OUTPUT_DIR)
    tokenizer.save_pretrained(OUTPUT_DIR)


if __name__ == "__main__":
    main()
