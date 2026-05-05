"""Evaluate the fine-tuned model and dump errors for analysis.

Run AFTER train.py finishes:
    python evaluate.py
"""

import json
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report, confusion_matrix
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from data import DataConfig, ID2LABEL, LABELS, load_raw


MODEL_DIR = "../outputs/roberta-base-spartqa-yn"
PRED_DIR = "../outputs"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def predict(model, tokenizer, examples, batch_size=32, max_length=256):
    """Run inference on a list of examples and return predicted label ids."""
    model.eval()
    preds = []
    with torch.no_grad():
        for i in range(0, len(examples), batch_size):
            batch = examples[i : i + batch_size]
            inputs = tokenizer(
                [e["story"] for e in batch],
                [e["question"] for e in batch],
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            ).to(DEVICE)
            logits = model(**inputs).logits
            preds.extend(logits.argmax(dim=-1).cpu().numpy().tolist())
    return preds


def main():
    config = DataConfig()
    raw = load_raw()

    tokenizer = AutoTokenizer.from_pretrained(MODEL_DIR)
    model = AutoModelForSequenceClassification.from_pretrained(MODEL_DIR).to(DEVICE)

    for split in ["validation", "test"]:
        examples = list(raw[split])
        gold = [LABELS.index(e["answer"]) for e in examples]
        pred = predict(model, tokenizer, examples)

        print(f"\n===== {split.upper()} =====")
        print(classification_report(gold, pred, target_names=LABELS, digits=4))
        print("Confusion matrix (rows=true, cols=pred):")
        print(pd.DataFrame(
            confusion_matrix(gold, pred),
            index=[f"true_{l}" for l in LABELS],
            columns=[f"pred_{l}" for l in LABELS],
        ))

        # Dump predictions to CSV for hand error analysis
        df = pd.DataFrame({
            "story": [e["story"] for e in examples],
            "question": [e["question"] for e in examples],
            "gold": [LABELS[g] for g in gold],
            "pred": [LABELS[p] for p in pred],
            "correct": [g == p for g, p in zip(gold, pred)],
        })
        out_path = os.path.join(PRED_DIR, f"{split}_predictions.csv")
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} predictions to {out_path}")

    # Majority-class baseline for comparison
    print("\n===== MAJORITY-CLASS BASELINE (test) =====")
    test_examples = list(raw["test"])
    gold = [LABELS.index(e["answer"]) for e in test_examples]
    majority_pred = [0] * len(gold)  # always predict "Yes" (label_id=0)
    print(classification_report(
        gold, majority_pred, target_names=LABELS, digits=4, zero_division=0
    ))


if __name__ == "__main__":
    main()
