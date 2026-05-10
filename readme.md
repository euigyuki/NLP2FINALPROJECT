Readme · MD
복사

# Closed-World Defaulting in Fine-Tuned Transformer Models on SpartQA-YN
 
Code and annotations for an error analysis of RoBERTa-base fine-tuned on the yes/no split of [SpartQA](https://aclanthology.org/2021.naacl-main.364/). Final project for COSI 115b (NLP II) at Brandeis, May 2026.
 
## Summary
 
Fine-tuned RoBERTa-base reaches 80.49% test accuracy on SpartQA-YN but trails Yes and No recall by 17 points on the "Don't Know" (DK) class (0.61 vs. 0.79–0.91). DK-related errors account for 77.6% of all test errors. A close reading of 50 errors stratified within the dominant DK-error categories shows that conventional spatial-reasoning failures (axis confusion, polarity, distractors) are absent from the sample; the errors are uniformly cases of *closed-world defaulting* — the model commits to a truth value when the queried relation is not in the asserted set. The aggregate evidence is most cleanly read as a calibration problem rather than a knowledge problem.
 
## Repository contents
 
```
src/                          fine-tuning and evaluation scripts
taxonomy.md                   the seven-category error taxonomy used for labeling
test_errors_sample50_final.xlsx   50 hand-labeled test errors (the analysis sample)
.gitignore
```
 
## Reproducing the model
 
Requirements: Python 3.10+, PyTorch, HuggingFace `transformers` and `datasets`, a single GPU with 16GB+ VRAM is sufficient.
 
```bash
pip install -r src/requirements.txt
python src/train.py        # fine-tune roberta-base on metaeval/spartqa-yn
python src/evaluate.py     # produces per-class metrics and confusion matrix
```
 
Hyperparameters used: 3 epochs, learning rate 2e-5, batch size 16, AdamW with weight decay 0.01, 500 warmup steps, FP16, max sequence length 256. Best checkpoint selected by validation macro-F1.
 
## The annotated sample
 
`test_errors_sample50_final.xlsx` contains the 50 errors used for the close reading: 18 from `DK_predicted_as_No` (6b) and 32 from `DK_predicted_as_Yes` (6a), sampled with seed 42 and stratified by error type. Each row has the story, question, gold label, predicted label, the assigned taxonomy category, and a free-text subpattern note describing which spatial relation or quantifier was implicated.
 
Single annotator. No inter-annotator reliability check. Per-cell counts in the directional analysis are small (4–14). These limits are spelled out in the paper and constrain how strongly any of the directional findings can be claimed.
 
## Data
 
SpartQA-Auto YN split, accessed via the HuggingFace dataset `metaeval/spartqa-yn`. 23,968 training, 3,860 validation, 3,896 test examples. Training distribution: 53.3% Yes, 18.8% No, 27.9% DK. Majority-class baseline: 50.6%.
 
