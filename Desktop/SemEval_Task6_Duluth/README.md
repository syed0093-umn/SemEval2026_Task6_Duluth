# Duluth at SemEval 2026 Task 6 - CLARITY

Political Question Evasion Detection: classifying Q&A responses from political interviews by their level of clarity.

## Task Description

This repository contains our system for [SemEval 2026 Task 6 (CLARITY)](https://ailsntua.github.io/SemEval2025-Task6/), which focuses on detecting question evasion in political interviews. The task uses the **QEvasion** dataset and involves two subtasks:

- **Subtask 1 (3-class)**: Classify responses as *Ambivalent*, *Clear Reply*, or *Clear Non-Reply*
- **Subtask 2 (9-class)**: Fine-grained evasion classification into 9 categories (Explicit, Implicit, Dodging, General, Deflection, Partial/half-answer, Declining to answer, Claims ignorance, Clarification)

**Evaluation metric**: Macro F1-Score

## Results

### Comparison with Baseline Models

| Model | Test F1 |
|-------|---------|
| Majority class (Ambivalent) | 0.2700 |
| TF-IDF + Logistic Regression | 0.4476 |
| SVM (linear) | 0.4270 |
| Random Forest | 0.4256 |
| DistilBERT | 0.5158 |
| BERT-base | 0.5628 |
| **DeBERTa-V3-base (Gemini)** | **0.66** |

### Performance of DeBERTa-V3-base Variants

| Configuration | Test F1 | Eval F1 |
|---------------|---------|---------|
| Original only | 0.64 | 0.69 |
| Claude-augmented | 0.65 | 0.74 |
| Gemini-augmented (submitted) | **0.66** | **0.76** |
| Top-System (TeleAI) | — | **0.89** |

## Repository Structure

```
.
├── training/                                    # Model training scripts
│   ├── train_deberta_improved.py                # DeBERTa-v3-base: LLRD + cosine scheduling (submission_final)
│   ├── train_deberta.py                         # Base DeBERTa-v3 training
│   ├── train_deberta_large.py                   # DeBERTa-v3-LARGE variant
│   ├── train_deberta_nli_large.py               # DeBERTa-v3-large NLI (best dev F1: 0.6678)
│   ├── train_political_debate.py                # Political DEBATE base model
│   ├── train_deberta_augmented.py               # Training with EDA/back-translation augmented data
│   ├── train_deberta_large_augmented.py         # DeBERTa-large with augmented data
│   ├── train_deberta_focal_features.py          # Focal loss + boolean features
│   ├── train_deberta_large_evasion.py           # Subtask 2: 9-class (DeBERTa-LARGE)
│   ├── train_deberta_large_evasion_augmented.py # Subtask 2 with augmented data
│   ├── train_deberta_evasion_augmented.py       # Subtask 2 evasion + augmentation
│   ├── train_evasion_corrected.py               # Subtask 2 with corrected label mapping
│   ├── train_multitask_deberta.py               # Multi-task learning (3-class + 9-class)
│   ├── train_multitask_large_evasion.py         # Multi-task with DeBERTa-LARGE
│   ├── train_modernbert_clarity.py              # ModernBERT alternative
│   ├── train_annotator_aware.py                 # Annotator-aware features
│   └── train_hierarchical_stage2.py             # Hierarchical two-stage classification
├── evaluation/                                  # Prediction and evaluation
│   ├── predict_eval_nli_large.py                # NLI large model evaluation
│   ├── predict_eval_deberta_augmented.py        # Augmented DeBERTa evaluation
│   ├── predict_eval_evasion.py                  # Subtask 2 evaluation
│   ├── predict_eval_evasion_augmented.py        # Subtask 2 augmented evaluation
│   ├── predict_eval_focal_features.py           # Focal loss evaluation
│   ├── predict_majority_vote.py                 # Annotator majority vote predictor
│   ├── predict_hierarchical_evasion.py          # Hierarchical evasion predictor
│   ├── create_ensemble.py                       # Ensemble methods (Subtask 1)
│   ├── create_ensemble_evasion.py               # Ensemble methods (Subtask 2)
│   └── ensemble_models.py                       # Traditional ML ensembles (RF, XGBoost)
├── baselines/                                   # Baseline models
│   ├── baseline_tfidf.py                        # TF-IDF + Logistic Regression
│   ├── baseline_svm.py                          # SVM variants
│   ├── baseline_svm_fixed.py                    # SVM with fixed class weighting
│   └── transformer_models.py                    # BERT / DistilBERT baselines
├── utils/                                       # Utilities
│   ├── download_data.py                         # Download QEvasion dataset
│   ├── convert_gemini_to_arrow.py               # Convert Gemini CSV augmentations to Arrow format
│   ├── fix_evasion_labels.py                    # Fix evasion label mapping inconsistencies
│   ├── run_hyperparameter_search.py             # Hyperparameter search utilities
│   ├── scoring.py                               # Competition scoring script
│   ├── error_analysis.py                        # BERT error analysis
│   └── load_data.py                             # Dataset inspection
├── data_augmentation/                           # Data augmentation pipeline
│   ├── augmenters.py                            # EDA, back-translation, LLM paraphrasing
│   ├── generate_synthetic_data.py               # Subtask 1 augmentation
│   └── generate_evasion_data.py                 # Subtask 2 augmentation
├── scripts/                                     # Shell scripts for training pipelines
│   ├── train_nli_large.sh                       # Run NLI large training
│   ├── train_political_debate.sh                # Run Political DEBATE training
│   ├── train_best_model.sh                      # Run best DeBERTa-base training
│   ├── train_deberta_large.sh                   # Run DeBERTa-large training
│   ├── run_phase1.sh / run_phase2_large.sh      # Phased training pipelines
│   └── ...                                      # Other experiment scripts
├── docs/                                        # Documentation
│   ├── EXPERIMENTAL_LOG.md                      # Full experiment tracking log
│   └── ANALYSIS_INSIGHTS.md                     # Error analysis findings
├── logs/                                        # Training logs (JSON + text)
│   ├── train_nli_large.json                     # NLI large epoch-by-epoch metrics
│   └── train_political_debate.json              # Political DEBATE epoch-by-epoch metrics
├── deberta_nli_large_model/                     # Saved DeBERTa-v3-large NLI model weights
├── political_debate_model/                      # Saved Political DEBATE model weights
├── QEvasion/                                    # Original QEvasion dataset (Arrow format)
├── QEvasion_Gemini/                             # Gemini-augmented balanced dataset (6120 train)
├── prediction/                                  # Output prediction files
├── clarity_task_evaluation_dataset.csv          # Evaluation dataset (CSV)
├── evaluation_set_agreed.csv                    # Agreed-label evaluation subset
├── confusion_matrix_final.png                   # Confusion matrix: final submission
├── confusion_matrix_submission1.png             # Confusion matrix: first submission
├── submission_1.zip                             # Submission 1: DeBERTa-v3-base (F1=0.61)
├── submission_final.zip                         # Final submission: DeBERTa-v3-base
├── submission_nli_large.zip                     # NLI large model submission
├── submission_political_debate.zip              # Political DEBATE model submission
└── requirements.txt
```

## Setup

### Requirements

- Python 3.8+
- CUDA-capable GPU (recommended: 24GB+ VRAM for LARGE models, 8GB+ for base models)

```bash
pip install -r requirements.txt
```

### Download Data

```bash
python utils/download_data.py
```

This downloads the [QEvasion dataset](https://huggingface.co/datasets/ailsntua/QEvasion) from HuggingFace to `./QEvasion/`.

The Gemini-augmented balanced dataset (`QEvasion_Gemini/`) is included in this repository (6,120 training samples, 2,040 per class, balanced via Gemini-generated paraphrases).

## Training

### DeBERTa-v3-large NLI (0.6678)

Fine-tunes `MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli` (435M params) on the Gemini-augmented balanced dataset.

```bash
bash scripts/train_nli_large.sh
# or directly:
python training/train_deberta_nli_large.py \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 6 \
    --warmup_ratio 0.15 \
    --dataset_dir ./QEvasion_Gemini \
    --output_dir ./deberta_nli_large_model
```

### DeBERTa-v3-base (F1=0.64, Rank 17/40)

```bash
bash scripts/train_best_model.sh
# or directly:
python training/train_deberta_improved.py \
    --learning_rate 3e-5 \
    --llrd_alpha 0.9 \
    --warmup_ratio 0.15 \
    --gradient_accumulation_steps 4 \
    --num_epochs 6 \
    --patience 3
```

### Political DEBATE base

Fine-tunes `mlburnham/Political_DEBATE_base_v1.0` — a DeBERTa-base model pre-trained on political text — on the Gemini-augmented balanced dataset.

```bash
bash scripts/train_political_debate.sh
# or directly:
python training/train_political_debate.py \
    --learning_rate 3e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 6 \
    --dataset_dir ./QEvasion_Gemini \
    --output_dir ./political_debate_model
```

### DeBERTa-v3-LARGE

```bash
python training/train_deberta_large.py
```

### Subtask 2 (9-class evasion)

```bash
python training/train_deberta_large_evasion.py
```

### Data Augmentation (EDA / Back-translation)

Generate synthetic training data to address class imbalance:

```bash
# Hybrid augmentation (EDA + back-translation)
python data_augmentation/generate_synthetic_data.py --method hybrid --output_dir ./QEvasion_augmented

# Train on augmented data
python training/train_deberta_augmented.py --data_dir ./QEvasion_augmented
```

### Convert Gemini Augmentations to Arrow Format

If you have Gemini-generated CSV augmentations, convert them:

```bash
python utils/convert_gemini_to_arrow.py
```

## Key Techniques

- **Layer-wise Learning Rate Decay (LLRD)**: Different learning rates per transformer layer, lower for earlier layers
- **Gradient Accumulation**: Effective batch size of 32 with batch size 4–8 and 4–8 accumulation steps
- **Cosine Annealing with Warmup**: Learning rate schedule with 15% linear warmup followed by cosine decay
- **Focal Loss**: Addresses class imbalance by down-weighting easy examples
- **Data Augmentation**: EDA, back-translation, and Gemini-based paraphrasing for minority class oversampling
- **NLI Pre-training**: Using a model pre-trained on large NLI corpora (MNLI, FEVER, ANLI) as initialization

## Dataset

The **QEvasion** dataset (Kalouli et al.) contains 3,448 training and 308 test examples of political Q&A pairs, annotated with clarity labels by 3 independent annotators.

**Original class distribution (training set)**:
- Ambivalent: 59.2%
- Clear Reply: 30.5%
- Clear Non-Reply: 10.3%

**QEvasion_Gemini (balanced)**:
- 6,120 training examples (2,040 per class) via Gemini-generated paraphrases
- 308 test examples (original, unchanged)

## Submissions

| File | Model | Notes |
|------|-------|-------|
| `submission_1.zip` | DeBERTa-v3-base | First submission; Test F1 = 0.61, Rank 17/40 |
| `submission_final.zip` | DeBERTa-v3-base | Final submission to eval phase with data augmentation |
| `submission_nli_large.zip` | DeBERTa-v3-large NLI |  Test F1 = 0.6678 |
| `submission_political_debate.zip` | Political DEBATE base | Test F1 = 0.5724 |

## License

This project is released for research and educational purposes.

## Acknowledgments

- SemEval 2026 Task 6 organizers
- [QEvasion dataset](https://huggingface.co/datasets/ailsntua/QEvasion) by Kalouli et al.
- Microsoft for [DeBERTa-v3](https://huggingface.co/microsoft/deberta-v3-base)
- [MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli](https://huggingface.co/MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli)
- [mlburnham/Political_DEBATE_base_v1.0](https://huggingface.co/mlburnham/Political_DEBATE_base_v1.0)
