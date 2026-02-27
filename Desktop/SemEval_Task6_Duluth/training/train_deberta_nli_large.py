"""
DeBERTa-v3-large NLI (MoritzLaurer) for 3-class Clarity Classification

Model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli
- Pre-trained on massive NLI datasets (MNLI, FEVER, ANLI, LingNLI, WANLI)
- Strong zero-shot and fine-tuning performance on classification tasks
- DeBERTa-v3-large architecture (434M params)

Techniques:
- Layer-wise Learning Rate Decay (LLRD)
- Gradient Accumulation (effective batch=32)
- Cosine Annealing with Warmup
- Class-weighted Cross-Entropy Loss
- Early Stopping
"""

from datasets import load_from_disk
import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import (
    AutoTokenizer,
    AutoModelForSequenceClassification,
    get_cosine_schedule_with_warmup
)
from sklearn.metrics import classification_report, accuracy_score, f1_score
from tqdm import tqdm
import time
import argparse
import json
import zipfile
import os

print("=" * 80)
print("DeBERTa-v3-large NLI for 3-Class Clarity Classification")
print("Model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli")
print("=" * 80)

# Parse arguments
parser = argparse.ArgumentParser()
parser.add_argument('--learning_rate', type=float, default=2e-5,
                    help='Peak learning rate (default: 2e-5, lower for large model)')
parser.add_argument('--batch_size', type=int, default=4,
                    help='Per-device batch size (default: 4, smaller for large model)')
parser.add_argument('--gradient_accumulation_steps', type=int, default=8,
                    help='Gradient accumulation steps (default: 8, effective batch=32)')
parser.add_argument('--num_epochs', type=int, default=6,
                    help='Maximum epochs (default: 6)')
parser.add_argument('--warmup_ratio', type=float, default=0.15,
                    help='Warmup ratio (default: 0.15)')
parser.add_argument('--weight_decay', type=float, default=0.01,
                    help='Weight decay (default: 0.01)')
parser.add_argument('--llrd_alpha', type=float, default=0.9,
                    help='Layer-wise LR decay factor (default: 0.9)')
parser.add_argument('--patience', type=int, default=3,
                    help='Early stopping patience (default: 3)')
parser.add_argument('--max_length', type=int, default=512,
                    help='Max sequence length (default: 512)')
parser.add_argument('--dataset_dir', type=str, default='./QEvasion_Gemini',
                    help='Dataset directory (default: ./QEvasion_Gemini)')
parser.add_argument('--output_dir', type=str, default='./deberta_nli_large_model',
                    help='Output directory for model (default: ./deberta_nli_large_model)')
parser.add_argument('--log_file', type=str, default='./logs/train_nli_large.json',
                    help='Training log file (default: ./logs/train_nli_large.json)')
args = parser.parse_args()

# Check for GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"\n Device: {device}")
if torch.cuda.is_available():
    print(f" GPU: {torch.cuda.get_device_name(0)}")
    print(f" Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB"
          if hasattr(torch.cuda.get_device_properties(0), 'total_mem')
          else f" Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

# Print hyperparameters
print(f"\n{'=' * 80}")
print("HYPERPARAMETERS")
print(f"{'=' * 80}")
print(f"Learning Rate: {args.learning_rate}")
print(f"Batch Size (per device): {args.batch_size}")
print(f"Gradient Accumulation Steps: {args.gradient_accumulation_steps}")
print(f"Effective Batch Size: {args.batch_size * args.gradient_accumulation_steps}")
print(f"Max Epochs: {args.num_epochs}")
print(f"Warmup Ratio: {args.warmup_ratio:.1%}")
print(f"Weight Decay: {args.weight_decay}")
print(f"LLRD Alpha: {args.llrd_alpha}")
print(f"Early Stopping Patience: {args.patience}")
print(f"Max Length: {args.max_length}")
print(f"Dataset: {args.dataset_dir}")

# Training log
training_log = {
    'model': 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli',
    'task': 'subtask1_clarity_3class',
    'hyperparameters': vars(args),
    'epochs': [],
    'best_epoch': None,
    'best_f1': None,
    'final_results': None,
}

# Load dataset
print("\n[1/8] Loading dataset...")
dataset = load_from_disk(args.dataset_dir)
train_df = pd.DataFrame(dataset['train'])
test_df = pd.DataFrame(dataset['test'])

print(f"* Training samples: {len(train_df)}")
print(f"* Test samples: {len(test_df)}")

# Prepare text
print("\n[2/8] Preparing text for DeBERTa...")


def prepare_text(row):
    """Format: Question: {q} [SEP] Answer: {a}"""
    question = row['question']
    answer = row['interview_answer']
    return f"Question: {question} [SEP] Answer: {answer}"


train_df['text'] = train_df.apply(prepare_text, axis=1)
test_df['text'] = test_df.apply(prepare_text, axis=1)

# Create label mapping
label_list = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
label2id = {label: idx for idx, label in enumerate(label_list)}
id2label = {idx: label for label, idx in label2id.items()}

train_df['label_id'] = train_df['clarity_label'].map(label2id)
test_df['label_id'] = test_df['clarity_label'].map(label2id)

print(f"* Label mapping: {label2id}")

# Calculate class weights for imbalance
class_counts = train_df['label_id'].value_counts().sort_index()
total = len(train_df)
class_weights = torch.tensor(
    [total / (len(class_counts) * count) for count in class_counts],
    dtype=torch.float
).to(device)

print(f"\n* Class distribution:")
for label, count in zip(label_list, class_counts):
    print(f"  {label}: {count} ({count / total * 100:.1f}%)")
print(f"* Class weights: {class_weights.cpu().numpy()}")

training_log['class_distribution'] = {
    label: int(count) for label, count in zip(label_list, class_counts)
}


# Custom Dataset
class ClarityDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=512):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        encoding = self.tokenizer(
            text,
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(label, dtype=torch.long)
        }


# Layer-wise Learning Rate Decay (LLRD)
def get_optimizer_grouped_parameters(model, learning_rate, weight_decay, llrd_alpha):
    no_decay = ["bias", "LayerNorm.weight"]
    num_layers = model.config.num_hidden_layers

    optimizer_grouped_parameters = []

    # Embeddings (lowest LR)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.deberta.embeddings.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
    })
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.deberta.embeddings.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": learning_rate * (llrd_alpha ** (num_layers + 1))
    })

    # Encoder layers
    for layer_idx in range(num_layers):
        layer = model.deberta.encoder.layer[layer_idx]
        layer_lr = learning_rate * (llrd_alpha ** (num_layers - layer_idx))

        optimizer_grouped_parameters.append({
            "params": [p for n, p in layer.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": layer_lr
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in layer.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": layer_lr
        })

    # Classifier head (highest LR)
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.classifier.named_parameters()
                   if not any(nd in n for nd in no_decay)],
        "weight_decay": weight_decay,
        "lr": learning_rate
    })
    optimizer_grouped_parameters.append({
        "params": [p for n, p in model.classifier.named_parameters()
                   if any(nd in n for nd in no_decay)],
        "weight_decay": 0.0,
        "lr": learning_rate
    })

    # Pooler if exists
    if hasattr(model.deberta, 'pooler') and model.deberta.pooler is not None:
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.deberta.pooler.named_parameters()
                       if not any(nd in n for nd in no_decay)],
            "weight_decay": weight_decay,
            "lr": learning_rate
        })
        optimizer_grouped_parameters.append({
            "params": [p for n, p in model.deberta.pooler.named_parameters()
                       if any(nd in n for nd in no_decay)],
            "weight_decay": 0.0,
            "lr": learning_rate
        })

    return optimizer_grouped_parameters


# Training function
def train_epoch(model, dataloader, optimizer, scheduler, device, class_weights,
                gradient_accumulation_steps):
    model.train()
    total_loss = 0
    predictions = []
    true_labels = []

    optimizer.zero_grad()

    progress_bar = tqdm(dataloader, desc="Training")
    for step, batch in enumerate(progress_bar):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels = batch['labels'].to(device)

        outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)

        # Class-weighted loss
        loss_fct = torch.nn.CrossEntropyLoss(weight=class_weights)
        loss = loss_fct(outputs.logits, labels)

        loss = loss / gradient_accumulation_steps
        total_loss += loss.item() * gradient_accumulation_steps

        loss.backward()

        if (step + 1) % gradient_accumulation_steps == 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()

        preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
        predictions.extend(preds)
        true_labels.extend(labels.cpu().numpy())

        progress_bar.set_postfix({'loss': loss.item() * gradient_accumulation_steps})

    avg_loss = total_loss / len(dataloader)
    f1 = f1_score(true_labels, predictions, average='macro')

    return avg_loss, f1


# Evaluation function
def evaluate(model, dataloader, device):
    model.eval()
    predictions = []
    true_labels = []

    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)

            preds = torch.argmax(outputs.logits, dim=1).cpu().numpy()
            predictions.extend(preds)
            true_labels.extend(labels.cpu().numpy())

    return predictions, true_labels


# ============================================================================
# TRAIN
# ============================================================================

model_name = 'MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli'

print(f"\n{'=' * 80}")
print(f"Training: {model_name}")
print(f"{'=' * 80}")

# Initialize tokenizer and model
print(f"\n[3/8] Loading {model_name}...")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=3,
    id2label=id2label,
    label2id=label2id,
    problem_type="single_label_classification",
    ignore_mismatched_sizes=True  # NLI head has different size, reinitialize
)
model.to(device)

num_params = sum(p.numel() for p in model.parameters())
print(f"* Model loaded: {num_params / 1e6:.1f}M parameters")
training_log['num_parameters'] = num_params

# Create datasets
print(f"\n[4/8] Creating datasets...")
train_dataset = ClarityDataset(
    train_df['text'].values,
    train_df['label_id'].values,
    tokenizer,
    max_length=args.max_length
)
test_dataset = ClarityDataset(
    test_df['text'].values,
    test_df['label_id'].values,
    tokenizer,
    max_length=args.max_length
)

train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=args.batch_size)

print(f"* Train batches: {len(train_loader)}")
print(f"* Test batches: {len(test_loader)}")

# Optimizer with LLRD
print(f"\n[5/8] Setting up optimizer with LLRD...")
optimizer_grouped_parameters = get_optimizer_grouped_parameters(
    model, args.learning_rate, args.weight_decay, args.llrd_alpha
)
optimizer = torch.optim.AdamW(optimizer_grouped_parameters)

# Scheduler
total_steps = (len(train_loader) // args.gradient_accumulation_steps) * args.num_epochs
warmup_steps = int(total_steps * args.warmup_ratio)

scheduler = get_cosine_schedule_with_warmup(
    optimizer,
    num_warmup_steps=warmup_steps,
    num_training_steps=total_steps
)

print(f"* Total steps: {total_steps}")
print(f"* Warmup steps: {warmup_steps}")

# Training loop
print(f"\n[6/8] Training for up to {args.num_epochs} epochs...")
start_time = time.time()

best_f1 = 0
best_model_state = None
best_epoch = 0
patience_counter = 0

for epoch in range(args.num_epochs):
    print(f"\n{'=' * 80}")
    print(f"Epoch {epoch + 1}/{args.num_epochs}")
    print(f"{'=' * 80}")

    train_loss, train_f1 = train_epoch(
        model, train_loader, optimizer, scheduler, device,
        class_weights, args.gradient_accumulation_steps
    )
    print(f"\n* Train Loss: {train_loss:.4f}, Train F1: {train_f1:.4f}")

    # Evaluate
    test_preds, test_labels = evaluate(model, test_loader, device)
    test_f1 = f1_score(test_labels, test_preds, average='macro')
    test_acc = accuracy_score(test_labels, test_preds)

    print(f"* Dev F1: {test_f1:.4f}, Dev Acc: {test_acc:.4f}")

    # Per-class F1
    per_class_f1 = f1_score(test_labels, test_preds, average=None)
    print(f"* Per-class F1: {dict(zip(label_list, [f'{f:.4f}' for f in per_class_f1]))}")

    epoch_log = {
        'epoch': epoch + 1,
        'train_loss': round(train_loss, 4),
        'train_f1': round(train_f1, 4),
        'dev_f1': round(test_f1, 4),
        'dev_acc': round(test_acc, 4),
        'per_class_f1': {label: round(f, 4) for label, f in zip(label_list, per_class_f1)},
        'elapsed_minutes': round((time.time() - start_time) / 60, 1),
    }
    training_log['epochs'].append(epoch_log)

    if test_f1 > best_f1:
        best_f1 = test_f1
        best_model_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
        best_epoch = epoch + 1
        patience_counter = 0
        print(f"  New best F1: {best_f1:.4f} (saved)")
    else:
        patience_counter += 1
        print(f"  No improvement for {patience_counter} epoch(s)")

    if patience_counter >= args.patience:
        print(f"\nEarly stopping triggered after {epoch + 1} epochs")
        break

train_time = time.time() - start_time
print(f"\n{'=' * 80}")
print(f"* Training completed in {train_time / 60:.1f} minutes")
print(f"* Best model from epoch {best_epoch} with F1: {best_f1:.4f}")

training_log['best_epoch'] = best_epoch
training_log['best_f1'] = round(best_f1, 4)
training_log['training_time_minutes'] = round(train_time / 60, 1)

# Load best model
model.load_state_dict(best_model_state)
model.to(device)

# Final evaluation
print(f"\n[7/8] Final evaluation with best model...")
test_preds, test_labels = evaluate(model, test_loader, device)

test_f1 = f1_score(test_labels, test_preds, average='macro')
test_acc = accuracy_score(test_labels, test_preds)

print(f"\n{'=' * 80}")
print("DeBERTa-v3-large NLI RESULTS")
print(f"{'=' * 80}")
print(f"\nBest Model (Epoch {best_epoch}):")
print(f"  Dev F1 (Macro): {test_f1:.4f}")
print(f"  Dev Accuracy: {test_acc:.4f}")
print(f"  Training Time: {train_time / 60:.1f} minutes")
print(f"\nDetailed Classification Report:")
report = classification_report(test_labels, test_preds, target_names=label_list, output_dict=True)
print(classification_report(test_labels, test_preds, target_names=label_list))

training_log['final_results'] = {
    'dev_f1': round(test_f1, 4),
    'dev_accuracy': round(test_acc, 4),
    'classification_report': report,
}

# ============================================================================
# GENERATE SUBMISSION
# ============================================================================
print(f"\n[8/8] Generating submission...")

pred_labels = [id2label[pred] for pred in test_preds]

# Save prediction file (extensionless)
with open('prediction', 'w') as f:
    for pred in pred_labels:
        f.write(f"{pred}\n")
print(f"* Prediction file created: prediction ({len(pred_labels)} lines)")

# Prediction distribution
unique, counts = np.unique(pred_labels, return_counts=True)
print(f"\nPrediction distribution:")
for label, count in zip(unique, counts):
    print(f"  {label}: {count} ({count / len(pred_labels) * 100:.1f}%)")

training_log['prediction_distribution'] = {
    str(label): int(count) for label, count in zip(unique, counts)
}

# Create submission zip (no subdirectories)
zip_name = 'submission_nli_large.zip'
with zipfile.ZipFile(zip_name, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('prediction', arcname='prediction')
print(f"* Submission ZIP created: {zip_name}")

# Save model
os.makedirs(args.output_dir, exist_ok=True)
model.save_pretrained(args.output_dir)
tokenizer.save_pretrained(args.output_dir)
print(f"* Model saved to: {args.output_dir}")

# Save training log
os.makedirs(os.path.dirname(args.log_file), exist_ok=True)
with open(args.log_file, 'w') as f:
    json.dump(training_log, f, indent=2, default=str)
print(f"* Training log saved to: {args.log_file}")

# Comparison
print(f"\n{'=' * 80}")
print("MODEL COMPARISON")
print(f"{'=' * 80}")
print(f"  DeBERTa-v3-base Improved:     F1 = 0.61")
print(f"  DeBERTa-v3-large NLI (this):  F1 = {test_f1:.4f}")
improvement = test_f1 - 0.61
print(f"  Difference: {improvement:+.4f}")

print(f"\n{'=' * 80}")
print("TRAINING COMPLETE")
print(f"{'=' * 80}")
