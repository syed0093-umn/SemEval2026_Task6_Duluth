"""
Generate predictions on the evaluation dataset using the trained
DeBERTa-v3-large NLI model (MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli).
"""

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
import zipfile
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--model_dir', type=str, default='./deberta_nli_large_model',
                    help='Directory with saved model')
parser.add_argument('--eval_csv', type=str, default='./clarity_task_evaluation_dataset.csv',
                    help='Path to evaluation CSV')
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--max_length', type=int, default=512)
args = parser.parse_args()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Device: {device}")

# Label mapping
label_list = ['Ambivalent', 'Clear Non-Reply', 'Clear Reply']
id2label = {idx: label for idx, label in enumerate(label_list)}


class EvalDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
        }


# Load evaluation data
print(f"Loading evaluation dataset: {args.eval_csv}")
eval_df = pd.read_csv(args.eval_csv)
print(f"Evaluation samples: {len(eval_df)}")


def prepare_text(row):
    question = row['question'] if pd.notna(row['question']) else ''
    answer = row['interview_answer'] if pd.notna(row['interview_answer']) else ''
    return f"Question: {question} [SEP] Answer: {answer}"


eval_df['text'] = eval_df.apply(prepare_text, axis=1)

# Load saved model
print(f"Loading model from: {args.model_dir}")
tokenizer = AutoTokenizer.from_pretrained(args.model_dir)
model = AutoModelForSequenceClassification.from_pretrained(args.model_dir)
model = model.to(device)
model.eval()
print(f"Model loaded: {sum(p.numel() for p in model.parameters()) / 1e6:.1f}M parameters")

# Run inference
eval_dataset = EvalDataset(
    texts=eval_df['text'].tolist(),
    tokenizer=tokenizer,
    max_length=args.max_length
)
eval_loader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False)

print("Running inference...")
all_preds = []
with torch.no_grad():
    for batch in tqdm(eval_loader):
        input_ids = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        preds = torch.argmax(outputs.logits, dim=1)
        all_preds.extend(preds.cpu().numpy())

pred_labels = [id2label[p] for p in all_preds]
print(f"Generated {len(pred_labels)} predictions")
print(f"Distribution: {pd.Series(pred_labels).value_counts().to_dict()}")

# Save prediction file (extensionless)
with open('prediction', 'w') as f:
    for label in pred_labels:
        f.write(label + '\n')
print(f"Saved: prediction ({len(pred_labels)} lines)")

# Create submission zip
submission_zip = 'submission_eval_nli_large.zip'
with zipfile.ZipFile(submission_zip, 'w', zipfile.ZIP_DEFLATED) as zf:
    zf.write('prediction', arcname='prediction')
print(f"Created: {submission_zip}")
print("Done. Upload submission_eval_nli_large.zip to Codabench.")
