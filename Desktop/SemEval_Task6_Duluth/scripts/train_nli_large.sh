#!/bin/bash
# Train DeBERTa-v3-large NLI model for Subtask 1 (3-class clarity)
# Model: MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli

set -e

cd "$(dirname "$0")/.."

echo "============================================="
echo "Training DeBERTa-v3-large NLI"
echo "============================================="

# Step 1: Ensure arrow format exists
if [ ! -f "QEvasion_Gemini/train/data-00000-of-00001.arrow" ]; then
    echo "Converting QEvasion_Gemini CSV to Arrow format..."
    python3 utils/convert_gemini_to_arrow.py
fi

# Step 2: Create logs directory
mkdir -p logs

# Step 3: Train
python3 training/train_deberta_nli_large.py \
    --learning_rate 2e-5 \
    --batch_size 4 \
    --gradient_accumulation_steps 8 \
    --num_epochs 6 \
    --warmup_ratio 0.15 \
    --weight_decay 0.01 \
    --llrd_alpha 0.9 \
    --patience 3 \
    --max_length 512 \
    --dataset_dir ./QEvasion_Gemini \
    --output_dir ./deberta_nli_large_model \
    --log_file ./logs/train_nli_large.json \
    2>&1 | tee logs/train_nli_large.log

echo ""
echo "============================================="
echo "Training complete!"
echo "Submission file: submission_nli_large.zip"
echo "Training log: logs/train_nli_large.json"
echo "Full output: logs/train_nli_large.log"
echo "============================================="
