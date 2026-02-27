#!/bin/bash
# Train Political DEBATE model for Subtask 1 (3-class clarity)
# Model: mlburnham/Political_DEBATE_base_v1.0

set -e

cd "$(dirname "$0")/.."

echo "============================================="
echo "Training Political DEBATE"
echo "============================================="

# Step 1: Ensure arrow format exists
if [ ! -f "QEvasion_Gemini/train/data-00000-of-00001.arrow" ]; then
    echo "Converting QEvasion_Gemini CSV to Arrow format..."
    python3 utils/convert_gemini_to_arrow.py
fi

# Step 2: Create logs directory
mkdir -p logs

# Step 3: Train
python3 training/train_political_debate.py \
    --learning_rate 3e-5 \
    --batch_size 8 \
    --gradient_accumulation_steps 4 \
    --num_epochs 6 \
    --warmup_ratio 0.15 \
    --weight_decay 0.01 \
    --llrd_alpha 0.9 \
    --patience 3 \
    --max_length 512 \
    --dataset_dir ./QEvasion_Gemini \
    --output_dir ./political_debate_model \
    --log_file ./logs/train_political_debate.json \
    2>&1 | tee logs/train_political_debate.log

echo ""
echo "============================================="
echo "Training complete!"
echo "Submission file: submission_political_debate.zip"
echo "Training log: logs/train_political_debate.json"
echo "Full output: logs/train_political_debate.log"
echo "============================================="
