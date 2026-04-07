#!/usr/bin/env bash
# train_llama33_70b_twostage.sh — Two-stage LoRA fine-tuning with Llama 3.3 70B
#
# Stage 1: Train on augmented data (LUAS or D0T) with standard learning rate
# Stage 2: Fine-tune on real data (MultiWOZ) with same learning rate
#
# Recommended hardware: 4× A100 80GB (or equivalent ~320GB VRAM total)
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/train_llama33_70b_twostage.sh luas
#   HF_TOKEN=hf_... bash scripts/train_llama33_70b_twostage.sh d0t
#
# Arguments:
#   $1  Dataset for stage 1: 'luas' or 'd0t'  (required)
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_llama33_70b_twostage.sh luas"
    exit 1
fi

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
    echo "Error: Dataset not specified. Use 'luas' or 'd0t'"
    echo "Usage: bash scripts/train_llama33_70b_twostage.sh [luas|d0t]"
    exit 1
fi

if [ "$DATASET" != "luas" ] && [ "$DATASET" != "d0t" ]; then
    echo "Error: Invalid dataset '$DATASET'. Must be 'luas' or 'd0t'"
    exit 1
fi

MODEL="meta-llama/Llama-3.3-70B-Instruct"

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"

DATASET_UPPER=$(echo "$DATASET" | tr '[:lower:]' '[:upper:]')
STAGE1_DIR="runs/llama33_70b_stage1_${DATASET}"
STAGE2_DIR="runs/llama33_70b_stage2_${DATASET}_mwoz_final"

echo
echo "============================================"
echo "TWO-STAGE LLAMA 3.3 70B FINE-TUNING"
echo "============================================"
echo "Stage 1 Dataset: $DATASET_UPPER"
echo "Model: $MODEL"
echo "Stage 1 Output: $STAGE1_DIR"
echo "Stage 2 Output: $STAGE2_DIR"
echo "Recommended: 4× A100 80GB GPUs"
echo "============================================"
echo

# ============================================================================
# STAGE 1: Train on augmented data (LUAS or D0T)
# ============================================================================
echo
echo "========================================"
echo "STAGE 1: Training on $DATASET_UPPER data"
echo "========================================"
echo

python -m dst.runners.train_llama \
  --train_path      "data_unified/${DATASET}/train.jsonl" \
  --stage           1 \
  --model           "$MODEL" \
  --out_dir         "$STAGE1_DIR" \
  --total_examples  8000 \
  --steps           500 \
  --warmup_steps    50 \
  --batch_size      1 \
  --grad_accum      16 \
  --max_length      256 \
  --load_in_4bit \
  --lr              2e-4

echo
echo "✓ Stage 1 complete. Checkpoint saved to: $STAGE1_DIR/final"
echo

# ============================================================================
# STAGE 2: Fine-tune on real data (MultiWOZ)
# ============================================================================
echo
echo "========================================"
echo "STAGE 2: Fine-tuning on MultiWOZ (real data)"
echo "========================================"
echo

python -m dst.runners.train_llama \
  --train_path      "data_unified/multiwoz24/train.jsonl" \
  --eval_path       "data_unified/multiwoz24/val.jsonl" \
  --stage           2 \
  --checkpoint      "$STAGE1_DIR/final" \
  --out_dir         "$STAGE2_DIR" \
  --total_examples  8000 \
  --steps           300 \
  --warmup_steps    50 \
  --batch_size      1 \
  --grad_accum      16 \
  --max_length      256 \
  --load_in_4bit \
  --lr              2e-4

echo
echo "============================================"
echo "✓ TWO-STAGE TRAINING COMPLETE"
echo "============================================"
echo "Final model: $STAGE2_DIR/final"
echo
echo "To evaluate:"
echo "  python -m dst.runners.eval_jga_llama \\"
echo "    --model $STAGE2_DIR/final \\"
echo "    --path data_unified/multiwoz24/test.jsonl \\"
echo "    --load_in_4bit"
echo "============================================"
echo
