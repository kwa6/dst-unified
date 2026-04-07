#!/usr/bin/env bash
# train_t5_raw_twostage.sh — Two-stage T5 raw (unbalanced) fine-tuning
#
# Stage 1: Train on augmented data (LUAS or D0T) with raw data distribution
# Stage 2: Fine-tune on real data (MultiWOZ) with raw data distribution
#
# This variant does NOT balance the dataset, using natural slot value distribution
# (like Llama training) for comparison.
#
# Usage:
#   bash scripts/train_t5_raw_twostage.sh luas
#   bash scripts/train_t5_raw_twostage.sh d0t
#
# Arguments:
#   $1  Dataset for stage 1: 'luas' or 'd0t'  (required)
set -euo pipefail

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
    echo "Error: Dataset not specified. Use 'luas' or 'd0t'"
    echo "Usage: bash scripts/train_t5_raw_twostage.sh [luas|d0t]"
    exit 1
fi

if [ "$DATASET" != "luas" ] && [ "$DATASET" != "d0t" ]; then
    echo "Error: Invalid dataset '$DATASET'. Must be 'luas' or 'd0t'"
    exit 1
fi

export PYTHONPATH=src

DATASET_UPPER=$(echo "$DATASET" | tr '[:lower:]' '[:upper:]')
STAGE1_DIR="runs/t5_raw_stage1_${DATASET}"
STAGE2_DIR="runs/t5_raw_stage2_${DATASET}_mwoz_final"

echo
echo "============================================"
echo "TWO-STAGE T5 RAW (UNBALANCED) FINE-TUNING"
echo "============================================"
echo "Stage 1 Dataset: $DATASET_UPPER"
echo "Model: google/flan-t5-base"
echo "Dataset type: RAW (natural distribution)"
echo "Stage 1 Output: $STAGE1_DIR"
echo "Stage 2 Output: $STAGE2_DIR"
echo "============================================"
echo

# ============================================================================
# STAGE 1: Train on augmented data (LUAS or D0T)
# ============================================================================
echo
echo "========================================"
echo "STAGE 1: Training on $DATASET_UPPER data (raw)"
echo "========================================"
echo

python -m dst.runners.train_t5_balanced \
  --train_path      "data_unified/${DATASET}/train.jsonl" \
  --stage           1 \
  --no-balanced \
  --out_dir         "$STAGE1_DIR" \
  --total_examples  4000 \
  --num_epochs      3 \
  --warmup_steps    50

echo
echo "✓ Stage 1 complete. Checkpoint saved to: $STAGE1_DIR/final"
echo

# ============================================================================
# STAGE 2: Fine-tune on real data (MultiWOZ)
# ============================================================================
echo
echo "========================================"
echo "STAGE 2: Fine-tuning on MultiWOZ (real data, raw)"
echo "========================================"
echo

python -m dst.runners.train_t5_balanced \
  --train_path      "data_unified/multiwoz24/train.jsonl" \
  --eval_path       "data_unified/multiwoz24/val.jsonl" \
  --stage           2 \
  --checkpoint      "$STAGE1_DIR/final" \
  --no-balanced \
  --out_dir         "$STAGE2_DIR" \
  --total_examples  2000 \
  --num_epochs      3 \
  --warmup_steps    50

echo
echo "============================================"
echo "✓ TWO-STAGE RAW TRAINING COMPLETE"
echo "============================================"
echo "Final model: $STAGE2_DIR/final"
echo
echo "To evaluate:"
echo "  python -m dst.runners.eval_jga \\"
echo "    --model $STAGE2_DIR/final \\"
echo "    --path data_unified/multiwoz24/test.jsonl"
echo "============================================"
echo
