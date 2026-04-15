#!/usr/bin/env bash
# train_t5_twostage.sh — Two-stage T5 balanced fine-tuning
#
# Stage 1: Train on augmented data (LUAS or D0T) with standard learning rate
# Stage 2: Fine-tune on real data (MultiWOZ) with reduced warmup
#
# Usage:
#   bash scripts/train_t5_twostage.sh luas
#   bash scripts/train_t5_twostage.sh d0t
#
# Arguments:
#   $1  Dataset for stage 1: 'luas' or 'd0t'  (required)
set -euo pipefail

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
    echo "Error: Dataset not specified. Use 'luas' or 'd0t'"
    echo "Usage: bash scripts/train_t5_twostage.sh [luas|d0t]"
    exit 1
fi

if [ "$DATASET" != "luas" ] && [ "$DATASET" != "d0t" ]; then
    echo "Error: Invalid dataset '$DATASET'. Must be 'luas' or 'd0t'"
    exit 1
fi

export PYTHONPATH=src

DATASET_UPPER=$(echo "$DATASET" | tr '[:lower:]' '[:upper:]')
STAGE1_DIR="runs/t5_stage1_${DATASET}"
STAGE2_DIR="runs/t5_stage2_${DATASET}_mwoz_final"

# Set flags for D0T: include descriptions and examples
# Set flags for LUAS: omit descriptions and examples (defaults)
if [ "$DATASET" = "d0t" ]; then
    STAGE1_FLAGS="--use_slot_description --use_value_examples"
else
    STAGE1_FLAGS=""
fi

echo
echo "============================================"
echo "TWO-STAGE T5 BALANCED FINE-TUNING"
echo "============================================"
echo "Stage 1 Dataset: $DATASET_UPPER"
echo "Model: google/flan-t5-base"
echo "Stage 1 Output: $STAGE1_DIR"
echo "Stage 2 Output: $STAGE2_DIR"
echo "============================================"
echo

# ============================================================================
# STAGE 1: Train on augmented data (LUAS or D0T)
# ============================================================================
if [ -d "$STAGE1_DIR/final" ]; then
    echo "========================================"
    echo "STAGE 1: SKIPPED (already trained)"
    echo "========================================"
    echo "Found existing checkpoint: $STAGE1_DIR/final"
    echo
else
    echo "========================================"
    echo "STAGE 1: Training on $DATASET_UPPER data"
    echo "========================================"
    echo

    python -m dst.runners.train_t5_balanced \
      --train_path      "data_unified/${DATASET}/train.jsonl" \
      --stage           1 \
      --out_dir         "$STAGE1_DIR" \
      --total_examples  4000 \
      --num_epochs      3 \
      --warmup_steps    50 \
      $STAGE1_FLAGS
    echo "✓ Stage 1 complete. Checkpoint saved to: $STAGE1_DIR/final"
    echo
fi

# ============================================================================
# STAGE 2: Fine-tune on real data (MultiWOZ)
# ============================================================================
echo
echo "========================================"
echo "STAGE 2: Fine-tuning on MultiWOZ (real data)"
echo "========================================"
echo

python -m dst.runners.train_t5_balanced \
  --train_path      "data_unified/multiwoz24/train.jsonl" \
  --eval_path       "data_unified/multiwoz24/val.jsonl" \
  --stage           2 \
  --checkpoint      "$STAGE1_DIR/final" \
  --out_dir         "$STAGE2_DIR" \
  --total_examples  2000 \
  --num_epochs      3 \
  --warmup_steps    50

echo
echo "============================================"
echo "✓ TWO-STAGE TRAINING COMPLETE"
echo "============================================"
echo "Final model: $STAGE2_DIR/final"
echo
echo "To evaluate:"
echo "  python -m dst.runners.eval_jga \\"
echo "    --model $STAGE2_DIR/final \\"
echo "    --path data_unified/multiwoz24/test.jsonl"
echo "============================================"
echo
