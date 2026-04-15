#!/usr/bin/env bash
# train_llama31_8b_twostage.sh — Two-stage LoRA fine-tuning with Llama 3.1 8B
#
# Stage 1: Train on augmented data (LUAS or D0T) with standard learning rate
# Stage 2: Fine-tune on real data (MultiWOZ) with same learning rate
#
# Recommended hardware: 1× A100 40GB (or similar)
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/train_llama31_8b_twostage.sh luas
#   HF_TOKEN=hf_... bash scripts/train_llama31_8b_twostage.sh d0t
#
# Arguments:
#   $1  Dataset for stage 1: 'luas' or 'd0t'  (required)
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_llama31_8b_twostage.sh luas"
    exit 1
fi

DATASET="${1:-}"
if [ -z "$DATASET" ]; then
    echo "Error: Dataset not specified. Use 'luas' or 'd0t'"
    echo "Usage: bash scripts/train_llama31_8b_twostage.sh [luas|d0t]"
    exit 1
fi

if [ "$DATASET" != "luas" ] && [ "$DATASET" != "d0t" ]; then
    echo "Error: Invalid dataset '$DATASET'. Must be 'luas' or 'd0t'"
    exit 1
fi

MODEL="meta-llama/Llama-3.1-8B-Instruct"

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"

DATASET_UPPER=$(echo "$DATASET" | tr '[:lower:]' '[:upper:]')
STAGE1_DIR="runs/llama31_8b_stage1_${DATASET}_128k"
STAGE2_DIR="runs/llama31_8b_stage2_${DATASET}_to_mwoz"

# Set flags for D0T: include descriptions and examples
# Set flags for LUAS: omit descriptions and examples (defaults)
if [ "$DATASET" = "d0t" ]; then
    STAGE1_FLAGS="--use_slot_description --use_value_examples"
else
    STAGE1_FLAGS=""
fi

echo
echo "============================================"
echo "TWO-STAGE LLAMA 3.1 8B FINE-TUNING"
echo "============================================"
echo "Stage 1 Dataset: $DATASET_UPPER"
echo "Model: $MODEL"
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

    python -m dst.runners.train_llama \
      --train_path      "data_unified/${DATASET}/train.jsonl" \
      --eval_path       "data_unified/multiwoz24/val.jsonl" \
      --stage           1 \
      --model           "$MODEL" \
      --out_dir         "$STAGE1_DIR" \
      --total_examples  128000 \
      --steps           8000 \
      --warmup_steps    800 \
      --balance_mode    50_50 \
      --batch_size      4 \
      --grad_accum      4 \
      --lr              2e-4 \
      --max_length      512 \
      --load_in_4bit \
      $STAGE1_FLAGS

    echo
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

python -m dst.runners.train_llama \
  --train_path          "data_unified/multiwoz24/train.jsonl" \
  --eval_path           "data_unified/multiwoz24/val.jsonl" \
  --stage               2 \
  --checkpoint          "$STAGE1_DIR/final" \
  --out_dir             "$STAGE2_DIR" \
  --total_examples      2000000 \
  --steps               1500 \
  --warmup_steps_stage2 100 \
  --balance_mode        none \
  --batch_size          4 \
  --grad_accum          4 \
  --lr_stage2           5e-5 \
  --max_length          512 \
  --load_in_4bit

echo
echo "============================================"
echo "✓ TWO-STAGE TRAINING COMPLETE"
echo "============================================"
echo "Final model: $STAGE2_DIR/final"
echo
echo "To evaluate:"
echo "  python -m dst.runners.eval_jga_llama \\"
echo "    --model $STAGE2_DIR/final \\"
echo "    --path data_unified/multiwoz24/test.jsonl"
echo "============================================"
echo
