#!/usr/bin/env bash
# block_train_t5_raw.sh — Train T5 models with raw (unbalanced) data (two-stage)
#
# Stage 1: Train on augmented data (LUAS or D0T) with raw distribution
# Stage 2: Fine-tune on real data (MultiWOZ) with raw distribution
#
# Trains two variants:
#   - LUAS (augmented) → MultiWOZ (real)
#   - D0T (augmented) → MultiWOZ (real)
#
# Usage:
#   bash scripts/block_train_t5_raw.sh
#
set -euo pipefail

run_step() {
  echo
  echo "------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------"
  "$@"
}

echo
echo "=========================================="
echo "BLOCK: TRAIN T5 MODELS (RAW, TWO-STAGE)"
echo "=========================================="
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Two-stage: LUAS (augmented) → MultiWOZ (real)
run_step bash scripts/train_t5_raw_twostage.sh luas

# Two-stage: D0T (augmented) → MultiWOZ (real)
run_step bash scripts/train_t5_raw_twostage.sh d0t

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: T5 RAW models trained"
echo "=========================================="
echo
