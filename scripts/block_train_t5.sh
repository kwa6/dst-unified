#!/usr/bin/env bash
# block_train_t5.sh — Train both T5 variants (balanced & raw) with two-stage fine-tuning
#
# Run 1: BALANCED dataset (50/50 none/non-none split)
#   - Stage 1: LUAS → MultiWOZ
#   - Stage 1: D0T → MultiWOZ
#
# Run 2: RAW dataset (natural slot value distribution)
#   - Stage 1: LUAS → MultiWOZ
#   - Stage 1: D0T → MultiWOZ
#
# Usage:
#   bash scripts/block_train_t5.sh
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
echo "BLOCK: TRAIN T5 MODELS (TWO VARIANTS)"
echo "=========================================="
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

echo
echo "======= Run 1: T5 BALANCED ======="
echo

# Two-stage: LUAS (augmented) → MultiWOZ (real)
run_step bash scripts/train_t5_twostage.sh luas

# Two-stage: D0T (augmented) → MultiWOZ (real)
run_step bash scripts/train_t5_twostage.sh d0t

echo
echo "======= Run 2: T5 RAW ======="
echo

# Two-stage: LUAS (augmented) → MultiWOZ (real)
run_step bash scripts/train_t5_raw_twostage.sh luas

# Two-stage: D0T (augmented) → MultiWOZ (real)
run_step bash scripts/train_t5_raw_twostage.sh d0t

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: All T5 models trained"
echo "=========================================="
echo "Models created:"
echo "  Balanced: runs/t5_stage2_*_mwoz_final/final"
echo "  Raw:      runs/t5_raw_stage2_*_mwoz_final/final"
echo

