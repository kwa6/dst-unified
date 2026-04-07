#!/usr/bin/env bash
# block_eval_t5.sh — Evaluate both T5 variants (balanced & raw) on MultiWOZ test set
#
# Evaluates:
#   - T5 balanced (stage 2): LUAS → MultiWOZ
#   - T5 balanced (stage 2): D0T → MultiWOZ
#   - T5 raw (stage 2): LUAS → MultiWOZ
#   - T5 raw (stage 2): D0T → MultiWOZ
#
# Usage:
#   bash scripts/block_eval_t5.sh
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
echo "BLOCK: EVALUATE T5 MODELS (BOTH VARIANTS)"
echo "=========================================="
echo "Evaluating on: data_unified/multiwoz24/test.jsonl"
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

echo
echo "======= T5 BALANCED (50/50 none/non-none) ======="
echo

# Evaluate T5 balanced models
run_step bash scripts/eval_jga.sh \
  runs/t5_stage2_luas_mwoz_final/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga.sh \
  runs/t5_stage2_d0t_mwoz_final/final \
  data_unified/multiwoz24/test.jsonl

echo
echo "======= T5 RAW (natural distribution) ======="
echo

# Evaluate T5 raw models (if they exist)
if [ -d "runs/t5_raw_stage2_luas_mwoz_final/final" ]; then
  run_step bash scripts/eval_jga.sh \
    runs/t5_raw_stage2_luas_mwoz_final/final \
    data_unified/multiwoz24/test.jsonl
fi

if [ -d "runs/t5_raw_stage2_d0t_mwoz_final/final" ]; then
  run_step bash scripts/eval_jga.sh \
    runs/t5_raw_stage2_d0t_mwoz_final/final \
    data_unified/multiwoz24/test.jsonl
fi

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: T5 evaluations done"
echo "=========================================="
echo "Review results above:"
echo "  Balanced: t5_stage2_*_mwoz_final JGA scores"
echo "  Raw:      t5_raw_stage2_*_mwoz_final JGA scores"
echo

