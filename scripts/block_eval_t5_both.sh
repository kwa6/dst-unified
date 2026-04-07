#!/usr/bin/env bash
# block_eval_t5_both.sh — Evaluate both T5 variants (balanced & raw) on MultiWOZ test
#
# Evaluates:
#   - T5 balanced (stage 2): LUAS → MultiWOZ
#   - T5 balanced (stage 2): D0T → MultiWOZ
#   - T5 raw (stage 2): LUAS → MultiWOZ
#   - T5 raw (stage 2): D0T → MultiWOZ
#
# Usage:
#   bash scripts/block_eval_t5_both.sh
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

# Evaluate T5 raw models
run_step bash scripts/eval_jga.sh \
  runs/t5_raw_stage2_luas_mwoz_final/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga.sh \
  runs/t5_raw_stage2_d0t_mwoz_final/final \
  data_unified/multiwoz24/test.jsonl

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: T5 evaluations done"
echo "=========================================="
echo "Compare results:"
echo "  Balanced vs Raw (LUAS) in output above"
echo "  Balanced vs Raw (D0T) in output above"
echo
