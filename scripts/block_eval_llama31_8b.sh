#!/usr/bin/env bash
# block_eval_llama31_8b.sh — Evaluate Llama 3.1 8B models on MultiWOZ test set
#
# Evaluates two-stage fine-tuned Llama 3.1 8B models:
#   - Llama 3.1 8B (LUAS → MultiWOZ) — no descriptions/examples
#   - Llama 3.1 8B (D0T → MultiWOZ) — with descriptions only
#
# Requires: HF_TOKEN set for gated Llama models
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/block_eval_llama31_8b.sh
#
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/block_eval_llama31_8b.sh"
    exit 1
fi

run_step() {
  echo
  echo "------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------"
  "$@"
}

echo
echo "=========================================="
echo "BLOCK: EVALUATE LLAMA 3.1 8B MODELS"
echo "=========================================="
echo "Evaluating on: data_unified/multiwoz24/test.jsonl"
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Llama 3.1 8B: Benchmark (trained directly on MultiWOZ)
if [ -d "runs/llama31_8b_mwoz_v2/final" ]; then
  echo "========================================"
  echo "Llama 3.1 8B: BENCHMARK (MultiWOZ only, minimal prompts)"
  echo "========================================"
  run_step bash scripts/eval_jga_llama.sh \
    runs/llama31_8b_mwoz_v2/final \
    data_unified/multiwoz24/test.jsonl \
    --mismatches_file results_llama31_8b_benchmark_mismatches.json
else
  echo "⚠️  Skipped: runs/llama31_8b_mwoz_v2/final not found"
fi

echo
if [ -d "runs/llama31_8b_stage2_luas_to_mwoz/final" ]; then
  echo "========================================"
  echo "Llama 3.1 8B: LUAS → MultiWOZ (minimal prompts)"
  echo "========================================"
  run_step bash scripts/eval_jga_llama.sh \
    runs/llama31_8b_stage2_luas_to_mwoz/final \
    data_unified/multiwoz24/test.jsonl \
    --mismatches_file results_llama31_8b_luas_mismatches.json
else
  echo "⚠️  Skipped: runs/llama31_8b_stage2_luas_to_mwoz/final not found"
fi

echo

# Llama 3.1 8B: D0T two-stage (with descriptions only for cross-dataset eval)
if [ -d "runs/llama31_8b_stage2_d0t_to_mwoz/final" ]; then
  echo "========================================"
  echo "Llama 3.1 8B: D0T → MultiWOZ (with descriptions)"
  echo "========================================"
  run_step bash scripts/eval_jga_llama.sh \
    runs/llama31_8b_stage2_d0t_to_mwoz/final \
    data_unified/multiwoz24/test.jsonl \
    --use_slot_description \
    --mismatches_file results_llama31_8b_d0t_mismatches.json
else
  echo "⚠️  Skipped: runs/llama31_8b_stage2_d0t_to_mwoz/final not found"
fi

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: Llama 3.1 8B models evaluated"
echo "=========================================="
echo "Results above show JGA (Joint Goal Accuracy) scores"
echo
