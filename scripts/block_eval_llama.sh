#!/usr/bin/env bash
# block_eval_llama.sh — Evaluate all Llama models on MultiWOZ test set
#
# Evaluates two-stage fine-tuned Llama models:
#   - Llama 2 7B (LUAS → MultiWOZ)
#   - Llama 2 7B (D0T → MultiWOZ)
#   - Llama 3.1 8B (LUAS → MultiWOZ) [if trained]
#   - Llama 3.1 8B (D0T → MultiWOZ) [if trained]
#   - Llama 3.3 70B (LUAS → MultiWOZ) [if trained]
#   - Llama 3.3 70B (D0T → MultiWOZ) [if trained]
#
# Requires: HF_TOKEN set for gated Llama models
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/block_eval_llama.sh
#
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/block_eval_llama.sh"
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
echo "BLOCK: EVALUATE LLAMA MODELS"
echo "=========================================="
echo "Evaluating on: data_unified/multiwoz24/test.jsonl"
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh


# Llama 3.1 8B: two-stage models (if they exist)
if [ -d "runs/llama31_8b_stage2_luas_mwoz_final/final" ]; then
  run_step bash scripts/eval_jga_llama.sh \
    runs/llama31_8b_stage2_luas_mwoz_final/final \
    data_unified/multiwoz24/test.jsonl
fi

if [ -d "runs/llama31_8b_stage2_d0t_mwoz_final/final" ]; then
  run_step bash scripts/eval_jga_llama.sh \
    runs/llama31_8b_stage2_d0t_mwoz_final/final \
    data_unified/multiwoz24/test.jsonl
fi

# Llama 3.3 70B: two-stage models (if they exist)
if [ -d "runs/llama33_70b_stage2_luas_mwoz_final/final" ]; then
  run_step bash scripts/eval_jga_llama.sh \
    runs/llama33_70b_stage2_luas_mwoz_final/final \
    data_unified/multiwoz24/test.jsonl
fi

if [ -d "runs/llama33_70b_stage2_d0t_mwoz_final/final" ]; then
  run_step bash scripts/eval_jga_llama.sh \
    runs/llama33_70b_stage2_d0t_mwoz_final/final \
    data_unified/multiwoz24/test.jsonl
fi

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: Llama models evaluated"
echo "=========================================="
echo
