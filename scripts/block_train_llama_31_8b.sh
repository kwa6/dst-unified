#!/usr/bin/env bash
# block_train_llama_31_8b.sh — Train Llama 3.1 8B (two-stage: augmented → real)
#
# Stage 1: Train on augmented data (LUAS or D0T)
# Stage 2: Fine-tune on real data (MultiWOZ)
#
# Trains two variants:
#   - LUAS (augmented) → MultiWOZ (real)
#   - D0T (augmented) → MultiWOZ (real)
#
# Recommended hardware: 1× A100 40GB
#
# Requires: HF_TOKEN set for gated Llama models
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/block_train_llama_31_8b.sh
#
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/block_train_llama_31_8b.sh"
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
echo "BLOCK: TRAIN LLAMA 3.1 8B (TWO-STAGE)"
echo "=========================================="
echo "Recommended: 1× A100 40GB"
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Two-stage: LUAS (augmented) → MultiWOZ (real)
run_step bash scripts/train_llama31_8b_twostage.sh luas

# Two-stage: D0T (augmented) → MultiWOZ (real)
run_step bash scripts/train_llama31_8b_twostage.sh d0t

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: Llama 3.1 8B models trained"
echo "=========================================="
echo
