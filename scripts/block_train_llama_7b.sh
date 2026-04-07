#!/usr/bin/env bash
# block_train_llama_7b.sh — Train Llama 2 7B (two-stage: augmented → real)
#
# Stage 1: Train on augmented data (LUAS or D0T)
# Stage 2: Fine-tune on real data (MultiWOZ)
#
# Trains two variants:
#   - LUAS (augmented) → MultiWOZ (real)
#   - D0T (augmented) → MultiWOZ (real)
#
# Requires: HF_TOKEN set for gated Llama models
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/block_train_llama_7b.sh
#
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/block_train_llama_7b.sh"
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
echo "BLOCK: TRAIN LLAMA 2 7B (TWO-STAGE)"
echo "=========================================="
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Two-stage: LUAS (augmented) → MultiWOZ (real)
run_step bash scripts/train_llama_twostage.sh luas

# Two-stage: D0T (augmented) → MultiWOZ (real)
run_step bash scripts/train_llama_twostage.sh d0t

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: Llama 2 7B models trained"
echo "=========================================="
echo
