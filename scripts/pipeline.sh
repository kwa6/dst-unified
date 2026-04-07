#!/usr/bin/env bash
# pipeline.sh — Complete DST-Unified pipeline orchestrator
#
# Runs the full pipeline composed of modular blocks:
#   1. System setup
#   2. Fetch data (MultiWOZ, D0T, LUAS)
#   3. Build unified datasets
#   4. Train T5 models (two-stage)
#   5. Train Llama 3.1 8B (two-stage)
#   6. Train Llama 3.3 70B (two-stage)
#   7. Evaluate T5 models
#   8. Evaluate Llama models
#
# Individual blocks can also be run independently:
#   bash scripts/block_fetch_data.sh
#   bash scripts/block_build_data.sh
#   bash scripts/block_train_t5.sh
#   bash scripts/block_train_llama_31_8b.sh [requires HF_TOKEN]
#   bash scripts/block_train_llama_33_70b.sh [requires HF_TOKEN]
#   bash scripts/block_eval_t5.sh
#   bash scripts/block_eval_llama.sh [requires HF_TOKEN]
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/pipeline.sh
#   (Requires HF_TOKEN for gated Llama 3 models)
#
set -euo pipefail

run_step() {
  echo
  echo "==========================================="
  echo "Running: $*"
  echo "==========================================="
  echo
  "$@"
}

echo
echo "==========================================="
echo "DST-UNIFIED FULL PIPELINE"
echo "==========================================="
echo

# System setup
run_step bash scripts/setup_ubuntu.sh

# Data preparation
run_step bash scripts/block_fetch_data.sh
run_step bash scripts/block_build_data.sh

# Model training
run_step bash scripts/block_train_t5.sh

if [ -z "${HF_TOKEN:-}" ]; then
  echo
  echo "==========================================="
  echo "ERROR: HF_TOKEN not set"
  echo "==========================================="
  echo "Llama 3.1 and 3.3 models require HF_TOKEN for access."
  echo "Get your token from: https://huggingface.co/settings/tokens"
  echo "Then run: export HF_TOKEN=hf_... && bash scripts/pipeline.sh"
  exit 1
fi

run_step bash scripts/block_train_llama_31_8b.sh
run_step bash scripts/block_train_llama_33_70b.sh

# Model evaluation
run_step bash scripts/block_eval_t5.sh
run_step bash scripts/block_eval_llama.sh

echo
echo "==========================================="
echo "✓ FULL PIPELINE COMPLETE"
echo "==========================================="
echo
echo "Summary of outputs:"
echo "  T5 BALANCED:  runs/t5_stage2_*_mwoz_final/final"
echo "  T5 RAW:       runs/t5_raw_stage2_*_mwoz_final/final"
echo "  Llama 3.1 8B: runs/llama31_8b_stage2_*_mwoz_final/final"
echo "  Llama 3.3 70B: runs/llama33_70b_stage2_*_mwoz_final/final"
echo
echo "Comparison:"
echo "  T5 Balanced vs Raw: Check block_eval_t5.sh output for JGA comparison"
echo "  Llama models: Check block_eval_llama.sh output"
echo
echo "==========================================="

