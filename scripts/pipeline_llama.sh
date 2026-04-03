#!/usr/bin/env bash
set -euo pipefail

run_step() {
  echo
  echo "------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------"
  "$@"
}

echo
echo "------------------------------------------"
echo "Sourcing: scripts/setup_env.sh"
echo "------------------------------------------"
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Train Llama 3.3 70B with LoRA (only if HF_TOKEN is set — model is gated)
if [ -n "${HF_TOKEN:-}" ]; then
  run_step bash scripts/train_llama33_70b.sh
  run_step bash scripts/train_llama33_70b_luas.sh
  run_step bash scripts/train_llama33_70b_d0t.sh
fi

# Evaluation — Llama 3.1 8B fine-tuned models
run_step bash scripts/eval_jga_llama.sh \
  runs/llama31_8b_mwoz_v1/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga_llama.sh \
  runs/llama31_8b_luas_v1/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga_llama.sh \
  runs/llama31_8b_d0t_v1/final \
  data_unified/multiwoz24/test.jsonl

# Evaluation — Llama 3.3 70B fine-tuned models (4-bit)
run_step bash scripts/eval_jga_llama.sh \
  runs/llama33_70b_mwoz_v1/final \
  data_unified/multiwoz24/test.jsonl \
  200 --load_in_4bit

run_step bash scripts/eval_jga_llama.sh \
  runs/llama33_70b_luas_v1/final \
  data_unified/multiwoz24/test.jsonl \
  200 --load_in_4bit

run_step bash scripts/eval_jga_llama.sh \
  runs/llama33_70b_d0t_v1/final \
  data_unified/multiwoz24/test.jsonl \
  200 --load_in_4bit
