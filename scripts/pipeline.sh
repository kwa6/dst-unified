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

# Ubuntu/system setup
run_step bash scripts/setup_ubuntu.sh

# Fetch sequence
run_step bash scripts/fetch_multiwoz.sh
run_step bash scripts/fetch_d0t.sh
run_step bash scripts/fetch_luas.sh

# Build sequence
run_step bash scripts/build_multiwoz.sh
run_step bash scripts/build_d0t.sh
run_step bash scripts/build_luas.sh

# Train sequence
run_step bash scripts/train_multiwoz.sh
run_step bash scripts/train_d0t.sh
run_step bash scripts/train_luas.sh

# Train Llama 3.1 8B with LoRA (only if HF_TOKEN is set — model is gated)
if [ -n "${HF_TOKEN:-}" ]; then
  run_step bash scripts/train_llama31_8b.sh
fi

# Final evaluation — T5 fine-tuned models
run_step bash scripts/eval_jga.sh \
  runs/t5_d0t_v1/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga.sh \
  runs/t5_luas_v1/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga.sh \
  runs/t5_mwoz_train_v1/final \
  data_unified/multiwoz24/test.jsonl

# Final evaluation — LLM zero-shot + fine-tuned
run_step bash scripts/eval_llm.sh


