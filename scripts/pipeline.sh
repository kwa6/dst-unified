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

# Final evaluation
run_step bash scripts/eval_jga.sh \
  runs/t5_d0t_v1/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga.sh \
  runs/t5_luas_v1/final \
  data_unified/multiwoz24/test.jsonl

run_step bash scripts/eval_jga.sh \
  runs/t5_mwoz_train_v1/final \
  data_unified/multiwoz24/test.jsonl

# ── LLM zero-shot evaluations ─────────────────────────────────────────────
# These require a HuggingFace token with access to gated Llama models.
# Set HF_TOKEN in the environment (e.g. via UCloud "Extra options") to enable.
if [ -n "${HF_TOKEN:-}" ]; then
  python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" \
    2>/dev/null && echo "HuggingFace login OK"
else
  echo "WARNING: HF_TOKEN not set — skipping LLM zero-shot evals"
fi

if [ -n "${HF_TOKEN:-}" ]; then
  run_step bash scripts/eval_jga_llama.sh \
    meta-llama/Llama-2-7b-chat-hf \
    data_unified/multiwoz24/test.jsonl
fi