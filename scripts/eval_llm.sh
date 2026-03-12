#!/usr/bin/env bash
# eval_llm.sh — zero-shot LLM evaluations on the test set.
#
# Requires HF_TOKEN to be set in the environment (passed from bootstrap_ucloud.sh
# or exported manually). Models are skipped with a warning if the token is missing.
#
# Usage:
#   HF_TOKEN=hf_xxx bash scripts/eval_llm.sh
#   bash scripts/eval_llm.sh          # skips all evals, just prints warning
set -euo pipefail

TEST_PATH="data_unified/multiwoz24/test.jsonl"

# ── HuggingFace login ──────────────────────────────────────────────────────
if [ -n "${HF_TOKEN:-}" ]; then
  echo "Logging in to HuggingFace..."
  python -c "from huggingface_hub import login; login(token='${HF_TOKEN}')" \
    2>/dev/null && echo "HuggingFace login OK"
else
  echo "WARNING: HF_TOKEN not set — skipping all LLM zero-shot evals"
  exit 0
fi

# ── Model list ─────────────────────────────────────────────────────────────
# Add or remove models here. Each entry is a HuggingFace model ID.
LLAMA_MODELS=(
  "meta-llama/Llama-2-7b-chat-hf"
)

# ── Run evals ─────────────────────────────────────────────────────────────
for MODEL in "${LLAMA_MODELS[@]}"; do
  echo
  echo "=========================================="
  echo "LLM eval: ${MODEL}"
  echo "=========================================="
  bash scripts/eval_jga_llama.sh "$MODEL" "$TEST_PATH" || \
    echo "WARNING: eval failed for ${MODEL}, continuing..."
done

# ── Also eval fine-tuned Llama checkpoints if they exist ──────────────────
FT_CHECKPOINT="runs/llama_mwoz_v1/final"
if [ -d "${FT_CHECKPOINT}" ]; then
  echo
  echo "=========================================="
  echo "LLM eval (fine-tuned): ${FT_CHECKPOINT}"
  echo "=========================================="
  bash scripts/eval_jga_llama.sh "$FT_CHECKPOINT" "$TEST_PATH" || \
    echo "WARNING: fine-tuned eval failed, continuing..."
fi

FT_CHECKPOINT_70B="runs/llama33_70b_mwoz_v1/final"
if [ -d "${FT_CHECKPOINT_70B}" ]; then
  echo
  echo "=========================================="
  echo "LLM eval (fine-tuned 4-bit): ${FT_CHECKPOINT_70B}"
  echo "=========================================="
  bash scripts/eval_jga_llama.sh "$FT_CHECKPOINT_70B" "$TEST_PATH" 200 --load_in_4bit || \
    echo "WARNING: fine-tuned 70B eval failed, continuing..."
fi

echo
echo "All LLM zero-shot evals complete."
