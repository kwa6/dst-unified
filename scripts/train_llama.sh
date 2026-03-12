#!/usr/bin/env bash
# train_llama.sh — LoRA fine-tune a Llama model on the DST training set.
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/train_llama.sh                          # defaults
#   HF_TOKEN=hf_... bash scripts/train_llama.sh meta-llama/Llama-3.2-3B-Instruct runs/llama_v2
#
# Arguments (all optional, positional):
#   $1  HF model ID or local path  (default: meta-llama/Llama-3.1-8B-Instruct)
#   $2  output directory           (default: runs/llama_mwoz_v1)
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_llama.sh"
    exit 1
fi

MODEL="${1:-meta-llama/Llama-3.1-8B-Instruct}"
OUT_DIR="${2:-runs/llama_mwoz_v1}"

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"
python -m dst.runners.train_llama \
  --train_path      data_unified/multiwoz24/train.jsonl \
  --eval_path       data_unified/multiwoz24/val.jsonl \
  --model           "$MODEL" \
  --out_dir         "$OUT_DIR" \
  --total_examples  8000 \
  --steps           500 \
  --warmup_steps    50 \
  --batch_size      4 \
  --grad_accum      4
