#!/usr/bin/env bash
# train_llama31_8b.sh — LoRA fine-tune meta-llama/Llama-3.1-8B-Instruct on the DST task.
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/train_llama31_8b.sh
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_llama31_8b.sh"
    exit 1
fi

MODEL="meta-llama/Llama-3.1-8B-Instruct"
OUT_DIR="runs/llama31_8b_mwoz_v1"

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"
python -m dst.runners.train_llama \
  --train_path      data_unified/multiwoz24/train.jsonl \
  --eval_path       data_unified/multiwoz24/val.jsonl \
  --model           "$MODEL" \
  --out_dir         "$OUT_DIR" \
  --total_examples  128000 \
  --steps           8000 \
  --warmup_steps    800 \
  --batch_size      4 \
  --grad_accum      4
