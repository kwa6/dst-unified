#!/usr/bin/env bash
# train_llama33_70b.sh — LoRA fine-tune meta-llama/Llama-3.3-70B-Instruct on the DST task.
#
# Recommended hardware: 4× A100 80GB (or equivalent ~320GB VRAM total).
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/train_llama33_70b.sh
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_llama33_70b.sh"
    exit 1
fi

MODEL="meta-llama/Llama-3.3-70B-Instruct"
OUT_DIR="runs/llama33_70b_mwoz_v1"

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
  --batch_size      1 \
  --grad_accum      16 \
  --max_length      256 \
  --load_in_4bit
