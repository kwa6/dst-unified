#!/usr/bin/env bash
# train_llama33_70b_d0t.sh — LoRA fine-tune meta-llama/Llama-3.3-70B-Instruct on the D0T DST task.
#
# Recommended hardware: 4× A100 80GB (or equivalent ~320GB VRAM total).
#
# Usage:
#   HF_TOKEN=hf_... bash scripts/train_llama33_70b_d0t.sh
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
    echo "Error: HF_TOKEN environment variable not set."
    echo "Get your token from: https://huggingface.co/settings/tokens"
    echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_llama33_70b_d0t.sh"
    exit 1
fi

MODEL="meta-llama/Llama-3.3-70B-Instruct"
OUT_DIR="runs/llama33_70b_d0t_v1"

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"
python -m dst.runners.train_llama \
  --train_path      data_unified/d0t/train.jsonl \
  --model           "$MODEL" \
  --out_dir         "$OUT_DIR" \
  --total_examples  8000 \
  --steps           500 \
  --warmup_steps    50 \
  --batch_size      1 \
  --grad_accum      16 \
  --use_slot_description \
  --use_value_examples \
  --max_length      256 \
  --load_in_4bit
