#!/usr/bin/env bash
# train_llama.sh — LoRA fine-tune a Llama model on the DST training set.
#
# Usage:
#   bash scripts/train_llama.sh                          # defaults
#   bash scripts/train_llama.sh meta-llama/Llama-2-7b-chat-hf runs/llama_mwoz_v1
#
# Arguments (all optional, positional):
#   $1  HF model ID or local path  (default: meta-llama/Llama-2-7b-chat-hf)
#   $2  output directory           (default: runs/llama_mwoz_v1)
set -euo pipefail

MODEL="${1:-meta-llama/Llama-2-7b-chat-hf}"
OUT_DIR="${2:-runs/llama_mwoz_v1}"

export PYTHONPATH=src
python -m dst.runners.train_llama \
  --train_path      data_unified/multiwoz24/train.jsonl \
  --model           "$MODEL" \
  --out_dir         "$OUT_DIR" \
  --total_examples  2000 \
  --steps           500 \
  --batch_size      4 \
  --grad_accum      4
