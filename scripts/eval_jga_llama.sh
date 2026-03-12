#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/eval_jga_llama.sh <model_id> <jsonl_path> [max_turns] [--load_in_4bit]"
  exit 1
fi

MODEL_ID="$1"
JSONL_PATH="$2"
MAX_TURNS="${3:-200}"
EXTRA_ARGS="${4:-}"

export PYTHONPATH=src
python -m dst.runners.eval_jga_llama \
  --model  "$MODEL_ID" \
  --path   "$JSONL_PATH" \
  --max_turns "$MAX_TURNS" \
  $EXTRA_ARGS
