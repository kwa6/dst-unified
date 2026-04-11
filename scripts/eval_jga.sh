#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/eval_jga.sh <model_path> <jsonl_path> [max_turns]"
  exit 1
fi

MODEL_PATH="$1"
JSONL_PATH="$2"
MAX_TURNS="${3:-}"

export PYTHONPATH=src
python -m dst.runners.eval_jga \
  --path "$JSONL_PATH" \
  --model "$MODEL_PATH" \
  ${MAX_TURNS:+--max_turns "$MAX_TURNS"}