#!/usr/bin/env bash
set -euo pipefail

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/eval_jga.sh <model_path> <jsonl_path> [max_turns]"
  exit 1
fi

MODEL_PATH="$1"
JSONL_PATH="$2"

# If $3 is a number, treat it as max_turns; otherwise start of optional args
if [[ "${3:-}" =~ ^[0-9]+$ ]]; then
    MAX_TURNS="$3"
    EXTRA_ARGS=("${@:4}")
else
    MAX_TURNS=""
    EXTRA_ARGS=("${@:3}")
fi

export PYTHONPATH=src
python -m dst.runners.eval_jga \
  --path "$JSONL_PATH" \
  --model "$MODEL_PATH" \
  ${MAX_TURNS:+--max_turns "$MAX_TURNS"} \
  "${EXTRA_ARGS[@]}"