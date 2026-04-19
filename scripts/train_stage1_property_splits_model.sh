#!/usr/bin/env bash
# Stage-1 smoke-test training for a specified Llama model.
#
# Usage:
#   bash scripts/train_stage1_property_splits_model.sh "<model_id>" "<run_tag>"
#
# Example:
#   bash scripts/train_stage1_property_splits_model.sh "meta-llama/Llama-3.1-70B-Instruct" "llama31_70b"
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable not set."
  echo "Get your token from: https://huggingface.co/settings/tokens"
  echo "Then run: export HF_TOKEN=hf_..."
  exit 1
fi

if [ "$#" -lt 2 ]; then
  echo "Usage: bash scripts/train_stage1_property_splits_model.sh \"<model_id>\" \"<run_tag>\""
  exit 1
fi

MODEL="$1"
RUN_TAG="$2"

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"

TRAIN_PATH="${TRAIN_PATH:-data_unified/luas/luas_aligned_34492_train.jsonl}"
EVAL_PATH="${EVAL_PATH:-data_unified/multiwoz24/val.jsonl}"
OUT_DIR="${OUT_DIR:-runs/${RUN_TAG}_stage1_smoke}"
LIMIT_READ="${LIMIT_READ:-200}"
TOTAL_EXAMPLES="${TOTAL_EXAMPLES:-200}"
STEPS="${STEPS:-20}"
WARMUP_STEPS="${WARMUP_STEPS:-2}"
BATCH_SIZE="${BATCH_SIZE:-1}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
USE_DESC="${USE_DESC:-0}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"

extra_args=()
if [ "$USE_DESC" = "1" ]; then
  extra_args+=(--use_slot_description)
fi
if [ "$LOAD_IN_4BIT" = "1" ]; then
  extra_args+=(--load_in_4bit)
fi

eval_args=()
if [ -n "${EVAL_PATH}" ]; then
  eval_args+=(--eval_path "$EVAL_PATH")
fi

python -m dst.runners.train_llama \
  --train_path "$TRAIN_PATH" \
  "${eval_args[@]}" \
  --model "$MODEL" \
  --out_dir "$OUT_DIR" \
  --limit_read "$LIMIT_READ" \
  --total_examples "$TOTAL_EXAMPLES" \
  --steps "$STEPS" \
  --warmup_steps "$WARMUP_STEPS" \
  --batch_size "$BATCH_SIZE" \
  --grad_accum "$GRAD_ACCUM" \
  --lr "$LR" \
  --balance_mode none \
  --stage 1 \
  "${extra_args[@]}"

printf '\nStage-1 smoke test complete.\n'
