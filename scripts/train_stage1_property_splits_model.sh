#!/usr/bin/env bash
# Stage-1 full-matrix training for a specified model.
#
# Usage:
#   bash scripts/train_stage1_property_splits_model.sh "<model_id>" "<run_tag>"
#
# Example:
#   CUDA_VISIBLE_DEVICES=0,1,2,3 LOAD_IN_4BIT=1 \
#   bash scripts/train_stage1_property_splits_model.sh \
#     "meta-llama/Llama-3.1-70B-Instruct" \
#     "llama31_70b"
#
# Notes:
# - For 70B, prefer single-process sharded loading, not torchrun/DDP.
# - Set CUDA_VISIBLE_DEVICES to the GPUs you want device_map="auto" to use.
# - Keep LOAD_IN_4BIT=1 for large models like 70B.

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

EVAL_PATH="${EVAL_PATH:-data_unified/multiwoz24/val.jsonl}"
BATCH_SIZE="${BATCH_SIZE:-4}"
GRAD_ACCUM="${GRAD_ACCUM:-4}"
LR="${LR:-2e-4}"
LOAD_IN_4BIT="${LOAD_IN_4BIT:-1}"
LIMIT_READ="${LIMIT_READ:-}"

run_train() {
  local train_path="$1"
  local out_dir="$2"
  local total_examples="$3"
  local steps="$4"
  local warmup="$5"
  local use_desc="${6:-0}"

  local extra_args=()
  local eval_args=()
  local limit_args=()

  if [ "$use_desc" = "1" ]; then
    extra_args+=(--use_slot_description)
  fi

  if [ "$LOAD_IN_4BIT" = "1" ]; then
    extra_args+=(--load_in_4bit)
  fi

  if [ -n "${EVAL_PATH}" ]; then
    eval_args+=(--eval_path "$EVAL_PATH")
  fi

  if [ -n "${LIMIT_READ}" ]; then
    limit_args+=(--limit_read "$LIMIT_READ")
  fi

  echo
  echo "============================================================"
  echo "MODEL:          $MODEL"
  echo "RUN:            $out_dir"
  echo "TRAIN_PATH:     $train_path"
  echo "TOTAL_EXAMPLES: $total_examples"
  echo "STEPS:          $steps"
  echo "WARMUP:         $warmup"
  echo "BATCH_SIZE:     $BATCH_SIZE"
  echo "GRAD_ACCUM:     $GRAD_ACCUM"
  echo "LR:             $LR"
  echo "USE_DESC:       $use_desc"
  echo "LOAD_IN_4BIT:   $LOAD_IN_4BIT"
  if [ -n "${LIMIT_READ}" ]; then
    echo "LIMIT_READ:     $LIMIT_READ"
  fi
  echo "============================================================"
  echo

  python -m dst.runners.train_llama \
    --train_path "$train_path" \
    "${eval_args[@]}" \
    --model "$MODEL" \
    --out_dir "$out_dir" \
    "${limit_args[@]}" \
    --total_examples "$total_examples" \
    --steps "$steps" \
    --warmup_steps "$warmup" \
    --batch_size "$BATCH_SIZE" \
    --grad_accum "$GRAD_ACCUM" \
    --lr "$LR" \
    --balance_mode none \
    --stage 1 \
    "${extra_args[@]}"
}

# 1. LUAS full 128k
run_train \
  "data_unified/luas/luas_full_128k_train.jsonl" \
  "runs/${RUN_TAG}_stage1_luas_full_128k" \
  128000 8000 800 0

# 2. D0T full 128k
run_train \
  "data_unified/d0t/d0t_full_128k_train.jsonl" \
  "runs/${RUN_TAG}_stage1_d0t_full_128k" \
  128000 8000 800 1

# 3. LUAS aligned
run_train \
  "data_unified/luas/luas_aligned_34492_train.jsonl" \
  "runs/${RUN_TAG}_stage1_luas_aligned_34492" \
  34516 2200 220 0

# 4. LUAS nonaligned
run_train \
  "data_unified/luas/luas_nonaligned_34492_train.jsonl" \
  "runs/${RUN_TAG}_stage1_luas_nonaligned_34492" \
  34516 2200 220 0

# 5. D0T aligned
run_train \
  "data_unified/d0t/d0t_aligned_34492_plus_native_none_train.jsonl" \
  "runs/${RUN_TAG}_stage1_d0t_aligned_34492_plus_native_none" \
  45039 2800 280 1

# 6. D0T nonaligned
run_train \
  "data_unified/d0t/d0t_nonaligned_34492_plus_same_native_none_train.jsonl" \
  "runs/${RUN_TAG}_stage1_d0t_nonaligned_34492_plus_same_native_none" \
  45039 2800 280 1

printf '\nStage-1 full matrix complete for model: %s\n' "$MODEL"