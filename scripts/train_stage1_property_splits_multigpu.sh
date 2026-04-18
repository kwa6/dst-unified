#!/usr/bin/env bash
# Stage-1 multi-GPU training for thesis-locked property splits.
#
# 1) Run: bash scripts/prepare_stage1_property_data.sh
# 2) Then: bash scripts/train_stage1_property_splits_multigpu.sh [NPROC]
# 3) Then: bash scripts/eval_stage1_property_splits_val.sh
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable not set."
  echo "Get your token from: https://huggingface.co/settings/tokens"
  echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_stage1_property_splits_multigpu.sh"
  exit 1
fi

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
EVAL_PATH="data_unified/multiwoz24/val.jsonl"
MASTER_PORT="${MASTER_PORT:-29500}"

NPROC_PER_NODE="${1:-${NPROC_PER_NODE:-}}"
if [ -z "$NPROC_PER_NODE" ]; then
  if command -v nvidia-smi >/dev/null 2>&1; then
    NPROC_PER_NODE=$(nvidia-smi -L | wc -l | tr -d ' ')
  fi
fi

if [ -z "$NPROC_PER_NODE" ]; then
  echo "Error: NPROC_PER_NODE not set and GPU count could not be detected."
  echo "Usage: bash scripts/train_stage1_property_splits_multigpu.sh <num_gpus>"
  exit 1
fi

run_train() {
  local train_path="$1"
  local out_dir="$2"
  local total_examples="$3"
  local steps="$4"
  local warmup="$5"
  local use_desc="${6:-0}"
  local extra_args=()

  if [ "$use_desc" = "1" ]; then
    extra_args+=(--use_slot_description)
  fi

  torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
    -m dst.runners.train_llama \
    --train_path "$train_path" \
    --eval_path "$EVAL_PATH" \
    --model "$MODEL" \
    --out_dir "$out_dir" \
    --total_examples "$total_examples" \
    --steps "$steps" \
    --warmup_steps "$warmup" \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4 \
    --balance_mode none \
    --stage 1 \
    "${extra_args[@]}"
}

# 1. LUAS full 128k
run_train "data_unified/luas/luas_full_128k_train.jsonl" \
  "runs/llama31_8b_stage1_luas_full_128k" \
  128000 8000 800 0

# 2. D0T full 128k
run_train "data_unified/d0t/d0t_full_128k_train.jsonl" \
  "runs/llama31_8b_stage1_d0t_full_128k" \
  128000 8000 800 1

# 3. LUAS aligned
run_train "data_unified/luas/luas_aligned_34492_train.jsonl" \
  "runs/llama31_8b_stage1_luas_aligned_34492" \
  34516 2200 220 0

# 4. LUAS nonaligned
run_train "data_unified/luas/luas_nonaligned_34492_train.jsonl" \
  "runs/llama31_8b_stage1_luas_nonaligned_34492" \
  34516 2200 220 0

# 5. D0T aligned
run_train "data_unified/d0t/d0t_aligned_34492_plus_native_none_train.jsonl" \
  "runs/llama31_8b_stage1_d0t_aligned_34492_plus_native_none" \
  45039 2800 280 1

# 6. D0T nonaligned
run_train "data_unified/d0t/d0t_nonaligned_34492_plus_same_native_none_train.jsonl" \
  "runs/llama31_8b_stage1_d0t_nonaligned_34492_plus_same_native_none" \
  45039 2800 280 1

printf '\nStage-1 multi-GPU training complete.\n'
