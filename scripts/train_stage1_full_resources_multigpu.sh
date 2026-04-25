#!/usr/bin/env bash
# Stage-1 multi-GPU training for full LUAS + full D0T unified datasets (2 GPUs).
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable not set."
  echo "Get your token from: https://huggingface.co/settings/tokens"
  echo "Then run: export HF_TOKEN=hf_... && bash scripts/train_stage1_full_resources_multigpu.sh"
  exit 1
fi

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
EVAL_PATH="data_unified/multiwoz24/val.jsonl"
MASTER_PORT="${MASTER_PORT:-29500}"
NPROC_PER_NODE=2

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

echo "[full-stage1] using 2 GPUs"
echo "[full-stage1] effective batch size = 4 x 4 x 2 = 32"
echo "[full-stage1] LUAS full: 278818 rows, 8700 steps, 870 warmup"
run_train "data_unified/luas/train.jsonl" \
  "runs/llama31_8b_stage1_luas_full_all" \
  278818 8700 870 0

echo "[full-stage1] D0T full: 325002 rows, 10200 steps, 1020 warmup"
run_train "data_unified/d0t/train.jsonl" \
  "runs/llama31_8b_stage1_d0t_full_all" \
  325002 10200 1020 1

printf '\nFull stage-1 multi-GPU training complete.\n'