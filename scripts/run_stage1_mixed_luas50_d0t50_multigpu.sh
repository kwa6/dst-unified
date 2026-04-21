#!/usr/bin/env bash
# Train + eval the mixed LUAS/D0T 50/50 stage-1 run (DDP multi-GPU).
set -euo pipefail

if [ -z "${HF_TOKEN:-}" ]; then
  echo "Error: HF_TOKEN environment variable not set."
  echo "Get your token from: https://huggingface.co/settings/tokens"
  echo "Then run: export HF_TOKEN=hf_... && bash scripts/run_stage1_mixed_luas50_d0t50_multigpu.sh <num_gpus>"
  exit 1
fi

export PYTHONPATH=src
export HF_TOKEN="$HF_TOKEN"

MODEL="meta-llama/Llama-3.1-8B-Instruct"
EVAL_PATH="data_unified/multiwoz24/val.jsonl"
TRAIN_PATH="data_unified/mixed/mixed_luas50_d0t50_full_128k_train.jsonl"
OUT_DIR="runs/llama31_8b_stage1_mixed_luas50_d0t50_full_128k"
OUTPUT_DIR="outputs"
MASTER_PORT="${MASTER_PORT:-29500}"

NPROC_PER_NODE="${1:-}"
if [ -z "$NPROC_PER_NODE" ]; then
  echo "Usage: bash scripts/run_stage1_mixed_luas50_d0t50_multigpu.sh <num_gpus>"
  exit 1
fi

mkdir -p "$OUTPUT_DIR"

train_args=()
train_args+=(--use_slot_description)

eval_args=()
eval_args+=(--use_slot_description)

echo "[stage1-mixed] training"
torchrun --standalone --nproc_per_node "$NPROC_PER_NODE" --master_port "$MASTER_PORT" \
  -m dst.runners.train_llama \
  --train_path "$TRAIN_PATH" \
  --eval_path "$EVAL_PATH" \
  --model "$MODEL" \
  --out_dir "$OUT_DIR" \
  --total_examples 128000 \
  --steps 8000 \
  --warmup_steps 800 \
  --batch_size 4 \
  --grad_accum 4 \
  --lr 2e-4 \
  --balance_mode none \
  --stage 1 \
  "${train_args[@]}"

echo "[stage1-mixed] evaluation"
python -m dst.runners.eval_jga_llama \
  --path "$EVAL_PATH" \
  --model "$OUT_DIR/final" \
  --mismatches_file "$OUTPUT_DIR/llama31_8b_stage1_mixed_luas50_d0t50_full_128k_mismatches_val.json" \
  --audit_file "$OUTPUT_DIR/llama31_8b_stage1_mixed_luas50_d0t50_full_128k_audit_val.json" \
  --audit_summary_file "$OUTPUT_DIR/llama31_8b_stage1_mixed_luas50_d0t50_full_128k_audit_summary_val.json" \
  "${eval_args[@]}"
