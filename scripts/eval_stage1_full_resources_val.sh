#!/usr/bin/env bash
# Stage-1 validation evaluation for full LUAS + full D0T unified runs.
set -euo pipefail

export PYTHONPATH=src

EVAL_PATH="data_unified/multiwoz24/val.jsonl"
RESULTS_FILE="results_llama_stage1_full_resources_val.csv"
OUTPUT_DIR="outputs"

mkdir -p "$OUTPUT_DIR"
rm -f "$RESULTS_FILE"

run_eval() {
  local model_path="$1"
  local run_name="$2"
  local use_desc="${3:-0}"
  local extra_args=()

  if [ "$use_desc" = "1" ]; then
    extra_args+=(--use_slot_description)
  fi

  python -m dst.runners.eval_jga_llama \
    --path "$EVAL_PATH" \
    --model "$model_path" \
    --results_file "$RESULTS_FILE" \
    --mismatches_file "$OUTPUT_DIR/${run_name}_mismatches_val.json" \
    --audit_file "$OUTPUT_DIR/${run_name}_audit_val.json" \
    --audit_summary_file "$OUTPUT_DIR/${run_name}_audit_summary_val.json" \
    "${extra_args[@]}"
}

echo "[full-stage1-eval] evaluating LUAS full"
run_eval "runs/llama31_8b_stage1_luas_full_all/final" \
  "llama31_8b_stage1_luas_full_all" \
  0

echo "[full-stage1-eval] evaluating D0T full"
run_eval "runs/llama31_8b_stage1_d0t_full_all/final" \
  "llama31_8b_stage1_d0t_full_all" \
  1

printf '\nFull stage-1 val evaluation complete. Results logged in %s\n' "$RESULTS_FILE"