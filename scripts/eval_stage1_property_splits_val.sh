#!/usr/bin/env bash
# Stage-1 validation evaluation for thesis-locked property splits.
#
# 1) Run: bash scripts/prepare_stage1_property_data.sh
# 2) Then: bash scripts/train_stage1_property_splits.sh
# 3) Then: bash scripts/eval_stage1_property_splits_val.sh
set -euo pipefail

export PYTHONPATH=src

EVAL_PATH="data_unified/multiwoz24/val.jsonl"
RESULTS_FILE="results_llama_stage1_val.csv"
rm -f "$RESULTS_FILE"

run_eval() {
  local out_dir="$1"
  local use_desc="${2:-0}"
  local extra_args=()

  if [ "$use_desc" = "1" ]; then
    extra_args+=(--use_slot_description)
  fi

  python -m dst.runners.eval_jga_llama \
    --path "$EVAL_PATH" \
    --model "$out_dir/final" \
    --results_file "$RESULTS_FILE" \
    --mismatches_file "$out_dir/mismatches_val.json" \
    --audit_file "$out_dir/audit_val.json" \
    --audit_summary_file "$out_dir/audit_summary_val.json" \
    "${extra_args[@]}"
}

# 1. LUAS full 128k
run_eval "runs/llama31_8b_stage1_luas_full_128k" 0

# 2. D0T full 128k
run_eval "runs/llama31_8b_stage1_d0t_full_128k" 1

# 3. LUAS aligned
run_eval "runs/llama31_8b_stage1_luas_aligned_34492" 0

# 4. LUAS nonaligned
run_eval "runs/llama31_8b_stage1_luas_nonaligned_34492" 0

# 5. D0T aligned
run_eval "runs/llama31_8b_stage1_d0t_aligned_34492_plus_native_none" 1

# 6. D0T nonaligned
run_eval "runs/llama31_8b_stage1_d0t_nonaligned_34492_plus_same_native_none" 1

printf '\nStage-1 val evaluation complete. Results logged in %s\n' "$RESULTS_FILE"
