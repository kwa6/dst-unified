#!/usr/bin/env bash
# Stage-1 workflow wrapper (training + eval scripts).
#
# Run in order:
# 1) bash scripts/prepare_stage1_property_data.sh
# 2) bash scripts/train_stage1_property_splits.sh
# 3) bash scripts/eval_stage1_property_splits_val.sh
set -euo pipefail

echo "Stage-1 workflow is now split into train and eval scripts."
echo "Run in order:"
echo "  1) bash scripts/prepare_stage1_property_data.sh"
echo "  2) bash scripts/train_stage1_property_splits.sh"
echo "  3) bash scripts/eval_stage1_property_splits_val.sh"