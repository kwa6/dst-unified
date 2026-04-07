#!/usr/bin/env bash
# block_build_data.sh — Build all unified datasets (MultiWOZ, D0T, LUAS)
#
# Usage:
#   bash scripts/block_build_data.sh
#
set -euo pipefail

run_step() {
  echo
  echo "------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------"
  "$@"
}

echo
echo "=========================================="
echo "BLOCK: BUILD DATA"
echo "=========================================="
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Build all datasets into unified format
run_step bash scripts/build_multiwoz.sh
run_step bash scripts/build_d0t.sh
run_step bash scripts/build_luas.sh

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: All datasets built"
echo "=========================================="
echo
