#!/usr/bin/env bash
# block_fetch_data.sh — Fetch all raw datasets (MultiWOZ, D0T, LUAS)
#
# Usage:
#   bash scripts/block_fetch_data.sh
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
echo "BLOCK: FETCH DATA"
echo "=========================================="
echo

# Source environment setup
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Fetch all datasets
run_step bash scripts/fetch_multiwoz.sh
run_step bash scripts/fetch_d0t.sh
run_step bash scripts/fetch_luas.sh

echo
echo "=========================================="
echo "✓ BLOCK COMPLETE: All datasets fetched"
echo "=========================================="
echo
