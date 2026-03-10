#!/usr/bin/env bash
set -euo pipefail

REPO_URL="https://github.com/kwa6/dst-unified"
REPO_DIR="dst-unified"

run_step() {
  echo
  echo "------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------"
  "$@"
}

echo "=========================================="
echo "uCloud bootstrap starting"
echo "=========================================="

# Clone repo if missing
if [ ! -d "$REPO_DIR/.git" ]; then
  echo "Cloning repository: $REPO_URL"
  git clone "$REPO_URL" "$REPO_DIR"
else
  echo "Repository already exists: $REPO_DIR"
fi

cd "$REPO_DIR"

echo
echo "=========================================="
echo "Now in repo: $(pwd)"
echo "=========================================="

# Source environment setup so it stays active in this shell
echo
echo "------------------------------------------"
echo "Sourcing: scripts/setup_env.sh"
echo "------------------------------------------"
# shellcheck disable=SC1091
source scripts/setup_env.sh

# Ubuntu/system setup
run_step bash scripts/setup_ubuntu.sh

# Build sequence
run_step bash scripts/build_multiwoz.sh
run_step bash scripts/build_d0t.sh
run_step bash scripts/build_luas.sh

# Repeated build sequence, kept exactly as requested
run_step bash scripts/build_multiwoz.sh
run_step bash scripts/build_d0t.sh
run_step bash scripts/build_luas.sh

# Final evaluation
run_step bash scripts/eval_jga.sh \
  runs/t5_d0t_v1/final \
  data_unified/multiwoz24/test.jsonl

echo
echo "=========================================="
echo "Bootstrap finished successfully"
echo "=========================================="