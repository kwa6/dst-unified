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

# Starting pipeline
run_step bash scripts/pipeline.sh
  
