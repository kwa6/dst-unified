#!/usr/bin/env bash
# Wrapper to analyze DST errors from eval_jga_llama.py or eval_jga.py
set -euo pipefail

if [ "$#" -lt 1 ]; then
  echo "Usage: bash scripts/analyze_errors.sh <errors.json> [--by-slot] [--output file.json] [--show-samples N]"
  exit 1
fi

export PYTHONPATH=src
python scripts/analyze_errors.py "$@"
