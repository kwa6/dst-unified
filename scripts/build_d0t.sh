#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m dst.data.d0t_adapter

echo "Built:"
echo "data_unified/d0t/train.jsonl"