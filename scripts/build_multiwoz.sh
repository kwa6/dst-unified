#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m dst.data.multiwoz_adapter

echo "Built:"
echo "data_unified/multiwoz24/train.jsonl"
echo "data_unified/multiwoz24/val.jsonl"
echo "data_unified/multiwoz24/test.jsonl"