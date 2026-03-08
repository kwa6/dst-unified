#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m dst.runners.build_luas_schema
python -m dst.data.luas_adapter

echo "Built:"
echo "schemas/luas_slots.json"
echo "data_unified/luas/train.jsonl"