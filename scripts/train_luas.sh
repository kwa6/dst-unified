#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m dst.runners.train_t5_balanced \
  --train_path data_unified/luas/train.jsonl \
  --total_examples 4000 \
  --steps 1000 \
  --out_dir runs/t5_luas_v1