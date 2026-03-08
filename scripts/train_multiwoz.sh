#!/usr/bin/env bash
set -euo pipefail

export PYTHONPATH=src
python -m dst.runners.train_t5_balanced \
  --train_path data_unified/multiwoz24/train.jsonl \
  --total_examples 2000 \
  --steps 500 \
  --out_dir runs/t5_mwoz_train_v1