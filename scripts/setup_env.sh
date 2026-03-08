#!/usr/bin/env bash
set -euo pipefail

python -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt

echo "Done. Activate with:"
echo "source .venv/bin/activate"
echo "export PYTHONPATH=src"