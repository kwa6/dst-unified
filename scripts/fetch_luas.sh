#!/usr/bin/env bash
set -euo pipefail

mkdir -p data_raw

if [ ! -d "data_raw/luas_repo" ]; then
  git lfs install
  git clone https://github.com/ParticleMedia/LUAS.git data_raw/luas_repo
fi

cd data_raw/luas_repo
git lfs pull --include="generation/multiwoz/datas/multiwoz.json"

echo "LUAS data ready at:"
echo "data_raw/luas_repo/generation/multiwoz/datas/multiwoz.json"