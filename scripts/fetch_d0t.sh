#!/usr/bin/env bash
set -euo pipefail

mkdir -p data_raw/d0t
cd data_raw/d0t

if [ ! -d ".git" ]; then
  git init
  git remote add origin https://github.com/emorynlp/Diverse0ShotTracking.git
  git config core.sparseCheckout true
  echo "data/dsg5k/*" > .git/info/sparse-checkout
  git pull origin main || git pull origin master
else
  git pull origin main || git pull origin master
fi

echo "D0T data ready under:"
echo "data_raw/d0t/data/dsg5k/train/"