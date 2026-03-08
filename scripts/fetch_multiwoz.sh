#!/usr/bin/env bash
set -euo pipefail

mkdir -p data_raw
cd data_raw

if [ ! -d "MultiWOZ2.4" ]; then
  git clone https://github.com/smartyfh/MultiWOZ2.4.git
fi

cd MultiWOZ2.4/data

if [ ! -d "MULTIWOZ2.4/MULTIWOZ2.4" ]; then
  unzip -q MULTIWOZ2.4.zip -d MULTIWOZ2.4
fi

echo "MultiWOZ 2.4 ready at:"
echo "data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/"