git clone https://github.com/kwa6/dst-unified.git
cd dst-unified

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt

mkdir -p data_raw
cd data_raw
git clone https://github.com/smartyfh/MultiWOZ2.4.git
cd MultiWOZ2.4/data
unzip MULTIWOZ2.4.zip -d MULTIWOZ2.4
cd ~/dst-unified

export PYTHONPATH=src

python -m dst.data.multiwoz_adapter
python -m dst.runners.train_t5_balanced --train_path data_unified/multiwoz24/train.jsonl --total_examples 2000 --steps 500 --out_dir runs/t5_mwoz_train_v1
python -m dst.runners.eval_jga --path data_unified/multiwoz24/val.jsonl --model runs/t5_mwoz_train_v1/final