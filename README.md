# dst-unified

A unified research pipeline for Dialogue State Tracking (DST).

This project converts different DST datasets into a shared slot-centric JSONL format, trains a model on that unified format, and evaluates with standard DST metrics such as Joint Goal Accuracy (JGA).

Current support:
- MultiWOZ 2.4
- D0T / DSG5K
- LUAS
- Flan-T5 training and inference
- slot-level exact match
- JGA evaluation

## Core idea

We reformulate DST as:

(dialogue context + slot name + slot description) -> slot value

This makes it possible to compare different datasets under the same:
- model
- prompt
- training loop
- evaluation pipeline

## Repository structure

```text
dst-unified/
├── configs/
├── schemas/
├── scripts/
├── src/
│   └── dst/
│       ├── data/
│       ├── models/
│       ├── runners/
│       └── schemas.py
├── requirements.txt
└── README.md