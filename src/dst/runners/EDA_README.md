# EDA Scripts for DST Unified

Three reusable EDA scripts for analyzing raw and unified DST data.

## Raw Data EDA: `eda_raw.py` (Unified for all datasets)

Analyzes raw data in their native formats for **MultiWOZ 2.4**, **D0T/DSG5K**, and **LUAS**.

### Basic usage:
```bash
cd src && python -m dst.runners.eda_raw --dataset multiwoz
cd src && python -m dst.runners.eda_raw --dataset d0t
cd src && python -m dst.runners.eda_raw --dataset luas
```

### With CSV export:
```bash
cd src && python -m dst.runners.eda_raw --dataset multiwoz --csv-prefix ../eda_raw_output
cd src && python -m dst.runners.eda_raw --dataset d0t --csv-prefix ../eda_raw_output
cd src && python -m dst.runners.eda_raw --dataset luas --csv-prefix ../eda_raw_output
```

### Dataset-specific configuration:

**MultiWOZ 2.4** (default paths):
```bash
python -m dst.runners.eda_raw --dataset multiwoz \
  --data-path data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/data.json \
  --val-path data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/valListFile.json \
  --test-path data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/testListFile.json
```

**D0T/DSG5K** (CSV-based, needs fetch first):
```bash
# First: fetch D0T raw data
# scripts/fetch_d0t.sh

# Then: analyze
python -m dst.runners.eda_raw --dataset d0t \
  --d0t-dir data_raw/d0t/data/dsg5k/train
```

**LUAS** (JSON-based, needs fetch first):
```bash
# First: fetch LUAS raw data
# scripts/fetch_luas.sh

# Then: analyze
python -m dst.runners.eda_raw --dataset luas \
  --luas-json data_raw/luas_repo/generation/multiwoz/datas/multiwoz.json
```

### Output:
- Console: statistics on dialogues, turns, domains, slot distribution, label imbalance (none/dontcare/filled)
- CSV files (if `--csv-prefix` is provided):
  - `{prefix}_{dataset}_summary.csv`: Key metrics (dialogue count, turn stats, label distribution)
  - `{prefix}_{dataset}_slot_stats.csv`: Per-slot observed/none/dontcare/total counts
  - `{prefix}_{dataset}_top_values.csv`: Top 10 values per slot

### Example output snapshot (MultiWOZ):
```
Dialogues total: 10438
  train=8438, val=1000, test=1000

Slot states total: 1,905,583
  none-like:    1,598,943 (83.91%)
  dontcare:        12,156 (0.64%)
  filled:         306,640 (16.09%)

Top filled slots: train-destination (21,411), restaurant-food (21,037), ...
```

---

## Legacy Raw Data EDA: `eda_raw_multiwoz.py`

(Deprecated: use `eda_raw.py --dataset multiwoz` instead)

MultiWOZ-specific EDA script. Kept for backward compatibility.

---

## Unified Data EDA: `eda_unified.py`

Analyzes UnifiedDSTExample JSONL files from `data_unified/multiwoz24/*.jsonl`.

### Basic usage (train split):
```bash
cd src && python -m dst.runners.eda_unified --split train
```

### With CSV export and row limit:
```bash
cd src && python -m dst.runners.eda_unified --split train \
  --csv-prefix ../eda_unified_output --limit 100000
```

### Available splits:
```bash
python -m dst.runners.eda_unified --split train
python -m dst.runners.eda_unified --split val
python -m dst.runners.eda_unified --split test
```

### Output:
- Console: examples count, dialogues, per-dialogue example count, slot coverage, target value distribution, context length stats, **target value alignment metrics**
- CSV files (if `--csv-prefix` is provided):
  - `{prefix}_{split}_summary.csv`: Key metrics + alignment metrics
  - `{prefix}_{split}_slot_coverage.csv`: Per-slot example count and dataset/split tags
  - `{prefix}_{split}_target_values.csv`: Top 10 target values per slot

### Example output snapshot:
```
Examples: 100,000 (from 2,996 dialogues)
Examples per dialogue (avg): 33.4

Target distribution:
  none:      75,295 (75.30%)
  dontcare:   1,825 (1.82%)
  filled:    22,880 (22.88%)

Top target values: none (75,295), centre (3,360), moderate (3,198), ...

================================================================================
TARGET VALUE ALIGNMENT (non-'none' targets only)
================================================================================

Turn-level alignment (substring match in context):
     21,430 /  24,705 examples (86.73%)

Dialogue-level alignment (≥1 target value in context):
      2,987 /   2,996 dialogues (99.70%)

Alignment by dataset:
  multiwoz24:
    Turn-level:        21,430 /  24,705 (86.73%)
    Dialogue-level:     2,987 /   2,996 (99.70%)
```

---

## Advanced Options

### `eda_raw.py`:
```bash
# MultiWOZ
python -m dst.runners.eda_raw --dataset multiwoz \
  --data-path <path> \
  --val-path <path> \
  --test-path <path> \
  --csv-prefix <output_prefix>

# D0T
python -m dst.runners.eda_raw --dataset d0t \
  --d0t-dir <path/to/turn.csv/slot.csv/slot_value.csv> \
  --csv-prefix <output_prefix>

# LUAS
python -m dst.runners.eda_raw --dataset luas \
  --luas-json <path/to/multiwoz.json> \
  --csv-prefix <output_prefix>
```

### `eda_unified.py`:
```bash
python -m dst.runners.eda_unified \
  --split train|val|test \
  --dataset multiwoz24 \
  --base-dir data_unified \
  --csv-prefix <output_prefix> \
  --limit <rows>
```

---

## Integration with Pipelines

### Run full EDA pipeline:
```bash
# Raw data EDA (all datasets)
export PYTHONPATH=src
python -m dst.runners.eda_raw --dataset multiwoz --csv-prefix eda_raw
python -m dst.runners.eda_raw --dataset d0t --csv-prefix eda_raw
python -m dst.runners.eda_raw --dataset luas --csv-prefix eda_raw

# Build unified data
python -m dst.data.multiwoz_adapter
python -m dst.data.d0t_adapter
python -m dst.data.luas_adapter

# Unified EDA (all splits, all datasets implicit)
python -m dst.runners.eda_unified --split train --csv-prefix eda_unified
python -m dst.runners.eda_unified --split val --csv-prefix eda_unified
python -m dst.runners.eda_unified --split test --csv-prefix eda_unified
```

---

## Key Metrics Tracked

### Raw EDA:
- **Dialogue stats**: Total, split breakdown, turns per dialogue quantiles
- **Domain coverage**: Turn presence, dialogue presence
- **Slot distribution**: Filled vs. none-like vs. dontcare (percentage)
- **Slot examples**: Top values with frequency counts

### Unified EDA:
- **Example density**: Examples per dialogue
- **Target distribution**: none vs. dontcare vs. filled (percentage)
- **Slot coverage**: Which slots appear, in which datasets/splits
- **Context length**: Character and turn count distribution
- **Target Value Alignment** (case-insensitive substring matching, non-'none' only):
  - **Turn-level %**: How many examples have the target value appearing in the dialogue context (per turn)
  - **Dialogue-level %**: How many dialogues have at least one target value appearing in the context
  - **By-dataset breakdown**: Separate alignment metrics for multiwoz24, d0t, luas

---

## Re-running After Data Changes

Both scripts are designed to be re-run safely. Simply re-execute with the same (or new) CSV prefix:

```bash
# Quick check on raw data
python -m dst.runners.eda_raw --dataset multiwoz

# Detailed export after modifications
python -m dst.runners.eda_raw --dataset multiwoz --csv-prefix eda_raw_v2
```

CSV exports are idempotent — running again overwrites previous files.

---

## Comparing Datasets

To compare raw characteristics across all three datasets:

```bash
export PYTHONPATH=src
python -m dst.runners.eda_raw --dataset multiwoz --csv-prefix eda_comparison
python -m dst.runners.eda_raw --dataset d0t --csv-prefix eda_comparison
python -m dst.runners.eda_raw --dataset luas --csv-prefix eda_comparison

# CSV files will be:
#   eda_comparison_multiwoz_summary.csv
#   eda_comparison_d0t_summary.csv
#   eda_comparison_luas_summary.csv
#
# Use your favorite spreadsheet tool or pandas to load and compare:
# import pandas as pd
# mw = pd.read_csv('eda_comparison_multiwoz_summary.csv')
# d0t = pd.read_csv('eda_comparison_d0t_summary.csv')
# luas = pd.read_csv('eda_comparison_luas_summary.csv')
```

Key metrics to compare across datasets:
- **dialogues_total**: Size of dataset
- **turns_per_dialogue_median**: Dialogue length distribution
- **slot_states_none_like_pct**: Label imbalance (higher = more sparse annotations)
- **slot_states_filled_pct**: Annotation density
- Unique slots per dataset (check slot_stats CSV)

---

## Target Value Alignment Metric (Unified EDA only)

**What it measures**: How often does the target value (slot value) appear directly in the dialogue context?

**Methodology**:
- **Substring matching**: Case-insensitive search for target value in dialogue_context
- **Filtered scope**: Only non-'none' target values (filled + dontcare) are measured
- **Turn-level**: Percentage of examples where target appears in context, grouped by turn
- **Dialogue-level**: Percentage of dialogues where at least one target value appears, grouped by dialogue

**Why it matters**: 
- High alignment suggests target values are grounded in dialogue context (good for training)
- Low alignment might indicate hallucinated values or out-of-vocabulary targets
- Differences across datasets (multiwoz24 vs. d0t vs. luas) reveal dataset characteristics

**Example interpretation**:
- 86.73% turn-level alignment: In 86.73% of non-'none' examples, the target value appears in the dialogue
- 99.70% dialogue-level alignment: In 99.70% of dialogues, at least one target value is mentioned somewhere
- By-dataset view: Compare alignment rates across multiwoz24, d0t, luas to identify data quality issues per dataset
