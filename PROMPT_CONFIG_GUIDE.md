# Prompt Configuration Guide

Configure when to use slot descriptions and value examples in training and evaluation.

## Summary

| Scenario | Dataset | Description | Examples | Command Flag |
|----------|---------|-------------|----------|--------------|
| **Training** | D0T | ✅ ON | ✅ ON | `--use_slot_description --use_value_examples` |
| **Training** | MultiWOZ | ❌ OFF | ❌ OFF | (no flags) |
| **Training** | LUAS | ❌ OFF | ❌ OFF | (no flags) |
| **Eval** | D0T on MultiWOZ | ✅ ON | ❌ OFF | `--use_slot_description` |
| **Eval** | LUAS on MultiWOZ | ❌ OFF | ❌ OFF | (no flags) |

---

## Training Commands

### D0T Training (with descriptions and examples ON)

```bash
python -m dst.runners.train_t5_balanced \
    --train_path data_unified/d0t/train.jsonl \
    --out_dir runs/t5_d0t_with_desc_examples \
    --use_slot_description \
    --use_value_examples

# OR for Llama:
python -m dst.runners.train_llama \
    --train_path data_unified/d0t/train.jsonl \
    --out_dir runs/llama_d0t_with_desc_examples \
    --use_slot_description \
    --use_value_examples
```

### MultiWOZ Training (default: descriptions and examples OFF)

```bash
python -m dst.runners.train_t5_balanced \
    --train_path data_unified/multiwoz24/train.jsonl \
    --out_dir runs/t5_multiwoz_minimal

# OR for Llama:
python -m dst.runners.train_llama \
    --train_path data_unified/multiwoz24/train.jsonl \
    --out_dir runs/llama_multiwoz_minimal
```

### LUAS Training (default: descriptions and examples OFF)

```bash
python -m dst.runners.train_t5_balanced \
    --train_path data_unified/luas/train.jsonl \
    --out_dir runs/t5_luas_minimal

# OR for Llama:
python -m dst.runners.train_llama \
    --train_path data_unified/luas/train.jsonl \
    --out_dir runs/llama_luas_minimal
```

---

## Evaluation Commands

### Evaluate D0T on MultiWOZ (descriptions ON, examples OFF)

This tests cross-dataset generalization with domain hints but no distribution mismatch from examples.

```bash
# T5:
python -m dst.runners.eval_jga \
    --path data_unified/multiwoz24/val.jsonl \
    --model runs/t5_d0t_with_desc_examples/final \
    --use_slot_description

# Llama:
python -m dst.runners.eval_jga_llama \
    --path data_unified/multiwoz24/val.jsonl \
    --model runs/llama_d0t_with_desc_examples/final \
    --use_slot_description

# Qwen:
python -m dst.runners.eval_jga_qwen \
    --path data_unified/multiwoz24/val.jsonl \
    --model runs/qwen_d0t_with_desc_examples/final \
    --use_slot_description
```

### Evaluate LUAS on MultiWOZ (no descriptions, no examples)

This tests pure generalization without any extra context.

```bash
# T5:
python -m dst.runners.eval_jga \
    --path data_unified/multiwoz24/val.jsonl \
    --model runs/t5_luas_minimal/final

# Llama:
python -m dst.runners.eval_jga_llama \
    --path data_unified/multiwoz24/val.jsonl \
    --model runs/llama_luas_minimal/final

# Qwen:
python -m dst.runners.eval_jga_qwen \
    --path data_unified/multiwoz24/val.jsonl \
    --model runs/qwen_luas_minimal/final
```

### Evaluate Model on Source Dataset (baseline)

When evaluating on the same dataset used for training, match the training config:

```bash
# D0T model on D0T data:
python -m dst.runners.eval_jga \
    --path data_unified/d0t/train.jsonl \
    --model runs/t5_d0t_with_desc_examples/final \
    --use_slot_description --use_value_examples

# MultiWOZ model on MultiWOZ data:
python -m dst.runners.eval_jga \
    --path data_unified/multiwoz24/val.jsonl \
    --model runs/t5_multiwoz_minimal/final
```

---

## Why These Settings?

### D0T: Both ON
- **Descriptions**: D0T's schema provides rich descriptions for each slot
- **Examples**: D0T has value_candidate.csv with real examples from data
- **Rationale**: Leverage all available signal from D0T's curated data

### MultiWOZ: Both OFF
- **Descriptions**: Already part of standard prompting, tested thoroughly
- **Examples**: MultiWOZ doesn't have explicit value candidates in schema
- **Rationale**: Efficiency-first approach; descriptions/examples OFF by default

### LUAS: Both OFF
- **Descriptions**: Available in schema but kept minimal for efficiency
- **Examples**: Available in schema but tested to work better without them
- **Rationale**: Consistency with minimal baseline; focus on task understanding

### D0T → MultiWOZ Eval: Desc ON, Examples OFF
- **Descriptions**: Help model understand domain shift (MultiWOZ context)
- **Examples**: Omit, since model wasn't trained on MultiWOZ examples
  - Prevents distribution mismatch
  - Tests pure cross-dataset generalization
  - Cleaner signal: did the model learn robust reasoning?

### LUAS → MultiWOZ Eval: Both OFF
- **Rationale**: Maximal domain shift test; minimal guidance
- **Alternative**: Could try `--use_slot_description` to see if it helps bridge domains

---

## Shell Scripts (Convenient Pre-Configured)

The training scripts in `scripts/` have been updated with the proper flags built-in:

### T5 Training (Direct)

```bash
# D0T (descriptions and examples ON)
bash scripts/train_d0t.sh

# MultiWOZ (descriptions and examples OFF)
bash scripts/train_multiwoz.sh

# LUAS (descriptions and examples OFF)
bash scripts/train_luas.sh
```

### T5 Two-Stage Training (Augmented → Real Data)

```bash
# Stage 1: D0T (with desc/examples) → Stage 2: MultiWOZ (without)
bash scripts/train_t5_twostage.sh d0t

# Stage 1: LUAS (without desc/examples) → Stage 2: MultiWOZ (without)
bash scripts/train_t5_twostage.sh luas

# Same as above but with raw (unbalanced) data
bash scripts/train_t5_raw_twostage.sh d0t
bash scripts/train_t5_raw_twostage.sh luas
```

### Llama Training (Direct)

```bash
export HF_TOKEN=hf_XXXXXXXXXXXX

# Llama 3.1 8B
bash scripts/train_llama31_8b_d0t.sh    # D0T with desc/examples
bash scripts/train_llama31_8b_luas.sh   # LUAS without desc/examples

# Llama 3.3 70B
bash scripts/train_llama33_70b_d0t.sh   # D0T with desc/examples
bash scripts/train_llama33_70b_luas.sh  # LUAS without desc/examples
```

### Llama Two-Stage Training

```bash
export HF_TOKEN=hf_XXXXXXXXXXXX

# Llama 2 generic (configurable model)
bash scripts/train_llama_twostage.sh d0t                           # D0T → MultiWOZ
bash scripts/train_llama_twostage.sh d0t meta-llama/Llama-3.1-8B-Instruct

# Llama 3.1 8B specific
bash scripts/train_llama31_8b_twostage.sh d0t
bash scripts/train_llama31_8b_twostage.sh luas

# Llama 3.3 70B specific
bash scripts/train_llama33_70b_twostage.sh d0t
bash scripts/train_llama33_70b_twostage.sh luas
```

---

## Evaluation with Shell Scripts

The `eval_jga.sh` and `eval_jga_llama.sh` scripts are flexible wrappers. Pass flags as extra arguments:

```bash
# D0T model on MultiWOZ (with descriptions only)
bash scripts/eval_jga.sh \
    runs/t5_d0t_v1/final \
    data_unified/multiwoz24/val.jsonl \
    --use_slot_description

# Llama D0T model on MultiWOZ (with descriptions only)
bash scripts/eval_jga_llama.sh \
    runs/llama31_8b_d0t_v2/final \
    data_unified/multiwoz24/val.jsonl \
    --use_slot_description

# LUAS model on MultiWOZ (no descriptions/examples)
bash scripts/eval_jga.sh \
    runs/t5_stage2_luas_mwoz_final/final \
    data_unified/multiwoz24/test.jsonl
```

---

**What gets included in the prompt:**

```
Dialogue Context (always included):
  Turn X [Speaker]: Text

Slot Name (always included):
  slot-name

[Description - if --use_slot_description flag is set]:
  The slot represents...

[Examples - if --use_value_examples flag is set]:
  value1, value2, value3, ...

Instructions (always included):
  - Extract the exact slot value...
  - Use only the most recent value...
  [etc.]
```

---

## Debugging: Check What's in Your Training Data

```bash
# Inspect first example with all components:
python -m dst.runners.inspect_prompt \
    --path data_unified/d0t/train.jsonl \
    --index 0 \
    --use_slot_description \
    --use_value_examples

# Compare minimal vs enhanced:
echo "=== Minimal (default) ==="
python -m dst.runners.inspect_prompt --path data_unified/d0t/train.jsonl --index 0

echo ""
echo "=== With descriptions ==="
python -m dst.runners.inspect_prompt --path data_unified/d0t/train.jsonl --index 0 --use_slot_description

echo ""
echo "=== With examples ==="
python -m dst.runners.inspect_prompt --path data_unified/d0t/train.jsonl --index 0 --use_value_examples

echo ""
echo "=== With both ==="
python -m dst.runners.inspect_prompt --path data_unified/d0t/train.jsonl --index 0 --use_slot_description --use_value_examples
```
