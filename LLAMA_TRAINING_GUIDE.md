# Llama Two-Stage Fine-Tuning Guide

This guide explains how to use two-stage fine-tuning in DST-Unified to train Llama models on dialogue state tracking.

## Overview

**Two-Stage Training Approach:**
1. **Stage 1:** Train on augmented data (LUAS or D0T) with LR=2e-4
2. **Stage 2:** Fine-tune on real data (MultiWOZ) with LR=2e-4 (same as Stage 1)

By training on augmented data first, the model learns broader patterns from diverse examples before specializing on the canonical MultiWOZ dataset.

## Requirements

```bash
export HF_TOKEN="hf_..."  # Your HuggingFace token
```

---

## Two-Stage Training

### Stage 1→2: LUAS → MultiWOZ
```bash
bash scripts/train_llama_twostage.sh luas
```

**Process:**
1. Stage 1: Train on 8000 examples from `data_unified/luas/train.jsonl` (500 steps)
2. Stage 2: Fine-tune on 8000 examples from `data_unified/multiwoz24/train.jsonl` (300 steps)

**Output:** `runs/llama_stage2_luas_mwoz_final/final`

### Stage 1→2: D0T → MultiWOZ
```bash
bash scripts/train_llama_twostage.sh d0t
```

**Process:**
1. Stage 1: Train on diverse D0T data (500 steps)
2. Stage 2: Fine-tune on MultiWOZ data (300 steps)

**Output:** `runs/llama_stage2_d0t_mwoz_final/final`

### Custom Model
```bash
bash scripts/train_llama_twostage.sh luas meta-llama/Llama-3.2-3B-Instruct
bash scripts/train_llama_twostage.sh d0t meta-llama/Llama-3.2-3B-Instruct
```

---

## Training Configuration

### Stage 1 (Augmented Data: LUAS or D0T)
- **Examples:** 8000 (balanced: 50% none / 50% non-none)
- **Learning Rate:** 2e-4
- **Warmup Steps:** 50
- **Training Steps:** 500
- **Purpose:** Learn broad patterns from augmented/diverse data

### Stage 2 (Real Data: MultiWOZ)
- **Examples:** 8000 (balanced: 50% none / 50% non-none)
- **Learning Rate:** 2e-4 (same as Stage 1)
- **Warmup Steps:** 50
- **Training Steps:** 300
- **Purpose:** Specialize on canonical real data

**Note:** Same learning rate both stages ensures equal treatment of augmented and real data patterns.

---

## Advanced: Manual Two-Stage Training

If you want more control, run stages separately:

```bash
# Stage 1: Train on LUAS
export HF_TOKEN="hf_..."
python -m dst.runners.train_llama \
    --train_path data_unified/luas/train.jsonl \
    --eval_path data_unified/luas/train_sample.jsonl \
    --stage 1 \
    --model meta-llama/Llama-2-7b-chat-hf \
    --out_dir runs/llama_stage1_luas \
    --steps 500 \
    --warmup_steps 50 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4

# Stage 2: Fine-tune on MultiWOZ (loads checkpoint from Stage 1)
python -m dst.runners.train_llama \
    --train_path data_unified/multiwoz24/train.jsonl \
    --eval_path data_unified/multiwoz24/val.jsonl \
    --stage 2 \
    --checkpoint runs/llama_stage1_luas/final \
    --out_dir runs/llama_stage2_mwoz_final \
    --steps 300 \
    --warmup_steps 50 \
    --batch_size 4 \
    --grad_accum 4 \
    --lr 2e-4
```

---

## Comparing Augmented Data Sources

| Dataset | Stage 1 Purpose | Characteristics |
|---------|-----------------|---|
| **LUAS** | LLM-backed agent simulation | Diverse dialogue moves, action-aware, synthetic |
| **D0T** | Zero-shot domain tracking | Cross-domain patterns, diverse domains, realistic |

Choose based on your goal:
- **LUAS:** Learning diverse dialogue patterns and action selection
- **D0T:** Learning cross-domain generalization

---

## Evaluation

After training, evaluate on MultiWOZ test set:

```bash
python -m dst.runners.eval_jga_llama \
    --model runs/llama_stage2_luas_mwoz_final/final \
    --path data_unified/multiwoz24/test.jsonl
```

Or with D0T:

```bash
python -m dst.runners.eval_jga_llama \
    --model runs/llama_stage2_d0t_mwoz_final/final \
    --path data_unified/multiwoz24/test.jsonl
```

---

## Training Timeline

**LUAS → MultiWOZ:**
- Stage 1: ~30-60 minutes (depending on GPU)
- Stage 2: ~20-40 minutes
- **Total:** ~50-100 minutes on A100 GPU

**D0T → MultiWOZ:**
- Similar timeline as LUAS

---

## Tips & Tricks

1. **Monitor GPU memory:** Each stage uses similar VRAM; no need to worry about memory doubling
2. **Same learning rate:** Both stages use LR=2e-4 to treat augmented and real data equally
3. **Shorter stage 1 warmup:** Auto-adjusted to 50 steps each stage
4. **Progressive adaptation:** Model learns patterns in Stage 1, specializes in Stage 2
5. **Checkpoints:** Each stage saved separately; Stage 1 becomes input for Stage 2

---

## Files

- `scripts/train_llama_twostage.sh` - Main two-stage training script
- `src/dst/runners/train_llama.py` - Core training code (supports both stages)
- `LLAMA_TRAINING_GUIDE.md` - This file

---

## Deprecated: Single-Stage Training

Single-stage training scripts have been deprecated in favor of two-stage training:
- ~~`train_llama.sh`~~ → Use `train_llama_twostage.sh luas` or `luas/d0t` instead  
- ~~`train_llama_luas.sh`~~ → Use `train_llama_twostage.sh luas`
- ~~`train_llama_d0t.sh`~~ → Use `train_llama_twostage.sh d0t`
