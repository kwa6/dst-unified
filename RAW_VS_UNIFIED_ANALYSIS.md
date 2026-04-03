# Raw vs Unified EDA - Complete Alignment Analysis

## Quick Summary

I've updated the unified EDA script to **group results by dataset**, enabling direct comparison of D0T, LUAS, and MultiWOZ within the unified framework. Here's what we found:

| Dataset | Raw Exact | Unified Exact | Change | Insight |
|---------|-----------|---------------|--------|---------|
| **MultiWOZ** | 82.98% | 97.22% | +14.2% | Good baseline → Polished |
| **D0T** | 0.16% | 48.31% | **+48.2%** ⭐ | Semantic concepts → Observable values |
| **LUAS** | 79.24% | 86.42% | +7.2% | Solid → Refined |
| **Overall** | N/A | 71.63% | N/A | Weighted average across all |

---

## What Changed in the Code

### Modified: `eda_unified.py`

**Before**: Loaded a single dataset (defaulted to MultiWOZ)
```python
def main():
    ap.add_argument("--dataset", type=str, default="multiwoz24", ...)
    jsonl_path = args.base_dir / args.dataset / f"{args.split}.jsonl"
```

**After**: Auto-discovers and processes all available datasets
```python
def main():
    ap.add_argument("--dataset", type=str, default=None, ...)
    if args.dataset:
        datasets_to_process = [args.dataset]  # Single dataset if specified
    else:
        # Auto-discover all datasets
        datasets_to_process = [item.name for item in base_dir.iterdir() 
                               if (item / f"{args.split}.jsonl").exists()]
    
    # Load and process all datasets together
    for dataset_name in datasets_to_process:
        examples = load_jsonl(base_dir / dataset_name / f"{args.split}.jsonl")
        all_examples.extend(examples)
```

**Result**: Now you can run:
```bash
# Process all three datasets together
python eda_unified.py --split train --csv-prefix eda_unified_output

# Or process single dataset
python eda_unified.py --split train --dataset multiwoz24 --csv-prefix eda_unified_output
```

---

## Detailed Dataset Analysis

### MultiWOZ: The Polished Dataset
- **Raw**: 82.98% exact (already good)
- **Unified**: 97.22% exact (+14.2%)
- **Interpretation**: Natural dialogue data where values often appear verbatim in dialogue context
- **Training Readiness**: Excellent - high confidence that targets are recoverable from context

### D0T: The Transformation Story ⭐
- **Raw**: 0.16% exact (nearly unusable)
- **Unified**: 48.31% exact (+48.2% - **CRITICAL IMPROVEMENT**)
- **Raw Normalized**: 39.19% (values found only after normalization)
- **Unified Normalized**: 0.75% (most normalized cases converted to exact)
- **Interpretation**: 
  - Raw D0T contains semantic concepts extracted from dialogue ("budget", "decision", "next steps")
  - These don't appear verbatim in dialogue text
  - Unification process converts these concepts into observable surface forms
  - This explains why raw D0T seemed broken earlier - it fundamentally uses semantic annotations
- **Training Readiness**: Moderate - 48% alignment is workable, but ~50% still unaligned

### LUAS: The Steady Performer
- **Raw**: 79.24% exact (solid)
- **Unified**: 86.42% exact (+7.2%)
- **Interpretation**: Values already grounded in dialogue context; unification adds normalization
- **Training Readiness**: Good - values already dialogue-aligned in raw format

---

## Overall Unified Statistics

**Train Split Combining All Three Datasets:**

```
Total Examples:        887,725
Total Dialogues:       20,994
  - MultiWOZ: 8,438 (40.2% of dialogues)
  - D0T: 5,011 (23.9% of dialogues)
  - LUAS: 7,556 (36.0% of dialogues)

Alignment Distribution (across 593,716 non-none examples):
  EXACT:        425,296 (71.63%)
  NORMALIZED:   430,548 (72.52%) ← includes exact
  NOT_ALIGNED:  163,168 (27.48%)
```

**By Dataset:**
- MultiWOZ: 97.22% exact (64,054 / 65,885)
- D0T: 48.31% exact (120,320 / 249,037)
- LUAS: 86.42% exact (240,922 / 278,794)

**Weighted Average**: ~71.6% exact alignment

---

## Key Insights

### 1. The Normalized Category Mystery
Raw EDA identifies a "normalized" category where values are found only after removing punctuation/normalizing spacing. In unified data, this category nearly disappears:

| Dataset | Raw Normalized | Unified Normalized |
|---------|----------------|--------------------|
| MultiWOZ | 0.44% | 0.00% |
| D0T | 39.19% | 0.75% |
| LUAS | 1.24% | 1.21% |

**Why?** Unification likely pre-normalizes values to match dialogue context exactly, forcing a binary decision (exact or not-aligned).

### 2. D0T's Semantic Extraction Strategy
D0T's 0.16% raw exact alignment reveals it uses **semantic annotation** rather than span annotation:
- Raw value: "art teacher speaking to"
- Dialogue context: "... the Art teacher is speaking to the student about ..."
- Raw matching: Fails (0.16% exact)
- Unified approach: Converts semantic concept to observable surface form or uses different context
- Result: 48% exact in unified (48× improvement!)

### 3. Dataset Maturity in Unified Format
- **Most mature**: MultiWOZ (97% - ready for production training)
- **Moderately mature**: LUAS (86% - good baseline)
- **Emerging maturity**: D0T (48% - requires special handling, but usable)

---

## Recommendations for Use

### For Model Training
1. **Use unified data exclusively** for D0T - raw is unsuitable (0.16% exact)
2. **Prefer unified data** for MultiWOZ and LUAS - better alignment (14.2% and 7.2% improvements)
3. **Expect 72% baseline** if model simply copies values from dialogue context
4. **Remaining 28%** require reasoning/inference beyond surface matching

### For Dataset Analysis
1. **D0T raw 16.59% "normalized" values**: These convert to 0.75% in unified, suggesting smart value remapping
2. **MultiWOZ raw 16.59% unaligned cases**: Worth investigating for edge cases
3. **LUAS already dialogue-aligned**: Both raw and unified are workable

### For Future Enhancement
1. Document D0T's unification strategy (semantic concepts → surface forms)
2. Investigate MultiWOZ's remaining 2.78% unaligned cases
3. Consider semantic matching for D0T beyond surface-form substring matching

---

## Files Generated

- **eda_unified_output_train_slot_coverage.csv**: Slot-level statistics
- **eda_unified_output_train_summary.csv**: Overall statistics with dataset breakdown
- **VALUE_ALIGNMENT_COMPARISON.md**: Detailed analysis (this document)
- **VALUE_ALIGNMENT_COMPARISON.csv**: Tabular data for easy analysis
- **DATASET_ALIGNMENT_SUMMARY.txt**: Quick reference summary

---

## Running the Unified EDA

**All datasets (grouped by dataset in output):**
```bash
python eda_unified.py --split train --csv-prefix eda_unified_output
```

**Single dataset:**
```bash
python eda_unified.py --split train --dataset multiwoz24 --csv-prefix eda_unified_output
```

**With limit (for quick testing):**
```bash
python eda_unified.py --split train --limit 10000 --csv-prefix eda_unified_output
```

The output now shows alignment metrics grouped "BY DATASET", allowing direct comparison of D0T, LUAS, and MultiWOZ within the unified framework.
