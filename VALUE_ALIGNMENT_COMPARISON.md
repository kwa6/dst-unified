# Value Alignment Comparison: Raw vs Unified EDA by Dataset

## Executive Summary

**Unified EDA now groups results by dataset**, allowing direct comparison of D0T, LUAS, and MultiWOZ alignment metrics:

| Dataset | Raw Exact | Unified Exact | Improvement |
|---------|-----------|---------------|------------|
| **MultiWOZ** | 82.98% | 97.22% | +14.2% |
| **D0T** | 0.16% | 48.31% | +48.2% ⭐ |
| **LUAS** | 79.24% | 86.42% | +7.2% |

**Key Finding**: Unification dramatically transforms D0T (0.16% → 48%), moderately improves MultiWOZ (82.98% → 97.22%), and incrementally improves LUAS (79.24% → 86.42%).

---

## Dataset-Level Comparison

### MultiWOZ 2.4

### MultiWOZ 2.4

| Metric | Raw | Unified |
|--------|-----|---------|
| **Exact Alignment** | 82.98% | 97.22% |
| **Normalized Alignment** | 0.44% | 0.00% |
| **Not Aligned** | 16.59% | 2.78% |
| **Examples** | 294,484 filled | 65,885 non-none |
| **Improvement** | - | +14.2% exact |

**Analysis**: 
- MultiWOZ shows dramatic improvement from raw to unified (82.98% → 97.22%)
- Normalized category disappears in unified (0.44% → 0.00%), suggesting exact pre-matching during unification
- Raw: 16.59% unaligned → Unified: 2.78% unaligned (-13.8%)

---

### D0T / DSG5K

| Metric | Raw | Unified |
|--------|-----|---------|
| **Exact Alignment** | 0.16% | 48.31% |
| **Normalized Alignment** | 39.19% | 0.75% |
| **Not Aligned** | 60.65% | 50.93% |
| **Examples** | 325,002 filled | 249,037 non-none |
| **Improvement** | - | +48.2% exact ⭐ |

**Analysis**:
- D0T undergoes **massive transformation** in unified processing
- Raw exact (0.16%) → Unified exact (48.31%): **+48.2% improvement**
- Raw normalized (39.19%) → Unified normalized (0.75%): -38.4%
  - This suggests unified processing converts most "normalized-matching" cases into "exact-matching"
  - Likely through value normalization/canonicalization
- D0T's semantic concepts (e.g., "budget", "decision") become dialogue-aligned through this process

---

### LUAS

| Metric | Raw | Unified |
|--------|-----|---------|
| **Exact Alignment** | 79.24% | 86.42% |
| **Normalized Alignment** | 1.24% | 1.21% |
| **Not Aligned** | 19.51% | 12.37% |
| **Examples** | 278,794 filled | 278,794 non-none |
| **Improvement** | - | +7.2% exact |

**Analysis**:
- LUAS shows modest improvement: 79.24% → 86.42% (+7.2%)
- Normalized category remains stable (1.24% → 1.21%)
- Not-aligned reduced from 19.51% to 12.37% (-7.1%)
- Suggests LUAS values were already well-aligned in raw data

---

## Overall Unified Statistics (All Datasets Combined)

### Unified Train Split - All Datasets

```
TOTAL:
  Total Examples:        887,725
  Total Dialogues:       20,994 (8,438 MultiWOZ + 5,011 D0T + 7,556 LUAS)
  None-like values:      294,009 (33.1%)
  Non-none examples:     593,716 (66.9%)
  
ALIGNMENT:
  EXACT alignment:       425,296 (71.63% of non-none)
  NORMALIZED alignment:  430,548 (72.52% of non-none)
  NOT_ALIGNED:           163,168 (27.48% of non-none)
```

### Unified Train Split - By Dataset Breakdown

```
MultiWOZ:
  Examples:        283,905 (31.9% of total)
  Non-none:         65,885
  Exact:         64,054 (97.22%)
  Not-aligned:    1,831 (2.78%)

D0T:
  Examples:        325,002 (36.6% of total)
  Non-none:       249,037
  Exact:        120,320 (48.31%)
  Not-aligned:  126,845 (50.93%)

LUAS:
  Examples:        278,818 (31.4% of total)
  Non-none:       278,794
  Exact:        240,922 (86.42%)
  Not-aligned:   34,492 (12.37%)
```

---

## Key Insights by Dataset

### 1. **MultiWOZ: Already Well-Aligned**
- Raw data: Natural dialogue → high exact match (83%)
- Unified processing: Further cleanup → very high match (97%)
- Interpretation: MultiWOZ annotations were already dialogue-centric; unification mainly polishes

### 2. **D0T: Major Transformation** ⭐
- Raw data: Semantic concepts → very low exact match (0.16%)
- Unified processing: Converts to observable values → moderate match (48%)
- Normalized category collapses (39% → 0.75%)
- **Critical insight**: D0T requires unification to produce usable training data
- Interpretation: Unification strategy converts D0T's abstract concepts into surface-level values that appear in dialogue

### 3. **LUAS: Incremental Improvement**
- Raw data: Good baseline (79% exact)
- Unified processing: Modest boost → 86% exact
- Normalized category stable (~1.2%)
- Interpretation: LUAS values were already well-grounded in dialogue; unification adds context normalization

---

## Processing Implications

### Why Unified Data is Better for Model Training

1. **Higher Alignment Confidence**: 
   - 71.6% of examples have exact alignment across all datasets
   - Combines 97% (MultiWOZ) + 48% (D0T) + 86% (LUAS) with weighted average
   - Models receive examples where targets are recoverable from dialogue

2. **Normalized Category Elimination**:
   - Unified removes the ambiguous "normalized" category
   - Forces binary decision: exact or not-aligned
   - Cleaner training signal

3. **Dataset Harmonization**:
   - D0T concepts mapped to dialogue surface forms
   - LUAS further refined for consistency
   - MultiWOZ already optimal

4. **Expected Model Performance**:
   - Models trained on unified data: ~72% of examples have exact context alignment
   - Remaining 27.48%: Require reasoning/inference beyond surface matching
   - Baseline expectation: 72% if model simply copies from context

---

## Dataset-Specific Recommendations

### For MultiWOZ
- Use unified data: 97% exact alignment is excellent
- Raw data 16.59% unaligned cases are likely edge cases worth studying

### For D0T
- **Critical**: Unified data is necessary
- Raw data (0.16% exact) shows D0T is fundamentally different from other datasets
- Unification converts semantic concepts to dialogue-aligned values
- Training on raw D0T would be ineffective; **always use unified**

### For LUAS
- Unified data preferred: 86% vs 79% exact alignment
- Raw data is reasonable fallback
- Values are already dialogue-grounded in both versions

---

## Technical Notes

### Raw EDA Alignment Method
- **Exact**: Value found as substring in dialogue context (case-insensitive)
- **Normalized**: Value found after punctuation removal and whitespace normalization
- **Not-aligned**: Value not found in any form

### Unified EDA Alignment Method
- Uses pre-computed `dialogue_context` field (turn-by-turn dialogue assembly)
- Same substring matching approach as raw
- Now groups results by dataset for direct comparison

### Data Coverage
- **Raw**: 23,005 total dialogues (MultiWOZ: 10,438 + D0T: 5,011 + LUAS: 7,556)
- **Unified Train**: 20,994 dialogues (MultiWOZ: 8,438 + D0T: 5,011 + LUAS: 7,556)
- **Note**: Unified train combines train split of MultiWOZ with train splits of D0T and LUAS
