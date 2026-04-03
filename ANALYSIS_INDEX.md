# EDA Analysis Index

## Quick Start

**Start here**: [RAW_VS_UNIFIED_ANALYSIS.md](RAW_VS_UNIFIED_ANALYSIS.md)
- Complete guide showing raw vs unified alignment by dataset
- Explains why D0T (0.16% raw) transforms to 48.31% unified
- Contains training recommendations and usage instructions

## Analysis Documents

### Comparison Analysis
1. **[RAW_VS_UNIFIED_ANALYSIS.md](RAW_VS_UNIFIED_ANALYSIS.md)** (6.8 KB)
   - Main analysis document
   - Code changes explained
   - Dataset-specific insights
   - Training recommendations
   - Usage instructions for unified EDA

2. **[VALUE_ALIGNMENT_COMPARISON.md](VALUE_ALIGNMENT_COMPARISON.md)** (6.6 KB)
   - Detailed per-dataset alignment metrics
   - Overall unified statistics
   - Key observations and insights
   - Dataset-specific recommendations
   - Technical notes on methodology

3. **[DATASET_ALIGNMENT_SUMMARY.txt](DATASET_ALIGNMENT_SUMMARY.txt)** (5.3 KB)
   - Quick reference summary
   - Side-by-side statistics
   - Dataset characteristics
   - Training readiness assessment
   - Text-based format for easy viewing

### Supporting Documentation
4. **[VALUE_ALIGNMENT_DEFINITION.md](VALUE_ALIGNMENT_DEFINITION.md)**
   - Alignment measurement methodology
   - Category definitions (exact/normalized/not-aligned)
   - How dialogue context is computed

5. **[VALUE_ALIGNMENT_ANALYSIS.md](VALUE_ALIGNMENT_ANALYSIS.md)**
   - Original framework documentation

## Data Files

### CSV Data (for analysis/plotting)
- **[VALUE_ALIGNMENT_COMPARISON.csv](VALUE_ALIGNMENT_COMPARISON.csv)**
  - Tabular format with all metrics
  - Suitable for Excel, Pandas, plotting tools

### Generated EDA Output
- `eda_unified_output_train_summary.csv` - Overall statistics with dataset breakdown
- `eda_unified_output_train_slot_coverage.csv` - Per-slot statistics
- `eda_unified_output_train_target_values.csv` - Value distributions

## Key Findings Summary

### Value Alignment by Dataset

| Dataset | Raw Exact | Unified Exact | Improvement | Status |
|---------|-----------|---------------|-------------|--------|
| **MultiWOZ** | 82.98% | 97.22% | +14.2% | ✅ Polished |
| **D0T** | 0.16% | 48.31% | +48.2% ⭐ | ⚠️ Semantic concepts |
| **LUAS** | 79.24% | 86.42% | +7.2% | ✅ Solid |
| **Overall** | N/A | 71.63% | N/A | ✅ Combined |

### Training Recommendations

- **D0T**: MUST use unified (0.16% raw is unusable)
- **MultiWOZ**: Use unified for best quality (97.22% exact)
- **LUAS**: Use unified (86% vs 79% raw)
- **Baseline expectation**: ~72% accuracy if model copies from dialogue

## Code Changes

**Modified File**: `src/dst/runners/eda_unified.py`

**Key Changes**:
1. Auto-discovers all available datasets
2. Processes all datasets together in one analysis
3. Groups alignment metrics by dataset in output
4. Maintains backward compatibility (can specify single dataset)

**Usage**:
```bash
# All datasets together (grouped by dataset)
python eda_unified.py --split train --csv-prefix eda_unified_output

# Single dataset
python eda_unified.py --split train --dataset d0t --csv-prefix eda_unified_output
```

## Why This Matters

### D0T Discovery
D0T's 0.16% raw exact alignment initially seemed wrong. Analysis revealed:
- D0T uses **semantic annotation** (extracted concepts like "budget", "decision")
- These don't appear verbatim in dialogue
- Unification converts semantic concepts to **observable surface forms**
- Result: 48.31% exact alignment (48× improvement)

### Data Quality
- **MultiWOZ**: High quality (83% raw → 97% unified)
- **LUAS**: Solid baseline (79% raw → 86% unified)
- **D0T**: Requires special handling (0.16% raw → 48% unified)

### Model Training
- Expect ~72% accuracy if model simply copies values from dialogue context
- Remaining 27.48% require reasoning beyond surface matching
- D0T and LUAS contribute most unaligned cases (~50% and ~12% respectively)

## Next Steps

1. Use unified EDA output to understand dataset characteristics
2. Train models with unified data (highest quality)
3. Investigate the 27.48% unaligned cases for error analysis
4. Consider semantic matching for D0T beyond surface-form matching

---

**Last updated**: April 3, 2026  
**Analysis type**: Raw vs Unified EDA comparison  
**Datasets analyzed**: MultiWOZ 2.4, D0T/DSG5K, LUAS  
**Total examples**: 887,725 across all datasets
