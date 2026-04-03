# Value Alignment Metric: Definition and Methodology

## Overview

This document defines the **value alignment** metric used for DST (Dialogue State Tracking) dataset analysis. The metric measures whether target slot values are **recoverable from the dialogue context** that the model actually receives during training and inference.

## Core Definition

### Unit of Measurement: Per Training Example

**NOT** per dialogue or per dataset abstractly.

Each training example consists of:
```
(dialogue_context, slot_name, target_value) → model_input
```

For each example:
- **Input**: The exact `dialogue_context` field - the accumulated dialogue history up to the current turn that the model receives
- **Label**: The `target_value` - the ground truth slot value for this turn
- **Question**: Does the target value appear recoverable from the dialogue context?

### Why This Unit?

1. **Alignment with training format**: Each example is processed independently by the model
2. **Honest context**:  Don't use full dialogue if model only receives turn-level or cumulative history
3. **Training-inference parity**: The context used for alignment matches what the model actually sees

## Layered Alignment Definition

We define alignment in layers, rather than a single binary metric:

### 1. **EXACT Alignment**
- **Definition**: Target value appears verbatim in dialogue_context
- **Matching**: Case-insensitive substring match
- **Example**:
  ```
  Context: "Turn 0: I need a hotel in the EAST"
  Value: "east"
  Result: EXACT ✓ (matched as substring, case-insensitive)
  ```

### 2. **NORMALIZED Alignment**
- **Definition**: Target value appears after normalization
- **Normalization includes**:
  - Lowercase conversion
  - Punctuation removal (but NOT space removal - spaces are preserved)
  - Internal whitespace normalization (multiple spaces → single space)
  - Trimming edges
- **Example - Successful**:
  ```
  Context: "Turn 0: I want Indian food, not Chinese"
  Value: "indian"
  Result: NORMALIZED ✓ (matches after lowercasing)
  ```
- **Example - Failed (spaces matter)**:
  ```
  Context: "Turn 0: I want modern-european cuisine"
  Value: "modern european"
  Result: NOT_ALIGNED (after punct removal, becomes "moderneuropean" ≠ "modern european")
  Note: The hyphen removal joins the words; space must be preserved in value
  ```

### 3. **SEMANTIC/Indirect Alignment** ⚠️
- **Definition**: Value does NOT appear verbatim, but dialogue clearly implies it
- **Status**: NOT counted as direct alignment (reported separately if needed)
- **Example**:
  ```
  Context: "Turn 0: I want to leave at end of work week"
  Value: "friday"
  Result: NOT_ALIGNED (indirect semantic connection, not direct grounding)
  ```

### 4. **NOT_ALIGNED**
- **Definition**: Value does not appear in context (verbatim or normalized)
- **Interpretation**: Model must generate or infer this value

### 5. **NONE-Like Values** (Excluded)
- **Values**: "none", "empty", "not mentioned", "not given", "dontcare", "?"
- **Status**: Tracked separately because "is none in the context?" is meaningless
- **Reporting**: Always reported but excluded from alignment percentages

## Percentage Calculations

Alignment percentages are computed **excluding none-like values**:

```
Exact alignment % = (count of exact matches) / (total non-none examples) × 100

Normalized alignment % = (count of normalized matches) / (total non-none examples) × 100
  Note: Includes exact matches (normalized ⊇ exact)

Not-aligned % = (count not found) / (total non-none examples) × 100
```

## Interpretation Guide

| Exact % | Normalized % | Interpretation |
|---------|-------------|---|
| 95%+ | 95%+ | Values are highly grounded in dialogue (extraction-friendly dataset) |
| 70-90% | 75-95% | Mixed grounding; some values require inference |
| 50-70% | 55-75% | Significant semantic understanding required |
| <50% | <55% | Values mostly out-of-dialogue (generative task) |

## Metrics by Dataset

### MultiWOZ 2.4
- **Train**: Exact 97.22% | Normalized 97.22% | Not-aligned 2.78%
- **Val**: Exact 98.98% | Normalized 98.98% | Not-aligned 1.02%
- **Test**: Exact 97.90% | Normalized 97.90% | Not-aligned 2.10%

**Interpretation**: Nearly all slot values appear verbatim in user utterances. This is an extraction-heavy dataset. Models can succeed with shallow pattern matching and retrieval strategies.

### D0T/DSG5K
- **Status**: Future analysis (raw data currently in CSV format, requires adapter)

### LUAS
- **Status**: Future analysis (raw data in JSON format, requires adapter)

## Implementation Details

### Code Location
- File: `src/dst/runners/eda_unified.py`
- Functions:
  - `categorize_alignment()`: Classifies each example
  - `check_exact_alignment()`: Substring matching
  - `check_normalized_alignment()`: After normalization
  - `normalize_value()`: Applies normalization rules

### CSV Export
Exported metrics in `{prefix}_{split}_summary.csv`:
```
VALUE_ALIGNMENT_LAYERED
  OVERALL
    none_examples: N
    non_none_examples: M
    exact_count: X
    exact_pct: X%
    normalized_count: Y (⊇ X)
    normalized_pct: Y%
    not_aligned_count: M - Y
    not_aligned_pct: (M-Y)%
  
  BY_DATASET
    dataset_NAME
      exact_count, exact_pct
      normalized_count, normalized_pct
      ...
```

## Limitations and Future Work

### Current Limitations
1. Binary membership test (substring match) - could miss partial matches
2. No semantic similarity scoring (e.g., synonyms not detected)
3. No word boundary detection (partial word matches possible)

### Potential Enhancements
1. **Whole-word matching**: Only match complete tokens, not substrings
2. **Fuzzy matching**: Edit distance for typos/variations
3. **Semantic matching**: Embeddings for synonym/paraphrase detection
4. **Configurable stoplist**: Exclude common generic values ("yes", "no", day names) from alignment calculations

## Validation

This metric validates the **Value Grounding Dependency Hypothesis**:
> Model performance on DST is heavily dependent on the proportion of target values appearing in dialogue context. When most values are in-dialogue (extractable), models can succeed with shallow pattern matching. When most values are out-of-dialogue (generative), models must perform deeper semantic understanding.

High alignment (>90%) → Extraction-friendly → Baseline models can succeed
Low alignment (<60%) → Generation-heavy → Deep semantic understanding required
