#!/usr/bin/env python3
"""
Analyze error patterns in DST model predictions.

Categorizes errors by whether the target value appears in the dialogue context:
  - MISSED (in_context=True):  Target is in context but model failed to extract it
  - INDIRECT_MISS: Target is indirectly expressible but model failed to infer it
  - HALLUCINATION (in_context=False): Target is not in context; model predicted wrong info
  - INDIRECT_HALLUCINATION: Predicted value not in context but should be inferrable
  - FALSE_POSITIVE/FALSE_NEGATIVE: Special handling for 'none' predictions

Usage:
    python scripts/analyze_errors.py errors.json
    python scripts/analyze_errors.py errors.json --output analysis.json
    python scripts/analyze_errors.py errors.json --show-indirect-only
"""
import argparse
import json
import re
from collections import defaultdict
from pathlib import Path
from typing import Optional, Tuple


def normalize(v: str) -> str:
    """Normalize value for comparison."""
    v = (v or "").strip().lower()
    if v in {"", "none", "not mentioned", "not given", "dontcare", "don't care"}:
        return "none"
    return v


def normalize_number(s: str) -> Optional[str]:
    """Extract number from text. E.g., '5 people' -> '5', 'at 3pm' -> '3'"""
    numbers = re.findall(r'\d+', s)
    return numbers[0] if numbers else None


def find_indirect_indicators(gold: str, context: str) -> Tuple[bool, str]:
    """
    Detect if gold value can be inferred indirectly from context.
    Returns (is_inferrable, reason)
    """
    gold_norm = normalize(gold)
    context_lower = context.lower()
    
    if gold_norm == "none":
        return False, ""
    
    # Pattern 1: Numbers expressed differently
    # E.g., "want 5 people" -> party_size: 5
    gold_num = normalize_number(gold_norm)
    if gold_num:
        # Look for the number in context
        if re.search(rf'\b{re.escape(gold_num)}\b', context_lower):
            return True, f"number_indirect: '{gold_num}' appears in context"
    
    # Pattern 2: Common time expressions
    # E.g., "5pm" vs "5 pm" vs "17:00"
    time_patterns = {
        r'(\d{1,2})\s*am': 'morning_time',
        r'(\d{1,2})\s*pm': 'afternoon_time',
        r'(\d{1,2}):(\d{2})': 'hm_time',
    }
    for pattern, label in time_patterns.items():
        if re.search(pattern, context_lower):
            matches = re.findall(pattern, gold_norm)
            if matches:
                return True, f"time_indirect: time expression in context"
    
    # Pattern 3: Days of week / dates
    days = {'monday', 'tuesday', 'wednesday', 'thursday', 'friday', 'saturday', 'sunday'}
    months = {'january', 'february', 'march', 'april', 'may', 'june', 'july', 'august', 'september', 'october', 'november', 'december'}
    
    if gold_norm in days or gold_norm in months:
        if gold_norm in context_lower:
            return True, f"date_indirect: day/month '{gold_norm}' in context"
    
    # Pattern 4: Keywords that strongly suggest semantic inference needed
    # (NOT indirect - these are truly semantic)
    semantic_keywords = {'yes', 'no', 'true', 'false', 'positive', 'negative', 'certain', 'uncertain', 
                        'excited', 'disappointed', 'satisfied', 'unsatisfied'}
    if any(kw in gold_norm for kw in semantic_keywords):
        return False, "semantic_generation_needed"
    
    return False, ""


def is_in_context(gold: str, context: str) -> bool:
    """Check if gold value appears in dialogue context (exact match)."""
    if normalize(gold) == "none":
        return True  # "none" is always conceptually in context
    
    gold_norm = normalize(gold)
    context_lower = context.lower()
    
    # Check for exact match or word boundaries
    return gold_norm in context_lower


def is_indirectly_in_context(gold: str, context: str) -> Tuple[bool, str]:
    """Check if gold value is indirectly expressed in context."""
    is_indirect, reason = find_indirect_indicators(gold, context)
    return is_indirect, reason


def categorize_error(error: dict) -> dict:
    """Categorize a single error with indirect expression detection."""
    gold = error["gold"]
    pred = error["pred"]
    context = error["context"]
    slot_name = error["slot_name"]
    
    gold_norm = normalize(gold)
    pred_norm = normalize(pred)
    
    # Check different alignment levels
    in_context = is_in_context(gold, context)
    is_indirect, indirect_reason = is_indirectly_in_context(gold, context)
    
    # Determine error type with indirect expression detection
    if gold_norm == "none" and pred_norm != "none":
        error_type = "false_positive"
    elif gold_norm != "none" and pred_norm == "none":
        error_type = "false_negative"
    elif in_context:
        error_type = "missed"
    elif is_indirect:
        error_type = "indirect_miss"  # Should have been inferrable
    else:
        # Further categorize hallucinations
        error_type = "hallucination"
    
    return {
        **error,
        "in_context": in_context,
        "is_indirect": is_indirect,
        "indirect_reason": indirect_reason,
        "error_type": error_type,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze DST error patterns with indirect expression detection")
    ap.add_argument("errors_file", help="JSON file with errors from eval_jga_llama.py or eval_jga.py")
    ap.add_argument("--output", default=None, help="Save categorized errors as JSON (optional)")
    ap.add_argument("--by-slot", action="store_true", help="Show breakdown by slot name")
    ap.add_argument("--show-samples", type=int, default=3, help="Show N example errors per category")
    ap.add_argument("--show-indirect-only", action="store_true", help="Show only indirect expression errors")
    args = ap.parse_args()

    # Load errors
    with open(args.errors_file) as f:
        errors = json.load(f)

    if not errors:
        print("No errors found in file.")
        return

    print(f"Loaded {len(errors)} errors from {args.errors_file}\n")

    # Categorize all errors
    categorized = [categorize_error(e) for e in errors]

    # Aggregate statistics
    stats = defaultdict(int)
    by_slot = defaultdict(lambda: defaultdict(int))
    samples_by_type = defaultdict(list)
    indirect_errors = []

    for err in categorized:
        error_type = err["error_type"]
        slot = err["slot_name"]
        
        stats[error_type] += 1
        by_slot[slot][error_type] += 1
        
        # Track indirect expression errors separately
        if err.get("is_indirect"):
            indirect_errors.append(err)
        
        if len(samples_by_type[error_type]) < args.show_samples:
            samples_by_type[error_type].append(err)

    # Filter to indirect-only if requested
    if args.show_indirect_only:
        print(f"Filtering to {len(indirect_errors)} indirect expression errors...\n")
        categorized = indirect_errors
        stats = defaultdict(int)
        samples_by_type = defaultdict(list)
        for err in indirect_errors:
            error_type = err["error_type"]
            stats[error_type] += 1
            if len(samples_by_type[error_type]) < args.show_samples:
                samples_by_type[error_type].append(err)

    # Print summary
    total = len(categorized)
    print("=" * 80)
    print("ERROR SUMMARY")
    print("=" * 80)
    
    for error_type in ["missed", "indirect_miss", "false_positive", "false_negative", "hallucination"]:
        count = stats.get(error_type, 0)
        pct = 100 * count / total if total else 0
        if count > 0:
            print(f"{error_type:25s}: {count:6d} ({pct:5.1f}%)")
    
    print("-" * 80)
    
    # Additional insights: alignment levels
    in_context = sum(1 for e in categorized if e["in_context"])
    is_indirect = sum(1 for e in categorized if e["is_indirect"])
    not_in_context = len(categorized) - in_context - is_indirect
    
    pct_in = 100 * in_context / total if total else 0
    pct_ind = 100 * is_indirect / total if total else 0
    pct_out = 100 * not_in_context / total if total else 0
    
    print(f"\nVALUE ALIGNMENT ANALYSIS:")
    print(f"{'  Direct in context':30s}: {in_context:6d} ({pct_in:5.1f}%)")
    print(f"{'  Indirectly expressible':30s}: {is_indirect:6d} ({pct_ind:5.1f}%)")
    print(f"{'  Truly not in context':30s}: {not_in_context:6d} ({pct_out:5.1f}%)")
    print()

    # Show by slot if requested
    if args.by_slot:
        print("=" * 80)
        print("ERRORS BY SLOT")
        print("=" * 80)
        for slot in sorted(by_slot.keys()):
            slot_errors = by_slot[slot]
            slot_total = sum(slot_errors.values())
            print(f"\n{slot} ({slot_total} errors):")
            for error_type in ["missed", "indirect_miss", "false_positive", "false_negative", "hallucination"]:
                count = slot_errors.get(error_type, 0)
                if count > 0:
                    pct = 100 * count / slot_total
                    print(f"  {error_type:23s}: {count:4d} ({pct:5.1f}%)")
        print()

    # Show samples
    print("=" * 80)
    print("SAMPLE ERRORS")
    print("=" * 80)
    
    for error_type in ["missed", "indirect_miss", "false_positive", "false_negative", "hallucination"]:
        samples = samples_by_type.get(error_type, [])
        if not samples:
            continue
        
        print(f"\n--- {error_type.upper()} ({len(samples)} samples shown) ---")
        for i, err in enumerate(samples, 1):
            print(f"\n{i}. Dialogue: {err['dialogue_id']}, Turn: {err['turn_id']}")
            print(f"   Slot: {err['slot_name']}")
            print(f"   Gold: {err['gold']!r}")
            print(f"   Pred: {err['pred']!r}")
            
            # Show indirect expression reason if applicable
            if err.get("is_indirect") and err.get("indirect_reason"):
                print(f"   Indirect Reason: {err['indirect_reason']}")
            
            ctx_preview = err["context"][:250].replace("\n", " ")
            print(f"   Context: {ctx_preview}...")
    
    print()
    print("=" * 80)

    # Save categorized errors if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(categorized, f, indent=2)
        print(f"✓ Categorized errors saved to: {args.output}\n")


if __name__ == "__main__":
    main()
