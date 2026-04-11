#!/usr/bin/env python3
"""
Analyze error patterns in DST model predictions.

Categorizes errors by whether the target value appears in the dialogue context:
  - MISSED (in_context=True):  Target is in context but model failed to extract it
  - HALLUCINATION (in_context=False): Target is not in context; model predicted wrong info
  - NONE_ERRORS: Special handling for 'none' predictions

Usage:
    python scripts/analyze_errors.py errors.json
    python scripts/analyze_errors.py errors.json --output analysis.json
"""
import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Optional


def normalize(v: str) -> str:
    """Normalize value for comparison."""
    v = (v or "").strip().lower()
    if v in {"", "none", "not mentioned", "not given"}:
        return "none"
    return v


def is_in_context(gold: str, context: str) -> bool:
    """Check if gold value appears in dialogue context."""
    if normalize(gold) == "none":
        return True  # "none" is always conceptually in context
    
    gold_norm = normalize(gold)
    context_lower = context.lower()
    
    # Check for exact match or word boundaries
    return gold_norm in context_lower


def categorize_error(error: dict) -> dict:
    """Categorize a single error."""
    gold = error["gold"]
    pred = error["pred"]
    context = error["context"]
    slot_name = error["slot_name"]
    
    gold_norm = normalize(gold)
    pred_norm = normalize(pred)
    
    # Determine error type based on whether gold is in context
    in_context = is_in_context(gold, context)
    
    error_type = "missed" if in_context else "hallucination"
    if gold_norm == "none" and pred_norm != "none":
        error_type = "false_positive"
    elif gold_norm != "none" and pred_norm == "none":
        error_type = "false_negative"
    
    return {
        **error,
        "in_context": in_context,
        "error_type": error_type,
    }


def main():
    ap = argparse.ArgumentParser(description="Analyze DST error patterns")
    ap.add_argument("errors_file", help="JSON file with errors from eval_jga_llama.py or eval_jga.py")
    ap.add_argument("--output", default=None, help="Save categorized errors as JSON (optional)")
    ap.add_argument("--by_slot", action="store_true", help="Show breakdown by slot name")
    ap.add_argument("--show_samples", type=int, default=3, help="Show N example errors per category")
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

    for err in categorized:
        error_type = err["error_type"]
        slot = err["slot_name"]
        
        stats[error_type] += 1
        by_slot[slot][error_type] += 1
        
        if len(samples_by_type[error_type]) < args.show_samples:
            samples_by_type[error_type].append(err)

    # Print summary
    total = len(categorized)
    print("=" * 70)
    print("ERROR SUMMARY")
    print("=" * 70)
    
    for error_type in ["missed", "hallucination", "false_positive", "false_negative"]:
        count = stats.get(error_type, 0)
        pct = 100 * count / total if total else 0
        print(f"{error_type:20s}: {count:5d} ({pct:5.1f}%)")
    
    print("-" * 70)
    
    # Additional insights
    in_context = sum(1 for e in categorized if e["in_context"])
    not_in_context = len(categorized) - in_context
    pct_in = 100 * in_context / total if total else 0
    pct_out = 100 * not_in_context / total if total else 0
    
    print(f"{'Gold IN context':20s}: {in_context:5d} ({pct_in:5.1f}%)")
    print(f"{'Gold NOT in context':20s}: {not_in_context:5d} ({pct_out:5.1f}%)")
    print()

    # Show by slot if requested
    if args.by_slot:
        print("=" * 70)
        print("ERRORS BY SLOT")
        print("=" * 70)
        for slot in sorted(by_slot.keys()):
            slot_errors = by_slot[slot]
            slot_total = sum(slot_errors.values())
            print(f"\n{slot} ({slot_total} errors):")
            for error_type in ["missed", "hallucination", "false_positive", "false_negative"]:
                count = slot_errors.get(error_type, 0)
                if count > 0:
                    pct = 100 * count / slot_total
                    print(f"  {error_type:20s}: {count:3d} ({pct:5.1f}%)")
        print()

    # Show samples
    print("=" * 70)
    print("SAMPLE ERRORS")
    print("=" * 70)
    
    for error_type in ["missed", "hallucination", "false_positive", "false_negative"]:
        samples = samples_by_type.get(error_type, [])
        if not samples:
            continue
        
        print(f"\n--- {error_type.upper()} ({len(samples)} samples shown) ---")
        for i, err in enumerate(samples, 1):
            print(f"\n{i}. Dialogue: {err['dialogue_id']}, Turn: {err['turn_id']}")
            print(f"   Slot: {err['slot_name']}")
            print(f"   Description: {err['slot_description']}")
            print(f"   Gold: {err['gold']!r}")
            print(f"   Pred: {err['pred']!r}")
            ctx_preview = err["context"][:200].replace("\n", " ")
            print(f"   Context: {ctx_preview}...")
    
    print()
    print("=" * 70)

    # Save categorized errors if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(categorized, f, indent=2)
        print(f"✓ Categorized errors saved to: {args.output}\n")


if __name__ == "__main__":
    main()
