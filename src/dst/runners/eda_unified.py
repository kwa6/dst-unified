"""
Exploratory Data Analysis for unified DST data (JSONL format).

Analyzes UnifiedDSTExample JSONL files from data_unified/ directory.
Outputs statistics on splits, slot coverage, target value distribution, and dialogue context.

DEFINITION: Value Alignment
===========================
Measured PER TRAINING EXAMPLE using the exact dialogue_context field.

For each example: dialogue_context + slot_name + target_value
    -> Does target_value appear in dialogue_context?

Layered metrics:
  1. EXACT ALIGNMENT: target_value appears verbatim (case-insensitive substring)
  2. NORMALIZED ALIGNMENT: target_value appears after normalization
     (lowercase, strip punctuation, normalize whitespace)
  3. SEMANTIC/NONE: Null values (none/empty/not mentioned/dontcare)
     tracked separately - "is none in context?" not meaningful

Usage:
    python eda_unified.py --split train [--csv-prefix OUTPUT_PREFIX] [--limit ROWS]

Example:
    python eda_unified.py --split train --csv-prefix eda_output --limit 50000
    python eda_unified.py --split val
"""

import argparse
import csv
import json
import re
import string
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any, Tuple


def load_jsonl(path: Path, limit: int = None) -> List[Dict[str, Any]]:
    """Load JSONL file with optional row limit."""
    examples = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            examples.append(json.loads(line))
            if limit and len(examples) >= limit:
                break
    return examples


# ============================================================================
# VALUE ALIGNMENT HELPERS
# ============================================================================

NONE_LIKE = {"none", "empty", "not mentioned", "not given", "dontcare", "?", ""}

def normalize_value(value: str) -> str:
    """
    Normalize value for matching:
    - lowercase
    - strip whitespace
    - strip punctuation
    - normalize internal whitespace
    """
    if not value:
        return ""
    v = value.strip().lower()
    # Remove punctuation
    v = v.translate(str.maketrans('', '', string.punctuation))
    # Normalize internal whitespace
    v = " ".join(v.split())
    return v


def check_exact_alignment(value: str, context: str) -> bool:
    """
    Check if value appears verbatim in context (case-insensitive).
    Uses substring matching on normalized strings.
    """
    if not value or not context:
        return False
    return value.lower() in context.lower()


def check_normalized_alignment(value: str, context: str) -> bool:
    """
    Check if value appears after normalization.
    If exact already matched, this is automatically true.
    """
    if not value or not context:
        return False
    v_norm = normalize_value(value)
    ctx_norm = normalize_value(context)
    if not v_norm:
        return False
    # Check substring match on normalized text
    return v_norm in ctx_norm


def categorize_alignment(value: str, context: str) -> str:
    """
    Categorize alignment into one of:
    - "none": null-like value
    - "exact": exact match in context
    - "normalized": normalized match (but not exact)
    - "not_aligned": value not found
    """
    v_lower = value.lower().strip()
    
    if v_lower in NONE_LIKE:
        return "none"
    
    if check_exact_alignment(value, context):
        return "exact"
    
    if check_normalized_alignment(value, context):
        return "normalized"
    
    return "not_aligned"


def compute_eda(examples: List[Dict[str, Any]], dataset_name: str = None):
    """Compute comprehensive EDA stats from unified JSONL examples."""
    
    if not examples:
        return {}
    
    n_examples = len(examples)
    n_dialogues = len(set(ex["dialogue_id"] for ex in examples))
    
    # Slot stats
    slot_counter = Counter()
    slot_target_dist = defaultdict(Counter)  # slot -> {target -> count}
    slot_datasets = defaultdict(set)
    slot_splits = defaultdict(set)
    
    # Target value distribution
    target_value_counter = Counter()
    target_is_none = 0
    target_is_filled = 0
    
    # Dialogue context stats
    context_length_dist = []
    context_turn_count = Counter()  # number of turns in dialogue
    
    # Speaker distribution
    speaker_counter = Counter()
    
    # Dataset/split distribution
    dataset_counter = Counter()
    split_counter = Counter()
    
    # Per-dialogue stats
    dialogues_seen = set()
    
    # ========================================================================
    # VALUE ALIGNMENT STATS (Per-example, layered definition)
    # ========================================================================
    # Track alignment category for each example
    # Categories: "none", "exact", "normalized", "not_aligned"
    alignment_categories = Counter()  # Overall counts
    alignment_by_dataset = defaultdict(Counter)  # dataset -> category -> count
    alignment_by_slot = defaultdict(Counter)  # slot -> category -> count
    
    # For percentage calculations
    alignment_non_none_total = 0  # Total non-none examples
    alignment_exact_total = 0     # Exact matches
    alignment_normalized_total = 0  # Normalized matches (includes exact)
    
    for ex in examples:
        slot = ex["slot_name"]
        target = ex["target_value"]
        dialogue_id = ex["dialogue_id"]
        dataset = ex.get("dataset", "unknown")
        ctx = ex.get("dialogue_context", "")
        
        slot_counter[slot] += 1
        slot_target_dist[slot][target] += 1
        slot_datasets[slot].add(dataset)
        slot_splits[slot].add(ex.get("split", "unknown"))
        
        target_value_counter[target] += 1
        if target.lower().strip() in NONE_LIKE:
            target_is_none += 1
        else:
            target_is_filled += 1
        
        # Context analysis
        context_length_dist.append(len(ctx))
        
        # Extract turn count from context (format: "Turn 0: ...\nTurn 1: ...\n")
        turn_count = ctx.count("\nTurn ") + 1 if "Turn" in ctx else 0
        context_turn_count[turn_count] += 1
        
        speaker_counter[ex.get("speaker", "unknown")] += 1
        dataset_counter[dataset] += 1
        split_counter[ex.get("split", "unknown")] += 1
        
        dialogues_seen.add(dialogue_id)
        
        # ====================================================================
        # LAYERED VALUE ALIGNMENT
        # ====================================================================
        alignment_cat = categorize_alignment(target, ctx)
        alignment_categories[alignment_cat] += 1
        alignment_by_dataset[dataset][alignment_cat] += 1
        alignment_by_slot[slot][alignment_cat] += 1
        
        # Count for percentage calculations
        if alignment_cat != "none":
            alignment_non_none_total += 1
            if alignment_cat == "exact":
                alignment_exact_total += 1
            if alignment_cat in {"exact", "normalized"}:
                alignment_normalized_total += 1
    
    # Compute quantiles for context length
    sorted_ctx_len = sorted(context_length_dist)
    
    def quantile(p):
        if not sorted_ctx_len:
            return 0
        idx = min(
            len(sorted_ctx_len) - 1,
            max(0, int(round((len(sorted_ctx_len) - 1) * p))),
        )
        return sorted_ctx_len[idx]
    
    # Top values per slot
    top_values = {}
    for slot, value_dist in slot_target_dist.items():
        top_values[slot] = value_dist.most_common(10)
    
    # Compute dataset-level alignment percentages
    dataset_alignment = {}
    for ds, cat_counts in alignment_by_dataset.items():
        exact = cat_counts.get("exact", 0)
        normalized = cat_counts.get("normalized", 0)
        not_aligned = cat_counts.get("not_aligned", 0)
        none_ct = cat_counts.get("none", 0)
        # non_none includes all non-none categories
        non_none_ct = exact + normalized + not_aligned
        
        exact_pct = (exact / non_none_ct * 100) if non_none_ct > 0 else 0
        normalized_pct = (normalized / non_none_ct * 100) if non_none_ct > 0 else 0
        
        dataset_alignment[ds] = {
            "exact": exact,
            "exact_pct": exact_pct,
            "normalized": normalized,
            "normalized_pct": normalized_pct,
            "not_aligned": not_aligned,
            "not_aligned_pct": (not_aligned / non_none_ct * 100) if non_none_ct > 0 else 0,
            "none": none_ct,
            "non_none_total": non_none_ct,
        }
    
    # Overall alignment percentages (non-none only)
    exact_pct_overall = (alignment_exact_total / alignment_non_none_total * 100) if alignment_non_none_total > 0 else 0
    normalized_pct_overall = (alignment_normalized_total / alignment_non_none_total * 100) if alignment_non_none_total > 0 else 0
    not_aligned_pct_overall = ((alignment_non_none_total - alignment_normalized_total) / alignment_non_none_total * 100) if alignment_non_none_total > 0 else 0
    
    return {
        "n_examples": n_examples,
        "n_dialogues": n_dialogues,
        "examples_per_dialogue_avg": n_examples / n_dialogues if n_dialogues else 0,
        "dataset_name": dataset_name,
        "dataset_counter": dict(dataset_counter),
        "split_counter": dict(split_counter),
        "speaker_counter": dict(speaker_counter),
        "n_unique_slots": len(slot_counter),
        "slot_coverage": dict(slot_counter.most_common()),
        "target_value_distribution": {
            "none": target_is_none,
            "none_pct": (target_is_none / n_examples * 100) if n_examples else 0,
            "filled": target_is_filled,
            "filled_pct": (target_is_filled / n_examples * 100) if n_examples else 0,
        },
        "target_value_examples": dict(target_value_counter.most_common(20)),
        "context_length": {
            "min": min(context_length_dist) if context_length_dist else 0,
            "p25": quantile(0.25),
            "median": quantile(0.50),
            "p75": quantile(0.75),
            "max": max(context_length_dist) if context_length_dist else 0,
            "mean": sum(context_length_dist) / len(context_length_dist)
            if context_length_dist
            else 0,
        },
        "context_turn_count": dict(context_turn_count.most_common()),
        "slot_target_examples": top_values,
        "slot_datasets": {k: list(v) for k, v in slot_datasets.items()},
        "slot_splits": {k: list(v) for k, v in slot_splits.items()},
        # Layered value alignment metrics
        "value_alignment": {
            "overall": {
                "exact": alignment_exact_total,
                "exact_pct": exact_pct_overall,
                "normalized": alignment_normalized_total,
                "normalized_pct": normalized_pct_overall,
                "not_aligned": alignment_non_none_total - alignment_normalized_total,
                "not_aligned_pct": not_aligned_pct_overall,
                "none": alignment_categories.get("none", 0),
                "non_none_total": alignment_non_none_total,
            },
            "by_dataset": dataset_alignment,
            "by_slot": {
                slot: dict(counts)
                for slot, counts in alignment_by_slot.items()
            }
        },
    }


def print_eda(eda: Dict[str, Any], split_name: str = "unknown"):
    """Pretty-print EDA results with layered value alignment metrics."""
    print("=" * 80)
    print(f"UNIFIED DST EDA ({split_name})")
    print("=" * 80)
    
    print(f"Examples: {eda['n_examples']}")
    print(f"Dialogues: {eda['n_dialogues']}")
    print(f"Examples per dialogue (avg): {eda['examples_per_dialogue_avg']:.1f}")
    
    if eda["dataset_counter"]:
        print("\nDatasets:")
        for ds, cnt in sorted(eda["dataset_counter"].items()):
            print(f"  {ds}: {cnt}")
    
    if eda["split_counter"]:
        print("\nSplits:")
        for sp, cnt in sorted(eda["split_counter"].items()):
            print(f"  {sp}: {cnt}")
    
    if eda["speaker_counter"]:
        print("\nSpeakers:")
        for sp, cnt in sorted(eda["speaker_counter"].items()):
            print(f"  {sp}: {cnt}")
    
    print(f"\nUnique slots: {eda['n_unique_slots']}")
    print("Top 20 slots by example count:")
    for slot, cnt in sorted(eda["slot_coverage"].items(), key=lambda x: -x[1])[:20]:
        print(f"  {slot:35s}: {cnt:6d}")
    
    tv = eda["target_value_distribution"]
    print(f"\nTarget value distribution:")
    print(f"  none:       {tv['none']:7d} ({tv['none_pct']:5.2f}%)")
    print(f"  filled:     {tv['filled']:7d} ({tv['filled_pct']:5.2f}%)")
    
    if eda["target_value_examples"]:
        print("\nTop 20 target values by frequency:")
        for value, cnt in sorted(
            eda["target_value_examples"].items(), key=lambda x: -x[1]
        )[:20]:
            disp_val = (value[:50] + "...") if len(value) > 50 else value
            print(f"  {disp_val:55s}: {cnt:6d}")
    
    ctx = eda["context_length"]
    print(
        f"\nDialogue context length (chars): "
        f"min={ctx['min']} p25={ctx['p25']} "
        f"median={ctx['median']} p75={ctx['p75']} "
        f"max={ctx['max']} mean={ctx['mean']:.0f}"
    )
    
    if eda.get("context_turn_count"):
        print("\nDialogue context turn count distribution:")
        for turns, cnt in sorted(eda["context_turn_count"].items()):
            print(f"  {turns} turns: {cnt}")
    
    # ========================================================================
    # LAYERED VALUE ALIGNMENT METRICS
    # ========================================================================
    if eda.get("value_alignment"):
        print("\n" + "=" * 80)
        print("VALUE ALIGNMENT (Per-Example, Layered Definition)")
        print("=" * 80)
        print("\nDEFINITION:")
        print("  Measured PER TRAINING EXAMPLE using dialogue_context the model receives.")
        print("  For each example: Is target_value recoverable from dialogue_context?")
        print()
        print("  1. EXACT:      Value appears verbatim (case-insensitive substring)")
        print("  2. NORMALIZED: Value appears after normalization (punct, spaces, case)")
        print("  3. NONE:       Null values (none/empty/not given/dontcare)")
        print("  4. NOT_ALIGNED: Value not found in context")
        print()
        
        va = eda["value_alignment"]
        overall = va["overall"]
        
        print(f"OVERALL ALIGNMENT (across all examples):")
        print(f"  None-like values:     {overall['none']:10d} examples (excluded from alignment calc)")
        print(f"  Non-none examples:    {overall['non_none_total']:10d} examples")
        print()
        print(f"  EXACT alignment:      {overall['exact']:10d} ({overall['exact_pct']:5.2f}%)")
        print(f"  NORMALIZED alignment: {overall['normalized']:10d} ({overall['normalized_pct']:5.2f}%)")
        print(f"    (includes exact matches)")
        print(f"  NOT_ALIGNED:          {overall['non_none_total'] - overall['normalized']:10d} ({overall['not_aligned_pct']:5.2f}%)")
        print()
        
        if va["by_dataset"]:
            print(f"ALIGNMENT BY DATASET:")
            for ds in sorted(va["by_dataset"].keys()):
                stats = va["by_dataset"][ds]
                print(f"\n  {ds}:")
                print(f"    None-like:     {stats['none']:6d}")
                print(f"    Non-none:      {stats['non_none_total']:6d}")
                print(f"    Exact:         {stats['exact']:6d} ({stats['exact_pct']:5.2f}%)")
                print(f"    Normalized:    {stats['normalized']:6d} ({stats['normalized_pct']:5.2f}%)")
                print(f"    Not-aligned:   {stats['not_aligned']:6d} ({stats['not_aligned_pct']:5.2f}%)")


def export_csv(eda: Dict[str, Any], csv_prefix: str, split_name: str):
    """Export EDA results to CSV files."""
    base = Path(csv_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    
    # Slot coverage
    slot_path = base.parent / f"{base.name}_{split_name}_slot_coverage.csv"
    with slot_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot_name", "example_count", "datasets", "splits"])
        
        for slot, cnt in sorted(eda["slot_coverage"].items(), key=lambda x: -x[1]):
            datasets = "|".join(sorted(eda.get("slot_datasets", {}).get(slot, [])))
            splits = "|".join(sorted(eda.get("slot_splits", {}).get(slot, [])))
            writer.writerow([slot, cnt, datasets, splits])
    
    print(f"Exported: {slot_path}")
    
    # Target values per slot
    values_path = base.parent / f"{base.name}_{split_name}_target_values.csv"
    with values_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot_name", "target_value", "count"])
        
        for slot in sorted(eda.get("slot_target_examples", {}).keys()):
            for value, count in eda["slot_target_examples"][slot]:
                writer.writerow([slot, value, count])
    
    print(f"Exported: {values_path}")
    
    # Summary
    summary_path = base.parent / f"{base.name}_{split_name}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        
        writer.writerow(["examples", eda["n_examples"]])
        writer.writerow(["dialogues", eda["n_dialogues"]])
        writer.writerow(
            ["examples_per_dialogue_avg", f"{eda['examples_per_dialogue_avg']:.2f}"]
        )
        writer.writerow(["unique_slots", eda["n_unique_slots"]])
        
        tv = eda["target_value_distribution"]
        writer.writerow(["target_none", tv["none"]])
        writer.writerow(["target_none_pct", f"{tv['none_pct']:.2f}"])
        writer.writerow(["target_filled", tv["filled"]])
        writer.writerow(["target_filled_pct", f"{tv['filled_pct']:.2f}"])
        
        ctx = eda["context_length"]
        writer.writerow(["context_length_min", ctx["min"]])
        writer.writerow(["context_length_p25", ctx["p25"]])
        writer.writerow(["context_length_median", ctx["median"]])
        writer.writerow(["context_length_p75", ctx["p75"]])
        writer.writerow(["context_length_max", ctx["max"]])
        writer.writerow(["context_length_mean", f"{ctx['mean']:.0f}"])
        
        # Layered value alignment
        if eda.get("value_alignment"):
            va = eda["value_alignment"]
            writer.writerow(["", ""])
            writer.writerow(["VALUE_ALIGNMENT_LAYERED", ""])
            writer.writerow(["", "Per-example, using dialogue_context model receives"])
            
            overall = va["overall"]
            writer.writerow(["", ""])
            writer.writerow(["OVERALL", ""])
            writer.writerow(["none_examples", overall["none"]])
            writer.writerow(["non_none_examples", overall["non_none_total"]])
            writer.writerow(["exact_count", overall["exact"]])
            writer.writerow(["exact_pct", f"{overall['exact_pct']:.2f}"])
            writer.writerow(["normalized_count", overall["normalized"]])
            writer.writerow(["normalized_pct", f"{overall['normalized_pct']:.2f}"])
            writer.writerow(["not_aligned_count", overall["not_aligned_count"] if "not_aligned_count" in overall else overall["non_none_total"] - overall["normalized"]])
            writer.writerow(["not_aligned_pct", f"{overall['not_aligned_pct']:.2f}"])
            
            if va["by_dataset"]:
                writer.writerow(["", ""])
                writer.writerow(["BY_DATASET", ""])
                for ds in sorted(va["by_dataset"].keys()):
                    stats = va["by_dataset"][ds]
                    writer.writerow(["", ""])
                    writer.writerow([f"dataset_{ds}", ""])
                    writer.writerow([f"  none", stats["none"]])
                    writer.writerow([f"  non_none_total", stats["non_none_total"]])
                    writer.writerow([f"  exact", stats["exact"]])
                    writer.writerow([f"  exact_pct", f"{stats['exact_pct']:.2f}"])
                    writer.writerow([f"  normalized", stats["normalized"]])
                    writer.writerow([f"  normalized_pct", f"{stats['normalized_pct']:.2f}"])
                    writer.writerow([f"  not_aligned", stats["not_aligned"]])
                    writer.writerow([f"  not_aligned_pct", f"{stats['not_aligned_pct']:.2f}"])
    
    print(f"Exported: {summary_path}")


def main():
    ap = argparse.ArgumentParser(description="EDA for unified DST JSONL data")
    ap.add_argument(
        "--split",
        required=True,
        choices=["train", "val", "test"],
        help="Which split to analyze",
    )
    ap.add_argument(
        "--dataset",
        type=str,
        default="multiwoz24",
        help="Dataset prefix (default: multiwoz24)",
    )
    ap.add_argument(
        "--base-dir",
        type=Path,
        default=Path("data_unified"),
        help="Base directory for unified data",
    )
    ap.add_argument(
        "--csv-prefix",
        type=str,
        default=None,
        help="If provided, export CSV files with this prefix",
    )
    ap.add_argument(
        "--limit",
        type=int,
        default=None,
        help="If provided, only analyze first N rows",
    )
    
    args = ap.parse_args()
    
    jsonl_path = args.base_dir / args.dataset / f"{args.split}.jsonl"
    
    if not jsonl_path.exists():
        print(f"Error: {jsonl_path} not found")
        return
    
    print(f"Loading examples from: {jsonl_path}")
    examples = load_jsonl(jsonl_path, limit=args.limit)
    
    if not examples:
        print("No examples found!")
        return
    
    print(f"Loaded {len(examples)} examples")
    
    eda = compute_eda(examples, dataset_name=args.dataset)
    print_eda(eda, split_name=args.split)
    
    if args.csv_prefix:
        export_csv(eda, args.csv_prefix, args.split)


if __name__ == "__main__":
    main()
