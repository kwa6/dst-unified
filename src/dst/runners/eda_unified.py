"""
Exploratory Data Analysis for unified DST data (JSONL format).

Analyzes UnifiedDSTExample JSONL files from data_unified/ directory.
Outputs statistics on splits, slot coverage, target value distribution, and dialogue context.

Usage:
    python eda_unified.py --split train [--csv-prefix OUTPUT_PREFIX] [--limit ROWS]

Example:
    python eda_unified.py --split train --csv-prefix eda_output --limit 50000
    python eda_unified.py --split val
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, List, Any


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
    target_is_dontcare = 0
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
    
    # Target value alignment stats (case-insensitive substring matching)
    # Track which examples have target appearing in context (non-none only)
    alignment_turn = defaultdict(lambda: {"aligned": 0, "total": 0})  # turn_id
    alignment_dialogue = defaultdict(lambda: {"aligned": False})  # dialogue_id
    alignment_dataset = defaultdict(lambda: {
        "turn_aligned": 0, "turn_total": 0,
        "dialogues_aligned": set(), "dialogues_total": set()
    })
    
    for ex in examples:
        slot = ex["slot_name"]
        target = ex["target_value"]
        dialogue_id = ex["dialogue_id"]
        dataset = ex.get("dataset", "unknown")
        turn_id = ex.get("turn_id", 0)
        ctx = ex.get("dialogue_context", "")
        
        slot_counter[slot] += 1
        slot_target_dist[slot][target] += 1
        slot_datasets[slot].add(dataset)
        slot_splits[slot].add(ex.get("split", "unknown"))
        
        target_value_counter[target] += 1
        if target == "none":
            target_is_none += 1
        elif target == "dontcare":
            target_is_dontcare += 1
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
        
        # Target value alignment (only for non-"none" values)
        if target != "none":
            # Turn-level tracking
            turn_key = (dialogue_id, turn_id)
            alignment_turn[turn_key]["total"] += 1
            
            # Check if target appears in dialogue context (case-insensitive substring)
            target_lower = target.lower()
            ctx_lower = ctx.lower()
            is_aligned = target_lower in ctx_lower
            
            if is_aligned:
                alignment_turn[turn_key]["aligned"] += 1
            
            # Dialogue-level tracking
            if is_aligned:
                alignment_dialogue[dialogue_id]["aligned"] = True
            
            # Dataset-level tracking
            alignment_dataset[dataset]["turn_total"] += 1
            if is_aligned:
                alignment_dataset[dataset]["turn_aligned"] += 1
            alignment_dataset[dataset]["dialogues_total"].add(dialogue_id)
            if is_aligned:
                alignment_dataset[dataset]["dialogues_aligned"].add(dialogue_id)
    
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
    
    # Compute alignment percentages at turn level
    turn_aligned_count = 0
    turn_total_count = 0
    for turn_stats in alignment_turn.values():
        turn_aligned_count += turn_stats["aligned"]
        turn_total_count += turn_stats["total"]
    
    turn_alignment_pct = (turn_aligned_count / turn_total_count * 100) if turn_total_count > 0 else 0
    
    # Compute alignment percentages at dialogue level
    dialogue_aligned_count = sum(1 for d in alignment_dialogue.values() if d["aligned"])
    dialogue_total_count = len(alignment_dialogue)
    dialogue_alignment_pct = (dialogue_aligned_count / dialogue_total_count * 100) if dialogue_total_count > 0 else 0
    
    # Compute dataset-level alignment
    dataset_alignment = {}
    for ds, stats in alignment_dataset.items():
        turn_pct = (stats["turn_aligned"] / stats["turn_total"] * 100) if stats["turn_total"] > 0 else 0
        dialogue_total = len(stats["dialogues_total"])
        dialogue_aligned = len(stats["dialogues_aligned"])
        dialogue_pct = (dialogue_aligned / dialogue_total * 100) if dialogue_total > 0 else 0
        dataset_alignment[ds] = {
            "turn_aligned": stats["turn_aligned"],
            "turn_total": stats["turn_total"],
            "turn_alignment_pct": turn_pct,
            "dialogue_aligned": dialogue_aligned,
            "dialogue_total": dialogue_total,
            "dialogue_alignment_pct": dialogue_pct,
        }
    
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
            "dontcare": target_is_dontcare,
            "dontcare_pct": (target_is_dontcare / n_examples * 100)
            if n_examples
            else 0,
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
        "target_value_alignment": {
            "turn_alignment": {
                "aligned": turn_aligned_count,
                "total": turn_total_count,
                "pct": turn_alignment_pct,
            },
            "dialogue_alignment": {
                "aligned": dialogue_aligned_count,
                "total": dialogue_total_count,
                "pct": dialogue_alignment_pct,
            },
            "by_dataset": dataset_alignment,
        },
    }


def print_eda(eda: Dict[str, Any], split_name: str = "unknown"):
    """Pretty-print EDA results."""
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
    print(f"  dontcare:   {tv['dontcare']:7d} ({tv['dontcare_pct']:5.2f}%)")
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
    
    # Target value alignment
    if eda.get("target_value_alignment"):
        print("\n" + "=" * 80)
        print("TARGET VALUE ALIGNMENT (non-'none' targets only)")
        print("=" * 80)
        
        ta = eda["target_value_alignment"]
        turn_align = ta["turn_alignment"]
        dial_align = ta["dialogue_alignment"]
        
        print(f"\nTurn-level alignment (substring match in context):")
        print(f"  {turn_align['aligned']:7d} / {turn_align['total']:7d} examples ({turn_align['pct']:5.2f}%)")
        
        print(f"\nDialogue-level alignment (≥1 target value in context):")
        print(f"  {dial_align['aligned']:7d} / {dial_align['total']:7d} dialogues ({dial_align['pct']:5.2f}%)")
        
        print(f"\nAlignment by dataset:")
        for ds in sorted(ta["by_dataset"].keys()):
            stats = ta["by_dataset"][ds]
            print(f"\n  {ds}:")
            print(f"    Turn-level:     {stats['turn_aligned']:7d} / {stats['turn_total']:7d} ({stats['turn_alignment_pct']:5.2f}%)")
            print(f"    Dialogue-level: {stats['dialogue_aligned']:7d} / {stats['dialogue_total']:7d} ({stats['dialogue_alignment_pct']:5.2f}%)")


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
        writer.writerow(["target_dontcare", tv["dontcare"]])
        writer.writerow(["target_dontcare_pct", f"{tv['dontcare_pct']:.2f}"])
        writer.writerow(["target_filled", tv["filled"]])
        writer.writerow(["target_filled_pct", f"{tv['filled_pct']:.2f}"])
        
        ctx = eda["context_length"]
        writer.writerow(["context_length_min", ctx["min"]])
        writer.writerow(["context_length_p25", ctx["p25"]])
        writer.writerow(["context_length_median", ctx["median"]])
        writer.writerow(["context_length_p75", ctx["p75"]])
        writer.writerow(["context_length_max", ctx["max"]])
        writer.writerow(["context_length_mean", f"{ctx['mean']:.0f}"])
        
        # Target value alignment
        if eda.get("target_value_alignment"):
            ta = eda["target_value_alignment"]
            writer.writerow(["", ""])
            writer.writerow(["TARGET_VALUE_ALIGNMENT_(non-none_only)", ""])
            writer.writerow(["turn_alignment_count", ta["turn_alignment"]["aligned"]])
            writer.writerow(["turn_alignment_total", ta["turn_alignment"]["total"]])
            writer.writerow(["turn_alignment_pct", f"{ta['turn_alignment']['pct']:.2f}"])
            writer.writerow(["dialogue_alignment_count", ta["dialogue_alignment"]["aligned"]])
            writer.writerow(["dialogue_alignment_total", ta["dialogue_alignment"]["total"]])
            writer.writerow(["dialogue_alignment_pct", f"{ta['dialogue_alignment']['pct']:.2f}"])
            
            for ds in sorted(ta["by_dataset"].keys()):
                stats = ta["by_dataset"][ds]
                writer.writerow(["", ""])
                writer.writerow([f"dataset_{ds}", ""])
                writer.writerow([f"  turn_aligned", stats["turn_aligned"]])
                writer.writerow([f"  turn_total", stats["turn_total"]])
                writer.writerow([f"  turn_pct", f"{stats['turn_alignment_pct']:.2f}"])
                writer.writerow([f"  dialogue_aligned", stats["dialogue_aligned"]])
                writer.writerow([f"  dialogue_total", stats["dialogue_total"]])
                writer.writerow([f"  dialogue_pct", f"{stats['dialogue_alignment_pct']:.2f}"])
    
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
