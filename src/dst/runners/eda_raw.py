"""
Exploratory Data Analysis for raw DST data (MultiWOZ 2.4, D0T/DSG5K, LUAS).

Analyzes raw data in their native formats and computes statistics for comparison.

Usage:
    python eda_raw.py --dataset multiwoz [--csv-prefix OUTPUT_PREFIX]
    python eda_raw.py --dataset d0t [--csv-prefix OUTPUT_PREFIX]
    python eda_raw.py --dataset luas [--csv-prefix OUTPUT_PREFIX]

Example:
    python eda_raw.py --dataset multiwoz --csv-prefix eda_raw_output
    python eda_raw.py --dataset d0t --csv-prefix eda_raw_output
    python eda_raw.py --dataset luas --csv-prefix eda_raw_output
"""

import argparse
import csv
import json
import re
from collections import Counter, defaultdict
from io import TextIOWrapper
from pathlib import Path
from typing import Dict, List, Optional, Tuple


# ============================================================================
# MULTIWOZ 2.4 PARSING
# ============================================================================

def parse_multiwoz(data_path: Path, val_path: Path, test_path: Path):
    """Parse raw MultiWOZ 2.4 data."""
    
    with data_path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    
    val_ids = {
        line.strip()
        for line in val_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    test_ids = {
        line.strip()
        for line in test_path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    all_ids = list(data.keys())
    train_ids = [i for i in all_ids if i not in val_ids and i not in test_ids]
    
    dialogues_by_split = {"train": len(train_ids), "val": len(val_ids), "test": len(test_ids)}
    
    n_turns = 0
    n_turns_with_metadata = 0
    n_user_turns = 0
    n_system_turns = 0
    turns_per_dialogue = []
    
    domain_turn_presence = Counter()
    domain_dialogue_presence = defaultdict(set)
    slot_observed_count = Counter()
    slot_none_like_count = Counter()
    slot_dontcare_count = Counter()
    slot_total_seen = Counter()
    slot_value_examples = defaultdict(lambda: defaultdict(int))
    
    none_like = {"", "none", "not mentioned", "not given"}
    dontcare_like = {"dontcare", "dont care", "don't care", "do not care"}
    
    for did, d in data.items():
        log = d.get("log", [])
        turns_per_dialogue.append(len(log))
        n_turns += len(log)
        
        for t_idx, turn in enumerate(log):
            if t_idx % 2 == 0:
                n_user_turns += 1
            else:
                n_system_turns += 1
            
            md = turn.get("metadata") or {}
            if not md:
                continue
            
            n_turns_with_metadata += 1
            
            for domain, domain_data in md.items():
                if not isinstance(domain_data, dict):
                    continue
                
                domain_dialogue_presence[domain].add(did)
                
                semi = domain_data.get("semi", {}) or {}
                if semi:
                    domain_turn_presence[domain] += 1
                
                for slot, raw_v in semi.items():
                    s = f"{domain}-{slot}"
                    slot_total_seen[s] += 1
                    
                    v = str(raw_v).strip().lower() if raw_v is not None else ""
                    slot_value_examples[s][v] += 1
                    
                    if v in none_like:
                        slot_none_like_count[s] += 1
                    elif v in dontcare_like:
                        slot_dontcare_count[s] += 1
                        slot_observed_count[s] += 1
                    else:
                        slot_observed_count[s] += 1
    
    return {
        "n_dialogues": len(all_ids),
        "dialogues_by_split": dialogues_by_split,
        "n_turns": n_turns,
        "n_turns_with_metadata": n_turns_with_metadata,
        "n_user_turns": n_user_turns,
        "n_system_turns": n_system_turns,
        "turns_per_dialogue": turns_per_dialogue,
        "domain_turn_presence": domain_turn_presence,
        "domain_dialogue_presence": {k: len(v) for k, v in domain_dialogue_presence.items()},
        "slot_observed_count": slot_observed_count,
        "slot_none_like_count": slot_none_like_count,
        "slot_dontcare_count": slot_dontcare_count,
        "slot_total_seen": slot_total_seen,
        "slot_value_examples": slot_value_examples,
    }


# ============================================================================
# D0T / DSG5K PARSING
# ============================================================================

def parse_d0t(base_dir: Path):
    """Parse raw D0T/DSG5K CSV data."""
    
    turn_csv = base_dir / "turn.csv"
    slot_csv = base_dir / "slot.csv"
    slot_value_csv = base_dir / "slot_value.csv"
    
    if not turn_csv.exists():
        raise FileNotFoundError(f"D0T turn.csv not found at {turn_csv}")
    
    # Load turns
    dialogues_by_split = {"train": 0, "val": 0, "test": 0}
    turns_by_dialogue = defaultdict(list)
    turns_by_id = {}
    
    with turn_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            turn_id = row["turn_id"]
            dialogue_id = row["dialogue"]
            split = row.get("split", "").lower() or "train"
            if split not in dialogues_by_split:
                split = "train"
            
            turns_by_id[turn_id] = {
                "dialogue": dialogue_id,
                "turn_index": int(row["turn_index"]),
                "speaker": row.get("speaker", "system").lower(),
                "split": split,
            }
            turns_by_dialogue[dialogue_id].append((int(row["turn_index"]), row.get("speaker", "system").lower()))
    
    # Count unique dialogues per split
    for dialogue_id, turns in turns_by_dialogue.items():
        # Infer split from turns if needed
        if turns:
            split = turns_by_id.get(list(turns_by_id.keys())[0], {}).get("split", "train")
            dialogues_by_split[split] += 1
    
    n_dialogues = len(turns_by_dialogue)
    n_turns = len(turns_by_id)
    n_user_turns = sum(1 for t in turns_by_id.values() if t["speaker"] == "user")
    n_system_turns = sum(1 for t in turns_by_id.values() if t["speaker"] == "system")
    
    turns_per_dialogue = [len(turns) for turns in turns_by_dialogue.values()]
    
    # Load slots
    slots_by_id = {}
    with slot_csv.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            slots_by_id[row["slot_id"]] = row["slot"]
    
    # Load slot values and compute stats
    domain_turn_presence = Counter()
    domain_dialogue_presence = defaultdict(set)
    slot_observed_count = Counter()
    slot_none_like_count = Counter()
    slot_dontcare_count = Counter()
    slot_total_seen = Counter()
    slot_value_examples = defaultdict(lambda: defaultdict(int))
    
    none_like = {"", "?", "none", "not mentioned", "not given"}
    dontcare_like = {"dontcare", "dont care", "don't care"}
    
    if slot_value_csv.exists():
        with slot_value_csv.open(newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                turn_id = row.get("turn_id")
                slot_id = row.get("slot_id")
                
                if turn_id not in turns_by_id or slot_id not in slots_by_id:
                    continue
                
                turn = turns_by_id[turn_id]
                slot_name = slots_by_id[slot_id]
                dialogue_id = turn["dialogue"]
                
                _, domain = slot_name.split("-", 1) if "-" in slot_name else ("unknown", slot_name)
                
                domain_dialogue_presence[domain].add(dialogue_id)
                domain_turn_presence[domain] += 1
                
                slot_total_seen[slot_name] += 1
                v = str(row.get("value", "")).strip().lower()
                slot_value_examples[slot_name][v] += 1
                
                if v in none_like:
                    slot_none_like_count[slot_name] += 1
                elif v in dontcare_like:
                    slot_dontcare_count[slot_name] += 1
                    slot_observed_count[slot_name] += 1
                else:
                    slot_observed_count[slot_name] += 1
    
    return {
        "n_dialogues": n_dialogues,
        "dialogues_by_split": dialogues_by_split,
        "n_turns": n_turns,
        "n_turns_with_metadata": n_turns,  # In D0T, all turns have annotations
        "n_user_turns": n_user_turns,
        "n_system_turns": n_system_turns,
        "turns_per_dialogue": turns_per_dialogue,
        "domain_turn_presence": domain_turn_presence,
        "domain_dialogue_presence": {k: len(v) for k, v in domain_dialogue_presence.items()},
        "slot_observed_count": slot_observed_count,
        "slot_none_like_count": slot_none_like_count,
        "slot_dontcare_count": slot_dontcare_count,
        "slot_total_seen": slot_total_seen,
        "slot_value_examples": slot_value_examples,
    }


# ============================================================================
# LUAS PARSING
# ============================================================================

def parse_luas(luas_json_path: Path):
    """Parse raw LUAS JSON data."""
    
    if not luas_json_path.exists():
        raise FileNotFoundError(f"LUAS JSON not found at {luas_json_path}")
    
    n_dialogues = 0
    n_turns = 0
    n_user_turns = 0
    n_system_turns = 0
    turns_per_dialogue = []
    
    domain_turn_presence = Counter()
    domain_dialogue_presence = defaultdict(set)
    slot_observed_count = Counter()
    slot_none_like_count = Counter()
    slot_dontcare_count = Counter()
    slot_total_seen = Counter()
    slot_value_examples = defaultdict(lambda: defaultdict(int))
    
    none_like = {"", "?", "none", "not mentioned", "not given"}
    dontcare_like = {"dontcare", "dont care", "don't care"}
    
    with luas_json_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            
            dialogue = json.loads(line)
            n_dialogues += 1
            dialogue_id = dialogue.get("dialogue_id", f"luas_{n_dialogues}")
            
            turns = dialogue.get("turns", [])
            turns_per_dialogue.append(len(turns))
            
            for turn in turns:
                n_turns += 1
                speaker = str(turn.get("speaker", "")).lower()
                if speaker == "user":
                    n_user_turns += 1
                elif speaker == "system":
                    n_system_turns += 1
                
                # Parse slot values from reference
                if "reference" in turn:
                    refs = turn["reference"]
                    if not isinstance(refs, list):
                        refs = [refs]
                    
                    for ref in refs:
                        if not isinstance(ref, dict):
                            continue
                        
                        slot_values = ref.get("slot_values", {})
                        if not isinstance(slot_values, dict):
                            continue
                        
                        for slot_name, values in slot_values.items():
                            # LUAS stores values as lists
                            if isinstance(values, list):
                                value = values[0] if values else ""
                            else:
                                value = values or ""
                            
                            # Extract domain
                            domain = slot_name.split("-")[0] if "-" in slot_name else "unknown"
                            
                            domain_dialogue_presence[domain].add(dialogue_id)
                            domain_turn_presence[domain] += 1
                            
                            slot_total_seen[slot_name] += 1
                            v = str(value).strip().lower()
                            slot_value_examples[slot_name][v] += 1
                            
                            if v in none_like:
                                slot_none_like_count[slot_name] += 1
                            elif v in dontcare_like:
                                slot_dontcare_count[slot_name] += 1
                                slot_observed_count[slot_name] += 1
                            else:
                                slot_observed_count[slot_name] += 1
    
    # LUAS doesn't have explicit splits, assume train
    dialogues_by_split = {"train": n_dialogues, "val": 0, "test": 0}
    
    return {
        "n_dialogues": n_dialogues,
        "dialogues_by_split": dialogues_by_split,
        "n_turns": n_turns,
        "n_turns_with_metadata": n_turns,
        "n_user_turns": n_user_turns,
        "n_system_turns": n_system_turns,
        "turns_per_dialogue": turns_per_dialogue,
        "domain_turn_presence": domain_turn_presence,
        "domain_dialogue_presence": {k: len(v) for k, v in domain_dialogue_presence.items()},
        "slot_observed_count": slot_observed_count,
        "slot_none_like_count": slot_none_like_count,
        "slot_dontcare_count": slot_dontcare_count,
        "slot_total_seen": slot_total_seen,
        "slot_value_examples": slot_value_examples,
    }


# ============================================================================
# COMMON EDA COMPUTATION
# ============================================================================

def compute_eda(raw_data: dict) -> dict:
    """Compute derived statistics from parsed raw data."""
    
    turns_per_dialogue = raw_data["turns_per_dialogue"]
    sorted_turns = sorted(turns_per_dialogue) if turns_per_dialogue else []
    
    def quantile(p):
        if not sorted_turns:
            return 0
        idx = min(len(sorted_turns) - 1, max(0, int(round((len(sorted_turns) - 1) * p))))
        return sorted_turns[idx]
    
    # Label distribution
    total_slot_states = sum(raw_data["slot_total_seen"].values())
    total_none_like = sum(raw_data["slot_none_like_count"].values())
    total_dontcare = sum(raw_data["slot_dontcare_count"].values())
    total_filled = sum(raw_data["slot_observed_count"].values())
    
    return {
        **raw_data,
        "turns_per_dialogue_stats": {
            "min": min(sorted_turns) if sorted_turns else 0,
            "p25": quantile(0.25),
            "median": quantile(0.50),
            "p75": quantile(0.75),
            "max": max(sorted_turns) if sorted_turns else 0,
            "mean": sum(turns_per_dialogue) / len(turns_per_dialogue) if turns_per_dialogue else 0,
        },
        "total_slot_states": total_slot_states,
        "label_distribution": {
            "none_like": total_none_like,
            "none_like_pct": (total_none_like / total_slot_states * 100) if total_slot_states else 0,
            "dontcare": total_dontcare,
            "dontcare_pct": (total_dontcare / total_slot_states * 100) if total_slot_states else 0,
            "filled": total_filled,
            "filled_pct": (total_filled / total_slot_states * 100) if total_slot_states else 0,
        },
    }


# ============================================================================
# OUTPUT
# ============================================================================

def print_eda(eda: dict, dataset_name: str):
    """Pretty-print EDA results."""
    print("=" * 80)
    print(f"RAW {dataset_name.upper()} EDA")
    print("=" * 80)
    
    print(f"Dialogues total: {eda['n_dialogues']}")
    splits = eda["dialogues_by_split"]
    print(f"  train={splits['train']}, val={splits['val']}, test={splits['test']}")
    
    print(f"\nTurns total: {eda['n_turns']}")
    print(f"Turns with metadata: {eda['n_turns_with_metadata']}")
    print(f"User turns: {eda['n_user_turns']}")
    print(f"System turns: {eda['n_system_turns']}")
    
    ts = eda["turns_per_dialogue_stats"]
    print(
        f"\nTurns per dialogue: "
        f"min={ts['min']} p25={ts['p25']} "
        f"median={ts['median']} p75={ts['p75']} "
        f"max={ts['max']} mean={ts['mean']:.1f}"
    )
    
    print("\nDomains by turn presence:")
    for dom, cnt in sorted(eda["domain_turn_presence"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {dom:15s}: {cnt}")
    
    print("\nDomains by dialogue presence:")
    for dom, cnt in sorted(eda["domain_dialogue_presence"].items(), key=lambda x: -x[1])[:10]:
        print(f"  {dom:15s}: {cnt}")
    
    ld = eda["label_distribution"]
    print(f"\nLabel distribution (across all slot states):")
    print(f"  total states: {eda['total_slot_states']}")
    print(f"  none-like:    {ld['none_like']:8d} ({ld['none_like_pct']:5.2f}%)")
    print(f"  dontcare:     {ld['dontcare']:8d} ({ld['dontcare_pct']:5.2f}%)")
    print(f"  filled:       {ld['filled']:8d} ({ld['filled_pct']:5.2f}%)")
    
    print("\nTop 10 slots by filled values:")
    for slot, cnt in sorted(eda["slot_observed_count"].most_common(10), key=lambda x: -x[1]):
        total = eda["slot_total_seen"].get(slot, 0)
        print(f"  {slot:30s}: {cnt:6d} / {total:6d}")
    
    print("\nTop 10 slots by none-like values:")
    for slot, cnt in sorted(eda["slot_none_like_count"].most_common(10), key=lambda x: -x[1]):
        total = eda["slot_total_seen"].get(slot, 0)
        print(f"  {slot:30s}: {cnt:6d} / {total:6d}")


def export_csv(eda: dict, csv_prefix: str, dataset_name: str):
    """Export EDA results to CSV files."""
    base = Path(csv_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    
    # Slot stats
    slot_stats_path = base.parent / f"{base.name}_{dataset_name}_slot_stats.csv"
    with slot_stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot_name", "observed", "none_like", "dontcare", "total_seen"])
        
        all_slots = set(eda["slot_total_seen"].keys())
        for slot in sorted(all_slots):
            observed = eda["slot_observed_count"].get(slot, 0)
            none_like = eda["slot_none_like_count"].get(slot, 0)
            dontcare = eda["slot_dontcare_count"].get(slot, 0)
            total = eda["slot_total_seen"].get(slot, 0)
            writer.writerow([slot, observed, none_like, dontcare, total])
    
    print(f"Exported: {slot_stats_path}")
    
    # Top slot values
    top_values_path = base.parent / f"{base.name}_{dataset_name}_top_values.csv"
    with top_values_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot_name", "value", "count"])
        
        for slot in sorted(eda["slot_value_examples"].keys()):
            for value, count in sorted(
                eda["slot_value_examples"][slot].items(),
                key=lambda x: -x[1],
            )[:10]:  # Top 10 values per slot
                writer.writerow([slot, value, count])
    
    print(f"Exported: {top_values_path}")
    
    # Summary stats
    summary_path = base.parent / f"{base.name}_{dataset_name}_summary.csv"
    with summary_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["metric", "value"])
        
        writer.writerow(["dialogues_total", eda["n_dialogues"]])
        writer.writerow(["dialogues_train", eda["dialogues_by_split"]["train"]])
        writer.writerow(["dialogues_val", eda["dialogues_by_split"]["val"]])
        writer.writerow(["dialogues_test", eda["dialogues_by_split"]["test"]])
        writer.writerow(["turns_total", eda["n_turns"]])
        writer.writerow(["turns_with_metadata", eda["n_turns_with_metadata"]])
        writer.writerow(["turns_user", eda["n_user_turns"]])
        writer.writerow(["turns_system", eda["n_system_turns"]])
        
        ts = eda["turns_per_dialogue_stats"]
        writer.writerow(["turns_per_dialogue_min", ts["min"]])
        writer.writerow(["turns_per_dialogue_p25", ts["p25"]])
        writer.writerow(["turns_per_dialogue_median", ts["median"]])
        writer.writerow(["turns_per_dialogue_p75", ts["p75"]])
        writer.writerow(["turns_per_dialogue_max", ts["max"]])
        writer.writerow(["turns_per_dialogue_mean", f"{ts['mean']:.2f}"])
        
        ld = eda["label_distribution"]
        writer.writerow(["slot_states_total", eda["total_slot_states"]])
        writer.writerow(["slot_states_none_like", ld["none_like"]])
        writer.writerow(["slot_states_none_like_pct", f"{ld['none_like_pct']:.2f}"])
        writer.writerow(["slot_states_dontcare", ld["dontcare"]])
        writer.writerow(["slot_states_dontcare_pct", f"{ld['dontcare_pct']:.2f}"])
        writer.writerow(["slot_states_filled", ld["filled"]])
        writer.writerow(["slot_states_filled_pct", f"{ld['filled_pct']:.2f}"])
    
    print(f"Exported: {summary_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    ap = argparse.ArgumentParser(description="EDA for raw DST data (MultiWOZ, D0T, LUAS)")
    ap.add_argument(
        "--dataset",
        required=True,
        choices=["multiwoz", "d0t", "luas"],
        help="Which dataset to analyze",
    )
    ap.add_argument(
        "--csv-prefix",
        type=str,
        default=None,
        help="If provided, export CSV files with this prefix",
    )
    
    # MultiWOZ specific args
    ap.add_argument(
        "--data-path",
        type=Path,
        default=Path("data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/data.json"),
        help="[MultiWOZ] Path to data.json",
    )
    ap.add_argument(
        "--val-path",
        type=Path,
        default=Path("data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/valListFile.json"),
        help="[MultiWOZ] Path to valListFile.json",
    )
    ap.add_argument(
        "--test-path",
        type=Path,
        default=Path("data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/testListFile.json"),
        help="[MultiWOZ] Path to testListFile.json",
    )
    
    # D0T specific args
    ap.add_argument(
        "--d0t-dir",
        type=Path,
        default=Path("data_raw/d0t/data/dsg5k/train"),
        help="[D0T] Base directory containing turn.csv, slot.csv, slot_value.csv",
    )
    
    # LUAS specific args
    ap.add_argument(
        "--luas-json",
        type=Path,
        default=Path("data_raw/luas_repo/generation/multiwoz/datas/multiwoz.json"),
        help="[LUAS] Path to JSONL file",
    )
    
    args = ap.parse_args()
    
    # Parse dataset
    if args.dataset == "multiwoz":
        print(f"Loading data from: {args.data_path}")
        raw_data = parse_multiwoz(args.data_path, args.val_path, args.test_path)
    elif args.dataset == "d0t":
        print(f"Loading data from: {args.d0t_dir}")
        raw_data = parse_d0t(args.d0t_dir)
    elif args.dataset == "luas":
        print(f"Loading data from: {args.luas_json}")
        raw_data = parse_luas(args.luas_json)
    
    # Compute EDA
    eda = compute_eda(raw_data)
    print_eda(eda, args.dataset)
    
    # Export CSV if requested
    if args.csv_prefix:
        export_csv(eda, args.csv_prefix, args.dataset)


if __name__ == "__main__":
    main()
