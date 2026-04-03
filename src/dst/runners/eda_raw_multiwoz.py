"""
Exploratory Data Analysis for raw MultiWOZ 2.4 data.

Analyzes data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/data.json and split files.
Outputs statistics on dialogues, turns, domains, slots, and label distribution.

Usage:
    python eda_raw_multiwoz.py [--csv-prefix OUTPUT_PREFIX]

Example:
    python eda_raw_multiwoz.py --csv-prefix eda_output
"""

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path


def compute_eda(data_path: Path, val_path: Path, test_path: Path):
    """Compute comprehensive EDA stats from raw MultiWOZ data."""
    
    # Load data and split IDs
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
    
    # Initialization
    turns_per_dialogue = []
    n_turns = 0
    n_turns_with_metadata = 0
    n_user_turns = 0
    n_system_turns = 0
    
    domain_turn_presence = Counter()
    domain_dialogue_presence = defaultdict(set)
    slot_observed_count = Counter()
    slot_none_like_count = Counter()
    slot_dontcare_count = Counter()
    slot_total_seen = Counter()
    slot_value_examples = defaultdict(lambda: defaultdict(int))
    
    dialogues_by_split = {"train": 0, "val": 0, "test": 0}
    
    none_like = {"", "none", "not mentioned", "not given"}
    dontcare_like = {"dontcare", "dont care", "don't care", "do not care"}
    
    # Traverse all dialogues
    for did, d in data.items():
        # Assign to split
        if did in test_ids:
            dialogues_by_split["test"] += 1
        elif did in val_ids:
            dialogues_by_split["val"] += 1
        else:
            dialogues_by_split["train"] += 1
        
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
    
    # Compute quantiles for turns per dialogue
    sorted_turns = sorted(turns_per_dialogue)
    
    def quantile(p):
        if not sorted_turns:
            return 0
        idx = min(
            len(sorted_turns) - 1,
            max(0, int(round((len(sorted_turns) - 1) * p))),
        )
        return sorted_turns[idx]
    
    # Aggregate label distribution
    total_slot_states = sum(slot_total_seen.values())
    total_none_like = sum(slot_none_like_count.values())
    total_dontcare = sum(slot_dontcare_count.values())
    total_filled = sum(slot_observed_count.values())
    
    return {
        "n_dialogues": len(all_ids),
        "dialogues_by_split": dialogues_by_split,
        "n_turns": n_turns,
        "n_turns_with_metadata": n_turns_with_metadata,
        "n_user_turns": n_user_turns,
        "n_system_turns": n_system_turns,
        "turns_per_dialogue_stats": {
            "min": min(sorted_turns) if sorted_turns else 0,
            "p25": quantile(0.25),
            "median": quantile(0.50),
            "p75": quantile(0.75),
            "max": max(sorted_turns) if sorted_turns else 0,
            "mean": sum(turns_per_dialogue) / len(turns_per_dialogue)
            if turns_per_dialogue
            else 0,
        },
        "domain_turn_presence": dict(domain_turn_presence.most_common()),
        "domain_dialogue_presence": {
            k: len(v) for k, v in sorted(domain_dialogue_presence.items())
        },
        "total_slot_states": total_slot_states,
        "label_distribution": {
            "none_like": total_none_like,
            "none_like_pct": (total_none_like / total_slot_states * 100)
            if total_slot_states
            else 0,
            "dontcare": total_dontcare,
            "dontcare_pct": (total_dontcare / total_slot_states * 100)
            if total_slot_states
            else 0,
            "filled": total_filled,
            "filled_pct": (total_filled / total_slot_states * 100)
            if total_slot_states
            else 0,
        },
        "slots": {
            "observed_count": slot_observed_count,
            "none_like_count": slot_none_like_count,
            "dontcare_count": slot_dontcare_count,
            "total_seen": slot_total_seen,
            "value_examples": slot_value_examples,
        },
    }


def print_eda(eda: dict):
    """Pretty-print EDA results."""
    print("=" * 80)
    print("RAW MULTIWOZ 2.4 EDA")
    print("=" * 80)
    
    print(f"Dialogues total: {eda['n_dialogues']}")
    splits = eda["dialogues_by_split"]
    print(
        f"  train={splits['train']}, val={splits['val']}, test={splits['test']}"
    )
    
    print(f"\nTurns total: {eda['n_turns']}")
    print(f"Turns with metadata: {eda['n_turns_with_metadata']}")
    print(f"User turns (by parity): {eda['n_user_turns']}")
    print(f"System turns (by parity): {eda['n_system_turns']}")
    
    ts = eda["turns_per_dialogue_stats"]
    print(
        f"\nTurns per dialogue: "
        f"min={ts['min']} p25={ts['p25']} "
        f"median={ts['median']} p75={ts['p75']} "
        f"max={ts['max']} mean={ts['mean']:.1f}"
    )
    
    print("\nDomains by turn presence in metadata:")
    for dom, cnt in sorted(
        eda["domain_turn_presence"].items(), key=lambda x: -x[1]
    ):
        print(f"  {dom:15s}: {cnt}")
    
    print("\nDomains by dialogue presence:")
    for dom, cnt in sorted(
        eda["domain_dialogue_presence"].items(), key=lambda x: -x[1]
    ):
        print(f"  {dom:15s}: {cnt}")
    
    ld = eda["label_distribution"]
    print(f"\nLabel distribution (across all slot states):")
    print(f"  total states: {eda['total_slot_states']}")
    print(f"  none-like:    {ld['none_like']:8d} ({ld['none_like_pct']:5.2f}%)")
    print(f"  dontcare:     {ld['dontcare']:8d} ({ld['dontcare_pct']:5.2f}%)")
    print(f"  filled:       {ld['filled']:8d} ({ld['filled_pct']:5.2f}%)")
    
    print("\nTop 15 slots by observed (filled) values:")
    for slot, cnt in sorted(
        eda["slots"]["observed_count"].most_common(15),
        key=lambda x: -x[1],
    ):
        total = eda["slots"]["total_seen"].get(slot, 0)
        print(f"  {slot:30s}: {cnt:6d} / {total:6d}")
    
    print("\nTop 15 slots by none-like values:")
    for slot, cnt in sorted(
        eda["slots"]["none_like_count"].most_common(15),
        key=lambda x: -x[1],
    ):
        total = eda["slots"]["total_seen"].get(slot, 0)
        print(f"  {slot:30s}: {cnt:6d} / {total:6d}")


def export_csv(eda: dict, csv_prefix: str):
    """Export EDA results to CSV files."""
    base = Path(csv_prefix)
    base.parent.mkdir(parents=True, exist_ok=True)
    
    # Slot stats
    slot_stats_path = base.parent / f"{base.name}_slot_stats.csv"
    with slot_stats_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["slot_name", "observed", "none_like", "dontcare", "total_seen"]
        )
        
        all_slots = set(eda["slots"]["total_seen"].keys())
        for slot in sorted(all_slots):
            observed = eda["slots"]["observed_count"].get(slot, 0)
            none_like = eda["slots"]["none_like_count"].get(slot, 0)
            dontcare = eda["slots"]["dontcare_count"].get(slot, 0)
            total = eda["slots"]["total_seen"].get(slot, 0)
            writer.writerow([slot, observed, none_like, dontcare, total])
    
    print(f"Exported: {slot_stats_path}")
    
    # Top slot values
    top_values_path = base.parent / f"{base.name}_top_values.csv"
    with top_values_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["slot_name", "value", "count"])
        
        for slot in sorted(eda["slots"]["value_examples"].keys()):
            for value, count in sorted(
                eda["slots"]["value_examples"][slot].items(),
                key=lambda x: -x[1],
            )[:10]:  # Top 10 values per slot
                writer.writerow([slot, value, count])
    
    print(f"Exported: {top_values_path}")
    
    # Summary stats
    summary_path = base.parent / f"{base.name}_summary.csv"
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


def main():
    ap = argparse.ArgumentParser(
        description="EDA for raw MultiWOZ 2.4 data"
    )
    ap.add_argument(
        "--data-path",
        type=Path,
        default=Path(
            "data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/data.json"
        ),
        help="Path to data.json",
    )
    ap.add_argument(
        "--val-path",
        type=Path,
        default=Path(
            "data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/valListFile.json"
        ),
        help="Path to valListFile.json",
    )
    ap.add_argument(
        "--test-path",
        type=Path,
        default=Path(
            "data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4/testListFile.json"
        ),
        help="Path to testListFile.json",
    )
    ap.add_argument(
        "--csv-prefix",
        type=str,
        default=None,
        help="If provided, export CSV files with this prefix",
    )
    
    args = ap.parse_args()
    
    print(f"Loading data from: {args.data_path}")
    eda = compute_eda(args.data_path, args.val_path, args.test_path)
    
    print_eda(eda)
    
    if args.csv_prefix:
        export_csv(eda, args.csv_prefix)


if __name__ == "__main__":
    main()
