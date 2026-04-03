import argparse
import csv
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dst.data.jsonl_dataset import iter_jsonl
from dst.models.prompting import make_prompt_example
from dst.models.t5_dst import T5DSTModel


def norm(v: str) -> str:
    v = (v or "").strip().lower()
    if v in {"", "none", "not mentioned", "not given"}:
        return "none"
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--path", required=True, help="JSONL split file, e.g. data_unified/multiwoz24/val.jsonl")
    ap.add_argument("--model", required=True, help="HF model name or local folder, e.g. runs/t5_mwoz_train_v1/final")
    ap.add_argument("--limit", type=int, default=None, help="Limit number of JSONL rows (debug)")
    ap.add_argument("--max_turns", type=int, default=None, help="Limit number of turns (debug)")
    ap.add_argument("--print_mismatches", type=int, default=0, help="Print first N incorrect turns")
    ap.add_argument("--results_file", default="results.csv", help="CSV file to log results (default: results.csv)")
    args = ap.parse_args()

    # 1) Load rows and group by (dialogue_id, turn_id)
    groups = defaultdict(list)
    for obj in iter_jsonl(args.path, limit=args.limit):
        key = (obj["dialogue_id"], int(obj["turn_id"]))
        groups[key].append(obj)

    # Optional: cap the number of turns evaluated
    turn_keys = sorted(groups.keys())
    if args.max_turns is not None:
        turn_keys = turn_keys[: args.max_turns]

    # 2) Load model
    model = T5DSTModel(args.model)

    # 3) Evaluate
    total_turns = 0
    correct_turns = 0

    total_slots = 0
    correct_slots = 0

    total_non_none = 0
    correct_non_none = 0

    mismatches_printed = 0

    for key in tqdm(turn_keys, desc="Evaluating", unit="turn"):
        rows = groups[key]
        total_turns += 1

        turn_all_correct = True
        turn_mismatches = []

        for r in rows:
            pe = make_prompt_example(
                r["dialogue_context"],
                r["slot_name"],
                r["slot_description"],
                r["target_value"],
            )
            pred = norm(model.predict(pe.input_text))
            gold = norm(r["target_value"])

            total_slots += 1
            if pred == gold:
                correct_slots += 1
            else:
                turn_all_correct = False
                turn_mismatches.append((r["slot_name"], gold, pred))

            if gold != "none":
                total_non_none += 1
                if pred == gold:
                    correct_non_none += 1

        if turn_all_correct:
            correct_turns += 1
        else:
            if args.print_mismatches and mismatches_printed < args.print_mismatches:
                mismatches_printed += 1
                d_id, t_id = key
                print("\n--- INCORRECT TURN ---")
                print("dialogue_id:", d_id, "turn_id:", t_id)
                # show one context (same for all rows in the turn)
                print("context (first 300 chars):", rows[0]["dialogue_context"][:300].replace("\n", "\\n"))
                for sn, g, p in turn_mismatches[:10]:
                    print(f"  {sn}: gold={g} pred={p}")

    jga = correct_turns / total_turns if total_turns else 0.0
    slot_acc = correct_slots / total_slots if total_slots else 0.0
    non_none_acc = correct_non_none / total_non_none if total_non_none else 0.0

    print("\n===== RESULTS =====")
    print("file:", args.path)
    print("model:", args.model)
    print(f"turns: {correct_turns}/{total_turns}  JGA={jga:.4f}")
    print(f"slots: {correct_slots}/{total_slots} slot_acc={slot_acc:.4f}")
    print(f"non-none: {correct_non_none}/{total_non_none} non_none_acc={non_none_acc:.4f}")
    
    # Log results to CSV
    results_path = Path(args.results_file)
    file_exists = results_path.exists()
    
    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'model', 'dataset', 'jga', 'slot_acc', 'non_none_acc'])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        model_name = Path(args.model).name
        dataset_name = Path(args.path).stem
        writer.writerow([timestamp, model_name, dataset_name, f'{jga:.4f}', f'{slot_acc:.4f}', f'{non_none_acc:.4f}'])
    
    print(f"\nResults saved to: {args.results_file}")


if __name__ == "__main__":
    main()