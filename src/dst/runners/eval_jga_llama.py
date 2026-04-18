"""
Evaluate Llama Instruct on the DST task using Joint Goal Accuracy.

Tested with:
  - meta-llama/Llama-3.3-70B-Instruct   (needs 4×A100 on UCloud)
  - meta-llama/Llama-3.1-8B-Instruct    (needs 1×A100 on UCloud)
  - meta-llama/Llama-3.2-3B-Instruct    (needs 1×A100 on UCloud)
  - meta-llama/Llama-3.2-1B-Instruct    (runs on CPU)

Usage:
    $env:PYTHONPATH = "src"
    python -m dst.runners.eval_jga_llama `
        --path  data_unified/multiwoz24/val.jsonl `
        --model meta-llama/Llama-3.3-70B-Instruct
"""
import argparse
import csv
import json
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path

from tqdm import tqdm

from dst.analysis.eval_audit import (
    build_audit_record,
    build_audit_summary,
    canonicalize_value,
    normalize_raw_value,
)
from dst.data.jsonl_dataset import iter_jsonl
from dst.models.prompting import make_prompt_example
from dst.models.llama_dst import LlamaDSTModel


def main():
    ap = argparse.ArgumentParser(description="Evaluate Llama 3.1 DST model with JGA.")
    ap.add_argument("--path", required=True,
                    help="JSONL split file, e.g. data_unified/multiwoz24/val.jsonl")
    ap.add_argument("--model", default=LlamaDSTModel.DEFAULT_MODEL,
                    help="HF model name or local folder "
                         "(default: meta-llama/Llama-3.1-70B-Instruct)")
    ap.add_argument("--limit", type=int, default=None,
                    help="Limit number of JSONL rows (debug)")
    ap.add_argument("--max_turns", type=int, default=None,
                    help="Limit number of turns evaluated (debug)")
    ap.add_argument("--print_mismatches", type=int, default=0,
                    help="Print first N incorrect turns")
    ap.add_argument("--load_in_4bit",    action="store_true",
                    help="Load base model in 4-bit (needed for large models like 70B)")
    ap.add_argument("--force_cuda",      action="store_true",
                    help="Force CUDA usage even if torch.cuda.is_available() returns False (for old drivers on UCloud)")
    ap.add_argument("--results_file", default="results.csv", help="CSV file to log results (default: results.csv)")
    ap.add_argument("--mismatches_file", default=None, help="JSON file to save all mismatches for analysis")
    ap.add_argument("--use_slot_description", action="store_true", help="Include slot descriptions in prompts (default: off)")
    ap.add_argument("--use_value_examples", action="store_true", help="Include value examples in prompts (default: off)")
    ap.add_argument("--audit_file", default=None, help="JSON file to save all predictions for audit analysis")
    ap.add_argument("--audit_summary_file", default=None, help="JSON file to save audit summary metrics")
    args = ap.parse_args()

    # 1) Load and group rows by (dialogue_id, turn_id)
    groups = defaultdict(list)
    for obj in iter_jsonl(args.path, limit=args.limit):
        key = (obj["dialogue_id"], int(obj["turn_id"]))
        groups[key].append(obj)

    turn_keys = sorted(groups.keys())
    if args.max_turns is not None:
        turn_keys = turn_keys[: args.max_turns]

    print(f"Evaluating {len(turn_keys)} turns ({sum(len(groups[k]) for k in turn_keys)} slot predictions)...")

    # 2) Load model
    model = LlamaDSTModel(args.model, load_in_4bit=args.load_in_4bit, force_cuda=args.force_cuda)

    # 3) Evaluate
    total_turns = 0
    correct_turns = 0
    total_slots = 0
    correct_slots = 0
    total_non_none = 0
    correct_non_none = 0
    mismatches_printed = 0
    mismatches_data = []  # Collect all mismatches for JSON export
    audit_records = []
    audit_enabled = bool(args.audit_file or args.audit_summary_file)
    canonical_correct_slots = 0

    for idx, key in enumerate(tqdm(turn_keys, desc="Evaluating", unit="turn")):
        rows = groups[key]
        total_turns += 1

        turn_all_correct = True
        turn_mismatches = []

        for r in rows:
            pe = make_prompt_example(
                r["dialogue_context"],
                r["slot_name"],
                r["target_value"],
                slot_description=r.get("slot_description"),
                use_desc=args.use_slot_description,
                value_examples=r.get("value_examples"),
                use_examples=args.use_value_examples,
            )
            pred = normalize_raw_value(model.predict(pe.input_text))
            gold = normalize_raw_value(r["target_value"])
            pred_canon = canonicalize_value(r["slot_name"], pred)
            gold_canon = canonicalize_value(r["slot_name"], gold)

            total_slots += 1
            if pred == gold:
                correct_slots += 1
            else:
                turn_all_correct = False
                turn_mismatches.append((r["slot_name"], gold, pred))
                # Collect for JSON export
                d_id, t_id = key
                mismatches_data.append({
                    "dialogue_id": d_id,
                    "turn_id": t_id,
                    "slot_name": r["slot_name"],
                    "slot_description": r["slot_description"],
                    "gold": gold,
                    "pred": pred,
                    "context": r["dialogue_context"]
                })

            if pred_canon == gold_canon:
                canonical_correct_slots += 1

            if gold != "none":
                total_non_none += 1
                if pred == gold:
                    correct_non_none += 1

            if audit_enabled:
                d_id, t_id = key
                audit_records.append(
                    build_audit_record(
                        dialogue_id=d_id,
                        turn_id=t_id,
                        slot_name=r["slot_name"],
                        slot_description=r.get("slot_description"),
                        context=r["dialogue_context"],
                        prompt_text=pe.input_text,
                        gold_raw=gold,
                        pred_raw=pred,
                    )
                )

        if turn_all_correct:
            correct_turns += 1
        elif args.print_mismatches and mismatches_printed < args.print_mismatches:
            mismatches_printed += 1
            d_id, t_id = key
            print("\n--- INCORRECT TURN ---")
            print("dialogue_id:", d_id, "turn_id:", t_id)
            print("context (first 300 chars):",
                  rows[0]["dialogue_context"][:300].replace("\n", "\\n"))
            for sn, g, p in turn_mismatches[:10]:
                print(f"  {sn}: gold={g!r}  pred={p!r}")

    jga          = correct_turns    / total_turns    if total_turns    else 0.0
    slot_acc     = correct_slots    / total_slots    if total_slots    else 0.0
    non_none_acc = correct_non_none / total_non_none if total_non_none else 0.0
    canonical_slot_acc = canonical_correct_slots / total_slots if total_slots else 0.0

    print("\n===== RESULTS =====")
    print("file: ", args.path)
    print("model:", args.model)
    print(f"turns: {correct_turns}/{total_turns}  JGA={jga:.4f}")
    print(f"slots: {correct_slots}/{total_slots}  slot_acc={slot_acc:.4f}")
    print(f"non-none: {correct_non_none}/{total_non_none}  non_none_acc={non_none_acc:.4f}")
    print(f"canonical_slot_acc={canonical_slot_acc:.4f}")
    
    # Log results to CSV
    results_path = Path(args.results_file)
    file_exists = results_path.exists()
    
    with open(results_path, 'a', newline='') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(['timestamp', 'model', 'dataset', 'jga', 'slot_acc', 'non_none_acc'])
        
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        # Use parent directory name, removing "_final" suffix for clarity
        model_path = Path(args.model)
        if model_path.name == "final" and model_path.parent.name:
            model_name = model_path.parent.name.replace("_final", "")
        else:
            model_name = model_path.name.replace("_final", "")
        dataset_name = Path(args.path).stem
        writer.writerow([timestamp, model_name, dataset_name, f'{jga:.4f}', f'{slot_acc:.4f}', f'{non_none_acc:.4f}'])
    
    print(f"Results saved to: {args.results_file}")
    
    # Save mismatches to JSON if requested
    if args.mismatches_file:
        with open(args.mismatches_file, 'w') as f:
            json.dump(mismatches_data, f, indent=2)
        print(f"Mismatches saved to: {args.mismatches_file} ({len(mismatches_data)} errors)")

    if args.audit_file:
        with open(args.audit_file, "w") as f:
            json.dump(audit_records, f, indent=2)
        print(f"Audit records saved to: {args.audit_file} ({len(audit_records)} rows)")

    if args.audit_summary_file:
        summary = build_audit_summary(audit_records)
        with open(args.audit_summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        print(f"Audit summary saved to: {args.audit_summary_file}")


if __name__ == "__main__":
    main()
