import csv
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional

from dst.data.unified_schema import UnifiedDSTExample


def norm_text(v: str) -> str:
    if v is None:
        return ""
    return str(v).strip().strip('"')


def norm_value(v: str) -> str:
    v = norm_text(v).lower()
    if v in {"", "?", "none", "not mentioned", "not given"}:
        return "none"
    return v


def load_turns(turn_csv_path: str | Path):
    """
    Returns:
      turns_by_raw_turn_id: raw turn_id -> turn row
      ordered_turns_by_dialogue: dialogue_id -> list of turns sorted by turn_index
    """
    turns_by_raw_turn_id: Dict[str, dict] = {}
    turns_by_dialogue: Dict[str, List[dict]] = defaultdict(list)

    with Path(turn_csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: norm_text(v) for k, v in row.items()}
            row["turn_index"] = int(row["turn_index"])

            raw_turn_id = row["turn_id"]
            dialogue_id = row["dialogue"]

            turns_by_raw_turn_id[raw_turn_id] = row
            turns_by_dialogue[dialogue_id].append(row)

    ordered_turns_by_dialogue: Dict[str, List[dict]] = {}
    for dialogue_id, rows in turns_by_dialogue.items():
        ordered_turns_by_dialogue[dialogue_id] = sorted(rows, key=lambda x: x["turn_index"])

    return turns_by_raw_turn_id, ordered_turns_by_dialogue


def load_slots(slot_csv_path: str | Path):
    """
    Returns:
      slots_by_slot_id: slot_id -> slot row
    """
    slots_by_slot_id: Dict[str, dict] = {}

    with Path(slot_csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: norm_text(v) for k, v in row.items()}
            slots_by_slot_id[row["slot_id"]] = row

    return slots_by_slot_id


def load_value_candidates(value_candidate_csv_path: str | Path) -> Dict[str, List[str]]:
    """
    Load candidate values for each slot.
    Returns:
      candidates_by_slot_id: slot_id -> list of candidate values
    """
    candidates_by_slot_id: Dict[str, List[str]] = defaultdict(list)

    with Path(value_candidate_csv_path).open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            row = {k: norm_text(v) for k, v in row.items()}
            slot_id = row["slot_id"]
            candidate_value = row["candidate_value"]
            if candidate_value:
                candidates_by_slot_id[slot_id].append(candidate_value)

    return dict(candidates_by_slot_id)


def build_dialogue_contexts(ordered_turns_by_dialogue: Dict[str, List[dict]]):
    """
    Returns:
      contexts[(dialogue_id, turn_index)] = cumulative dialogue text up to that turn
    """
    contexts: Dict[tuple, str] = {}

    for dialogue_id, turns in ordered_turns_by_dialogue.items():
        pieces = []
        for t in turns:
            pieces.append(f'Turn {t["turn_index"]} [{t["speaker"]}]: {t["text"]}')
            contexts[(dialogue_id, t["turn_index"])] = "\n".join(pieces)

    return contexts


def convert_d0t_train(
    d0t_train_dir: str | Path = "data_raw/d0t/data/dsg5k/train",
    out_path: str | Path = "data_unified/d0t/train.jsonl",
    limit_rows: Optional[int] = None,
):
    """
    Converts D0T train split into unified JSONL.

    Expected files under d0t_train_dir:
      - turn.csv
      - slot.csv
      - slot_value.csv
      - value_candidate.csv
    """
    d0t_train_dir = Path(d0t_train_dir)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    turn_csv = d0t_train_dir / "turn.csv"
    slot_csv = d0t_train_dir / "slot.csv"
    slot_value_csv = d0t_train_dir / "slot_value.csv"
    value_candidate_csv = d0t_train_dir / "value_candidate.csv"

    print("Loading turns...")
    turns_by_raw_turn_id, ordered_turns_by_dialogue = load_turns(turn_csv)

    print("Loading slots...")
    slots_by_slot_id = load_slots(slot_csv)

    print("Loading value candidates...")
    candidates_by_slot_id = load_value_candidates(value_candidate_csv)

    print("Building dialogue contexts...")
    contexts = build_dialogue_contexts(ordered_turns_by_dialogue)

    n_written = 0

    print("Converting slot_value rows...")
    with slot_value_csv.open(newline="", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        reader = csv.DictReader(f_in)

        for i, row in enumerate(reader):
            if limit_rows is not None and i >= limit_rows:
                break

            row = {k: norm_text(v) for k, v in row.items()}

            raw_turn_id = row["turn_id"]
            slot_id = row["slot_id"]

            if raw_turn_id not in turns_by_raw_turn_id:
                continue
            if slot_id not in slots_by_slot_id:
                continue

            turn = turns_by_raw_turn_id[raw_turn_id]
            slot = slots_by_slot_id[slot_id]

            dialogue_id = turn["dialogue"]
            turn_index = int(turn["turn_index"])
            context = contexts[(dialogue_id, turn_index)]

            # Get candidate values for this slot
            value_examples = candidates_by_slot_id.get(slot_id)

            ex = UnifiedDSTExample(
                dataset="d0t",
                split="train",
                dialogue_id=dialogue_id,
                turn_id=turn_index,
                speaker=turn["speaker"] if turn["speaker"] else "unknown",
                dialogue_context=context,
                slot_name=slot["slot"],
                slot_description=slot["description"],
                value_examples=value_examples,
                target_value=norm_value(row["value"]),
            )

            f_out.write(json.dumps(ex.to_json_dict(), ensure_ascii=False) + "\n")
            n_written += 1

    print(f"Wrote {n_written} examples to {out_path}")


if __name__ == "__main__":
    convert_d0t_train(
        d0t_train_dir="data_raw/d0t/data/dsg5k/train",
        out_path="data_unified/d0t/train.jsonl",
        limit_rows=None,
    )