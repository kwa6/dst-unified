import json
from pathlib import Path
from typing import Optional

from dst.data.unified_schema import UnifiedDSTExample
from dst.schemas import SlotSchema


def norm_value(v):
    if v is None:
        return "none"
    v = str(v).strip().lower()
    if v in {"", "none", "not mentioned", "not given"}:
        return "none"
    return v


def load_split_ids(path: Path) -> list[str]:
    txt = path.read_text(encoding="utf-8").strip()
    if not txt:
        raise ValueError(f"Split file is empty: {path}")
    if txt[0] in "[{":
        obj = json.loads(txt)
        if isinstance(obj, list):
            return [str(x).strip() for x in obj if str(x).strip()]
        raise ValueError(f"Unexpected JSON format in {path}: {type(obj)}")
    return [line.strip() for line in txt.splitlines() if line.strip()]


def infer_speaker_from_turn_id(turn_id: int) -> str:
    return "user" if turn_id % 2 == 0 else "system"


def write_split(
    split: str,
    ids: list[str],
    data: dict,
    schema: SlotSchema,
    out_path: Path,
    limit_dialogues: Optional[int] = None,
):
    out_path.parent.mkdir(parents=True, exist_ok=True)

    if limit_dialogues is not None:
        ids = ids[:limit_dialogues]

    n_written = 0
    n_dialogues = 0

    with out_path.open("w", encoding="utf-8") as f:
        for dialogue_id in ids:
            if dialogue_id not in data:
                continue

            dialogue = data[dialogue_id]["log"]
            context = ""
            n_dialogues += 1

            for turn_id, turn in enumerate(dialogue):
                text = turn["text"]
                speaker = infer_speaker_from_turn_id(turn_id)
                context += f"Turn {turn_id} [{speaker}]: {text}\n"

                metadata = turn.get("metadata") or {}
                if not metadata:
                    continue

                for domain, domain_data in metadata.items():
                    if not isinstance(domain_data, dict):
                        continue

                    semi = domain_data.get("semi", {}) or {}
                    book = domain_data.get("book", {}) or {}

                    # observed slot values from both informable and bookable slots
                    observed = {}
                    
                    # Extract informable slots
                    for slot, value in semi.items():
                        slot_name = f"{domain}-{slot}"
                        observed[slot_name] = norm_value(value)
                    
                    # Extract bookable slots (e.g., book day, book people, book time, book stay)
                    for slot, value in book.items():
                        slot_name = f"{domain}-book {slot}"
                        observed[slot_name] = norm_value(value)

                    # write examples for all schema slots in this domain
                    domain_prefix = f"{domain}-"
                    for slot_name in schema.slot_map.keys():
                        if not slot_name.startswith(domain_prefix):
                            continue

                        slot_info = schema.get(slot_name)
                        target = observed.get(slot_name, "none")

                        ex = UnifiedDSTExample(
                            dataset="multiwoz24",
                            split=split,
                            dialogue_id=dialogue_id,
                            turn_id=turn_id,
                            speaker="system",
                            dialogue_context=context.strip(),
                            slot_name=slot_name,
                            slot_description=slot_info.description,
                            value_examples=slot_info.examples,
                            target_value=target,
                        )
                        f.write(json.dumps(ex.to_json_dict(), ensure_ascii=False) + "\n")
                        n_written += 1

    print(f"[{split}] dialogues={n_dialogues} examples={n_written} -> {out_path}")


def build_multiwoz_splits(limit_train=None, limit_val=None, limit_test=None):
    schema = SlotSchema.from_json("schemas/multiwoz24_slots.json")

    base = Path("data_raw/MultiWOZ2.4/data/MULTIWOZ2.4/MULTIWOZ2.4")
    data_path = base / "data.json"
    val_path = base / "valListFile.json"
    test_path = base / "testListFile.json"

    data = json.loads(data_path.read_text(encoding="utf-8"))
    all_ids = list(data.keys())

    val_ids = set(load_split_ids(val_path))
    test_ids = set(load_split_ids(test_path))

    train_ids = [i for i in all_ids if i not in val_ids and i not in test_ids]
    val_ids = [i for i in all_ids if i in val_ids]
    test_ids = [i for i in all_ids if i in test_ids]

    out_base = Path("data_unified/multiwoz24")

    write_split("train", train_ids, data, schema, out_base / "train.jsonl", limit_dialogues=limit_train)
    write_split("val", val_ids, data, schema, out_base / "val.jsonl", limit_dialogues=limit_val)
    write_split("test", test_ids, data, schema, out_base / "test.jsonl", limit_dialogues=limit_test)


if __name__ == "__main__":
    # Build full splits for script-driven use
    build_multiwoz_splits()