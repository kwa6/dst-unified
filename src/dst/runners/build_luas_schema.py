import json
from pathlib import Path


def fallback_description(slot_name: str) -> str:
    return f"value for {slot_name.replace('-', ' ')}"


def main():
    multiwoz_schema_path = Path("schemas/multiwoz24_slots.json")
    luas_json_path = Path("data_raw/luas_repo/generation/multiwoz/datas/multiwoz.json")
    out_path = Path("schemas/luas_slots.json")

    multiwoz_schema = json.loads(multiwoz_schema_path.read_text(encoding="utf-8"))

    seen_slots = set()

    with luas_json_path.open("r", encoding="utf-8") as f:
        for line in f:
            d = json.loads(line)
            for turn in d.get("turns", []):
                refs = turn.get("reference", [])
                if not isinstance(refs, list):
                    continue
                for ref in refs:
                    if not isinstance(ref, dict):
                        continue
                    slot_values = ref.get("slot_values")
                    if not isinstance(slot_values, dict):
                        continue
                    for slot_name in slot_values.keys():
                        seen_slots.add(slot_name)

    out = {}
    for slot_name in sorted(seen_slots):
        if slot_name in multiwoz_schema:
            out[slot_name] = multiwoz_schema[slot_name]
        else:
            out[slot_name] = {
                "description": fallback_description(slot_name),
                "examples": None,
            }

    out_path.write_text(json.dumps(out, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"Wrote {len(out)} slots to {out_path}")


if __name__ == "__main__":
    main()