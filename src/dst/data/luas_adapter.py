import json
import re
from pathlib import Path
from typing import Optional

from dst.data.unified_schema import UnifiedDSTExample
from dst.schemas import SlotSchema


def norm_text(v):
    if v is None:
        return ""
    return str(v).strip()


def norm_value(v):
    v = norm_text(v).lower()
    if v in {"", "none", "not mentioned", "not given", "?"}:
        return "none"
    return v


def norm_speaker(v):
    v = norm_text(v).lower()
    if v in {"user", "system"}:
        return v
    return "unknown"

def parse_turn_id(v) -> int:
    """
    LUAS turn_id can look like:
      '2'
      '3::follow_by_user_select_api'
      '3:follow_by_user_call_api'
    We extract the leading integer prefix.
    """
    v = norm_text(v)
    m = re.match(r"^(\d+)", v)
    if not m:
        raise ValueError(f"Could not parse LUAS turn_id: {v}")
    return int(m.group(1))

def is_meta_utterance(text: str) -> bool:
    """
    Filters obvious LUAS generation/internal artifacts from dialogue context.
    """
    text = norm_text(text)
    if not text:
        return True

    bad_exact = {
        "GenAPIConfig",
        "DoAPICall",
    }

    if text in bad_exact:
        return True

    return False

def fallback_description(slot_name: str) -> str:
    """
    Simple fallback if a LUAS slot is missing from schemas/luas_slots.json.
    Example: train-leaveat -> value for train leaveat
    """
    return f"value for {slot_name.replace('-', ' ')}"


def get_slot_description(slot_name: str, schema: SlotSchema) -> tuple[str, Optional[list[str]]]:
    if slot_name in schema.slot_map:
        info = schema.get(slot_name)
        return info.description, info.examples
    return fallback_description(slot_name), None


def convert_luas_train(
    luas_json_path: str | Path = "data_raw/luas_repo/generation/multiwoz/datas/multiwoz.json",
    out_path: str | Path = "data_unified/luas/train.jsonl",
    schema_path: str | Path = "schemas/luas_slots.json",
    limit_dialogues: Optional[int] = None,
):
    luas_json_path = Path(luas_json_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    schema = SlotSchema.from_json(schema_path)

    n_dialogues = 0
    n_turns = 0
    n_written = 0
    n_refs_without_slot_values = 0
    n_missing_schema = 0

    with luas_json_path.open("r", encoding="utf-8") as f_in, out_path.open("w", encoding="utf-8") as f_out:
        for d_idx, line in enumerate(f_in):
            if limit_dialogues is not None and d_idx >= limit_dialogues:
                break

            dialogue = json.loads(line)
            n_dialogues += 1

            dialogue_id = f"luas_{d_idx:06d}"
            turns = dialogue.get("turns", [])

            context_parts = []

            for turn in turns:
                turn_id = parse_turn_id(turn["turn_id"])
                speaker = norm_speaker(turn.get("speaker"))
                utterance = norm_text(turn.get("utterance"))

                # Build cumulative context, but skip LUAS internal/meta turns
                if not is_meta_utterance(utterance):
                    context_parts.append(f"Turn {turn_id} [{speaker}]: {utterance}")

                dialogue_context = "\n".join(context_parts)
                n_turns += 1

                # Only turns with structured state references are useful for DST supervision
                if "reference" not in turn:
                    continue

                refs = turn["reference"]
                if not isinstance(refs, list):
                    continue

                for ref in refs:
                    if not isinstance(ref, dict):
                        continue

                    # Keep only belief-state style references
                    if "slot_values" not in ref:
                        n_refs_without_slot_values += 1
                        continue

                    slot_values = ref.get("slot_values", {})
                    if not isinstance(slot_values, dict):
                        continue

                    for slot_name, values in slot_values.items():
                        # LUAS stores values as lists
                        if isinstance(values, list) and values:
                            value = norm_value(values[0])
                        else:
                            value = norm_value(values)

                        desc, value_examples = get_slot_description(slot_name, schema)
                        if slot_name not in schema.slot_map:
                            n_missing_schema += 1

                        ex = UnifiedDSTExample(
                            dataset="luas",
                            split="train",
                            dialogue_id=dialogue_id,
                            turn_id=turn_id,
                            speaker=speaker,
                            dialogue_context=dialogue_context,
                            slot_name=slot_name,
                            slot_description=desc,
                            value_examples=value_examples,
                            target_value=value,
                        )

                        f_out.write(json.dumps(ex.to_json_dict(), ensure_ascii=False) + "\n")
                        n_written += 1

    print(f"Wrote {n_written} examples to {out_path}")
    print(f"dialogues={n_dialogues} turns={n_turns}")
    print(f"refs_without_slot_values={n_refs_without_slot_values}")
    print(f"missing_schema_slots={n_missing_schema}")


if __name__ == "__main__":
    # Safe default: build a sample first
    convert_luas_train(
        luas_json_path="data_raw/luas_repo/generation/multiwoz/datas/multiwoz.json",
        out_path="data_unified/luas/train.jsonl",
        schema_path="schemas/luas_slots.json",
        limit_dialogues=None,
    )