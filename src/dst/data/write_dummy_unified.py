import json
from pathlib import Path

from dst.data.unified_schema import UnifiedDSTExample
from dst.schemas import SlotSchema


def main():
    schema = SlotSchema.from_json("schemas/multiwoz24_slots.json")

    out_path = Path("data_unified/dummy_from_schema.jsonl")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    ctx0 = "User: I want a cheap hotel in the centre."
    ctx1 = ctx0 + "\nSystem: Sure, do you have a preferred star rating?"

    def mk(slot_name: str, target: str, turn_id: int, speaker: str, ctx: str):
        info = schema.get(slot_name)
        return UnifiedDSTExample(
            dataset="dummy",
            split="train",
            dialogue_id="dlg-001",
            turn_id=turn_id,
            speaker=speaker,
            dialogue_context=ctx,
            slot_name=slot_name,
            slot_description=info.description,
            value_examples=info.examples,
            target_value=target,
        )

    examples = [
        mk("hotel-pricerange", "cheap", 0, "user", ctx0),
        mk("hotel-area", "centre", 0, "user", ctx0),
        mk("hotel-pricerange", "none", 1, "system", ctx1),
    ]

    with out_path.open("w", encoding="utf-8") as f:
        for ex in examples:
            f.write(json.dumps(ex.to_json_dict(), ensure_ascii=False) + "\n")

    print(f"Wrote {len(examples)} examples to {out_path}")


if __name__ == "__main__":
    main()