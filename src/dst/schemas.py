import json
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, List, Dict, Any


@dataclass(frozen=True)
class SlotInfo:
    description: str
    examples: Optional[List[str]] = None


class SlotSchema:
    def __init__(self, slot_map: Dict[str, SlotInfo]):
        self.slot_map = slot_map

    @classmethod
    def from_json(cls, path: str | Path) -> "SlotSchema":
        p = Path(path)
        raw = json.loads(p.read_text(encoding="utf-8"))
        slot_map: Dict[str, SlotInfo] = {}
        for slot_name, info in raw.items():
            desc = info.get("description")
            if not desc:
                raise ValueError(f"Missing description for slot {slot_name} in {p}")
            slot_map[slot_name] = SlotInfo(
                description=desc,
                examples=info.get("examples"),
            )
        return cls(slot_map)

    def get(self, slot_name: str) -> SlotInfo:
        if slot_name not in self.slot_map:
            raise KeyError(f"Slot '{slot_name}' not found in schema.")
        return self.slot_map[slot_name]