from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Literal, Optional, List, Dict, Any


Split = Literal["train", "dev", "test"]
Speaker = Literal["user", "system", "unknown"]


@dataclass(frozen=True)
class UnifiedDSTExample:
    dataset: str
    split: Split
    dialogue_id: str
    turn_id: int
    speaker: Speaker
    dialogue_context: str
    slot_name: str
    target_value: str  # value or "none"
    slot_description: Optional[str] = None
    value_examples: Optional[List[str]] = None

    def to_json_dict(self) -> Dict[str, Any]:
        d = asdict(self)
        # JSONL cleanliness: omit nulls
        return {k: v for k, v in d.items() if v is not None}