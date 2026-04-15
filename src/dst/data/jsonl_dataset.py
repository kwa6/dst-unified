import json
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Optional, List, Dict, Any


@dataclass(frozen=True)
class JsonlExample:
    # keep it flexible; we validate key fields we need
    data: Dict[str, Any]

    @property
    def dialogue_context(self) -> str:
        return self.data["dialogue_context"]

    @property
    def slot_name(self) -> str:
        return self.data["slot_name"]

    @property
    def slot_description(self) -> str:
        return self.data["slot_description"]

    @property
    def target_value(self) -> str:
        return self.data["target_value"]

    @property
    def value_examples(self) -> Optional[List[str]]:
        return self.data.get("value_examples")


def iter_jsonl(path: str | Path, limit: Optional[int] = None) -> Iterator[Dict[str, Any]]:
    p = Path(path)
    n = 0
    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)
            n += 1
            if limit is not None and n >= limit:
                return


def load_jsonl(path: str | Path, limit: Optional[int] = None) -> List[JsonlExample]:
    out: List[JsonlExample] = []
    for obj in iter_jsonl(path, limit=limit):
        # minimal validation
        for k in ("dialogue_context", "slot_name", "slot_description", "target_value"):
            if k not in obj:
                raise ValueError(f"Missing key '{k}' in {path}")
        out.append(JsonlExample(obj))
    return out