from __future__ import annotations

import re
import string


NONE_LIKE = {
    "",
    "none",
    "not mentioned",
    "not given",
    "empty",
    "unknown",
    "null",
    "?",
}

DONTCARE_LIKE = {
    "dontcare",
    "dont care",
    "dont-care",
    "don't care",
    "dontcare?",
}

TURN_LINE_RE = re.compile(
    r"^\s*Turn\s+(?P<turn_id>\d+)\s*(?:\[(?P<speaker>[^\]]+)\])?\s*:\s*(?P<text>.*)$",
    re.IGNORECASE,
)


def _strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def normalize_text(value: str) -> str:
    if value is None:
        return ""
    v = str(value).strip().lower()
    if not v:
        return ""
    v = _strip_punctuation(v)
    v = " ".join(v.split())
    return v


def normalize_value(value: str) -> str:
    v = normalize_text(value)
    if v in DONTCARE_LIKE:
        return "dontcare"
    if v in NONE_LIKE:
        return "none"
    return v


def is_none_value(value: str) -> bool:
    return normalize_value(value) == "none"


def is_dontcare_value(value: str) -> bool:
    return normalize_value(value) == "dontcare"


def check_exact_alignment(value: str, context: str) -> bool:
    if not value or not context:
        return False
    return str(value).lower() in str(context).lower()


def check_canonical_alignment(value: str, context: str) -> bool:
    if not value or not context:
        return False
    v_norm = normalize_text(value)
    ctx_norm = normalize_text(context)
    if not v_norm or not ctx_norm:
        return False
    return v_norm in ctx_norm


def normalize_speaker_label(label: str | None) -> str | None:
    if label is None:
        return None
    label = str(label).strip().lower()
    if label in {"user", "system"}:
        return label
    return "unknown"


def split_context_turns(context: str) -> list[dict]:
    turns: list[dict] = []
    if not context:
        return turns
    for line in str(context).splitlines():
        if not line.strip():
            continue
        match = TURN_LINE_RE.match(line)
        if not match:
            continue
        speaker_raw = match.group("speaker")
        speaker = normalize_speaker_label(speaker_raw) if speaker_raw is not None else None
        turns.append(
            {
                "turn_id": int(match.group("turn_id")),
                "speaker": speaker,
                "text": match.group("text"),
                "has_speaker_tag": speaker_raw is not None,
            }
        )
    return turns


def extract_user_context(context: str) -> str:
    turns = split_context_turns(context)
    if not turns:
        return ""
    has_tag = any(t["has_speaker_tag"] for t in turns)
    user_lines: list[str] = []
    if has_tag:
        for turn in turns:
            if turn["speaker"] == "user":
                user_lines.append(turn["text"])
    else:
        for turn in turns:
            if turn["turn_id"] % 2 == 0:
                user_lines.append(turn["text"])
    return "\n".join(user_lines)


def classify_alignment(value: str, context: str) -> str:
    user_ctx = extract_user_context(context)
    if check_exact_alignment(value, user_ctx):
        return "direct_user_exact"
    if check_canonical_alignment(value, user_ctx):
        return "direct_user_canonical"
    if check_exact_alignment(value, context):
        return "direct_full_exact"
    if check_canonical_alignment(value, context):
        return "direct_full_canonical"
    return "not_directly_aligned"
