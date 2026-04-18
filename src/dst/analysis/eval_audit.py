from __future__ import annotations

import re
import string
from collections import defaultdict
from typing import Any


_NONE_VALUES = {"", "none", "not mentioned", "not given"}

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


def normalize_raw_value(value: str) -> str:
    """Normalize raw slot values using the current evaluator behavior."""
    value = (value or "").strip().lower()
    if value in _NONE_VALUES:
        return "none"
    return value


def _strip_punctuation(text: str) -> str:
    return text.translate(str.maketrans("", "", string.punctuation))


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _normalize_punctuation_spacing(text: str) -> str:
    # Only remove spaces before simple punctuation to avoid over-normalization.
    return re.sub(r"\s+([,.;:!?])", r"\1", text)


def _apply_lexical_normalizations(text: str) -> str:
    text = re.sub(r"\bguesthouse\b", "guest house", text)
    text = re.sub(r"\bguest-house\b", "guest house", text)
    text = re.sub(r"\bb\s*&\s*b\b", "bed and breakfast", text)
    text = re.sub(r"\bb\s+and\s+b\b", "bed and breakfast", text)
    text = re.sub(r"\bbed\s*&\s*breakfast\b", "bed and breakfast", text)
    return text


def canonicalize_value(slot_name: str, value: str) -> str:
    """Conservative canonicalization for alignment analysis."""
    value = normalize_raw_value(value)
    value = _collapse_whitespace(value)
    value = _normalize_punctuation_spacing(value)
    value = _apply_lexical_normalizations(value)

    if slot_name == "restaurant-name" and value.endswith(" restaurant"):
        trimmed = value[: -len(" restaurant")].strip()
        if trimmed:
            value = trimmed

    return value


def _normalize_context_text(text: str) -> str:
    text = (text or "").strip().lower()
    text = _collapse_whitespace(text)
    text = _normalize_punctuation_spacing(text)
    return text


def _canonicalize_context_text(text: str) -> str:
    text = _normalize_context_text(text)
    text = _apply_lexical_normalizations(text)
    text = _collapse_whitespace(text)
    return text


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


def split_context_turns(dialogue_context: str) -> list[dict[str, Any]]:
    """Split context into turns using Turn X prefixes with optional speaker tags."""
    turns: list[dict[str, Any]] = []

    for line in (dialogue_context or "").splitlines():
        if not line.strip():
            continue
        match = TURN_LINE_RE.match(line)
        if not match:
            continue
        turn_id = int(match.group("turn_id"))
        speaker_raw = match.group("speaker")
        speaker = normalize_speaker_label(speaker_raw) if speaker_raw is not None else None
        has_speaker_tag = speaker_raw is not None
        is_user_guess = (speaker == "user") if has_speaker_tag else (turn_id % 2 == 0)
        turns.append({
            "turn_id": turn_id,
            "text": match.group("text").strip(),
            "speaker": speaker,
            "has_speaker_tag": has_speaker_tag,
            "is_user_guess": is_user_guess,
        })

    if turns:
        return turns

    return [{
        "turn_id": 0,
        "text": (dialogue_context or "").strip(),
        "speaker": None,
        "has_speaker_tag": False,
        "is_user_guess": True,
    }]


def extract_user_context(context: str) -> str:
    turns = split_context_turns(context)
    if not turns:
        return ""
    has_tag = any(t.get("has_speaker_tag") for t in turns)
    user_lines: list[str] = []
    if has_tag:
        for turn in turns:
            if turn.get("speaker") == "user":
                user_lines.append(turn.get("text", ""))
    else:
        for turn in turns:
            if turn.get("turn_id", 0) % 2 == 0:
                user_lines.append(turn.get("text", ""))
    return "\n".join([line for line in user_lines if line])


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


def _value_in_context(value_raw: str, value_canon: str, dialogue_context: str) -> bool:
    if not value_raw or value_raw in {"none", "dontcare"}:
        return False
    context_raw = _normalize_context_text(dialogue_context)
    if value_raw in context_raw:
        return True
    context_canon = _canonicalize_context_text(dialogue_context)
    return value_canon in context_canon


def compute_alignment_features(
    slot_name: str,
    gold_raw: str,
    gold_canon: str,
    dialogue_context: str,
) -> dict[str, bool]:
    """Compute conservative alignment features for a slot value."""
    gold_raw = normalize_raw_value(gold_raw)
    gold_canon = canonicalize_value(slot_name, gold_canon)

    if gold_raw in {"none", "dontcare"}:
        return {
            "gold_in_full_context_exact": False,
            "gold_in_full_context_canon": False,
            "gold_in_user_turns_exact": False,
            "gold_in_user_turns_canon": False,
        }

    context_raw = _normalize_context_text(dialogue_context)
    context_canon = _canonicalize_context_text(dialogue_context)

    in_full_exact = gold_raw in context_raw
    in_full_canon = gold_canon in context_canon

    in_user_exact = False
    in_user_canon = False
    for turn in split_context_turns(dialogue_context):
        if not turn.get("is_user_guess"):
            continue
        turn_raw = _normalize_context_text(turn.get("text", ""))
        turn_canon = _canonicalize_context_text(turn.get("text", ""))
        if gold_raw in turn_raw:
            in_user_exact = True
        if gold_canon in turn_canon:
            in_user_canon = True

    return {
        "gold_in_full_context_exact": in_full_exact,
        "gold_in_full_context_canon": in_full_canon,
        "gold_in_user_turns_exact": in_user_exact,
        "gold_in_user_turns_canon": in_user_canon,
    }


def derive_alignment_class(gold_raw: str, features: dict[str, bool]) -> str:
    """Assign a coarse alignment class based on contextual features."""
    gold_raw = normalize_raw_value(gold_raw)
    if gold_raw == "none":
        return "none_value"
    if gold_raw == "dontcare":
        return "dontcare_value"

    if features.get("gold_in_user_turns_exact"):
        return "direct_user_exact"
    if features.get("gold_in_user_turns_canon"):
        return "direct_user_canonical"
    if features.get("gold_in_full_context_exact"):
        return "direct_full_exact"
    if features.get("gold_in_full_context_canon"):
        return "direct_full_canonical"
    return "not_directly_aligned"


def derive_error_family(
    slot_name: str,
    gold_raw: str,
    pred_raw: str,
    gold_canon: str,
    pred_canon: str,
    alignment_features: dict[str, bool],
    dialogue_context: str,
) -> str:
    """Heuristic error classification for audit analysis."""
    gold_raw = normalize_raw_value(gold_raw)
    pred_raw = normalize_raw_value(pred_raw)
    gold_canon = canonicalize_value(slot_name, gold_canon)
    pred_canon = canonicalize_value(slot_name, pred_canon)

    if pred_raw == gold_raw:
        return "correct_raw"
    if pred_canon == gold_canon:
        return "correct_canonical_only"
    if gold_raw == "dontcare" and pred_raw == "none":
        return "dontcare_to_none"

    pred_in_context = _value_in_context(pred_raw, pred_canon, dialogue_context)

    if gold_raw == "none" and pred_raw != "none":
        if pred_in_context:
            return "carryover_from_context"
        return "none_to_value_overprediction"

    if gold_raw not in {"none", "dontcare"} and pred_raw == "none":
        if (
            alignment_features.get("gold_in_full_context_exact")
            or alignment_features.get("gold_in_full_context_canon")
            or alignment_features.get("gold_in_user_turns_exact")
            or alignment_features.get("gold_in_user_turns_canon")
        ):
            return "missed_explicit_value"

    if pred_raw not in {"none", "dontcare"}:
        if pred_in_context and pred_canon != gold_canon:
            return "wrong_value_from_context"

    return "other_value_error"


def build_audit_record(
    dialogue_id: str,
    turn_id: int,
    slot_name: str,
    slot_description: str | None,
    context: str,
    prompt_text: str,
    gold_raw: str,
    pred_raw: str,
) -> dict[str, Any]:
    """Build a JSON-serializable audit record for one slot row."""
    gold_raw_norm = normalize_raw_value(gold_raw)
    pred_raw_norm = normalize_raw_value(pred_raw)
    gold_canon = canonicalize_value(slot_name, gold_raw_norm)
    pred_canon = canonicalize_value(slot_name, pred_raw_norm)
    is_correct_raw = pred_raw_norm == gold_raw_norm
    is_correct_canon = pred_canon == gold_canon
    features = compute_alignment_features(slot_name, gold_raw_norm, gold_canon, context)
    alignment_class = derive_alignment_class(gold_raw_norm, features)
    error_family = derive_error_family(
        slot_name,
        gold_raw_norm,
        pred_raw_norm,
        gold_canon,
        pred_canon,
        features,
        context,
    )

    return {
        "dialogue_id": dialogue_id,
        "turn_id": turn_id,
        "slot_name": slot_name,
        "slot_description": slot_description,
        "context": context,
        "prompt_text": prompt_text,
        "gold_raw": gold_raw_norm,
        "pred_raw": pred_raw_norm,
        "gold_canon": gold_canon,
        "pred_canon": pred_canon,
        "is_correct_raw": is_correct_raw,
        "is_correct_canon": is_correct_canon,
        "gold_in_full_context_exact": features["gold_in_full_context_exact"],
        "gold_in_full_context_canon": features["gold_in_full_context_canon"],
        "gold_in_user_turns_exact": features["gold_in_user_turns_exact"],
        "gold_in_user_turns_canon": features["gold_in_user_turns_canon"],
        "alignment_class": alignment_class,
        "error_family": error_family,
    }


def _bucket_stats(records: list[dict[str, Any]], key: str) -> dict[str, Any]:
    buckets: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for record in records:
        buckets[str(record.get(key, "unknown"))].append(record)

    summary: dict[str, Any] = {}
    for bucket, items in buckets.items():
        total = len(items)
        raw_correct = sum(1 for r in items if r.get("is_correct_raw"))
        canon_correct = sum(1 for r in items if r.get("is_correct_canon"))
        summary[bucket] = {
            "count": total,
            "raw_accuracy": (raw_correct / total) if total else 0.0,
            "canonical_accuracy": (canon_correct / total) if total else 0.0,
        }
    return summary


def build_audit_summary(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Aggregate audit records into summary JSON metrics."""
    total = len(records)
    raw_correct = sum(1 for r in records if r.get("is_correct_raw"))
    canon_correct = sum(1 for r in records if r.get("is_correct_canon"))

    return {
        "n_rows": total,
        "raw_slot_accuracy": (raw_correct / total) if total else 0.0,
        "canonical_slot_accuracy": (canon_correct / total) if total else 0.0,
        "n_mismatches_raw": total - raw_correct,
        "n_mismatches_canonical": total - canon_correct,
        "by_alignment_class": _bucket_stats(records, "alignment_class"),
        "by_error_family": _bucket_stats(records, "error_family"),
        "by_slot_name": _bucket_stats(records, "slot_name"),
    }
