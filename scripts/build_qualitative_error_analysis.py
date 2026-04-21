#!/usr/bin/env python3
"""Build a qualitative error analysis markdown from audit_val.json files."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

RUNS = [
    {
        "name": "llama31_8b_stage1_luas_full_128k",
        "audit_file": "llama31_8b_stage1_luas_full_128k_audit_val.json",
    },
    {
        "name": "llama31_8b_stage1_d0t_full_128k",
        "audit_file": "llama31_8b_stage1_d0t_full_128k_audit_val.json",
    },
    {
        "name": "llama31_8b_stage1_d0t_nonaligned_34492_plus_same_native_none",
        "audit_file": "llama31_8b_stage1_d0t_nonaligned_34492_plus_same_native_none_audit_val.json",
    },
    {
        "name": "llama31_8b_stage1_mixed_luas50_d0t50_full_128k",
        "audit_file": "llama31_8b_stage1_mixed_luas50_d0t50_full_128k_audit_val.json",
    },
]

TARGET_FAMILIES = [
    "none_to_value_overprediction",
    "carryover_from_context",
    "missed_explicit_value",
    "wrong_value_from_context",
]

SAMPLES_PER_FAMILY = 5
MAX_CONTEXT_CHARS = 220
MAX_PROMPT_CHARS = 260
CHUNK_SIZE = 1024 * 1024

FAMILY_NOTES = {
    "none_to_value_overprediction": "Overpredicts a value when gold is none.",
    "carryover_from_context": "Carries over a context value despite gold none.",
    "missed_explicit_value": "Misses explicitly stated value; predicts none.",
    "wrong_value_from_context": "Selects wrong context value instead of gold.",
}


def iter_json_array(path: Path) -> Iterable[Dict[str, object]]:
    decoder = json.JSONDecoder()
    buffer = ""
    idx = 0
    with path.open("r", encoding="utf-8") as handle:
        # Find the opening bracket.
        while True:
            if idx >= len(buffer):
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    return
                buffer += chunk
            while idx < len(buffer) and buffer[idx].isspace():
                idx += 1
            if idx < len(buffer):
                if buffer[idx] == "[":
                    idx += 1
                    break
                idx += 1
        while True:
            while True:
                if idx >= len(buffer):
                    chunk = handle.read(CHUNK_SIZE)
                    if not chunk:
                        return
                    buffer += chunk
                if buffer[idx].isspace() or buffer[idx] == ",":
                    idx += 1
                    continue
                if buffer[idx] == "]":
                    return
                break
            try:
                obj, end = decoder.raw_decode(buffer, idx)
            except json.JSONDecodeError:
                chunk = handle.read(CHUNK_SIZE)
                if not chunk:
                    raise
                buffer += chunk
                continue
            idx = end
            if idx > CHUNK_SIZE:
                buffer = buffer[idx:]
                idx = 0
            if isinstance(obj, dict):
                yield obj


def clean_table_text(text: str) -> str:
    cleaned = " ".join(text.replace("|", "/").split())
    return cleaned


def make_context_snippet(context: str) -> str:
    lines = [line.strip() for line in context.splitlines() if line.strip()]
    if len(lines) >= 2:
        snippet = " | ".join(lines[-2:])
    elif lines:
        snippet = lines[-1]
    else:
        snippet = ""
    snippet = clean_table_text(snippet)
    if len(snippet) > MAX_CONTEXT_CHARS:
        snippet = snippet[: MAX_CONTEXT_CHARS - 3].rstrip() + "..."
    return snippet


def make_prompt_snippet(prompt_text: str) -> str:
    snippet = clean_table_text(prompt_text)
    if len(snippet) > MAX_PROMPT_CHARS:
        snippet = snippet[: MAX_PROMPT_CHARS - 3].rstrip() + "..."
    return snippet


def collect_samples_and_counts(
    path: Path,
) -> Tuple[Dict[str, int], Dict[str, List[Dict[str, str]]]]:
    counts = {family: 0 for family in TARGET_FAMILIES}
    samples: Dict[str, List[Dict[str, str]]] = {family: [] for family in TARGET_FAMILIES}

    for record in iter_json_array(path):
        family = str(record.get("error_family", ""))
        if family not in TARGET_FAMILIES:
            continue
        counts[family] += 1
        if len(samples[family]) >= SAMPLES_PER_FAMILY:
            continue
        context = str(record.get("context", ""))
        prompt_text = str(record.get("prompt_text", ""))
        sample = {
            "dialogue_id": str(record.get("dialogue_id", "")),
            "turn_id": str(record.get("turn_id", "")),
            "slot_name": str(record.get("slot_name", "")),
            "alignment_class": str(record.get("alignment_class", "")),
            "error_family": family,
            "gold_raw": str(record.get("gold_raw", "")),
            "pred_raw": str(record.get("pred_raw", "")),
            "gold_canon": str(record.get("gold_canon", "")),
            "pred_canon": str(record.get("pred_canon", "")),
            "context_snippet": make_context_snippet(context),
            "prompt_text": make_prompt_snippet(prompt_text),
            "note": FAMILY_NOTES.get(family, ""),
        }
        samples[family].append(sample)
    return counts, samples


def render_markdown(
    output_path: Path,
    summary_counts: Dict[str, Dict[str, int]],
    samples_by_run: Dict[str, Dict[str, List[Dict[str, str]]]],
) -> None:
    lines: List[str] = []
    lines.append("# Qualitative Error Analysis (Stage 1)")
    lines.append("")
    lines.append("## Summary")
    lines.append("")
    run_names = [run["name"] for run in RUNS]
    audit_names = [run["audit_file"] for run in RUNS]
    lines.append("- Runs analyzed: " + ", ".join(run_names))
    lines.append("- Audit files used: " + ", ".join(audit_names))
    lines.append("- Error families sampled: " + ", ".join(TARGET_FAMILIES))
    lines.append(
        "- Notes: This version uses full audit_val.json files and exact audit labels for error_family and alignment_class."
    )
    lines.append("")
    lines.append("### Audit counts (by error family)")
    lines.append("")
    lines.append("| run | none_to_value_overprediction | carryover_from_context | missed_explicit_value | wrong_value_from_context |")
    lines.append("| --- | --- | --- | --- | --- |")
    for run in RUNS:
        run_name = run["name"]
        counts = summary_counts.get(run_name, {})
        row = [
            run_name,
            str(counts.get("none_to_value_overprediction", 0)),
            str(counts.get("carryover_from_context", 0)),
            str(counts.get("missed_explicit_value", 0)),
            str(counts.get("wrong_value_from_context", 0)),
        ]
        lines.append("| " + " | ".join(row) + " |")
    lines.append("")

    for run in RUNS:
        run_name = run["name"]
        lines.append(f"## Run: {run_name}")
        lines.append("")
        lines.append("| run | dialogue_id | turn_id | slot_name | alignment_class | error_family | gold_raw | pred_raw | gold_canon | pred_canon | context_snippet | prompt_text | note |")
        lines.append("| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |")
        run_samples = samples_by_run.get(run_name, {})
        for family in TARGET_FAMILIES:
            for sample in run_samples.get(family, []):
                row = [
                    run_name,
                    clean_table_text(sample["dialogue_id"]),
                    clean_table_text(sample["turn_id"]),
                    clean_table_text(sample["slot_name"]),
                    clean_table_text(sample["alignment_class"]),
                    clean_table_text(sample["error_family"]),
                    clean_table_text(sample["gold_raw"]),
                    clean_table_text(sample["pred_raw"]),
                    clean_table_text(sample["gold_canon"]),
                    clean_table_text(sample["pred_canon"]),
                    clean_table_text(sample["context_snippet"]),
                    clean_table_text(sample["prompt_text"]),
                    clean_table_text(sample["note"]),
                ]
                lines.append("| " + " | ".join(row) + " |")
        lines.append("")

    output_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    outputs_dir = repo_root / "outputs"
    outputs_dir.mkdir(parents=True, exist_ok=True)
    output_path = outputs_dir / "qualitative_error_analysis_stage1.md"

    summary_counts: Dict[str, Dict[str, int]] = {}
    samples_by_run: Dict[str, Dict[str, List[Dict[str, str]]]] = {}
    for run in RUNS:
        run_name = run["name"]
        audit_path = repo_root / run["audit_file"]
        counts, samples = collect_samples_and_counts(audit_path)
        summary_counts[run_name] = counts
        samples_by_run[run_name] = samples

    render_markdown(output_path, summary_counts, samples_by_run)


if __name__ == "__main__":
    main()
