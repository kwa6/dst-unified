#!/usr/bin/env python3
"""Report bucket distributions and raw dontcare-like counts for unified JSONL files."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Tuple

from dst.data.build_property_splits import (
    BUCKET_ACTIVE_ALIGNED,
    BUCKET_ACTIVE_NONALIGNED,
    BUCKET_DONTCARE,
    BUCKET_NONE,
    classify_row,
)
from dst.data.jsonl_dataset import iter_jsonl


RAW_DONTCARE_LIKE = {
    "dontcare",
    "dont care",
    "don't care",
}


@dataclass(frozen=True)
class FileSpec:
    dataset: str
    split: str
    path: Path


@dataclass(frozen=True)
class FileResult:
    dataset: str
    split: str
    total_rows: int
    none_count: int
    dontcare_count: int
    active_aligned: int
    active_nonaligned: int
    raw_dontcare_like: int


def _normalize_raw_value(value: object) -> str:
    raw = "" if value is None else str(value)
    raw = raw.strip().lower()
    return " ".join(raw.split())


def _count_raw_dontcare(value: object) -> bool:
    return _normalize_raw_value(value) in RAW_DONTCARE_LIKE


def _gather_file_specs(base_dir: Path) -> List[FileSpec]:
    required = [
        FileSpec("multiwoz24", "val", base_dir / "data_unified/multiwoz24/val.jsonl"),
        FileSpec("multiwoz24", "test", base_dir / "data_unified/multiwoz24/test.jsonl"),
        FileSpec("luas", "train", base_dir / "data_unified/luas/train.jsonl"),
        FileSpec("d0t", "train", base_dir / "data_unified/d0t/train.jsonl"),
    ]
    optional = [
        FileSpec("luas", "val", base_dir / "data_unified/luas/val.jsonl"),
        FileSpec("luas", "test", base_dir / "data_unified/luas/test.jsonl"),
        FileSpec("d0t", "val", base_dir / "data_unified/d0t/val.jsonl"),
        FileSpec("d0t", "test", base_dir / "data_unified/d0t/test.jsonl"),
    ]
    specs = [spec for spec in required if spec.path.exists()]
    specs.extend([spec for spec in optional if spec.path.exists()])
    missing_required = [spec for spec in required if not spec.path.exists()]
    if missing_required:
        missing_list = ", ".join(str(spec.path) for spec in missing_required)
        print(f"WARNING: Missing required files: {missing_list}")
    return specs


def _analyze_file(spec: FileSpec) -> FileResult:
    bucket_counts = {
        BUCKET_NONE: 0,
        BUCKET_DONTCARE: 0,
        BUCKET_ACTIVE_ALIGNED: 0,
        BUCKET_ACTIVE_NONALIGNED: 0,
    }
    total = 0
    raw_dontcare_like = 0

    for row in iter_jsonl(spec.path):
        total += 1
        bucket, _ = classify_row(row)
        bucket_counts[bucket] = bucket_counts.get(bucket, 0) + 1
        if _count_raw_dontcare(row.get("target_value")):
            raw_dontcare_like += 1

    return FileResult(
        dataset=spec.dataset,
        split=spec.split,
        total_rows=total,
        none_count=bucket_counts[BUCKET_NONE],
        dontcare_count=bucket_counts[BUCKET_DONTCARE],
        active_aligned=bucket_counts[BUCKET_ACTIVE_ALIGNED],
        active_nonaligned=bucket_counts[BUCKET_ACTIVE_NONALIGNED],
        raw_dontcare_like=raw_dontcare_like,
    )


def _print_bucket_table(results: Iterable[FileResult]) -> None:
    print("| Dataset | Split | Total rows | None | Dontcare | Active aligned | Active nonaligned |")
    print("| --- | --- | --- | --- | --- | --- | --- |")
    for r in results:
        print(
            f"| {r.dataset} | {r.split} | {r.total_rows} | {r.none_count} | {r.dontcare_count} | "
            f"{r.active_aligned} | {r.active_nonaligned} |"
        )


def _print_verification_table(results: Iterable[FileResult]) -> None:
    print("| Dataset | Split | Raw dontcare-like targets found | Final dontcare_value rows |")
    print("| --- | --- | --- | --- |")
    for r in results:
        print(f"| {r.dataset} | {r.split} | {r.raw_dontcare_like} | {r.dontcare_count} |")


def _print_conversion_notes(results: Iterable[FileResult]) -> None:
    for r in results:
        if r.raw_dontcare_like > 0 and r.dontcare_count == 0:
            print(
                "NOTE: Raw dontcare-like values found but none classified as dontcare_value for "
                f"{r.dataset}/{r.split}. Conversion is likely happening in "
                "eval_audit.normalize_value() (DONTCARE_LIKE) or in the dataset adapter "
                "norm_value() before writing the unified JSONL."
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Report bucket distributions for unified JSONL files.")
    parser.add_argument(
        "--base_dir",
        default=Path(".").resolve(),
        type=Path,
        help="Workspace root directory (default: current working directory)",
    )
    args = parser.parse_args()

    specs = _gather_file_specs(args.base_dir)
    results = [_analyze_file(spec) for spec in specs]

    print("\nBucket distribution summary:")
    _print_bucket_table(results)

    print("\nRaw dontcare verification:")
    _print_verification_table(results)

    print("\nConversion notes:")
    _print_conversion_notes(results)


if __name__ == "__main__":
    main()
