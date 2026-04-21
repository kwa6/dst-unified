#!/usr/bin/env python3
"""Build a 50/50 LUAS+D0T mixed stage-one dataset without normalizing rows."""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Dict, List

from dst.data.build_property_splits import (
    BUCKET_ACTIVE_ALIGNED,
    BUCKET_ACTIVE_NONALIGNED,
    BUCKET_DONTCARE,
    BUCKET_NONE,
    classify_row,
)
from dst.data.jsonl_dataset import iter_jsonl


def reservoir_sample(source_path: Path, sample_size: int, rng: random.Random) -> List[dict]:
    """Sample rows with reservoir sampling to preserve source row contents."""
    reservoir: List[dict] = []
    for idx, row in enumerate(iter_jsonl(source_path)):
        if idx < sample_size:
            reservoir.append(row)
            continue
        j = rng.randint(0, idx)
        if j < sample_size:
            reservoir[j] = row

    if len(reservoir) < sample_size:
        raise ValueError(
            f"Requested {sample_size} rows from {source_path} but only found {len(reservoir)}."
        )
    return reservoir


def count_buckets(rows: List[dict]) -> Dict[str, int]:
    counts = {
        BUCKET_NONE: 0,
        BUCKET_DONTCARE: 0,
        BUCKET_ACTIVE_ALIGNED: 0,
        BUCKET_ACTIVE_NONALIGNED: 0,
    }
    for row in rows:
        bucket, _ = classify_row(row)
        counts[bucket] = counts.get(bucket, 0) + 1
    return counts


def write_jsonl(rows: List[dict], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f_out:
        for row in rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build a 50/50 LUAS+D0T mixed stage-one dataset without normalizing rows."
    )
    parser.add_argument("--luas_path", default="data_unified/luas/train.jsonl")
    parser.add_argument("--d0t_path", default="data_unified/d0t/train.jsonl")
    parser.add_argument(
        "--out_path",
        default="data_unified/mixed/mixed_luas50_d0t50_full_128k_train.jsonl",
    )
    parser.add_argument(
        "--manifest_path",
        default="data_unified/mixed/mixed_luas50_d0t50_full_128k.manifest.json",
    )
    parser.add_argument("--sample_size", type=int, default=64000)
    parser.add_argument("--seed", type=int, default=13)
    args = parser.parse_args()

    luas_path = Path(args.luas_path)
    d0t_path = Path(args.d0t_path)
    out_path = Path(args.out_path)
    manifest_path = Path(args.manifest_path)

    rng_luas = random.Random(args.seed)
    rng_d0t = random.Random(args.seed + 1)
    rng_shuffle = random.Random(args.seed + 2)

    luas_rows = reservoir_sample(luas_path, args.sample_size, rng_luas)
    d0t_rows = reservoir_sample(d0t_path, args.sample_size, rng_d0t)

    mixed_rows = list(luas_rows) + list(d0t_rows)
    rng_shuffle.shuffle(mixed_rows)

    write_jsonl(mixed_rows, out_path)

    bucket_counts = count_buckets(mixed_rows)
    manifest = {
        "output_path": str(out_path),
        "seed": args.seed,
        "sample_size_per_source": args.sample_size,
        "rows": {
            "total": len(mixed_rows),
            "luas": len(luas_rows),
            "d0t": len(d0t_rows),
        },
        "sources": {
            "luas": str(luas_path),
            "d0t": str(d0t_path),
        },
        "bucket_counts": bucket_counts,
    }

    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    with manifest_path.open("w", encoding="utf-8") as f_out:
        json.dump(manifest, f_out, indent=2)

    print("Wrote mixed dataset:", out_path)
    print("Wrote manifest:", manifest_path)
    print("Bucket counts:", bucket_counts)


if __name__ == "__main__":
    main()
