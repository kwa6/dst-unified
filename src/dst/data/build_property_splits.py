from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from dst.analysis import eval_audit
from dst.data.jsonl_dataset import iter_jsonl


BUCKET_NONE = "none_value"
BUCKET_DONTCARE = "dontcare_value"
BUCKET_ACTIVE_ALIGNED = "active_aligned"
BUCKET_ACTIVE_NONALIGNED = "active_nonaligned"

ALIGNMENT_ALIGNED = {
    "direct_user_exact",
    "direct_user_canonical",
    "direct_full_exact",
    "direct_full_canonical",
}
ALIGNMENT_NONALIGNED = {"not_directly_aligned"}


@dataclass(frozen=True)
class AnalyzeSummary:
    source: str
    resource_name: str
    total_rows: int
    bucket_counts: Dict[str, int]
    bucket_slot_counts: Dict[str, Dict[str, int]]
    active_total: int
    active_aligned: int
    active_nonaligned: int
    max_feasible_active_target: int
    active_target_requested: int
    active_target_feasible: bool
    implied_split_total: int


@dataclass
class SamplerPlan:
    target_total: int
    slot_targets: Dict[str, int]
    stratified_used: bool
    shortfall: int


class StratifiedReservoirSampler:
    def __init__(self, slot_targets: Dict[str, int], rng: random.Random):
        self.slot_targets = slot_targets
        self.rng = rng
        self.reservoirs: Dict[str, List[dict]] = {slot: [] for slot in slot_targets}
        self.seen: Counter[str] = Counter()

    def _slot_key(self, slot_name: str) -> str:
        if "__all__" in self.slot_targets:
            return "__all__"
        return slot_name

    def consider(self, row: dict, slot_name: str) -> None:
        slot_key = self._slot_key(slot_name)
        target = self.slot_targets.get(slot_key, 0)
        if target <= 0:
            return
        self.seen[slot_key] += 1
        reservoir = self.reservoirs[slot_key]
        if len(reservoir) < target:
            reservoir.append(row)
            return
        j = self.rng.randint(0, self.seen[slot_key] - 1)
        if j < target:
            reservoir[j] = row

    def sampled_rows(self) -> List[dict]:
        out: List[dict] = []
        for rows in self.reservoirs.values():
            out.extend(rows)
        return out


def _seed_for_label(seed: int, label: str) -> int:
    h = hashlib.sha256(f"{seed}:{label}".encode("utf-8")).hexdigest()
    return int(h[:16], 16)


def classify_row(row: dict) -> Tuple[str, str | None]:
    value = row.get("target_value", "")
    if eval_audit.is_none_value(value):
        return BUCKET_NONE, None
    if eval_audit.is_dontcare_value(value):
        return BUCKET_DONTCARE, None
    alignment_class = eval_audit.classify_alignment(value, row.get("dialogue_context", ""))
    if alignment_class in ALIGNMENT_ALIGNED:
        return BUCKET_ACTIVE_ALIGNED, alignment_class
    return BUCKET_ACTIVE_NONALIGNED, alignment_class


def _counter_to_dict(counter: Counter) -> Dict[str, int]:
    return {k: int(v) for k, v in counter.items()}


def _resolve_target(active_target: int, multiplier: float, explicit_target: int | None) -> int:
    if explicit_target is not None:
        return explicit_target
    return int(round(active_target * multiplier))


def analyze_source(
    source_path: str | Path,
    resource_name: str,
    active_target: int,
    none_multiplier: float,
    dontcare_multiplier: float,
    none_target: int | None = None,
    dontcare_target: int | None = None,
) -> AnalyzeSummary:
    bucket_counts: Counter[str] = Counter()
    bucket_slot_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    total = 0

    for row in iter_jsonl(source_path):
        total += 1
        bucket, _ = classify_row(row)
        bucket_counts[bucket] += 1
        bucket_slot_counts[bucket][row.get("slot_name", "unknown")] += 1

    for bucket in (
        BUCKET_NONE,
        BUCKET_DONTCARE,
        BUCKET_ACTIVE_ALIGNED,
        BUCKET_ACTIVE_NONALIGNED,
    ):
        bucket_counts.setdefault(bucket, 0)
        bucket_slot_counts.setdefault(bucket, Counter())

    active_aligned = bucket_counts.get(BUCKET_ACTIVE_ALIGNED, 0)
    active_nonaligned = bucket_counts.get(BUCKET_ACTIVE_NONALIGNED, 0)
    active_total = active_aligned + active_nonaligned
    max_feasible = min(active_aligned, active_nonaligned)
    effective_none_target = _resolve_target(active_target, none_multiplier, none_target)
    effective_dontcare_target = _resolve_target(active_target, dontcare_multiplier, dontcare_target)
    implied_total = active_target + effective_none_target + effective_dontcare_target

    summary = AnalyzeSummary(
        source=str(source_path),
        resource_name=resource_name,
        total_rows=total,
        bucket_counts=_counter_to_dict(bucket_counts),
        bucket_slot_counts={
            bucket: _counter_to_dict(counter) for bucket, counter in bucket_slot_counts.items()
        },
        active_total=active_total,
        active_aligned=active_aligned,
        active_nonaligned=active_nonaligned,
        max_feasible_active_target=max_feasible,
        active_target_requested=active_target,
        active_target_feasible=active_target <= max_feasible,
        implied_split_total=implied_total,
    )
    return summary


def _print_analyze_summary(summary: AnalyzeSummary) -> None:
    print("Analyze summary")
    print(f"source: {summary.source}")
    print(f"resource_name: {summary.resource_name}")
    print(f"total_rows: {summary.total_rows}")
    print("bucket_counts:")
    for bucket, count in summary.bucket_counts.items():
        print(f"  {bucket}: {count}")
    print("active_total:", summary.active_total)
    print("active_aligned:", summary.active_aligned)
    print("active_nonaligned:", summary.active_nonaligned)
    print("max_feasible_active_target:", summary.max_feasible_active_target)
    print("active_target_requested:", summary.active_target_requested)
    print("active_target_feasible:", summary.active_target_feasible)
    print("implied_split_total:", summary.implied_split_total)
    print("bucket_slot_counts:")
    for bucket, slot_counts in summary.bucket_slot_counts.items():
        print(f"  {bucket}:")
        for slot, count in sorted(slot_counts.items()):
            print(f"    {slot}: {count}")


def _allocate_bucket_targets(total_target: int, bucket_counts: Dict[str, int], rng: random.Random) -> Dict[str, int]:
    buckets = list(bucket_counts.keys())
    total_available = sum(bucket_counts.values())
    if total_available == 0 or total_target <= 0:
        return {bucket: 0 for bucket in buckets}
    if total_target >= total_available:
        return {bucket: int(bucket_counts[bucket]) for bucket in buckets}

    fractional = []
    targets: Dict[str, int] = {}
    for bucket, count in bucket_counts.items():
        exact = total_target * (count / total_available)
        base = int(math.floor(exact))
        targets[bucket] = base
        fractional.append((exact - base, bucket))

    remainder = total_target - sum(targets.values())
    rng.shuffle(fractional)
    fractional.sort(key=lambda x: x[0], reverse=True)

    idx = 0
    while remainder > 0 and fractional:
        _, bucket = fractional[idx % len(fractional)]
        if targets[bucket] < bucket_counts[bucket]:
            targets[bucket] += 1
            remainder -= 1
        idx += 1

    return targets


def _allocate_slot_targets(
    slot_counts: Dict[str, int],
    target_total: int,
    rng: random.Random,
    stratify: bool,
) -> SamplerPlan:
    total_available = sum(slot_counts.values())
    requested_total = target_total
    if target_total <= 0:
        return SamplerPlan(target_total=0, slot_targets={}, stratified_used=stratify, shortfall=0)

    if total_available <= 0:
        return SamplerPlan(target_total=0, slot_targets={}, stratified_used=stratify, shortfall=requested_total)

    if target_total > total_available:
        target_total = total_available
        shortfall = requested_total - total_available
    else:
        shortfall = 0

    if not stratify or len(slot_counts) <= 1:
        return SamplerPlan(
            target_total=target_total,
            slot_targets={"__all__": target_total},
            stratified_used=False,
            shortfall=shortfall,
        )

    fractional = []
    slot_targets: Dict[str, int] = {}
    for slot, count in slot_counts.items():
        exact = target_total * (count / total_available)
        base = int(math.floor(exact))
        slot_targets[slot] = base
        fractional.append((exact - base, slot))

    remainder = target_total - sum(slot_targets.values())
    rng.shuffle(fractional)
    fractional.sort(key=lambda x: x[0], reverse=True)

    idx = 0
    while remainder > 0 and fractional:
        _, slot = fractional[idx % len(fractional)]
        if slot_targets[slot] < slot_counts[slot]:
            slot_targets[slot] += 1
            remainder -= 1
        idx += 1

    return SamplerPlan(
        target_total=target_total,
        slot_targets=slot_targets,
        stratified_used=True,
        shortfall=shortfall,
    )


def _summarize_rows(rows: Iterable[dict]) -> Dict[str, object]:
    bucket_counts: Counter[str] = Counter()
    bucket_slot_counts: Dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        bucket, _ = classify_row(row)
        bucket_counts[bucket] += 1
        bucket_slot_counts[bucket][row.get("slot_name", "unknown")] += 1
    for bucket in (
        BUCKET_NONE,
        BUCKET_DONTCARE,
        BUCKET_ACTIVE_ALIGNED,
        BUCKET_ACTIVE_NONALIGNED,
    ):
        bucket_counts.setdefault(bucket, 0)
        bucket_slot_counts.setdefault(bucket, Counter())
    return {
        "bucket_counts": _counter_to_dict(bucket_counts),
        "bucket_slot_counts": {
            bucket: _counter_to_dict(counter) for bucket, counter in bucket_slot_counts.items()
        },
        "total_rows": sum(bucket_counts.values()),
        "slot_distribution": _counter_to_dict(
            Counter(row.get("slot_name", "unknown") for row in rows)
        ),
    }


def _write_jsonl(path: Path, rows: Iterable[dict]) -> None:
    with path.open("w", encoding="utf-8") as f_out:
        for row in rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_splits(
    source_path: str | Path,
    resource_name: str,
    out_dir: str | Path,
    seed: int,
    full_target_total: int,
    active_target: int,
    none_multiplier: float,
    dontcare_multiplier: float,
    include_randommatched: bool,
    slot_stratified: bool,
    allow_downscale: bool,
    summary: AnalyzeSummary | None = None,
    none_target: int | None = None,
    dontcare_target: int | None = None,
) -> Dict[str, object]:
    if summary is None:
        summary = analyze_source(
            source_path=source_path,
            resource_name=resource_name,
            active_target=active_target,
            none_multiplier=none_multiplier,
            dontcare_multiplier=dontcare_multiplier,
            none_target=none_target,
            dontcare_target=dontcare_target,
        )

    if not summary.active_target_feasible:
        if not allow_downscale:
            raise ValueError(
                "active_target exceeds feasible matched size; use --allow_downscale to proceed"
            )
        active_target = summary.max_feasible_active_target

    none_target = _resolve_target(active_target, none_multiplier, none_target)
    dontcare_target = _resolve_target(active_target, dontcare_multiplier, dontcare_target)
    aligned_total_target = active_target + none_target + dontcare_target

    bucket_counts = summary.bucket_counts
    rng_full = random.Random(_seed_for_label(seed, "full_bucket_targets"))
    full_bucket_targets = _allocate_bucket_targets(full_target_total, bucket_counts, rng_full)

    slot_counts: Dict[str, Dict[str, int]] = summary.bucket_slot_counts
    active_slot_counts: Dict[str, int] = Counter()
    active_slot_counts.update(slot_counts.get(BUCKET_ACTIVE_ALIGNED, {}))
    active_slot_counts.update(slot_counts.get(BUCKET_ACTIVE_NONALIGNED, {}))

    plans: Dict[str, SamplerPlan] = {}
    samplers: Dict[str, StratifiedReservoirSampler] = {}

    def init_sampler(label: str, slot_count_map: Dict[str, int], target_total: int) -> None:
        rng = random.Random(_seed_for_label(seed, label))
        plan = _allocate_slot_targets(slot_count_map, target_total, rng, slot_stratified)
        plans[label] = plan
        samplers[label] = StratifiedReservoirSampler(plan.slot_targets, rng)

    init_sampler("none_pool", slot_counts.get(BUCKET_NONE, {}), none_target)
    init_sampler("dontcare_pool", slot_counts.get(BUCKET_DONTCARE, {}), dontcare_target)
    init_sampler("active_aligned", slot_counts.get(BUCKET_ACTIVE_ALIGNED, {}), active_target)
    init_sampler("active_nonaligned", slot_counts.get(BUCKET_ACTIVE_NONALIGNED, {}), active_target)
    init_sampler("active_random", dict(active_slot_counts), active_target)

    for bucket, target_total in full_bucket_targets.items():
        init_sampler(f"full::{bucket}", slot_counts.get(bucket, {}), target_total)

    for row in iter_jsonl(source_path):
        bucket, _ = classify_row(row)
        slot_name = row.get("slot_name", "unknown")

        if bucket == BUCKET_NONE:
            samplers["none_pool"].consider(row, slot_name)
        elif bucket == BUCKET_DONTCARE:
            samplers["dontcare_pool"].consider(row, slot_name)
        elif bucket == BUCKET_ACTIVE_ALIGNED:
            samplers["active_aligned"].consider(row, slot_name)
            samplers["active_random"].consider(row, slot_name)
        elif bucket == BUCKET_ACTIVE_NONALIGNED:
            samplers["active_nonaligned"].consider(row, slot_name)
            samplers["active_random"].consider(row, slot_name)

        sampler_key = f"full::{bucket}"
        if sampler_key in samplers:
            samplers[sampler_key].consider(row, slot_name)

    none_pool = samplers["none_pool"].sampled_rows()
    dontcare_pool = samplers["dontcare_pool"].sampled_rows()
    aligned_active = samplers["active_aligned"].sampled_rows()
    nonaligned_active = samplers["active_nonaligned"].sampled_rows()
    random_active = samplers["active_random"].sampled_rows()

    full_rows: List[dict] = []
    for bucket in bucket_counts.keys():
        full_rows.extend(samplers[f"full::{bucket}"].sampled_rows())

    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    def shuffle_rows(rows: List[dict], label: str) -> List[dict]:
        rng = random.Random(_seed_for_label(seed, label))
        rows = list(rows)
        rng.shuffle(rows)
        return rows

    full_rows = shuffle_rows(full_rows, "full_shuffle")
    aligned_rows = shuffle_rows(aligned_active + none_pool + dontcare_pool, "aligned_shuffle")
    nonaligned_rows = shuffle_rows(nonaligned_active + none_pool + dontcare_pool, "nonaligned_shuffle")

    split_outputs: Dict[str, Path] = {}

    full_label = int(round(sum(full_bucket_targets.values())))
    aligned_label = int(round(aligned_total_target))

    full_name = f"{resource_name}_full_{full_label}_train.jsonl"
    aligned_name = f"{resource_name}_alignedheavy_{aligned_label}_train.jsonl"
    nonaligned_name = f"{resource_name}_nonalignedheavy_{aligned_label}_train.jsonl"

    split_outputs["full"] = out_path / full_name
    split_outputs["alignedheavy"] = out_path / aligned_name
    split_outputs["nonalignedheavy"] = out_path / nonaligned_name

    if include_randommatched:
        random_rows = shuffle_rows(random_active + none_pool + dontcare_pool, "random_shuffle")
        random_name = f"{resource_name}_randommatched_{aligned_label}_train.jsonl"
        split_outputs["randommatched"] = out_path / random_name
    else:
        random_rows = []

    _write_jsonl(split_outputs["full"], full_rows)
    _write_jsonl(split_outputs["alignedheavy"], aligned_rows)
    _write_jsonl(split_outputs["nonalignedheavy"], nonaligned_rows)
    if include_randommatched:
        _write_jsonl(split_outputs["randommatched"], random_rows)

    manifests: Dict[str, Dict[str, object]] = {}
    for split_name, path in split_outputs.items():
        rows = {
            "full": full_rows,
            "alignedheavy": aligned_rows,
            "nonalignedheavy": nonaligned_rows,
            "randommatched": random_rows,
        }[split_name]
        summary_rows = _summarize_rows(rows)
        manifests[split_name] = {
            "source_path": str(source_path),
            "resource_name": resource_name,
            "seed": seed,
            "output_path": str(path),
            "actual_counts": summary_rows["bucket_counts"],
            "actual_total_rows": summary_rows["total_rows"],
            "slot_distribution": summary_rows["slot_distribution"],
            "slot_distribution_by_bucket": summary_rows["bucket_slot_counts"],
            "slot_stratified_requested": slot_stratified,
            "slot_stratified_used": {
                label: plan.stratified_used for label, plan in plans.items() if "full::" not in label
            },
            "active_target_requested": summary.active_target_requested,
            "active_target_used": active_target,
            "active_target_downscaled": summary.active_target_requested != active_target,
            "sampling_plans": {
                label: {
                    "target_total": plan.target_total,
                    "stratified_used": plan.stratified_used,
                    "shortfall": plan.shortfall,
                }
                for label, plan in plans.items()
            },
        }

        manifest_path = path.with_suffix(".manifest.json")
        with manifest_path.open("w", encoding="utf-8") as f_out:
            json.dump(manifests[split_name], f_out, indent=2)
        manifests[split_name]["manifest_path"] = str(manifest_path)

    top_manifest = {
        "source_path": str(source_path),
        "resource_name": resource_name,
        "seed": seed,
        "analysis": {
            "total_rows": summary.total_rows,
            "bucket_counts": summary.bucket_counts,
            "active_total": summary.active_total,
            "max_feasible_active_target": summary.max_feasible_active_target,
        },
        "splits": manifests,
    }
    top_manifest_path = out_path / f"{resource_name}_property_splits_manifest.json"
    with top_manifest_path.open("w", encoding="utf-8") as f_out:
        json.dump(top_manifest, f_out, indent=2)

    return {
        "summary": summary,
        "split_outputs": {k: str(v) for k, v in split_outputs.items()},
        "top_manifest_path": str(top_manifest_path),
    }


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Build DST property splits from unified JSONL")
    ap.add_argument("--source", required=True)
    ap.add_argument("--resource_name", required=True)
    ap.add_argument("--out_dir", required=True)
    ap.add_argument("--analyze_only", action="store_true")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--full_target_total", type=int, default=128000)
    ap.add_argument("--active_target", type=int, default=40000)
    ap.add_argument("--none_target", type=int, default=None)
    ap.add_argument("--dontcare_target", type=int, default=None)
    ap.add_argument("--none_multiplier", type=float, default=2.0)
    ap.add_argument("--dontcare_multiplier", type=float, default=0.2)
    ap.add_argument("--include_randommatched", action="store_true")
    ap.add_argument("--slot_stratified", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--allow_downscale", action="store_true")
    return ap.parse_args()


def main() -> None:
    args = parse_args()
    summary = analyze_source(
        source_path=args.source,
        resource_name=args.resource_name,
        active_target=args.active_target,
        none_multiplier=args.none_multiplier,
        dontcare_multiplier=args.dontcare_multiplier,
        none_target=args.none_target,
        dontcare_target=args.dontcare_target,
    )

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    _print_analyze_summary(summary)

    analysis_manifest_path = out_dir / f"{args.resource_name}_analysis_manifest.json"
    with analysis_manifest_path.open("w", encoding="utf-8") as f_out:
        json.dump(summary.__dict__, f_out, indent=2)
    print(f"Saved analysis manifest to {analysis_manifest_path}")

    if args.analyze_only:
        return

    result = build_splits(
        source_path=args.source,
        resource_name=args.resource_name,
        out_dir=args.out_dir,
        seed=args.seed,
        full_target_total=args.full_target_total,
        active_target=args.active_target,
        none_multiplier=args.none_multiplier,
        dontcare_multiplier=args.dontcare_multiplier,
        include_randommatched=args.include_randommatched,
        slot_stratified=args.slot_stratified,
        allow_downscale=args.allow_downscale,
        summary=summary,
        none_target=args.none_target,
        dontcare_target=args.dontcare_target,
    )
    print(f"Saved split manifest to {result['top_manifest_path']}")


if __name__ == "__main__":
    main()
