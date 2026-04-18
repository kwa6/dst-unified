import json
from pathlib import Path

import pytest

from dst.analysis import eval_audit
from dst.data import build_property_splits as bps


def make_row(idx, slot, target_value, context):
    return {
        "dataset": "test",
        "split": "train",
        "dialogue_id": f"dlg-{idx:03d}",
        "turn_id": idx,
        "speaker": "user",
        "dialogue_context": context,
        "slot_name": slot,
        "slot_description": "desc",
        "target_value": target_value,
    }


def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f_out:
        for row in rows:
            f_out.write(json.dumps(row, ensure_ascii=False) + "\n")


def read_jsonl(path: Path):
    rows = []
    with path.open("r", encoding="utf-8") as f_in:
        for line in f_in:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def row_sig(row: dict) -> str:
    return json.dumps(row, sort_keys=True)


def test_classify_row_buckets():
    ctx_aligned = "Turn 0 [USER]: I want cheap"
    ctx_nonaligned = "Turn 0 [USER]: I want expensive"

    row_none = make_row(1, "hotel-pricerange", "none", ctx_aligned)
    row_dontcare = make_row(2, "hotel-pricerange", "dontcare", ctx_aligned)
    row_aligned = make_row(3, "hotel-pricerange", "cheap", ctx_aligned)
    row_nonaligned = make_row(4, "hotel-pricerange", "cheap", ctx_nonaligned)

    assert bps.classify_row(row_none)[0] == bps.BUCKET_NONE
    assert bps.classify_row(row_dontcare)[0] == bps.BUCKET_DONTCARE
    assert bps.classify_row(row_aligned)[0] == bps.BUCKET_ACTIVE_ALIGNED
    assert bps.classify_row(row_nonaligned)[0] == bps.BUCKET_ACTIVE_NONALIGNED


def test_alignment_classes():
    ctx_user = "Turn 0 [USER]: I want cheap"
    ctx_user_canon = "Turn 0 [USER]: I want cheap"
    ctx_system = "Turn 0 [SYSTEM]: cheap"
    ctx_none = "Turn 0 [USER]: hello"

    assert eval_audit.classify_alignment("cheap", ctx_user) == "direct_user_exact"
    assert eval_audit.classify_alignment("cheap!", ctx_user_canon) == "direct_user_canonical"
    assert eval_audit.classify_alignment("cheap", ctx_system) == "direct_full_exact"
    assert eval_audit.classify_alignment("cheap", ctx_none) == "not_directly_aligned"


def test_negative_pool_reuse(tmp_path):
    rows = []
    for i in range(6):
        rows.append(make_row(i, "slot-a", "none", "Turn 0 [USER]: hi"))
    for i in range(6, 10):
        rows.append(make_row(i, "slot-b", "dontcare", "Turn 0 [USER]: hi"))
    for i in range(10, 14):
        rows.append(make_row(i, "slot-a", "cheap", "Turn 0 [USER]: I want cheap"))
    for i in range(14, 18):
        rows.append(make_row(i, "slot-a", "cheap", "Turn 0 [USER]: I want expensive"))

    source_path = tmp_path / "train.jsonl"
    write_jsonl(source_path, rows)

    out_dir = tmp_path / "out"
    result = bps.build_splits(
        source_path=source_path,
        resource_name="test",
        out_dir=out_dir,
        seed=123,
        full_target_total=20,
        active_target=3,
        none_multiplier=1.0,
        dontcare_multiplier=1.0,
        include_randommatched=False,
        slot_stratified=True,
        allow_downscale=True,
    )

    aligned_path = Path(result["split_outputs"]["alignedheavy"])
    nonaligned_path = Path(result["split_outputs"]["nonalignedheavy"])

    aligned_rows = read_jsonl(aligned_path)
    nonaligned_rows = read_jsonl(nonaligned_path)

    aligned_neg = {row_sig(r) for r in aligned_rows if r["target_value"] in {"none", "dontcare"}}
    nonaligned_neg = {row_sig(r) for r in nonaligned_rows if r["target_value"] in {"none", "dontcare"}}

    assert aligned_neg == nonaligned_neg


def test_deterministic_sampling(tmp_path):
    rows = []
    for i in range(8):
        rows.append(make_row(i, "slot-a", "none", "Turn 0 [USER]: hi"))
    for i in range(8, 12):
        rows.append(make_row(i, "slot-b", "dontcare", "Turn 0 [USER]: hi"))
    for i in range(12, 18):
        rows.append(make_row(i, "slot-a", "cheap", "Turn 0 [USER]: I want cheap"))
    for i in range(18, 24):
        rows.append(make_row(i, "slot-a", "cheap", "Turn 0 [USER]: I want expensive"))

    source_path = tmp_path / "train.jsonl"
    write_jsonl(source_path, rows)

    out_a = tmp_path / "out_a"
    out_b = tmp_path / "out_b"

    bps.build_splits(
        source_path=source_path,
        resource_name="test",
        out_dir=out_a,
        seed=7,
        full_target_total=16,
        active_target=4,
        none_multiplier=1.0,
        dontcare_multiplier=0.0,
        include_randommatched=True,
        slot_stratified=True,
        allow_downscale=True,
    )
    bps.build_splits(
        source_path=source_path,
        resource_name="test",
        out_dir=out_b,
        seed=7,
        full_target_total=16,
        active_target=4,
        none_multiplier=1.0,
        dontcare_multiplier=0.0,
        include_randommatched=True,
        slot_stratified=True,
        allow_downscale=True,
    )

    aligned_a = read_jsonl(out_a / "test_alignedheavy_8_train.jsonl")
    aligned_b = read_jsonl(out_b / "test_alignedheavy_8_train.jsonl")

    assert [row_sig(r) for r in aligned_a] == [row_sig(r) for r in aligned_b]


def test_downscale_behavior(tmp_path):
    rows = [
        make_row(1, "slot-a", "cheap", "Turn 0 [USER]: cheap"),
        make_row(2, "slot-a", "cheap", "Turn 0 [USER]: expensive"),
    ]
    source_path = tmp_path / "train.jsonl"
    write_jsonl(source_path, rows)

    out_dir = tmp_path / "out"

    with pytest.raises(ValueError):
        bps.build_splits(
            source_path=source_path,
            resource_name="test",
            out_dir=out_dir,
            seed=1,
            full_target_total=4,
            active_target=3,
            none_multiplier=1.0,
            dontcare_multiplier=1.0,
            include_randommatched=False,
            slot_stratified=True,
            allow_downscale=False,
        )

    result = bps.build_splits(
        source_path=source_path,
        resource_name="test",
        out_dir=out_dir,
        seed=1,
        full_target_total=4,
        active_target=3,
        none_multiplier=1.0,
        dontcare_multiplier=1.0,
        include_randommatched=False,
        slot_stratified=True,
        allow_downscale=True,
    )

    manifest_path = Path(result["top_manifest_path"])
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    assert manifest["splits"]["alignedheavy"]["active_target_downscaled"] is True
