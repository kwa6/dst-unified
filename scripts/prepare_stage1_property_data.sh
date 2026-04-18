#!/usr/bin/env bash
# Prepare stage-1 property splits from source datasets.
#
# 1) Run: bash scripts/prepare_stage1_property_data.sh
# 2) Then: bash scripts/run_stage1_property_splits.sh
set -euo pipefail

export PYTHONPATH=src

SEED=42
ACTIVE_TARGET=34492
FULL_TARGET=128000

LUAS_NONE_TARGET=24
LUAS_DONTCARE_TARGET=0

D0T_NONE_TARGET=10547
D0T_DONTCARE_TARGET=0

run_step() {
  echo
  echo "------------------------------------------"
  echo "Running: $*"
  echo "------------------------------------------"
  "$@"
}

run_step bash scripts/build_multiwoz.sh
run_step bash scripts/build_d0t.sh
run_step bash scripts/build_luas.sh

run_step python -m dst.data.build_property_splits \
  --source data_unified/luas/train.jsonl \
  --resource_name luas \
  --out_dir data_unified/luas/splits \
  --analyze_only \
  --seed "$SEED" \
  --active_target "$ACTIVE_TARGET" \
  --none_target "$LUAS_NONE_TARGET" \
  --dontcare_target "$LUAS_DONTCARE_TARGET"

mv -f data_unified/luas/splits/luas_analysis_manifest.json \
  data_unified/luas/splits/luas_analysis_manifest_v2.json

run_step python -m dst.data.build_property_splits \
  --source data_unified/d0t/train.jsonl \
  --resource_name d0t \
  --out_dir data_unified/d0t/splits \
  --analyze_only \
  --seed "$SEED" \
  --active_target "$ACTIVE_TARGET" \
  --none_target "$D0T_NONE_TARGET" \
  --dontcare_target "$D0T_DONTCARE_TARGET"

mv -f data_unified/d0t/splits/d0t_analysis_manifest.json \
  data_unified/d0t/splits/d0t_analysis_manifest_v2.json

run_step python -m dst.data.build_property_splits \
  --source data_unified/luas/train.jsonl \
  --resource_name luas \
  --out_dir data_unified/luas \
  --seed "$SEED" \
  --full_target_total "$FULL_TARGET" \
  --active_target "$ACTIVE_TARGET" \
  --none_target "$LUAS_NONE_TARGET" \
  --dontcare_target "$LUAS_DONTCARE_TARGET" \
  --slot_stratified \
  --allow_downscale

run_step python -m dst.data.build_property_splits \
  --source data_unified/d0t/train.jsonl \
  --resource_name d0t \
  --out_dir data_unified/d0t \
  --seed "$SEED" \
  --full_target_total "$FULL_TARGET" \
  --active_target "$ACTIVE_TARGET" \
  --none_target "$D0T_NONE_TARGET" \
  --dontcare_target "$D0T_DONTCARE_TARGET" \
  --slot_stratified \
  --allow_downscale

PYTHONPATH=src python - <<'PY'
import json
from pathlib import Path

def rename_with_manifest(base_dir: Path, mapping: dict, top_manifest: Path):
    for old_base, new_base in mapping.items():
        old_jsonl = base_dir / f"{old_base}.jsonl"
        old_manifest = base_dir / f"{old_base}.manifest.json"
        new_jsonl = base_dir / f"{new_base}.jsonl"
        new_manifest = base_dir / f"{new_base}.manifest.json"
        if old_jsonl.exists():
            old_jsonl.replace(new_jsonl)
        if old_manifest.exists():
            data = json.loads(old_manifest.read_text(encoding="utf-8"))
            data["output_path"] = str(new_jsonl)
            with new_manifest.open("w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            old_manifest.unlink()

    if top_manifest.exists():
        top = json.loads(top_manifest.read_text(encoding="utf-8"))
        for info in top.get("splits", {}).values():
            out_path = info.get("output_path")
            if out_path:
                for old_base, new_base in mapping.items():
                    old_jsonl = str(base_dir / f"{old_base}.jsonl")
                    new_jsonl = str(base_dir / f"{new_base}.jsonl")
                    if out_path == old_jsonl:
                        info["output_path"] = new_jsonl
            manifest_path = info.get("manifest_path")
            if manifest_path:
                for old_base, new_base in mapping.items():
                    old_manifest = str(base_dir / f"{old_base}.manifest.json")
                    new_manifest = str(base_dir / f"{new_base}.manifest.json")
                    if manifest_path == old_manifest:
                        info["manifest_path"] = new_manifest
        with top_manifest.open("w", encoding="utf-8") as f:
            json.dump(top, f, indent=2)

luas_dir = Path("data_unified/luas")
rename_with_manifest(
    luas_dir,
    {
        "luas_full_128000_train": "luas_full_128k_train",
        "luas_alignedheavy_34516_train": "luas_aligned_34492_train",
        "luas_nonalignedheavy_34516_train": "luas_nonaligned_34492_train",
    },
    luas_dir / "luas_property_splits_manifest.json",
)

d0t_dir = Path("data_unified/d0t")
rename_with_manifest(
    d0t_dir,
    {
        "d0t_full_128000_train": "d0t_full_128k_train",
        "d0t_alignedheavy_45039_train": "d0t_aligned_34492_plus_native_none_train",
        "d0t_nonalignedheavy_45039_train": "d0t_nonaligned_34492_plus_same_native_none_train",
    },
    d0t_dir / "d0t_property_splits_manifest.json",
)
PY

echo
printf "Done. Manifests written to:\n  data_unified/luas/splits/luas_analysis_manifest_v2.json\n  data_unified/d0t/splits/d0t_analysis_manifest_v2.json\n"
