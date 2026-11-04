#!/usr/bin/env python3
"""Freeze deterministic ACDC10 VAL cases for V23 mechanism visualization."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
ACDC = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC")
AUDIT_ROOT = ROOT / "audits/paper_internal_evidence_v23_0"
RESULT_ROOT = ROOT / "final_report/paper_internal_evidence_v23_0"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def freeze_cases(limit: int) -> list[dict[str, object]]:
    val_manifest = ACDC / "val.list"
    cases = sorted([line.strip() for line in val_manifest.read_text().splitlines() if line.strip()])
    selected = []
    for case in cases[:limit]:
        volume_path = ACDC / "data" / f"{case}.h5"
        with h5py.File(volume_path, "r") as handle:
            image_shape = list(handle["image"].shape)
            label = handle["label"][:]
        foreground = (label > 0).reshape(label.shape[0], -1).sum(axis=1)
        slice_index = int(np.argmax(foreground))
        selected.append(
            {
                "CaseOrder": len(selected) + 1,
                "CaseID": case,
                "Split": "VAL",
                "VolumePath": str(volume_path),
                "ImageShape": "x".join(str(v) for v in image_shape),
                "SelectedSliceIndex0Based": slice_index,
                "SelectedSliceForegroundPixels": int(foreground[slice_index]),
                "SelectionRule": "val.list sorted ascending; first four volumes; slice=max_GT_foreground",
                "TestAccess": 0,
            }
        )
    return selected


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "run"], default="run")
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "val_manifest": str((ACDC / "val.list")),
                    "selection_rule": "val.list sorted ascending; first N volumes; slice=max_GT_foreground",
                    "limit": args.limit,
                    "test_access": 0,
                },
                indent=2,
            )
        )
        return 0

    rows = freeze_cases(args.limit)
    tsv = RESULT_ROOT / "ACDC10_VAL_MECHANISM_VIS_CASE_SELECTION.tsv"
    write_tsv(tsv, rows)
    payload = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection_tsv": str(tsv.relative_to(ROOT)),
        "case_count": len(rows),
        "split": "VAL",
        "test_access": 0,
        "selection_rule": rows[0]["SelectionRule"] if rows else "",
        "cases": rows,
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "ACDC10_VAL_MECHANISM_VIS_CASE_SELECTION.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
