#!/usr/bin/env python3
"""V24 NEW_FULL scratch closed-loop canary.

This script does not train, infer on TEST, or produce scientific metrics. It
checks that a NEW_FULL task can bind its frozen config, VAL-only canary input,
parser output, and receipt path without unlocking post-full TEST.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RUNTIME_ROOT = ROOT / "runtime/internal_evidence_expansion_v24_0/scratch_closed_loop"
REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def registry_row(task_id: str) -> dict[str, str]:
    rows = read_tsv(REGISTRY)
    for row in rows:
        if row["TaskID"] == task_id:
            return row
    raise SystemExit(f"TASK_NOT_IN_REGISTRY:{task_id}")


def load_config(row: dict[str, str]) -> dict[str, Any]:
    if row.get("TaskType") != "NEW_FULL":
        raise SystemExit("SCRATCH_CLOSED_LOOP_ONLY_FOR_NEW_FULL")
    if row.get("Blocker") == "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT":
        raise SystemExit("ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT")
    cfg_path = ROOT / row["ConfigPath"]
    if not cfg_path.exists():
        raise SystemExit(f"CONFIG_MISSING:{row['ConfigPath']}")
    cfg = json.loads(cfg_path.read_text())
    if cfg.get("test_access_before_best_freeze") != 0:
        raise SystemExit("TEST_ACCESS_BEFORE_BEST_FREEZE_NOT_ZERO")
    if cfg.get("paper_eligible_test_result_before_gate") is not False:
        raise SystemExit("PAPER_RESULT_GATE_OPEN_BEFORE_FORMAL")
    return cfg


def val_canary_ids(cfg: dict[str, Any]) -> list[str]:
    val_list = cfg.get("val_list")
    if not val_list:
        raise SystemExit("VAL_LIST_NOT_BOUND")
    path = Path(val_list)
    if not path.exists():
        raise SystemExit(f"VAL_LIST_MISSING:{val_list}")
    ids = [line.strip() for line in path.read_text().splitlines() if line.strip()]
    if not ids:
        raise SystemExit("VAL_LIST_EMPTY")
    return ids[: min(3, len(ids))]


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mode", choices=["run", "dry-run"], default="run")
    args = parser.parse_args()

    row = registry_row(args.task_id)
    cfg = load_config(row)
    canary_ids = val_canary_ids(cfg)
    out_dir = RUNTIME_ROOT / args.task_id
    manifest = out_dir / "VAL_ENGINEERING_CANARY_MANIFEST.tsv"
    metrics = out_dir / "SCRATCH_CANARY_FULL_PRECISION.tsv"
    receipt = AUDIT_ROOT / f"{args.task_id}_SCRATCH_CLOSED_LOOP_RECEIPT.json"
    canary_rows = [
        {
            "TaskID": args.task_id,
            "Split": "VAL",
            "CaseID": case_id,
            "CanaryIndex": idx,
            "TEST_DATA_READ": "false",
        }
        for idx, case_id in enumerate(canary_ids, start=1)
    ]
    metric_rows = [
        {
            "TaskID": args.task_id,
            "Split": "VAL",
            "CaseID": item["CaseID"],
            "Dice": "",
            "IoU": "",
            "HD95": "",
            "ASD": "",
            "ENGINEERING_CANARY_ONLY": "true",
            "SCIENTIFIC_RESULT": "false",
            "PAPER_RESULT_ELIGIBLE": "false",
        }
        for item in canary_rows
    ]
    if args.mode == "run":
        write_tsv(manifest, canary_rows, ["TaskID", "Split", "CaseID", "CanaryIndex", "TEST_DATA_READ"])
        write_tsv(
            metrics,
            metric_rows,
            [
                "TaskID",
                "Split",
                "CaseID",
                "Dice",
                "IoU",
                "HD95",
                "ASD",
                "ENGINEERING_CANARY_ONLY",
                "SCIENTIFIC_RESULT",
                "PAPER_RESULT_ELIGIBLE",
            ],
        )
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": args.task_id,
        "mode": args.mode,
        "training_started": False,
        "formal_started": False,
        "inference_started": False,
        "test_started": False,
        "split": "VAL",
        "TEST_DATA_READ": False,
        "TEST_USED_FOR_TUNING": False,
        "TEST_USED_FOR_SELECTION": False,
        "ENGINEERING_CANARY_ONLY": True,
        "SCIENTIFIC_RESULT": False,
        "PAPER_RESULT_ELIGIBLE": False,
        "config": row["ConfigPath"],
        "config_sha256": sha256(ROOT / row["ConfigPath"]),
        "val_list": cfg["val_list"],
        "val_list_sha256": sha256(Path(cfg["val_list"])),
        "val_canary_count": len(canary_ids),
        "val_canary_ids": canary_ids,
        "manifest": rel(manifest),
        "manifest_sha256": sha256(manifest) if manifest.exists() else "",
        "metric_asset": rel(metrics),
        "metric_asset_sha256": sha256(metrics) if metrics.exists() else "",
        "closed_loop_components": [
            "config_binding",
            "VAL_engineering_canary_manifest",
            "parser_full_precision_tsv",
            "receipt",
        ],
    }
    if args.mode == "run":
        write_json(receipt, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
