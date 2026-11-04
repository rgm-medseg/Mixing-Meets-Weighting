#!/usr/bin/env python3
"""Preflight audit for V24 frozen-replay Plane-A tasks.

The audit does not run inference and does not access TEST manifests. It checks
whether canonical frozen outputs already exist, and whether checkpoints/specs
needed for a later audited replay are present.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"

CANONICAL_CASE_METRICS = {
    "ACDC10_ARM_000": ROOT / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/case_class_metrics.tsv",
    "ACDC10_ARM_110": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/OURS/case_class_metrics.tsv",
    "ACDC10_ARM_111": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/X03/case_class_metrics.tsv",
}

CANONICAL_CHECKPOINTS = {
    "ACDC10_ARM_000": ROOT / "runs/beta_fft_acdc_official_full_seed42_fresh_ckpt/checkpoints/best.ckpt",
    "ACDC10_ARM_110": ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/checkpoints/best.ckpt",
    "ACDC10_ARM_111": ROOT / "runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt",
}

SPECS = {
    "BOUNDARY_F1": ROOT / "audits/internal_evidence_expansion_v24_0/V24_0_BOUNDARY_SPEC.json",
    "CORRUPTION_ROBUSTNESS": ROOT / "audits/internal_evidence_expansion_v24_0/V24_0_CORRUPTION_SPEC.json",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def audit_case_metrics() -> dict:
    rows = []
    missing = []
    for method, path in CANONICAL_CASE_METRICS.items():
        exists = path.exists()
        case_count = 0
        class_rows = 0
        if exists:
            data = read_tsv(path)
            class_rows = len(data)
            case_count = len({row["case"] for row in data})
        else:
            missing.append(method)
        rows.append(
            {
                "MethodID": method,
                "CanonicalCaseMetrics": str(path.relative_to(ROOT)),
                "Exists": str(exists),
                "SHA256": sha256(path) if exists else "",
                "CaseCount": case_count,
                "ClassMetricRowCount": class_rows,
                "DirectReadEligible": str(exists and case_count > 0),
            }
        )
    out = AUDIT_ROOT / "V24_0_CANONICAL_ACDC10_CASE_METRICS_AUDIT.tsv"
    write_tsv(out, rows)
    receipt = {
        "status": "BLOCKED_MISSING_CANONICAL_CASE_METRICS" if missing else "PASS",
        "created_at": now(),
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "audit_asset": str(out.relative_to(ROOT)),
        "audit_sha256": sha256(out),
        "missing_methods": missing,
        "affected_tasks": ["PATIENT_LEVEL_STATS", "HARD_MEDIUM_EASY", "BOUNDARY_F1"],
        "identity_binding": "ACDC10_ARM_110 uses V21.5 raw_test/OURS; ACDC10_ARM_111 uses V21.5 raw_test/X03. See ACDC10_ARM_000_110_111_CASE_METRICS_BINDING.json.",
    }
    write_json(AUDIT_ROOT / "V24_0_CANONICAL_ACDC10_CASE_METRICS_AUDIT.json", receipt)
    return receipt


def audit_replay_readiness() -> dict:
    rows = []
    missing = []
    for method, path in CANONICAL_CHECKPOINTS.items():
        exists = path.exists()
        if not exists:
            missing.append(method)
        rows.append(
            {
                "MethodID": method,
                "CheckpointPath": str(path.relative_to(ROOT)),
                "Exists": str(exists),
                "SHA256": sha256(path) if exists else "",
                "FrozenReplayCheckpointReady": str(exists),
            }
        )
    for task, path in SPECS.items():
        exists = path.exists()
        if not exists:
            missing.append(task)
        rows.append(
            {
                "MethodID": task,
                "CheckpointPath": str(path.relative_to(ROOT)),
                "Exists": str(exists),
                "SHA256": sha256(path) if exists else "",
                "FrozenReplayCheckpointReady": str(exists),
            }
        )
    out = AUDIT_ROOT / "V24_0_FROZEN_REPLAY_PREFLIGHT_AUDIT.tsv"
    write_tsv(out, rows)
    receipt = {
        "status": "PASS_PREFLIGHT_ASSETS_PRESENT" if not missing else "BLOCKED_MISSING_REPLAY_PREFLIGHT_ASSET",
        "created_at": now(),
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "audit_asset": str(out.relative_to(ROOT)),
        "audit_sha256": sha256(out),
        "missing_assets": missing,
        "boundary_f1_status": "FROZEN_REPLAY_SCRIPT_REQUIRED",
        "corruption_robustness_status": "FROZEN_REPLAY_SCRIPT_REQUIRED",
        "hard_medium_easy_status": "DIRECT_READ_READY_AFTER_CASE_METRICS_BINDING_PASS",
    }
    write_json(AUDIT_ROOT / "V24_0_FROZEN_REPLAY_PREFLIGHT_AUDIT.json", receipt)
    return receipt


def main() -> int:
    summary = {
        "status": "PASS_PREFLIGHT_MAPPED",
        "created_at": now(),
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "case_metric_audit": audit_case_metrics(),
        "replay_preflight_audit": audit_replay_readiness(),
    }
    write_json(AUDIT_ROOT / "V24_0_REPLAY_PREFLIGHT_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
