#!/usr/bin/env python3
"""Create the V24 formal start manifest only after prelaunch PASS."""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
FROZEN_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY.tsv"
WORKING_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"
PRELAUNCH = AUDIT_ROOT / "V24_0_PRELAUNCH_FIVE_REVIEW.json"
L1_SUMMARY = AUDIT_ROOT / "V24_0_PRELAUNCH_L1_L10_AUDIT_SUMMARY.json"
READINESS = AUDIT_ROOT / "V24_0_FINAL_LAUNCH_READINESS_SUMMARY.json"
ENV_GATE = AUDIT_ROOT / "V24_0_PRESTART_ENVIRONMENT_GATE.json"
RESUME_AUDIT = AUDIT_ROOT / "V24_0_RESUME_SAFETY_AUDIT.json"
CONTROLLER = ROOT / "controllers/internal_evidence_expansion_controller_v24_0.py"
OUT = AUDIT_ROOT / "V24_0_FORMAL_START_MANIFEST.json"

PRE_REGISTERED_SPECS = [
    AUDIT_ROOT / "V24_0_SEED_SPEC.json",
    AUDIT_ROOT / "V24_0_CORRUPTION_SPEC.json",
    AUDIT_ROOT / "V24_0_BOUNDARY_SPEC.json",
    AUDIT_ROOT / "V24_0_STATISTICS_SPEC.json",
    AUDIT_ROOT / "X03_SCORE_SEMANTIC_CONTRACT.json",
    AUDIT_ROOT / "ACDC2_PROTOCOL_DECISION.json",
    ROOT / "configs/internal_evidence_expansion_v24_0/ACDC2_SPLIT.json",
]

EVALUATOR_PROTOCOLS = [
    ROOT / "train_ACDC/val_2D.py",
    ROOT / "train_ACDC/test_ACDC_beta_FFT.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_frozen_best_test_launcher.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_acdc10_frozen_replay_analyses.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_patient_level_statistics.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_x03_reliability_faithfulness.py",
]

PRELAUNCH_BOOTSTRAP_SCRIPTS = [
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_prelaunch_l1_l10_audit.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_final_launch_readiness.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_prelaunch_five_review.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_formal_start_manifest.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_failure_injection_audit.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_hard_gate_semantic_audits.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_master_registry_freeze.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_prestart_environment_gate.py",
    ROOT / "scripts/internal_evidence_expansion_v24_0/v24_resume_safety_audit.py",
]
PROTOCOL_EXCLUDED = "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def manifest_registry() -> Path:
    return FROZEN_REGISTRY if FROZEN_REGISTRY.exists() else WORKING_REGISTRY


def display_path(path: Path) -> str:
    path = path if path.is_absolute() else ROOT / path
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def asset_ref(path: Path) -> dict[str, Any]:
    path = path if path.is_absolute() else ROOT / path
    return {
        "path": display_path(path),
        "exists": path.exists(),
        "sha256": sha256(path) if path.exists() else "",
    }


def collect_sha(rows: list[dict[str, str]], field: str, sha_field: str) -> list[dict[str, str]]:
    assets = []
    seen = set()
    for row in rows:
        value = row.get(field, "")
        if not value or value in seen:
            continue
        seen.add(value)
        path = ROOT / value
        assets.append({
            "path": value,
            "exists": path.exists(),
            "registry_sha256": row.get(sha_field, ""),
            "current_sha256": sha256(path) if path.exists() else "",
        })
    return assets


def collect_config_manifests(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    assets: list[dict[str, Any]] = []
    seen: set[str] = set()
    manifest_keys = ["labeled_list", "unlabeled_list", "val_list", "test_list"]
    for row in rows:
        cfg_value = row.get("ConfigPath", "")
        if not cfg_value:
            continue
        cfg_path = ROOT / cfg_value
        if not cfg_path.exists():
            continue
        try:
            cfg = json.loads(cfg_path.read_text())
        except json.JSONDecodeError:
            continue
        for key in manifest_keys:
            value = cfg.get(key, "")
            if not value or value in seen:
                continue
            seen.add(value)
            path = Path(value)
            assets.append({"config": cfg_value, "key": key, **asset_ref(path)})
        dataset_root = cfg.get("dataset_root", "")
        if dataset_root:
            for name in ["train.list", "val.list", "test.list"]:
                value = str(Path(dataset_root) / name)
                if value in seen:
                    continue
                seen.add(value)
                path = Path(value)
                assets.append({"config": cfg_value, "key": f"dataset_root/{name}", **asset_ref(path)})
    return assets


def collect_protocol_locks(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    assets = []
    for row in rows:
        if row.get("TaskType") == "NEW_FULL" and row.get("FinalLaunchReady") == "PASS":
            path = AUDIT_ROOT / f"{row['TaskID']}_TEST_PROTOCOL_LOCK.json"
            assets.append({"task_id": row["TaskID"], **asset_ref(path)})
    return assets


def collect_named_assets(paths: list[Path]) -> list[dict[str, Any]]:
    return [asset_ref(path) for path in paths]


def all_assets_exist(groups: list[list[dict[str, Any]]]) -> bool:
    return all(asset.get("exists") is True for group in groups for asset in group)


def blocker_details(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    blocked = [row for row in rows if row.get("Blocker") or row.get("FinalLaunchReady") != "PASS"]
    launch_blocking = [row for row in blocked if row.get("FinalLaunchReady") not in {"BLOCKED_NONCRITICAL", PROTOCOL_EXCLUDED}]
    noncritical = [row for row in blocked if row.get("FinalLaunchReady") == "BLOCKED_NONCRITICAL"]
    protocol_excluded = [row for row in blocked if row.get("FinalLaunchReady") == PROTOCOL_EXCLUDED]

    def compact(row: dict[str, str]) -> dict[str, str]:
        return {
            "TaskID": row.get("TaskID", ""),
            "TaskType": row.get("TaskType", ""),
            "Action": row.get("Action", ""),
            "FinalLaunchReady": row.get("FinalLaunchReady", ""),
            "Blocker": row.get("Blocker", ""),
        }

    return {
        "launch_blocking_tasks": [compact(row) for row in launch_blocking],
        "noncritical_terminal_gap_tasks": [compact(row) for row in noncritical],
        "protocol_excluded_tasks": [compact(row) for row in protocol_excluded],
    }


def main() -> int:
    prelaunch = read_json(PRELAUNCH)
    registry = manifest_registry()
    rows = read_tsv(registry) if registry.exists() else []
    dataset_manifests = collect_config_manifests(rows)
    pre_registered_specs = collect_named_assets(PRE_REGISTERED_SPECS)
    evaluator_protocols = collect_named_assets(EVALUATOR_PROTOCOLS)
    prelaunch_bootstrap_scripts = collect_named_assets(PRELAUNCH_BOOTSTRAP_SCRIPTS)
    test_protocol_locks = collect_protocol_locks(rows)
    blockers = blocker_details(rows)
    if prelaunch.get("status") != "PASS" or prelaunch.get("FORMAL_START_AUTHORIZATION") is not True:
        payload = {
            "status": "REFUSE_PRELAUNCH_NOT_PASS",
            "created_at": now(),
            "FORMAL_START_AUTHORIZATION": False,
            "formal_started": False,
            "tmux_started": False,
            "registry": asset_ref(registry),
            "controller": asset_ref(CONTROLLER),
            "l1_l10_summary": asset_ref(L1_SUMMARY),
            "readiness_summary": asset_ref(READINESS),
            "prestart_environment_gate": asset_ref(ENV_GATE),
            "resume_safety_audit": asset_ref(RESUME_AUDIT),
            "dataset_manifests": dataset_manifests,
            "pre_registered_specs": pre_registered_specs,
            "evaluator_protocols": evaluator_protocols,
            "prelaunch_bootstrap_scripts": prelaunch_bootstrap_scripts,
            "test_protocol_locks": test_protocol_locks,
            "prelaunch_review": str(PRELAUNCH.relative_to(ROOT)),
            "prelaunch_status": prelaunch.get("status", "MISSING"),
            "launch_blocking_task_ids": prelaunch.get("launch_blocking_task_ids", []),
            **blockers,
        }
        write_json(AUDIT_ROOT / "V24_0_FORMAL_START_MANIFEST_REFUSAL.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 3
    counts = {
        "TOTAL": len(rows),
        "NEW_FULL_READY": sum(1 for row in rows if row.get("TaskType") == "NEW_FULL" and row.get("FinalLaunchReady") == "PASS"),
        "EXPECTED_NEW_FULL_READY": 4,
        "PROTOCOL_EXCLUDED": sum(1 for row in rows if row.get("FinalLaunchReady") == PROTOCOL_EXCLUDED),
        "BLOCKED_NONCRITICAL": sum(1 for row in rows if row.get("FinalLaunchReady") == "BLOCKED_NONCRITICAL"),
        "LAUNCH_BLOCKING": len(blockers["launch_blocking_tasks"]),
        "PENDING": sum(1 for row in rows if row.get("FinalLaunchReady") == "False"),
        "TEST_UNLOCKED_COUNT": 0,
    }
    expected_counts_ok = (
        counts["TOTAL"] == 27
        and counts["NEW_FULL_READY"] == 4
        and counts["PROTOCOL_EXCLUDED"] == 2
        and counts["BLOCKED_NONCRITICAL"] == 3
        and counts["LAUNCH_BLOCKING"] == 0
        and counts["PENDING"] == 0
        and counts["TEST_UNLOCKED_COUNT"] == 0
    )
    asset_groups_ok = all_assets_exist([dataset_manifests, pre_registered_specs, evaluator_protocols, prelaunch_bootstrap_scripts, test_protocol_locks])
    if not expected_counts_ok or not asset_groups_ok:
        payload = {
            "status": "REFUSE_START_MANIFEST_ASSERTION_FAIL",
            "created_at": now(),
            "FORMAL_START_AUTHORIZATION": False,
            "formal_started": False,
            "tmux_started": False,
            "registry": asset_ref(registry),
            "controller": asset_ref(CONTROLLER),
            "task_counts": counts,
            "expected_counts_ok": expected_counts_ok,
            "asset_groups_ok": asset_groups_ok,
            **blockers,
            "dataset_manifests": dataset_manifests,
            "pre_registered_specs": pre_registered_specs,
            "evaluator_protocols": evaluator_protocols,
            "prelaunch_bootstrap_scripts": prelaunch_bootstrap_scripts,
            "test_protocol_locks": test_protocol_locks,
        }
        write_json(AUDIT_ROOT / "V24_0_FORMAL_START_MANIFEST_REFUSAL.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 3
    payload = {
        "status": "PASS",
        "created_at": now(),
        "FORMAL_START_AUTHORIZATION": True,
        "formal_started": False,
        "tmux_started": False,
        "registry": asset_ref(registry),
        "FinalRegistrySHA256": sha256(registry),
        "ControllerSHA256": sha256(CONTROLLER),
        "L1L10SummarySHA256": sha256(L1_SUMMARY),
        "ReadinessSummarySHA256": sha256(READINESS),
        "PrestartEnvironmentGateSHA256": sha256(ENV_GATE),
        "ResumeSafetyAuditSHA256": sha256(RESUME_AUDIT),
        "task_counts": counts,
        "protocol_excluded_tasks": blockers["protocol_excluded_tasks"],
        "noncritical_terminal_gap_tasks": blockers["noncritical_terminal_gap_tasks"],
        "launch_blocking_tasks": blockers["launch_blocking_tasks"],
        "sources": collect_sha(rows, "SourcePath", "SourceSHA"),
        "configs": collect_sha(rows, "ConfigPath", "ConfigSHA"),
        "launchers": collect_sha(rows, "Launcher", "LauncherSHA"),
        "checkpoints": collect_sha(rows, "CheckpointPath", "CheckpointSHA"),
        "dataset_manifests": dataset_manifests,
        "pre_registered_specs": pre_registered_specs,
        "evaluator_protocols": evaluator_protocols,
        "prelaunch_bootstrap_scripts": prelaunch_bootstrap_scripts,
        "test_protocol_locks": test_protocol_locks,
        "seeds": [1337, 2025, 2026],
        "X03_WEIGHT": 0.05,
        "X03_WEIGHT_RESELECTION": False,
        "TEST_UNLOCKED_COUNT": 0,
        "launch_blocking_count": counts["LAUNCH_BLOCKING"],
        "protocol_excluded_count": counts["PROTOCOL_EXCLUDED"],
        "noncritical_terminal_gap_count": counts["BLOCKED_NONCRITICAL"],
    }
    write_json(OUT, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
