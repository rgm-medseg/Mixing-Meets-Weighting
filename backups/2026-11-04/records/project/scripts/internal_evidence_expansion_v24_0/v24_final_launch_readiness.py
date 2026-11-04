#!/usr/bin/env python3
"""Generate per-task V24 final launch-readiness receipts.

The registry's FinalLaunchReady field is launch permission only. This audit
keeps that separate from post-full completion and refuses to mark NEW_FULL
tasks ready until their own L7 smoke and L8 scratch closed-loop receipts exist.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
FROZEN_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY.tsv"
WORKING_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"
L1_L10 = AUDIT_ROOT / "V24_0_PRELAUNCH_L1_L10_AUDIT.tsv"
SUMMARY = AUDIT_ROOT / "V24_0_FINAL_LAUNCH_READINESS_SUMMARY.json"

NEW_FULL_PENDING_BLOCKER = "WAITING_NEW_FULL_L7_SMOKE_AND_L8_SCRATCH_CLOSED_LOOP"
ACDC2_BLOCKER = "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT"
PROTOCOL_EXCLUDED = "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE"
NON_LAUNCH_BLOCKING_STATES = {"PASS", "BLOCKED_NONCRITICAL", PROTOCOL_EXCLUDED}


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


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def registry_path() -> Path:
    return FROZEN_REGISTRY if FROZEN_REGISTRY.exists() else WORKING_REGISTRY


def passish(value: str) -> bool:
    return value in {"PASS", "PASS_OR_NA", "PASS_WITH_BLOCKERS", "NA", "NA_RECEIPT_CHECK_ONLY"}


def receipt_pass(path: Path) -> bool:
    if not path.exists():
        return False
    try:
        return json.loads(path.read_text()).get("status") == "PASS"
    except json.JSONDecodeError:
        return False


def current_sha_ok(row: dict[str, str], path_key: str, sha_key: str) -> dict[str, Any]:
    value = row.get(path_key, "")
    expected = row.get(sha_key, "")
    if not value:
        return {"applicable": False, "pass": True, "path": "", "expected_sha256": "", "current_sha256": ""}
    if expected and not re.fullmatch(r"[0-9a-f]{64}", expected):
        return {
            "applicable": False,
            "pass": True,
            "path": value,
            "expected_sha256": expected,
            "current_sha256": "",
            "reason": "descriptive_non_sha_field",
        }
    path = ROOT / value
    current = sha256(path) if path.exists() else ""
    ok = path.exists() and (not expected or current == expected)
    return {
        "applicable": True,
        "pass": ok,
        "path": value,
        "exists": path.exists(),
        "expected_sha256": expected,
        "current_sha256": current,
    }


def update_registry_launch_readiness(rows: list[dict[str, str]]) -> list[str]:
    changed = []
    for row in rows:
        if row["TaskID"] in {"ACDC2_BETAFFT_FULL", "ACDC2_OURS_FULL"}:
            if row.get("FinalLaunchReady") != PROTOCOL_EXCLUDED or row.get("Action") != "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED" or row.get("Blocker") != ACDC2_BLOCKER:
                row["Action"] = "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED"
                row["FinalLaunchReady"] = PROTOCOL_EXCLUDED
                row["Blocker"] = ACDC2_BLOCKER
                changed.append(row["TaskID"])
            continue
        if row.get("TaskType") == "NEW_FULL":
            smoke = receipt_pass(AUDIT_ROOT / f"{row['TaskID']}_SMOKE_RECEIPT.json")
            scratch = receipt_pass(AUDIT_ROOT / f"{row['TaskID']}_SCRATCH_CLOSED_LOOP_RECEIPT.json")
            if smoke and scratch:
                if row.get("FinalLaunchReady") != "PASS" or row.get("Blocker"):
                    row["Action"] = "NEW_FULL_READY"
                    row["FinalLaunchReady"] = "PASS"
                    row["Blocker"] = ""
                    changed.append(row["TaskID"])
            else:
                if row.get("FinalLaunchReady") != "False" or row.get("Blocker") != NEW_FULL_PENDING_BLOCKER:
                    row["Action"] = "NEW_FULL_PRELAUNCH_PENDING"
                    row["FinalLaunchReady"] = "False"
                    row["Blocker"] = NEW_FULL_PENDING_BLOCKER
                    changed.append(row["TaskID"])
    if changed and registry_path() == WORKING_REGISTRY:
        write_tsv(WORKING_REGISTRY, rows, list(rows[0].keys()))
    return changed


def build_task_payload(row: dict[str, str], lrow: dict[str, str]) -> dict[str, Any]:
    task_id = row["TaskID"]
    source = current_sha_ok(row, "SourcePath", "SourceSHA")
    launcher = current_sha_ok(row, "Launcher", "LauncherSHA")
    config = current_sha_ok(row, "ConfigPath", "ConfigSHA")
    checkpoint = current_sha_ok(row, "CheckpointPath", "CheckpointSHA")
    source_freeze = all(item["pass"] for item in [source, launcher])
    config_freeze = config["pass"] and checkpoint["pass"]
    static_pass = lrow.get("L1StaticPaths") == "PASS"
    argv_pass = passish(lrow.get("L2DryRunOrReceiptMode", ""))
    path_pass = lrow.get("L3Paths") == "PASS"
    sci_pass = passish(lrow.get("L4ScientificIdentity", ""))
    import_pass = passish(lrow.get("L5ImportCompile", ""))
    role_pass = lrow.get("L6TestFirewall") == "PASS"
    smoke_pass = passish(lrow.get("L7SmokeGate", "")) and not lrow.get("L7SmokeGate", "").startswith("WAITING")
    closed_pass = lrow.get("L8ClosedLoopReceipt") == "PASS"
    failure_pass = passish(lrow.get("L9FailureInjection", ""))
    controller_pass = passish(lrow.get("L10ControllerDryConsumption", ""))
    output_clean = True
    if row.get("TaskType") == "NEW_FULL" and row.get("FinalLaunchReady") != "PASS":
        output_clean = not Path(ROOT / row.get("ExpectedReceipt", "")).exists()
    checks = {
        "StaticPass": static_pass,
        "ArgvPass": argv_pass,
        "PathPass": path_pass,
        "ScientificIdentityPass": sci_pass,
        "ImportPass": import_pass,
        "RoleBindingPass": role_pass,
        "SmokePass": smoke_pass,
        "ClosedLoopPass": closed_pass,
        "FailureInjectionPass": failure_pass,
        "ControllerDryPass": controller_pass,
        "OutputClean": output_clean,
        "SourceFreezePass": source_freeze,
        "ConfigFreezePass": config_freeze,
    }
    registry_ready = row.get("FinalLaunchReady", "")
    launch_blocking = registry_ready not in NON_LAUNCH_BLOCKING_STATES
    protocol_excluded = registry_ready == PROTOCOL_EXCLUDED
    hard_blocked = bool(row.get("Blocker")) or registry_ready != "PASS"
    final_launch_ready = all(checks.values()) and not hard_blocked
    payload = {
        "status": "PASS" if final_launch_ready else "NOT_READY",
        "created_at": now(),
        "task_id": task_id,
        "task_type": row.get("TaskType"),
        "action": row.get("Action"),
        "registry_FinalLaunchReady": row.get("FinalLaunchReady"),
        "registry_Blocker": row.get("Blocker"),
        "registry_BlockerCategory": row.get("Blocker", "") or "NONE",
        "LaunchBlocking": launch_blocking,
        "NoncriticalTerminalGap": registry_ready == "BLOCKED_NONCRITICAL",
        "ProtocolExcluded": protocol_excluded,
        "FinalLaunchReady": final_launch_ready,
        "FinalLaunchReady_semantics": "launch permission only; not completion evidence",
        "PostFullCompletionEvidenceRequired": row.get("TaskType") in {"POST_FULL", "FINALIZATION"},
        "FormalFullStarted": False,
        "TmuxStarted": False,
        "checks": checks,
        "l1_l10": lrow,
        "freeze_evidence": {
            "source": source,
            "launcher": launcher,
            "config": config,
            "checkpoint": checkpoint,
        },
        "expected_receipt": row.get("ExpectedReceipt", ""),
        "test_semantics": {
            "TESTAccessAllowed": row.get("TESTAccessAllowed", ""),
            "TEST_DATA_READ_FOR_PRE_REGISTERED_EVALUATION": row.get("TESTAccessAllowed") == "PRE_REGISTERED_EVALUATION_ONLY",
            "TEST_USED_FOR_TUNING": False,
            "TEST_USED_FOR_SELECTION": False,
        },
    }
    write_json(AUDIT_ROOT / f"task_{task_id}_FINAL_LAUNCH_READINESS.json", payload)
    return payload


def main() -> int:
    registry = registry_path()
    rows = read_tsv(registry)
    registry_changes = update_registry_launch_readiness(rows) if registry == WORKING_REGISTRY else []
    lrows = {row["TaskID"]: row for row in read_tsv(L1_L10)}
    task_payloads = [build_task_payload(row, lrows.get(row["TaskID"], {})) for row in rows]
    pass_count = sum(1 for payload in task_payloads if payload["FinalLaunchReady"])
    pending = [payload["task_id"] for payload in task_payloads if not payload["FinalLaunchReady"]]
    launch_blocking = [
        {
            "TaskID": payload["task_id"],
            "TaskType": payload.get("task_type", ""),
            "Action": payload.get("action", ""),
            "FinalLaunchReady": payload.get("registry_FinalLaunchReady", ""),
            "Blocker": payload.get("registry_Blocker", ""),
        }
        for payload in task_payloads
        if payload.get("LaunchBlocking")
    ]
    noncritical_gaps = [
        {
            "TaskID": payload["task_id"],
            "TaskType": payload.get("task_type", ""),
            "Action": payload.get("action", ""),
            "FinalLaunchReady": payload.get("registry_FinalLaunchReady", ""),
            "Blocker": payload.get("registry_Blocker", ""),
        }
        for payload in task_payloads
        if payload.get("NoncriticalTerminalGap")
    ]
    protocol_excluded = [
        {
            "TaskID": payload["task_id"],
            "TaskType": payload.get("task_type", ""),
            "Action": payload.get("action", ""),
            "FinalLaunchReady": payload.get("registry_FinalLaunchReady", ""),
            "Blocker": payload.get("registry_Blocker", ""),
        }
        for payload in task_payloads
        if payload.get("ProtocolExcluded")
    ]
    active_new_full_ready = [
        payload for payload in task_payloads
        if payload.get("task_type") == "NEW_FULL" and payload.get("registry_FinalLaunchReady") == "PASS"
    ]
    prelaunch_ready = len(launch_blocking) == 0 and len(active_new_full_ready) == 4
    payload = {
        "status": "PRELAUNCH_READY" if prelaunch_ready else "NOT_PRELAUNCH_READY",
        "created_at": now(),
        "formal_started": False,
        "tmux_started": False,
        "V24_0_PRELAUNCH_READY": prelaunch_ready,
        "EXPECTED_NEW_FULL_COUNT": 4,
        "EXPECTED_NEW_FULL_READY": 4,
        "active_new_full_ready_count": len(active_new_full_ready),
        "task_count": len(task_payloads),
        "final_launch_ready_count": pass_count,
        "not_ready_count": len(pending),
        "not_ready_task_ids": pending,
        "launch_blocking_count": len(launch_blocking),
        "launch_blocking_tasks": launch_blocking,
        "noncritical_terminal_gap_count": len(noncritical_gaps),
        "noncritical_terminal_gap_tasks": noncritical_gaps,
        "protocol_excluded_count": len(protocol_excluded),
        "protocol_excluded_tasks": protocol_excluded,
        "registry_changes": registry_changes,
        "registry": rel(registry),
        "registry_sha256": sha256(registry),
        "l1_l10_audit": rel(L1_L10),
        "l1_l10_audit_sha256": sha256(L1_L10),
        "generated_task_readiness_glob": "audits/internal_evidence_expansion_v24_0/task_*_FINAL_LAUNCH_READINESS.json",
        "hard_gate": "No NEW_FULL task is FinalLaunchReady without its own real TRAIN/VAL smoke receipt and scratch closed-loop receipt.",
    }
    write_json(SUMMARY, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
