#!/usr/bin/env python3
"""V24 prelaunch five-review gate.

This produces an explicit machine-readable prelaunch decision. It never starts
training, TEST, tmux, or the formal controller.
"""

from __future__ import annotations

import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
FROZEN_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY.tsv"
WORKING_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"
L1_SUMMARY = AUDIT_ROOT / "V24_0_PRELAUNCH_L1_L10_AUDIT_SUMMARY.json"
READINESS = AUDIT_ROOT / "V24_0_FINAL_LAUNCH_READINESS_SUMMARY.json"
ENV_GATE = AUDIT_ROOT / "V24_0_PRESTART_ENVIRONMENT_GATE.json"
RESUME_AUDIT = AUDIT_ROOT / "V24_0_RESUME_SAFETY_AUDIT.json"
OUT = AUDIT_ROOT / "V24_0_PRELAUNCH_FIVE_REVIEW.json"
PROTOCOL_EXCLUDED = "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def registry_path() -> Path:
    return FROZEN_REGISTRY if FROZEN_REGISTRY.exists() else WORKING_REGISTRY


def main() -> int:
    registry = registry_path()
    rows = read_tsv(registry)
    l1 = read_json(L1_SUMMARY)
    readiness = read_json(READINESS)
    env_gate = read_json(ENV_GATE)
    resume_audit = read_json(RESUME_AUDIT)
    task_ids = {row["TaskID"] for row in rows}
    actions = {row["Action"] for row in rows}
    blocked = [row for row in rows if row.get("Blocker") or row.get("FinalLaunchReady") != "PASS"]
    noncritical = [row for row in blocked if row.get("FinalLaunchReady") == "BLOCKED_NONCRITICAL"]
    protocol_excluded = [row for row in blocked if row.get("FinalLaunchReady") == PROTOCOL_EXCLUDED]
    launch_blocking = [row for row in blocked if row.get("FinalLaunchReady") not in {"BLOCKED_NONCRITICAL", PROTOCOL_EXCLUDED}]
    new_full = [row for row in rows if row.get("TaskType") == "NEW_FULL"]
    active_new_full_ready = [row for row in new_full if row.get("FinalLaunchReady") == "PASS"]
    external_forbidden = any(token in " ".join(task_ids | actions) for token in ["DTC", "URPC", "MC_NET", "HCCP", "SAM2"])
    checks = {
        "PRELAUNCH_REVIEW_1_SCOPE": (
            not external_forbidden
            and len(active_new_full_ready) == 4
            and len(protocol_excluded) == 2
            and "PROMISE10" not in task_ids
            and "MSCMR" not in task_ids
        ),
        "PRELAUNCH_REVIEW_2_PREREG": (
            (AUDIT_ROOT / "V24_0_SEED_SPEC.json").exists()
            and (AUDIT_ROOT / "V24_0_CORRUPTION_SPEC.json").exists()
            and (AUDIT_ROOT / "V24_0_BOUNDARY_SPEC.json").exists()
            and (AUDIT_ROOT / "V24_0_STATISTICS_SPEC.json").exists()
            and all(row.get("Action") == "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED" for row in protocol_excluded)
            and all(row.get("Blocker") == "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT" for row in protocol_excluded)
        ),
        "PRELAUNCH_REVIEW_3_SCRIPTS": (
            l1.get("required_scripts_pass") is True
            and not launch_blocking
            and l1.get("controller_dry_run_failed_count") == 0
        ),
        "PRELAUNCH_REVIEW_4_DATA": (
            l1.get("hard_gate_status", {}).get("V24_TEST_SEMANTIC_SPLIT_AUDIT") == "PASS"
            and l1.get("hard_gate_status", {}).get("X03_RAW_SCORE_EFFECTIVE_WEIGHT_SEMANTIC_AUDIT") == "PASS"
        ),
        "PRELAUNCH_REVIEW_5_CONTROLLER": (
            l1.get("controller_dry_run_status") in {"PASS", "PASS_WITH_NONCRITICAL_BLOCKERS"}
            and l1.get("controller_dry_run_failed_count") == 0
            and not launch_blocking
            and env_gate.get("status") == "PASS"
            and resume_audit.get("status") == "PASS"
        ),
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at": now(),
        "task_id": "V24_0_PRELAUNCH_FIVE_REVIEW",
        "registry": str(registry.relative_to(ROOT)),
        "EXPECTED_NEW_FULL_COUNT": 4,
        "EXPECTED_NEW_FULL_READY": 4,
        "NEW_FULL_READY": len(active_new_full_ready),
        "protocol_excluded_count": len(protocol_excluded),
        "noncritical_terminal_gap_count": len(noncritical),
        "launch_blocking_count": len(launch_blocking),
        **{key: "PASS" if value else "FAIL" for key, value in checks.items()},
        "formal_started": False,
        "tmux_started": False,
        "FORMAL_START_AUTHORIZATION": all(checks.values()),
        "blocked_or_pending_task_ids": [row["TaskID"] for row in blocked],
        "noncritical_terminal_gap_task_ids": [row["TaskID"] for row in noncritical],
        "protocol_excluded_task_ids": [row["TaskID"] for row in protocol_excluded],
        "launch_blocking_task_ids": [row["TaskID"] for row in launch_blocking],
        "hard_blockers": [
            {"TaskID": row["TaskID"], "Blocker": row.get("Blocker"), "FinalLaunchReady": row.get("FinalLaunchReady")}
            for row in launch_blocking
        ],
        "noncritical_terminal_gaps": [
            {"TaskID": row["TaskID"], "Blocker": row.get("Blocker"), "FinalLaunchReady": row.get("FinalLaunchReady")}
            for row in noncritical
        ],
        "protocol_excluded_tasks": [
            {"TaskID": row["TaskID"], "Blocker": row.get("Blocker"), "FinalLaunchReady": row.get("FinalLaunchReady"), "Action": row.get("Action")}
            for row in protocol_excluded
        ],
        "controller_dry_run_status": l1.get("controller_dry_run_status"),
        "controller_dry_run_failed_count": l1.get("controller_dry_run_failed_count"),
        "controller_dry_run_blocked_count": l1.get("controller_dry_run_blocked_count"),
        "prestart_environment_gate_status": env_gate.get("status", "MISSING"),
        "resume_safety_audit_status": resume_audit.get("status", "MISSING"),
        "tmux_start_allowed": False,
        "prelaunch_ready": all(checks.values()),
    }
    write_json(OUT, payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
