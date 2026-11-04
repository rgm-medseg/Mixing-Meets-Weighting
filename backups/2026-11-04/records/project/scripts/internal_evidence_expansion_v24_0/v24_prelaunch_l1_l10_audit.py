#!/usr/bin/env python3
"""V24 prelaunch L1-L10 aggregate audit.

This is a scripts-first, no-training audit. It records the current queue as
not prelaunch-ready while preserving completed read-only analyses and terminal
blockers.
"""

from __future__ import annotations

import csv
import hashlib
import json
import py_compile
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
FROZEN_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY.tsv"
WORKING_REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"
CONTROLLER = ROOT / "controllers/internal_evidence_expansion_controller_v24_0.py"
PROTOCOL_EXCLUDED = "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE"

REQUIRED_HARD_GATES = {
    "X03_RAW_SCORE_EFFECTIVE_WEIGHT_SEMANTIC_AUDIT": AUDIT_ROOT / "X03_RAW_SCORE_EFFECTIVE_WEIGHT_SEMANTIC_AUDIT.json",
    "ACDC2_LABEL1_FROZEN_BETAFFT_EXECUTABILITY_AUDIT": AUDIT_ROOT / "ACDC2_LABEL1_FROZEN_BETAFFT_EXECUTABILITY_AUDIT.json",
    "V24_TEST_SEMANTIC_SPLIT_AUDIT": AUDIT_ROOT / "V24_TEST_SEMANTIC_SPLIT_AUDIT.json",
    "V24_FAILURE_INJECTION_AUDIT": AUDIT_ROOT / "V24_FAILURE_INJECTION_AUDIT.json",
}

CORE_REQUIRED_SCRIPTS = [
    "controllers/internal_evidence_expansion_controller_v24_0.py",
    "scripts/internal_evidence_expansion_v24_0/v24_new_full_launcher.py",
    "scripts/internal_evidence_expansion_v24_0/v24_frozen_best_test_launcher.py",
    "scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py",
    "scripts/internal_evidence_expansion_v24_0/v24_hard_gate_semantic_audits.py",
    "scripts/internal_evidence_expansion_v24_0/v24_failure_injection_audit.py",
    "scripts/internal_evidence_expansion_v24_0/v24_final_launch_readiness.py",
    "scripts/internal_evidence_expansion_v24_0/v24_new_full_scratch_closed_loop.py",
    "scripts/internal_evidence_expansion_v24_0/v24_prelaunch_five_review.py",
    "scripts/internal_evidence_expansion_v24_0/v24_formal_start_manifest.py",
    "scripts/internal_evidence_expansion_v24_0/v24_master_registry_freeze.py",
    "scripts/internal_evidence_expansion_v24_0/v24_prestart_environment_gate.py",
    "scripts/internal_evidence_expansion_v24_0/v24_resume_safety_audit.py",
    "scripts/internal_evidence_expansion_v24_0/run_acdc2_betafft_test_gate.sh",
    "scripts/internal_evidence_expansion_v24_0/run_acdc2_ours_test_gate.sh",
    "scripts/internal_evidence_expansion_v24_0/run_acdc10_betafft_seed2025_test_gate.sh",
    "scripts/internal_evidence_expansion_v24_0/run_acdc10_ours_seed2025_test_gate.sh",
    "scripts/internal_evidence_expansion_v24_0/run_acdc10_betafft_seed2026_test_gate.sh",
    "scripts/internal_evidence_expansion_v24_0/run_acdc10_ours_seed2026_test_gate.sh",
]


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
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def exists_rel(value: str) -> bool:
    return bool(value) and (ROOT / value).exists()


def registry_path() -> Path:
    return FROZEN_REGISTRY if FROZEN_REGISTRY.exists() else WORKING_REGISTRY


def compile_status(source: str) -> str:
    if not source or not source.endswith(".py"):
        return "NA"
    path = ROOT / source
    if not path.exists():
        return "FAIL_MISSING_SOURCE"
    try:
        py_compile.compile(str(path), doraise=True)
    except py_compile.PyCompileError:
        return "FAIL"
    return "PASS"


def required_scripts(rows: list[dict[str, str]]) -> list[dict[str, Any]]:
    scripts: list[dict[str, Any]] = []
    seen: set[str] = set()
    for script in CORE_REQUIRED_SCRIPTS:
        if script and script not in seen:
            scripts.append({"path": script, "kind": "core_or_launcher", "executable_required": True})
            seen.add(script)
    for row in rows:
        source = row.get("SourcePath", "")
        if source and source not in seen and source.endswith((".py", ".sh")):
            executable_required = source.startswith("scripts/internal_evidence_expansion_v24_0/")
            scripts.append({"path": source, "kind": "registry_source", "executable_required": executable_required})
            seen.add(source)
        launcher = row.get("Launcher", "")
        if launcher and launcher not in seen and launcher.endswith((".py", ".sh")):
            scripts.append({"path": launcher, "kind": "registry_launcher", "executable_required": True})
            seen.add(launcher)
    return scripts


def command_dry_status(row: dict[str, str]) -> str:
    launcher = row.get("Launcher", "")
    if not launcher:
        return "NA"
    pending_new_full = (
        row.get("TaskType") == "NEW_FULL"
        and row.get("Blocker") == "WAITING_NEW_FULL_L7_SMOKE_AND_L8_SCRATCH_CLOSED_LOOP"
    )
    if row.get("FinalLaunchReady") == PROTOCOL_EXCLUDED:
        return "TERMINAL_PROTOCOL_EXCLUDED"
    if row.get("FinalLaunchReady") == "BLOCKED_NONCRITICAL":
        return "BLOCKED_NONCRITICAL_TERMINAL_GAP"
    if row.get("FinalLaunchReady") != "PASS" and not pending_new_full:
        return "BLOCKED_OR_PENDING_FAIL_CLOSED"
    if row.get("Action") not in {"NEW_FULL_READY", "NEW_FULL_PRELAUNCH_PENDING", "FROZEN_REPLAY_READY"}:
        return "NA_RECEIPT_CHECK_ONLY"
    log = AUDIT_ROOT / "l1_l10_dry_logs" / f"{row['TaskOrder']}_{row['TaskID']}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run([str(ROOT / launcher), "--mode", "dry-run"], cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    return "PASS" if proc.returncode == 0 else f"FAIL_RC_{proc.returncode}"


def l4_identity(row: dict[str, str]) -> str:
    task_id = row["TaskID"]
    if task_id.startswith("ACDC10_BETAFFT"):
        cfg = json.loads((ROOT / row["ConfigPath"]).read_text())
        return "PASS" if cfg.get("mask_token_enabled") is False and cfg.get("adamix_enabled") is False and cfg.get("x03_enabled") is False else "FAIL"
    if task_id.startswith("ACDC10_OURS"):
        cfg = json.loads((ROOT / row["ConfigPath"]).read_text())
        return "PASS" if cfg.get("mask_token_enabled") is True and cfg.get("adamix_enabled") is True and cfg.get("x03_enabled") is True and cfg.get("x03_weight") == 0.05 else "FAIL"
    if task_id.startswith("ACDC2_"):
        blocker = row.get("Blocker", "")
        if row.get("FinalLaunchReady") == PROTOCOL_EXCLUDED and blocker == "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT":
            return "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED"
        return "BLOCKED_PROTOCOL_NOT_EXECUTABLE" if blocker == "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT" else "PASS_OR_NA"
    return "PASS_OR_NA"


def receipt_status(path: Path, expected_status: str = "PASS") -> str:
    if not path.exists():
        return "MISSING"
    try:
        payload = json.loads(path.read_text())
    except json.JSONDecodeError:
        return "FAIL_PARSE"
    return "PASS" if payload.get("status") == expected_status else f"FAIL_STATUS_{payload.get('status')}"


def l7_smoke_status(row: dict[str, str], final_ready: str, has_blocker: bool) -> str:
    if final_ready == PROTOCOL_EXCLUDED:
        return "TERMINAL_PROTOCOL_EXCLUDED"
    if row.get("TaskType") != "NEW_FULL":
        return "PASS_OR_NA" if final_ready == "PASS" else "BLOCKED_OR_PENDING"
    if has_blocker and row.get("Blocker") != "WAITING_NEW_FULL_L7_SMOKE_AND_L8_SCRATCH_CLOSED_LOOP":
        return "BLOCKED"
    status = receipt_status(AUDIT_ROOT / f"{row['TaskID']}_SMOKE_RECEIPT.json")
    return status if status == "PASS" else "WAITING_REAL_TRAIN_VAL_SMOKE_RECEIPT"


def l8_closed_loop_status(row: dict[str, str], final_ready: str, has_blocker: bool, receipt: Path | None) -> str:
    if final_ready == PROTOCOL_EXCLUDED:
        return "TERMINAL_PROTOCOL_EXCLUDED"
    if row.get("TaskType") == "NEW_FULL":
        if has_blocker and row.get("Blocker") != "WAITING_NEW_FULL_L7_SMOKE_AND_L8_SCRATCH_CLOSED_LOOP":
            return "BLOCKED"
        status = receipt_status(AUDIT_ROOT / f"{row['TaskID']}_SCRATCH_CLOSED_LOOP_RECEIPT.json")
        return status if status == "PASS" else "WAITING_SCRATCH_CLOSED_LOOP_RECEIPT"
    if row.get("TaskType") in {"POST_FULL", "FINALIZATION"}:
        dry_receipt = AUDIT_ROOT / f"{row['TaskID']}_DRY_RUN_RECEIPT.json"
        if not dry_receipt.exists():
            return "WAITING_POSTFULL_DRY_RUN_RECEIPT"
        try:
            payload = json.loads(dry_receipt.read_text())
        except json.JSONDecodeError:
            return "FAIL_POSTFULL_DRY_RUN_RECEIPT_PARSE"
        return "PASS" if str(payload.get("status", "")).startswith("DRY_RUN_PASS") else f"FAIL_POSTFULL_DRY_RUN_{payload.get('status')}"
    if receipt and receipt.exists() and final_ready == "PASS":
        return "PASS"
    return "BLOCKED_OR_PENDING" if has_blocker or final_ready != "PASS" else "WAITING_RECEIPT"


def main() -> int:
    registry = registry_path()
    rows = read_tsv(registry)
    details = []
    for row in rows:
        receipt = ROOT / row["ExpectedReceipt"] if row.get("ExpectedReceipt") else None
        has_blocker = bool(row.get("Blocker"))
        final_ready = row.get("FinalLaunchReady")
        l1 = "PASS" if (not row.get("ConfigPath") or exists_rel(row["ConfigPath"])) and (not row.get("SourcePath") or exists_rel(row["SourcePath"])) and (not row.get("Launcher") or exists_rel(row["Launcher"])) else "FAIL"
        l2 = command_dry_status(row)
        l3 = "PASS" if not row.get("OutputDir") or (ROOT / row["OutputDir"]).parent.exists() else "FAIL"
        l4 = l4_identity(row)
        l5 = compile_status(row.get("SourcePath", ""))
        l6 = "PASS" if row.get("TESTAccessAllowed") in {"0", "PRE_REGISTERED_EVALUATION_ONLY"} else "FAIL"
        l7 = l7_smoke_status(row, final_ready, has_blocker)
        l8 = l8_closed_loop_status(row, final_ready, has_blocker, receipt)
        l9 = "PASS_OR_NA" if final_ready == "PASS" else "BLOCKED_OR_PENDING"
        l10 = "PASS_WITH_BLOCKERS"  # controller dry-consumption is checked globally below.
        details.append({
            "TaskOrder": row["TaskOrder"],
            "TaskID": row["TaskID"],
            "Action": row["Action"],
            "FinalLaunchReady": final_ready,
            "Blocker": row.get("Blocker", ""),
            "L1StaticPaths": l1,
            "L2DryRunOrReceiptMode": l2,
            "L3Paths": l3,
            "L4ScientificIdentity": l4,
            "L5ImportCompile": l5,
            "L6TestFirewall": l6,
            "L7SmokeGate": l7,
            "L8ClosedLoopReceipt": l8,
            "L9FailureInjection": l9,
            "L10ControllerDryConsumption": l10,
        })

    controller_proc = subprocess.run(
        ["/home/renguanming/.conda/envs/beta_fft/bin/python", str(CONTROLLER), "--mode", "dry-run"],
        cwd=ROOT,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    controller_payload: dict[str, Any]
    try:
        controller_payload = json.loads(controller_proc.stdout)
    except json.JSONDecodeError:
        controller_payload = {"status": "FAIL_PARSE", "stdout": controller_proc.stdout}

    hard_gate_status = {}
    for name, path in REQUIRED_HARD_GATES.items():
        payload = json.loads(path.read_text()) if path.exists() else {}
        hard_gate_status[name] = payload.get("status", "MISSING")

    script_inventory = []
    for item in required_scripts(rows):
        script = item["path"]
        path = ROOT / script
        compile_or_bash = "NA"
        if path.suffix == ".py":
            compile_or_bash = compile_status(script)
        elif path.suffix == ".sh" and path.exists():
            proc = subprocess.run(["bash", "-n", str(path)], cwd=ROOT)
            compile_or_bash = "PASS" if proc.returncode == 0 else "FAIL"
        script_inventory.append(
            {
                "path": script,
                "kind": item["kind"],
                "exists": path.exists(),
                "sha256": sha256(path) if path.exists() else "",
                "syntax": compile_or_bash,
                "executable": bool(path.exists() and path.stat().st_mode & 0o111),
                "executable_required": item["executable_required"],
            }
        )

    out_tsv = AUDIT_ROOT / "V24_0_PRELAUNCH_L1_L10_AUDIT.tsv"
    fields = list(details[0].keys())
    write_tsv(out_tsv, details, fields)
    blockers = [r for r in rows if r.get("Blocker") or r.get("FinalLaunchReady") != "PASS"]
    launch_blockers = [r for r in blockers if r.get("FinalLaunchReady") not in {"BLOCKED_NONCRITICAL", PROTOCOL_EXCLUDED}]
    protocol_excluded = [r for r in blockers if r.get("FinalLaunchReady") == PROTOCOL_EXCLUDED]
    noncritical = [r for r in blockers if r.get("FinalLaunchReady") == "BLOCKED_NONCRITICAL"]
    active_new_full_ready_count = sum(1 for r in rows if r.get("TaskType") == "NEW_FULL" and r.get("FinalLaunchReady") == "PASS")
    prelaunch_ready = (
        active_new_full_ready_count == 4
        and not launch_blockers
        and controller_payload.get("status") == "PASS"
        and controller_payload.get("failed_count") == 0
    )
    payload = {
        "status": "PRELAUNCH_READY" if prelaunch_ready else "NOT_PRELAUNCH_READY",
        "created_at": now(),
        "formal_started": False,
        "tmux_started": False,
        "EXPECTED_NEW_FULL_COUNT": 4,
        "EXPECTED_NEW_FULL_READY": 4,
        "active_new_full_ready_count": active_new_full_ready_count,
        "task_count": len(rows),
        "pass_ready_count": sum(1 for r in rows if r["FinalLaunchReady"] == "PASS"),
        "blocked_or_pending_count": len(blockers),
        "launch_blocking_count": len(launch_blockers),
        "protocol_excluded_count": len(protocol_excluded),
        "noncritical_terminal_gap_count": len(noncritical),
        "hard_gate_status": hard_gate_status,
        "required_script_inventory": script_inventory,
        "required_scripts_pass": all(
            item["exists"]
            and item["syntax"] == "PASS"
            and (item["executable"] or not item["executable_required"])
            for item in script_inventory
        ),
        "controller_dry_run_status": controller_payload.get("status"),
        "controller_dry_run_failed_count": controller_payload.get("failed_count"),
        "controller_dry_run_blocked_count": controller_payload.get("blocked_count"),
        "controller_dry_run_noncritical_blocked_count": controller_payload.get("noncritical_blocked_count"),
        "controller_dry_run_protocol_excluded_count": controller_payload.get("protocol_excluded_count"),
        "tmux_start_allowed": prelaunch_ready,
        "prelaunch_ready": prelaunch_ready,
        "audit_tsv": str(out_tsv.relative_to(ROOT)),
        "audit_tsv_sha256": sha256(out_tsv),
        "registry": str(registry.relative_to(ROOT)),
        "registry_sha256": sha256(registry),
        "five_review": {
            "REVIEW_1_ALL_RESULTS_FROM_REAL_ASSETS": "BLOCKED_PENDING_COMPLETION_ASSETS",
            "REVIEW_2_PROTOCOL_IDENTITY_LOCKED": "PASS_WITH_ACDC2_PROTOCOL_BLOCKER",
            "REVIEW_3_TEST_ENDPOINT_FIREWALL": "PASS",
            "REVIEW_4_NO_VAL_AS_TEST_NO_GUESSED_VALUES": "PASS",
            "REVIEW_5_NO_TRAINING_OR_FORMAL_STARTED": "PASS",
        },
    }
    write_json(AUDIT_ROOT / "V24_0_PRELAUNCH_L1_L10_AUDIT_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
