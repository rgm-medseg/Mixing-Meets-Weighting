#!/usr/bin/env python3
"""V24 failure-injection audit.

All probes are engineered to fail before training or TEST evaluation starts.
They exercise the launch/test/finalization gates without mutating scientific
source, config, checkpoint, or result assets.
"""

from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
PY = "/home/renguanming/.conda/envs/beta_fft/bin/python"
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RUNTIME_ROOT = ROOT / "runtime/internal_evidence_expansion_v24_0/failure_injection"
NEW_FULL = ROOT / "scripts/internal_evidence_expansion_v24_0/v24_new_full_launcher.py"
TEST_GATE = ROOT / "scripts/internal_evidence_expansion_v24_0/v24_frozen_best_test_launcher.py"
POSTFULL = ROOT / "scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py"
CONTROLLER = ROOT / "controllers/internal_evidence_expansion_controller_v24_0.py"
START_MANIFEST = ROOT / "scripts/internal_evidence_expansion_v24_0/v24_formal_start_manifest.py"
BASE_CFG = ROOT / "configs/internal_evidence_expansion_v24_0/ACDC10_BETAFFT_SEED2025.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def run_probe(name: str, cmd: list[str]) -> dict[str, Any]:
    log = RUNTIME_ROOT / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    with log.open("w") as handle:
        proc = subprocess.run(cmd, cwd=ROOT, stdout=handle, stderr=subprocess.STDOUT, text=True)
    text = log.read_text(errors="replace")
    unsafe_tokens = ["iteration 1:", "Epoch", '"TEST_COMPLETION_RECEIPT": true', '"test_started": true']
    status_fail_closed = any(
        token in text
        for token in [
            '"status": "FAIL"',
            '"status": "LOCKED"',
            '"status": "REFUSE_TEST_LOCKED"',
            '"status": "REFUSE_RUN_WAITING_UPSTREAM"',
            "invalid choice:",
            "MISSING_JSON:",
            "PRETRAINED_MODEL_MISSING:",
            "OUTPUT_COLLISION:",
            "REFUSE_RUN_NOT_PRELAUNCH_READY",
            "REFUSE_RESUME_NOT_PRELAUNCH_READY",
            "RESUME_REQUIRES_MANUAL_IDENTITY_REVIEW",
            "REFUSE_PRELAUNCH_NOT_PASS",
            "REFUSE_START_MANIFEST_ASSERTION_FAIL",
        ]
    )
    return {
        "Probe": name,
        "Command": cmd,
        "ReturnCode": proc.returncode,
        "FailClosed": proc.returncode != 0 or status_fail_closed,
        "TrainingStartedEvidence": any(token in text for token in ["iteration 1:", "Epoch"]),
        "TestStartedEvidence": '"test_started": true' in text or '"TEST_COMPLETION_RECEIPT": true' in text,
        "UnsafeTokenFound": any(token in text for token in unsafe_tokens),
        "Log": str(log.relative_to(ROOT)),
    }


def run_probe_with_restore(name: str, cmd: list[str], restore_paths: list[Path]) -> dict[str, Any]:
    snapshots = {path: path.read_bytes() if path.exists() else None for path in restore_paths}
    try:
        return run_probe(name, cmd)
    finally:
        for path, data in snapshots.items():
            if data is None:
                if path.exists():
                    path.unlink()
            else:
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_bytes(data)


def make_collision_config() -> Path:
    cfg = json.loads(BASE_CFG.read_text())
    collision = RUNTIME_ROOT / "existing_output_dir"
    collision.mkdir(parents=True, exist_ok=True)
    cfg["task_id"] = "ACDC10_BETAFFT_SEED2025_OUTPUT_COLLISION_PROBE"
    cfg["run_root"] = str(collision)
    out = RUNTIME_ROOT / "output_collision_config.json"
    write_json(out, cfg)
    return out


def main() -> int:
    collision_cfg = make_collision_config()
    probes = [
        run_probe("missing_config_new_full", [PY, str(NEW_FULL), "--config", "configs/internal_evidence_expansion_v24_0/DOES_NOT_EXIST.json", "--mode", "dry-run"]),
        run_probe("acdc2_missing_pretrain_fail_closed", [str(ROOT / "scripts/internal_evidence_expansion_v24_0/run_acdc2_betafft_full.sh"), "--mode", "dry-run"]),
        run_probe("output_collision_before_full", [PY, str(NEW_FULL), "--config", str(collision_cfg.relative_to(ROOT)), "--mode", "full"]),
        run_probe("unknown_test_gate_task", [PY, str(TEST_GATE), "--task-id", "NOT_A_V24_TASK", "--mode", "dry-run"]),
        run_probe("premature_test_locked", [str(ROOT / "scripts/internal_evidence_expansion_v24_0/run_acdc10_betafft_seed2025_test_gate.sh"), "--mode", "run"]),
        run_probe_with_restore(
            "postfull_run_missing_upstream",
            [str(ROOT / "scripts/internal_evidence_expansion_v24_0/run_seed_test_closure.sh"), "--mode", "run"],
            [AUDIT_ROOT / "SEED_TEST_CLOSURE.json"],
        ),
        run_probe("unknown_postfull_task", [PY, str(POSTFULL), "--task-id", "NOT_A_TASK", "--mode", "dry-run"]),
        run_probe("controller_run_without_prelaunch", [PY, str(CONTROLLER), "--mode", "run"]),
        run_probe("controller_resume_without_prelaunch", [PY, str(CONTROLLER), "--mode", "resume"]),
        run_probe("start_manifest_without_prelaunch", [PY, str(START_MANIFEST)]),
    ]
    status = "PASS" if all(p["FailClosed"] and not p["TrainingStartedEvidence"] and not p["UnsafeTokenFound"] for p in probes) else "FAIL"
    payload = {
        "status": status,
        "created_at": now(),
        "task_id": "V24_FAILURE_INJECTION_AUDIT",
        "training_started": False,
        "formal_started": False,
        "test_started": False,
        "probes": probes,
        "review": {
            "missing_config": "PASS" if probes[0]["FailClosed"] else "FAIL",
            "missing_pretrain": "PASS" if probes[1]["FailClosed"] else "FAIL",
            "output_collision": "PASS" if probes[2]["FailClosed"] else "FAIL",
            "premature_test": "PASS" if probes[4]["FailClosed"] else "FAIL",
            "missing_receipt": "PASS" if probes[5]["FailClosed"] else "FAIL",
            "controller_run_without_prelaunch": "PASS" if probes[7]["FailClosed"] else "FAIL",
            "controller_resume_without_prelaunch": "PASS" if probes[8]["FailClosed"] else "FAIL",
            "start_manifest_without_prelaunch": "PASS" if probes[9]["FailClosed"] else "FAIL",
        },
    }
    write_json(AUDIT_ROOT / "V24_FAILURE_INJECTION_AUDIT.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if status == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
