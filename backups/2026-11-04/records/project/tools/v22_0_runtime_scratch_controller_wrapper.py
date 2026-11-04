#!/usr/bin/env python3
"""Run the V22 controller against an isolated scratch runtime."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import controllers.three_innovation_ablation_transfer_controller_v22_0 as controller


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scratch-root", required=True)
    parser.add_argument("--mode", choices=["run", "resume"], required=True)
    args = parser.parse_args()

    scratch = Path(args.scratch_root).resolve()
    audit_root = scratch / "audits"
    runtime_root = scratch / "runtime"
    script_root = scratch / "scripts"
    audit_root.mkdir(parents=True, exist_ok=True)
    runtime_root.mkdir(parents=True, exist_ok=True)
    script_root.mkdir(parents=True, exist_ok=True)

    controller.AUDIT_ROOT = audit_root
    controller.RUNTIME_ROOT = runtime_root
    controller.SCRIPT_ROOT = script_root
    controller.STATE = runtime_root / "controller_dry_state.json"
    controller.FORMAL_STATE = runtime_root / "controller_formal_state.json"
    controller.FORMAL_LEDGER = runtime_root / "controller_formal_ledger.tsv"
    controller.FORMAL_LOG_ROOT = runtime_root / "controller_formal_logs"
    controller.STOP_AFTER_CURRENT_FLAG = runtime_root / "STOP_AFTER_CURRENT.flag"
    controller.ACTIVE_TASK = runtime_root / "active_task.json"
    controller.HEARTBEAT = runtime_root / "heartbeat.json"
    controller.COMPLETED = runtime_root / "completed.tsv"
    controller.PAUSE_RECEIPT = runtime_root / "FINAL_COLD_AUDIT_PAUSE_RECEIPT.json"
    controller.FINAL_REGISTRY = audit_root / "V22_0_MASTER_TASK_REGISTRY_FINAL.tsv"
    controller.FINAL_REGISTRY_SUMMARY = audit_root / "V22_0_MASTER_TASK_REGISTRY_FINAL_SUMMARY.json"
    controller.GLOBAL_L10_SUMMARY = audit_root / "GLOBAL_MASTER_QUEUE_FINAL_L10_SUMMARY.json"
    controller.PRELAUNCH_FIVE_REVIEW = audit_root / "V22_0_PRELAUNCH_FIVE_REVIEW_SUMMARY.json"
    controller.FORMAL_UNLOCK_RECEIPT = audit_root / "V22_0_FORMAL_UNLOCK_RECEIPT.json"
    controller.FORMAL_START_AUTHORIZATION = audit_root / "V22_0_FORMAL_START_AUTHORIZATION.json"

    payload = controller.run_formal_global_final(resume=args.mode == "resume")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
