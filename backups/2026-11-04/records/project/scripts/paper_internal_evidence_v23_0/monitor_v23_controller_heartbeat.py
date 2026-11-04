#!/usr/bin/env python3
"""Sidecar heartbeat monitor for the V23 tmux controller.

This is read-only with respect to scientific assets.  It only records process
liveness, active task identity, and the absence of TEST processes.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RUNTIME_ROOT = ROOT / "runtime/paper_internal_evidence_v23_0"
HEARTBEAT = RUNTIME_ROOT / "monitor_heartbeat.json"
SESSION = "paper_internal_evidence_controller_v23_0"


def pgrep(pattern: str) -> list[str]:
    proc = subprocess.run(["pgrep", "-af", pattern], stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
    return [line for line in proc.stdout.splitlines() if line.strip() and "pgrep -af" not in line]


def tmux_exists() -> bool:
    proc = subprocess.run(["tmux", "has-session", "-t", SESSION], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return proc.returncode == 0


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2) + "\n")
    tmp.replace(path)


def snapshot() -> dict:
    controller = pgrep(r"controllers/paper_internal_evidence_controller_v23_0.py --mode run")
    trainers = pgrep(r"train_ACDC/v21_3_acdc10_third_innovation_transfer.py")
    launchers = pgrep(r"tools/v23_0_acdc10_x03_sensitivity_launcher.py")
    test_like = [
        line
        for line in pgrep(r"v22_0_acdc_test|v21_5_acdc10_test|postfull_test|formal_test|test_")
        if "monitor_v23_controller_heartbeat.py" not in line
    ]
    active_task = ""
    active_weight = ""
    if trainers:
        cmd = trainers[0]
        for task in ["ACDC10_X03_WEIGHT_001", "ACDC10_X03_WEIGHT_003", "ACDC10_X03_WEIGHT_007", "ACDC10_X03_WEIGHT_010"]:
            if task in cmd:
                active_task = task
        marker = "--v21_3_innovation_x_weight "
        if marker in cmd:
            active_weight = cmd.split(marker, 1)[1].split()[0]
    return {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tmux_session": SESSION,
        "tmux_session_exists": tmux_exists(),
        "controller_pid_alive": bool(controller),
        "controller_processes": controller,
        "launcher_processes": launchers,
        "trainer_processes": trainers,
        "active_task_inferred": active_task,
        "active_x03_weight_inferred": active_weight,
        "unexpected_test_process_count": len(test_like),
        "unexpected_test_processes": test_like,
        "formal_started": bool(trainers),
        "test_access": 0,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--interval", type=int, default=60)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    while True:
        payload = snapshot()
        write_json(HEARTBEAT, payload)
        print(json.dumps(payload, indent=2), flush=True)
        if args.once:
            return 0
        time.sleep(args.interval)


if __name__ == "__main__":
    raise SystemExit(main())
