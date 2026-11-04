#!/usr/bin/env python3
"""Small non-scientific child process for V22 controller resume canaries."""

from __future__ import annotations

import argparse
import json
import os
import time
from datetime import datetime, timezone
from pathlib import Path


def atomic_write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as handle:
        handle.write(json.dumps(payload, indent=2) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
    tmp.replace(path)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--run-root", required=True)
    parser.add_argument("--sleep-seconds", type=float, default=0.0)
    parser.add_argument("--heartbeat-interval", type=float, default=0.2)
    parser.add_argument("--resume_ckpt", default="")
    parser.add_argument("--self_train_resume_checkpoint", default="")
    parser.add_argument("--resume_checkpoint", default="")
    args = parser.parse_args()

    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    checkpoint = run_root / "checkpoints" / "last.ckpt"
    resumed_from = args.resume_ckpt or args.self_train_resume_checkpoint or args.resume_checkpoint
    if resumed_from and not Path(resumed_from).exists():
        raise SystemExit(f"SCRATCH_RESUME_CHECKPOINT_MISSING:{resumed_from}")

    start = time.time()
    ticks = 0
    while time.time() - start < args.sleep_seconds:
        ticks += 1
        atomic_write_json(run_root / "scratch_child_heartbeat.json", {
            "task_id": args.task_id,
            "pid": os.getpid(),
            "tick": ticks,
            "resumed_from": resumed_from,
            "created_at": datetime.now(timezone.utc).isoformat(),
        })
        time.sleep(args.heartbeat_interval)

    atomic_write_json(checkpoint, {
        "format": "V22_RUNTIME_SCRATCH_CANARY_CHECKPOINT",
        "task_id": args.task_id,
        "pid": os.getpid(),
        "ticks": ticks,
        "resumed_from": resumed_from,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    atomic_write_json(run_root / "SCRATCH_CHILD_RECEIPT.json", {
        "status": "PASS",
        "task_id": args.task_id,
        "pid": os.getpid(),
        "checkpoint": str(checkpoint),
        "resumed": bool(resumed_from),
        "resumed_from": resumed_from,
        "created_at": datetime.now(timezone.utc).isoformat(),
    })
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
