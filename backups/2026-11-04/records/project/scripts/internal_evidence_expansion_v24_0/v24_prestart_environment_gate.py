#!/usr/bin/env python3
"""V24 prestart GPU/disk/process environment gate.

This is read-only and never starts tmux, training, or TEST.
"""

from __future__ import annotations

import json
import os
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
OUT = AUDIT_ROOT / "V24_0_PRESTART_ENVIRONMENT_GATE.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def run(cmd: list[str]) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=ROOT, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)


def process_scan() -> dict[str, Any]:
    proc = run(["ps", "-eo", "pid,ppid,stat,lstart,cmd"])
    needles = [
        "internal_evidence_expansion_controller_v24_0",
        "v24_new_full_launcher",
        "beta_fft_acdc_repro.py --",
        "v21_3_acdc10_third_innovation_transfer.py",
    ]
    rows = []
    for line in proc.stdout.splitlines():
        if any(token in line for token in needles) and "v24_prestart_environment_gate.py" not in line:
            if "ps -eo" not in line and " rg " not in line:
                rows.append(line)
    tmux = run(["bash", "-lc", "tmux ls 2>/dev/null | rg 'internal_evidence_expansion_controller_v24_0' || true"])
    return {
        "CONFLICTING_CONTROLLER_COUNT": len(rows),
        "conflicting_process_lines": rows,
        "tmux_session_exists": bool(tmux.stdout.strip()),
        "tmux_lines": tmux.stdout.strip().splitlines(),
    }


def gpu_scan() -> dict[str, Any]:
    proc = run(["bash", "-lc", "nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits 2>/dev/null || true"])
    gpus = []
    for line in proc.stdout.splitlines():
        parts = [part.strip() for part in line.split(",")]
        if len(parts) >= 5:
            gpus.append({
                "index": parts[0],
                "name": parts[1],
                "memory_used_mib": int(parts[2]),
                "memory_total_mib": int(parts[3]),
                "utilization_gpu_percent": int(parts[4]),
            })
    return {
        "gpu_visible": bool(gpus),
        "gpus": gpus,
        "GPU_CONFLICTING_TASK_COUNT": sum(1 for gpu in gpus if gpu["memory_used_mib"] > 2048 or gpu["utilization_gpu_percent"] > 10),
    }


def disk_scan() -> dict[str, Any]:
    usage = shutil.disk_usage(ROOT)
    stat = os.statvfs(ROOT)
    return {
        "path": str(ROOT),
        "disk_total_bytes": usage.total,
        "disk_free_bytes": usage.free,
        "disk_free_gib": round(usage.free / (1024**3), 3),
        "disk_free_fraction": usage.free / usage.total if usage.total else 0.0,
        "inode_total": stat.f_files,
        "inode_free": stat.f_ffree,
        "inode_free_fraction": stat.f_ffree / stat.f_files if stat.f_files else 0.0,
        "required_disk_free_gib_minimum": 100.0,
        "required_inode_free_minimum": 100000,
    }


def main() -> int:
    process = process_scan()
    gpu = gpu_scan()
    disk = disk_scan()
    pass_state = (
        process["CONFLICTING_CONTROLLER_COUNT"] == 0
        and not process["tmux_session_exists"]
        and gpu["GPU_CONFLICTING_TASK_COUNT"] == 0
        and disk["disk_free_gib"] >= disk["required_disk_free_gib_minimum"]
        and disk["inode_free"] >= disk["required_inode_free_minimum"]
    )
    payload = {
        "status": "PASS" if pass_state else "WAIT",
        "created_at": now(),
        "task_id": "V24_0_PRESTART_ENVIRONMENT_GATE",
        "formal_started": False,
        "tmux_started": False,
        "training_started": False,
        "process": process,
        "gpu": gpu,
        "disk": disk,
        "LOCAL_PC_OFF_EXECUTION_SAFE": "READY_AFTER_TMUX_DETACH_CANARY",
        "SERVER_REBOOT_AUTO_RESUME": "NOT_CONFIGURED_ENVIRONMENT_BLOCKER",
        "manual_resume_available": True,
    }
    OUT.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
