#!/usr/bin/env python3
"""V24 NEW_FULL resume-safety audit.

This script is read-only. It classifies resume evidence without claiming
bitwise exact continuation unless full checkpoint state is present.
"""

from __future__ import annotations

import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import torch


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"
OUT_JSON = AUDIT_ROOT / "V24_0_RESUME_SAFETY_AUDIT.json"
OUT_TSV = AUDIT_ROOT / "V24_0_RESUME_SAFETY_AUDIT.tsv"


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


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def checkpoint_classification(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {
            "CheckpointExists": False,
            "CheckpointSHA256": "",
            "ResumeClass": "NO_CHECKPOINT_YET",
            "MissingKeys": "ALL",
        }
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    keys = set(ckpt) if isinstance(ckpt, dict) else set()
    required = {"model_1", "model_2", "ema_model", "optimizer1", "optimizer2", "epoch", "global_step", "best_metrics", "random_state"}
    missing = sorted(required - keys)
    random_state = ckpt.get("random_state", {}) if isinstance(ckpt, dict) else {}
    rng_ok = isinstance(random_state, dict) and {"python_random", "numpy_random", "torch_random"}.issubset(random_state)
    if not missing and rng_ok:
        resume_class = "SCIENTIFIC_CONTINUATION_WITH_RNG_STATE"
    elif {"model_1", "model_2", "global_step"}.issubset(keys):
        resume_class = "CHECKPOINT_RESTART_ONLY"
    else:
        resume_class = "NOT_RESUMABLE"
    return {
        "CheckpointExists": True,
        "CheckpointSHA256": sha256(path),
        "ResumeClass": resume_class,
        "MissingKeys": ",".join(missing),
        "RNGStatePresent": rng_ok,
    }


def main() -> int:
    rows = [row for row in read_tsv(REGISTRY) if row.get("TaskType") == "NEW_FULL"]
    out_rows = []
    for row in rows:
        cfg_path = ROOT / row["ConfigPath"]
        cfg = json.loads(cfg_path.read_text()) if cfg_path.exists() else {}
        run_root = Path(cfg.get("run_root", "")) if cfg else ROOT / "MISSING"
        candidates = [run_root / "checkpoints" / "last.ckpt", run_root / "checkpoints" / "best.ckpt", run_root / "checkpoints" / "final.ckpt"]
        existing = [path for path in candidates if path.exists()]
        selected = existing[0] if existing else candidates[0]
        cls = checkpoint_classification(selected)
        out_rows.append({
            "TaskID": row["TaskID"],
            "FinalLaunchReady": row.get("FinalLaunchReady", ""),
            "Blocker": row.get("Blocker", ""),
            "RunRoot": str(run_root),
            "SelectedResumeCheckpoint": str(selected),
            **cls,
            "BitwiseExactResumeClaimAllowed": False,
            "ResumePolicy": "monitor_live_child_or_resume_same_task_only; no silent restart from scratch",
        })
    write_tsv(OUT_TSV, out_rows)
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "V24_0_RESUME_SAFETY_AUDIT",
        "formal_started": False,
        "tmux_started": False,
        "training_started": False,
        "new_full_task_count": len(out_rows),
        "bitwise_exact_claim_count": 0,
        "resume_class_counts": {name: sum(1 for row in out_rows if row["ResumeClass"] == name) for name in sorted({row["ResumeClass"] for row in out_rows})},
        "audit_tsv": str(OUT_TSV.relative_to(ROOT)),
        "audit_tsv_sha256": sha256(OUT_TSV),
        "policy": "If live child exists and identity matches, monitor only. If dead child has completion receipt, mark complete. If dead child has valid checkpoint, resume same task. Without checkpoint, block.",
    }
    OUT_JSON.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
