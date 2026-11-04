#!/usr/bin/env python3
"""Freeze the current V24 working registry into the canonical registry path."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
WORKING = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"
FROZEN = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY.tsv"
SUMMARY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_FREEZE.json"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    if not WORKING.exists():
        raise SystemExit("WORKING_REGISTRY_MISSING")
    FROZEN.write_bytes(WORKING.read_bytes())
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "V24_0_MASTER_TASK_REGISTRY_FREEZE",
        "formal_started": False,
        "tmux_started": False,
        "working_registry": str(WORKING.relative_to(ROOT)),
        "working_registry_sha256": sha256(WORKING),
        "frozen_registry": str(FROZEN.relative_to(ROOT)),
        "frozen_registry_sha256": sha256(FROZEN),
        "note": "This freezes the current scripts-first registry state; it does not authorize formal start.",
    }
    SUMMARY.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
