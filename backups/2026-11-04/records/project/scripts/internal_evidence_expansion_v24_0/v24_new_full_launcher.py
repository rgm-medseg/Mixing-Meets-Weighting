#!/usr/bin/env python3
"""Authoritative V24.0 NEW_FULL launcher.

This wrapper does not implement any scientific method. It binds a V24 task
config to the already frozen ACDC training entries, performs fail-closed gates,
and then dispatches dry-run/smoke/full/resume modes.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
PYTHON = Path("/home/renguanming/.conda/envs/beta_fft/bin/python")
BETA_ENTRY = ROOT / "train_ACDC/beta_fft_acdc_repro.py"
OURS_ENTRY = ROOT / "train_ACDC/v21_3_acdc10_third_innovation_transfer.py"
IDENTITY_GATE = ROOT / "audits/internal_evidence_expansion_v24_0/V24_0_FROZEN_METHOD_IDENTITY.json"
ACDC2_GATE = ROOT / "audits/internal_evidence_expansion_v24_0/ACDC2_PROTOCOL_DECISION.json"
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def rel(path: Path) -> str:
    return str(path.relative_to(ROOT))


def bool_word(value: bool) -> str:
    return "true" if bool(value) else "false"


def load_json(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"MISSING_JSON:{path.relative_to(ROOT)}")
    return json.loads(path.read_text())


def require_identity() -> None:
    gate = load_json(IDENTITY_GATE)
    if gate.get("status") != "PASS":
        raise SystemExit("V24_FROZEN_METHOD_IDENTITY_NOT_PASS")
    if gate.get("X03_WEIGHT") != 0.05 or gate.get("X03_WEIGHT_RESELECTION_ALLOWED") is not False:
        raise SystemExit("V24_X03_IDENTITY_NOT_FROZEN")


def load_config(path: Path) -> dict:
    cfg = load_json(path)
    required = [
        "task_id",
        "dataset_root",
        "method",
        "seed",
        "labelnum",
        "run_root",
        "max_iterations",
        "save_interval",
        "pretrained_model",
        "mask_token_enabled",
        "adamix_enabled",
        "x03_enabled",
        "x03_weight",
        "test_access_before_best_freeze",
        "paper_eligible_test_result_before_gate",
        "only_allowed_scientific_difference",
    ]
    missing = [key for key in required if key not in cfg]
    if missing:
        raise SystemExit(f"CONFIG_MISSING_KEYS:{missing}")
    if cfg["test_access_before_best_freeze"] != 0:
        raise SystemExit("PRE_FULL_TEST_ACCESS_NOT_ZERO")
    if cfg["paper_eligible_test_result_before_gate"] is not False:
        raise SystemExit("PAPER_TEST_RESULT_UNLOCKED_TOO_EARLY")
    if cfg["method"] not in {"BETAFFT", "OURS"}:
        raise SystemExit("UNKNOWN_METHOD")
    if cfg["x03_enabled"] and float(cfg["x03_weight"]) != 0.05:
        raise SystemExit("X03_WEIGHT_MUST_BE_0_05")
    if cfg["task_id"].startswith("ACDC2_"):
        acdc2 = load_json(ACDC2_GATE)
        if acdc2.get("status") != "PASS":
            raise SystemExit("ACDC2_PROTOCOL_NOT_PASS")
        if int(cfg["labelnum"]) != int(acdc2["Labelnum"]):
            raise SystemExit("ACDC2_LABELNUM_MISMATCH")
    return cfg


def path_gate(cfg: dict, mode: str) -> None:
    dataset_root = Path(cfg["dataset_root"])
    pretrained = Path(cfg["pretrained_model"])
    run_root = Path(cfg["run_root"])
    if not dataset_root.is_dir():
        raise SystemExit(f"DATASET_ROOT_MISSING:{dataset_root}")
    if not pretrained.exists():
        raise SystemExit(f"PRETRAINED_MODEL_MISSING:{pretrained}")
    for key in ["labeled_list", "unlabeled_list", "val_list", "test_list"]:
        if cfg.get(key) and not Path(cfg[key]).exists():
            raise SystemExit(f"{key.upper()}_MISSING:{cfg[key]}")
    if mode == "full" and run_root.exists():
        raise SystemExit(f"OUTPUT_COLLISION:{run_root}")


def build_argv(cfg: dict, mode: str) -> list[str]:
    smoke = mode == "smoke"
    run_root = Path(cfg["run_root"])
    max_iterations = int(cfg["max_iterations"])
    save_interval = int(cfg["save_interval"])
    mechanism_interval = int(cfg.get("mechanism_log_interval", 200))
    if smoke:
        run_root = ROOT / "runtime/internal_evidence_expansion_v24_0/smoke" / cfg["task_id"]
        max_iterations = int(cfg.get("smoke_iterations", 1))
        save_interval = max_iterations
        mechanism_interval = max_iterations
    pretrained_arg = Path(cfg["pretrained_model"])
    if pretrained_arg.is_file():
        pretrained_arg = pretrained_arg.parent

    if cfg["method"] == "BETAFFT":
        argv = [
            str(PYTHON),
            str(BETA_ENTRY),
            "--root_path",
            cfg["dataset_root"],
            "--labelnum",
            str(cfg["labelnum"]),
            "--seed",
            str(cfg["seed"]),
            "--gpu",
            str(cfg.get("gpu", 0)),
            "--max_iterations",
            str(max_iterations),
            "--save_interval",
            str(save_interval),
            "--run_root",
            str(run_root),
            "--pretrained_model",
            str(pretrained_arg),
            "--disable_intrain_test_eval",
            "true",
            "--split_name",
            cfg.get("split_name", "official"),
        ]
        for key, flag in [
            ("labeled_list", "--labeled_list"),
            ("unlabeled_list", "--unlabeled_list"),
            ("val_list", "--val_list"),
        ]:
            if cfg.get(key):
                argv.extend([flag, cfg[key]])
        return argv

    return [
        str(PYTHON),
        str(OURS_ENTRY),
        "--root_path",
        cfg["dataset_root"],
        "--labelnum",
        str(cfg["labelnum"]),
        "--seed",
        str(cfg["seed"]),
        "--gpu",
        str(cfg.get("gpu", 0)),
        "--max_iterations",
        str(max_iterations),
        "--save_interval",
        str(save_interval),
        "--run_root",
        str(run_root),
        "--pretrained_model",
        str(pretrained_arg),
        "--mask_token_enabled",
        bool_word(cfg["mask_token_enabled"]),
        "--adamix_enabled",
        bool_word(cfg["adamix_enabled"]),
        "--mechanism_log_interval",
        str(mechanism_interval),
        "--innovation_x_id",
        cfg.get("x03_id", "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT"),
        "--enable_v21_3_innovation_x",
        bool_word(cfg["x03_enabled"]),
        "--v21_3_innovation_x_weight",
        str(cfg["x03_weight"]),
    ]


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_post_full_receipts(cfg: dict, argv: list[str], mode: str, return_code: int) -> None:
    if mode not in {"full", "resume"} or return_code != 0:
        return
    run_root = Path(cfg["run_root"])
    best_ckpt = run_root / "checkpoints" / "best.ckpt"
    train_log = run_root / "train.log"
    if not best_ckpt.exists():
        raise SystemExit(f"BEST_CHECKPOINT_MISSING_AFTER_FULL:{best_ckpt}")
    checkpoint_sha = sha256(best_ckpt)
    completion = {
        "status": "PASS",
        "task_id": cfg["task_id"],
        "created_at": now(),
        "mode": mode,
        "FormalFullComplete": True,
        "training_started": True,
        "training_command": argv,
        "run_root": str(run_root),
        "train_log": rel(train_log) if train_log.exists() else "",
        "train_log_exists": train_log.exists(),
        "BestCheckpointPath": rel(best_ckpt),
        "BestCheckpointSHA256": checkpoint_sha,
        "TEST_STARTED_BY_LAUNCHER": False,
        "TEST_COMPLETION_RECEIPT_CREATED": False,
    }
    best_val = {
        "status": "PASS",
        "task_id": cfg["task_id"],
        "created_at": now(),
        "BestValSelected": True,
        "selection_endpoint": "BEST_VAL",
        "BestCheckpointPath": rel(best_ckpt),
        "BestCheckpointSHA256": checkpoint_sha,
        "test_used_for_checkpoint_selection": False,
        "test_access_before_best_freeze": cfg["test_access_before_best_freeze"],
    }
    frozen = {
        "status": "PASS",
        "task_id": cfg["task_id"],
        "created_at": now(),
        "BestCheckpointSHA256Frozen": True,
        "BestCheckpointPath": rel(best_ckpt),
        "BestCheckpointSHA256": checkpoint_sha,
    }
    write_json(run_root / "FORMAL_FULL_COMPLETION_RECEIPT.json", completion)
    write_json(run_root / "BEST_VAL_SELECTION.json", best_val)
    write_json(run_root / "FROZEN_CHECKPOINT_SHA256.json", frozen)
    write_json(AUDIT_ROOT / f"{cfg['task_id']}_FORMAL_FULL_COMPLETION_RECEIPT.json", completion)
    write_json(AUDIT_ROOT / f"{cfg['task_id']}_BEST_VAL_SELECTION.json", best_val)
    write_json(AUDIT_ROOT / f"{cfg['task_id']}_FROZEN_CHECKPOINT_SHA256.json", frozen)


def write_smoke_receipt(cfg: dict, argv: list[str], return_code: int) -> None:
    if return_code != 0:
        return
    smoke_root = ROOT / "runtime/internal_evidence_expansion_v24_0/smoke" / cfg["task_id"]
    payload = {
        "status": "PASS",
        "task_id": cfg["task_id"],
        "created_at": now(),
        "mode": "smoke",
        "training_started": True,
        "formal_started": False,
        "formal_full_started": False,
        "test_started": False,
        "TEST_DATA_READ": False,
        "TEST_USED_FOR_TUNING": False,
        "TEST_USED_FOR_SELECTION": False,
        "disable_intrain_test_eval": cfg["method"] == "BETAFFT",
        "test_access_before_best_freeze": cfg["test_access_before_best_freeze"],
        "dataset": "ACDC",
        "split": "TRAIN_VAL",
        "canonical_initialization": cfg["pretrained_model"],
        "smoke_iterations": cfg.get("smoke_iterations", 1),
        "argv": argv,
        "run_root": rel(smoke_root),
        "scientific_result": False,
        "paper_result_eligible": False,
    }
    write_json(smoke_root / "SMOKE_RECEIPT.json", payload)
    write_json(AUDIT_ROOT / f"{cfg['task_id']}_SMOKE_RECEIPT.json", payload)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dry-run", "smoke", "full", "resume"], required=True)
    args = parser.parse_args()

    require_identity()
    cfg_path = ROOT / args.config
    cfg = load_config(cfg_path)
    path_gate(cfg, args.mode)
    argv = build_argv(cfg, "full" if args.mode == "resume" else args.mode)

    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "task_id": cfg["task_id"],
                    "mode": args.mode,
                    "config": args.config,
                    "method": cfg["method"],
                    "seed": cfg["seed"],
                    "labelnum": cfg["labelnum"],
                    "test_access_before_best_freeze": cfg["test_access_before_best_freeze"],
                    "argv": argv,
                    "cwd": str(ROOT),
                    "training_started": False,
                    "test_started": False,
                },
                indent=2,
            )
        )
        return 0

    proc = subprocess.run(argv, cwd=ROOT)
    if args.mode == "smoke":
        write_smoke_receipt(cfg, argv, proc.returncode)
    write_post_full_receipts(cfg, argv, args.mode, proc.returncode)
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
