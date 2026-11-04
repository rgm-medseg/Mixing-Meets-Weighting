#!/usr/bin/env python3
"""PROMISE12 10% paper4 formal training launcher."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/renguanming/.conda/envs/beta_fft/bin/python")
RUNTIME = ROOT / "runtime/promise10_paper4"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def load_config(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    required = [
        "task_id",
        "labeled_count",
        "slice_indices_json",
        "formal_seed",
        "budget",
        "training_output_dir",
        "train_entry",
        "train_entry_sha256",
        "mask_token_enabled",
        "adamix_enabled",
        "x03_enabled",
        "x03_weight",
        "x03_source_sha256",
        "REAL_TEST_ACCESS_BEFORE_FULL",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(f"CONFIG_MISSING_KEYS:{missing}")
    return payload


def bool_flag(argv: list[str], enabled: bool, *parts: str) -> None:
    if enabled:
        argv.extend(parts)


def build_argv(config: dict[str, Any], mode: str) -> list[str]:
    budget = config["budget"]
    smoke = mode == "smoke"
    output_dir = ROOT / config["engineering_smoke_output_dir"] if smoke else ROOT / config["training_output_dir"]
    argv = [
        str(PYTHON),
        "-u",
        str(ROOT / config["train_entry"]),
        "--root_path",
        str(ROOT / config["root_path"]),
        "--model_1",
        "unet",
        "--model_2",
        "swin_unet",
        "--max_iterations",
        "1" if smoke else str(budget["self_train_iterations"]),
        "--batch_size",
        str(budget["batch_size"]),
        "--labeled_bs",
        str(budget["labeled_batch_size"]),
        "--labeled_num",
        str(config["labeled_count"]),
        "--base_lr",
        str(budget["base_lr"]),
        "--seed",
        str(config["formal_seed"]),
        "--gpu",
        "0",
        "--deterministic",
        "1",
        "--pretrain_max_iterations_override",
        "1" if smoke else str(budget["pretrain_iterations"]),
        "--self_train_max_iterations_override",
        "1" if smoke else str(budget["self_train_iterations"]),
        "--self_train_checkpoint_interval",
        "1" if smoke else str(budget["checkpoint_cadence"]),
        "--evaluation_policy",
        "validation_only",
        "--promise12_slice_indices_json",
        str(ROOT / config["slice_indices_json"]),
        "--exp",
        config["task_id"],
        "--v22_output_root_override",
        str(output_dir),
    ]
    bool_flag(argv, bool(config["mask_token_enabled"]), "--use_mask_token_prehead_router", "--mask_token_num", "4", "--mask_token_dim", "16")
    bool_flag(argv, bool(config["adamix_enabled"]), "--use_adamix_bcp_lite_v1")
    bool_flag(
        argv,
        bool(config["x03_enabled"]),
        "--enable_v22_frozen_x03",
        "--v22_frozen_x03_id",
        config["x03_candidate_id"],
        "--v22_frozen_x03_weight",
        str(config["x03_weight"]),
    )
    return argv


def identity_checks(config: dict[str, Any]) -> dict[str, Any]:
    train_entry = ROOT / config["train_entry"]
    slice_json = ROOT / config["slice_indices_json"]
    x03_source = ROOT / config["x03_source"]
    return {
        "train_entry_exists": train_entry.exists(),
        "train_entry_sha256_matches": train_entry.exists() and sha256(train_entry) == config["train_entry_sha256"],
        "slice_indices_exists": slice_json.exists(),
        "slice_indices_sha256_matches": slice_json.exists() and sha256(slice_json) == config["slice_indices_json_sha256"],
        "x03_source_exists": x03_source.exists(),
        "x03_source_sha256_matches": x03_source.exists() and sha256(x03_source) == config["x03_source_sha256"],
        "REAL_TEST_ACCESS_BEFORE_FULL": config["REAL_TEST_ACCESS_BEFORE_FULL"],
    }


def find_best_checkpoint(run_root: Path, *, allow_final_fallback: bool = False) -> dict[str, Any]:
    self_train = run_root / "self_train"
    candidates = sorted(self_train.glob("model1_iter_*_dice_*.pth"))
    if not candidates:
        best = self_train / "unet_best_model.pth"
        if not best.exists() and allow_final_fallback:
            best = self_train / "unet_final_model.pth"
        if not best.exists():
            raise SystemExit(f"BEST_CHECKPOINT_MISSING:{self_train}")
        return {"best_checkpoint": best, "best_iter": None, "best_val_dice": None}
    best_item = None
    for path in candidates:
        stem = path.stem
        parts = stem.split("_")
        try:
            iter_idx = parts.index("iter")
        except ValueError:
            iter_num = int(parts[2])
        else:
            iter_num = int(parts[iter_idx + 1])
        dice = float(parts[-1])
        key = (dice, -iter_num)
        if best_item is None or key > best_item[0]:
            best_item = (key, path, iter_num, dice)
    assert best_item is not None
    return {"best_checkpoint": best_item[1], "best_iter": best_item[2], "best_val_dice": best_item[3]}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dry-run", "smoke", "run"], default="dry-run")
    args = parser.parse_args()

    config_path = ROOT / args.config
    config = load_config(config_path)
    checks = identity_checks(config)
    checks_pass = all(v is True for k, v in checks.items() if k != "REAL_TEST_ACCESS_BEFORE_FULL") and checks["REAL_TEST_ACCESS_BEFORE_FULL"] == 0
    argv = build_argv(config, "smoke" if args.mode == "smoke" else "run")
    if args.mode == "dry-run":
        payload = {
            "status": "DRY_RUN_PASS" if checks_pass else "DRY_RUN_FAIL",
            "task_id": config["task_id"],
            "argv": argv,
            "checks": checks,
            "formal_started": False,
            "test_access": 0,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if checks_pass else 2
    if not checks_pass:
        raise SystemExit(json.dumps({"status": "IDENTITY_CHECK_FAIL", "checks": checks}, indent=2, sort_keys=True))

    env = os.environ.copy()
    env["REAL_TEST_ACCESS_BEFORE_FULL"] = "0"
    proc = subprocess.run(argv, cwd=ROOT, env=env, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    out_root = ROOT / (config["engineering_smoke_output_dir"] if args.mode == "smoke" else config["training_output_dir"])
    out_root.mkdir(parents=True, exist_ok=True)
    (out_root / "launcher_stdout.log").write_text(proc.stdout, encoding="utf-8")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)
    best = find_best_checkpoint(out_root, allow_final_fallback=args.mode == "smoke")
    best_path = best["best_checkpoint"]
    receipt = {
        "schema": "PROMISE10_PAPER4_TRAINING_RECEIPT_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": config["task_id"],
        "mode": args.mode,
        "returncode": proc.returncode,
        "best_checkpoint": str(best_path.relative_to(ROOT)),
        "best_checkpoint_sha256": sha256(best_path),
        "best_iter": best["best_iter"],
        "best_val_dice": best["best_val_dice"],
        "FormalFullComplete": args.mode == "run",
        "BestValCheckpointSelected": True,
        "BestCheckpointSHA256Frozen": args.mode == "run",
        "TestUnlocked": False,
        "training_started": True,
        "REAL_TEST_ACCESS_BEFORE_FULL": 0,
        "stdout_log": str((out_root / "launcher_stdout.log").relative_to(ROOT)),
    }
    write_json(out_root / ("FORMAL_TRAINING_RECEIPT.json" if args.mode == "run" else "SMOKE_RECEIPT.json"), receipt)
    print(json.dumps({"status": "RUN_PASS" if args.mode == "run" else "SMOKE_PASS", "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
