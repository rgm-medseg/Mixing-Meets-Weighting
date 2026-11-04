#!/usr/bin/env python3
"""Authoritative V22.0 ACDC10 factorial task launcher.

The same config-to-argv path is used for parse-only, smoke, and full modes.
This file deliberately does not contain performance-based branching.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/renguanming/.conda/envs/beta_fft/bin/python")
TRAIN_ENTRY = ROOT / "train_ACDC/v21_3_acdc10_third_innovation_transfer.py"


def bool_word(value: bool) -> str:
    return "true" if bool(value) else "false"


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    required = [
        "task_id", "dataset_root", "labelnum", "seed", "max_iterations",
        "save_interval", "mechanism_log_interval", "run_root",
        "pretrained_model", "mask_token_enabled", "adamix_enabled",
        "innovation_x_enabled", "innovation_x_id", "innovation_x_weight",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(f"CONFIG_MISSING_KEYS:{missing}")
    return payload


def build_argv(config: dict, mode: str) -> list[str]:
    if mode == "parse-only":
        return [str(PYTHON), str(TRAIN_ENTRY), "--help"]

    run_root = Path(config["run_root"])
    max_iterations = int(config["max_iterations"])
    save_interval = int(config["save_interval"])
    mechanism_interval = int(config["mechanism_log_interval"])
    if mode == "smoke":
        run_root = ROOT / "runtime/three_innovation_ablation_transfer_v22_0/smoke" / config["task_id"]
        max_iterations = int(config.get("smoke_iterations", 1))
        save_interval = max_iterations
        mechanism_interval = max_iterations

    return [
        str(PYTHON), str(TRAIN_ENTRY),
        "--root_path", config["dataset_root"],
        "--labelnum", str(config["labelnum"]),
        "--seed", str(config["seed"]),
        "--gpu", str(config.get("gpu", 0)),
        "--max_iterations", str(max_iterations),
        "--save_interval", str(save_interval),
        "--run_root", str(run_root),
        "--pretrained_model", str(ROOT / config["pretrained_model"] if not Path(config["pretrained_model"]).is_absolute() else Path(config["pretrained_model"])),
        "--mask_token_enabled", bool_word(config["mask_token_enabled"]),
        "--adamix_enabled", bool_word(config["adamix_enabled"]),
        "--mechanism_log_interval", str(mechanism_interval),
        "--innovation_x_id", config["innovation_x_id"],
        "--enable_v21_3_innovation_x", bool_word(config["innovation_x_enabled"]),
        "--v21_3_innovation_x_weight", str(config["innovation_x_weight"]),
    ]


def path_gate(config: dict, mode: str):
    dataset_root = Path(config["dataset_root"])
    pretrained = Path(config["pretrained_model"])
    if not pretrained.is_absolute():
        pretrained = ROOT / pretrained
    if not dataset_root.is_dir():
        raise SystemExit(f"DATASET_ROOT_MISSING:{dataset_root}")
    if not pretrained.exists():
        raise SystemExit(f"PRETRAINED_MODEL_MISSING:{pretrained}")
    if mode == "full" and Path(config["run_root"]).exists():
        raise SystemExit(f"OUTPUT_COLLISION:{config['run_root']}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["parse-only", "dry-run", "smoke", "full"], required=True)
    args = parser.parse_args()

    config = load_config(ROOT / args.config)
    path_gate(config, args.mode)
    argv = build_argv(config, "parse-only" if args.mode == "parse-only" else args.mode)
    if args.mode == "dry-run":
        print(json.dumps({
            "status": "DRY_RUN_PASS",
            "task_id": config["task_id"],
            "mode": args.mode,
            "argv": argv,
            "cwd": str(ROOT),
            "training_started": False,
        }, indent=2))
        return

    proc = subprocess.run(argv, cwd=ROOT)
    raise SystemExit(proc.returncode)


if __name__ == "__main__":
    main()
