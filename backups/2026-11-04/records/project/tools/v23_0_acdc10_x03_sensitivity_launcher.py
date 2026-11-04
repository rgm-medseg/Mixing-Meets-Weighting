#!/usr/bin/env python3
"""Authoritative V23.0 ACDC10 X03 weight sensitivity launcher.

This launcher never performs TEST.  It only drives validation-selected full
training for the four non-reuse X03 weight points after the added hard gates
have passed.
"""

from __future__ import annotations

import argparse
import json
import subprocess
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/renguanming/.conda/envs/beta_fft/bin/python")
TRAIN_ENTRY = ROOT / "train_ACDC/v21_3_acdc10_third_innovation_transfer.py"
HARD_GATE = ROOT / "audits/paper_internal_evidence_v23_0/ACDC10_ARM_111_CANONICAL_PARENT_IDENTITY_AUDIT.json"


def bool_word(value: bool) -> str:
    return "true" if bool(value) else "false"


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    required = [
        "task_id",
        "dataset_root",
        "labelnum",
        "seed",
        "max_iterations",
        "save_interval",
        "mechanism_log_interval",
        "run_root",
        "pretrained_model",
        "mask_token_enabled",
        "adamix_enabled",
        "innovation_x_enabled",
        "innovation_x_id",
        "innovation_x_weight",
        "TEST_ACCESS_FOR_SENSITIVITY",
        "PAPER_ELIGIBLE_TEST_RESULT",
        "only_allowed_scientific_difference",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(f"CONFIG_MISSING_KEYS:{missing}")
    if payload["TEST_ACCESS_FOR_SENSITIVITY"] != 0:
        raise SystemExit("SENSITIVITY_TEST_ACCESS_NOT_ZERO")
    if payload["PAPER_ELIGIBLE_TEST_RESULT"] is not False:
        raise SystemExit("SENSITIVITY_TEST_RESULT_MUST_BE_INELIGIBLE")
    if payload["only_allowed_scientific_difference"] != "innovation_x_weight":
        raise SystemExit("SENSITIVITY_ALLOWED_DIFF_NOT_WEIGHT_ONLY")
    if not payload["mask_token_enabled"] or not payload["adamix_enabled"] or not payload["innovation_x_enabled"]:
        raise SystemExit("SENSITIVITY_REQUIRES_FINAL3_SWITCHES_ON")
    return payload


def hard_gate() -> dict:
    if not HARD_GATE.exists():
        raise SystemExit("ACDC10_ARM_111_PARENT_IDENTITY_GATE_MISSING")
    payload = json.loads(HARD_GATE.read_text())
    if payload.get("status") != "PASS" or payload.get("UnexpectedScientificDifferenceCount") != 0:
        raise SystemExit("ACDC10_ARM_111_PARENT_IDENTITY_GATE_NOT_PASS")
    if payload.get("sensitivity_parent_allowed") is not True:
        raise SystemExit("SENSITIVITY_PARENT_NOT_ALLOWED")
    return payload


def build_argv(config: dict, mode: str) -> list[str]:
    if mode == "parse-only":
        return [str(PYTHON), str(TRAIN_ENTRY), "--help"]

    run_root = Path(config["run_root"])
    max_iterations = int(config["max_iterations"])
    save_interval = int(config["save_interval"])
    mechanism_interval = int(config["mechanism_log_interval"])
    if mode == "smoke":
        run_root = ROOT / "runtime/paper_internal_evidence_v23_0/smoke" / config["task_id"]
        max_iterations = int(config.get("smoke_iterations", 1))
        save_interval = max_iterations
        mechanism_interval = max_iterations

    pretrained = Path(config["pretrained_model"])
    if not pretrained.is_absolute():
        pretrained = ROOT / pretrained
    return [
        str(PYTHON),
        str(TRAIN_ENTRY),
        "--root_path",
        config["dataset_root"],
        "--labelnum",
        str(config["labelnum"]),
        "--seed",
        str(config["seed"]),
        "--gpu",
        str(config.get("gpu", 0)),
        "--max_iterations",
        str(max_iterations),
        "--save_interval",
        str(save_interval),
        "--run_root",
        str(run_root),
        "--pretrained_model",
        str(pretrained),
        "--mask_token_enabled",
        bool_word(config["mask_token_enabled"]),
        "--adamix_enabled",
        bool_word(config["adamix_enabled"]),
        "--mechanism_log_interval",
        str(mechanism_interval),
        "--innovation_x_id",
        config["innovation_x_id"],
        "--enable_v21_3_innovation_x",
        bool_word(config["innovation_x_enabled"]),
        "--v21_3_innovation_x_weight",
        str(config["innovation_x_weight"]),
    ]


def path_gate(config: dict, mode: str) -> None:
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


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["parse-only", "dry-run", "smoke", "full"], required=True)
    args = parser.parse_args()

    gate = hard_gate()
    config_path = ROOT / args.config
    config = load_config(config_path)
    path_gate(config, args.mode)
    argv = build_argv(config, "parse-only" if args.mode == "parse-only" else args.mode)
    if args.mode == "dry-run":
        print(
            json.dumps(
                {
                    "status": "DRY_RUN_PASS",
                    "task_id": config["task_id"],
                    "mode": args.mode,
                    "config": args.config,
                    "parent_identity_gate": gate.get("status"),
                    "test_access_for_sensitivity": config["TEST_ACCESS_FOR_SENSITIVITY"],
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
    return proc.returncode


if __name__ == "__main__":
    raise SystemExit(main())
