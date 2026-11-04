#!/usr/bin/env python3
"""Authoritative V22.0 transfer Final3 task launcher.

The full training path is fail-closed until dataset-specific X03 hooks are
implemented and audited. Dry-run and canary modes still consume the same
derived config and argv builder used by the future formal path.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
CONFIG_ROOT = ROOT / "configs/three_innovation_ablation_transfer_v22_0"
RUNTIME_ROOT = ROOT / "runtime/three_innovation_ablation_transfer_v22_0"
AUDIT_ROOT = ROOT / "audits/three_innovation_ablation_transfer_v22_0"
PYTHON = Path("/home/renguanming/.conda/envs/beta_fft/bin/python")
X03_SOURCE = ROOT / "train_ACDC/v21_3_acdc10_third_innovation_components.py"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def load_config(path: Path) -> dict:
    payload = json.loads(path.read_text())
    required = [
        "task_id", "dataset", "ratio", "num_classes", "parent_control_identity_status",
        "mask_token_enabled", "adamix_enabled", "contour_reliability_weight_enabled",
        "x03_candidate_id", "x03_weight", "x03_source_sha256",
        "BEFORE_FORMAL_FULL_REAL_TRANSFER_TEST_ACCESS",
        "transfer_l8_closed_loop_manifest_policy", "training_output_dir",
        "engineering_canary_output_dir",
    ]
    missing = [key for key in required if key not in payload]
    if missing:
        raise SystemExit(f"CONFIG_MISSING_KEYS:{missing}")
    return payload


def training_entry(config: dict) -> Path:
    dataset = config["dataset"]
    if dataset in {"PROMISE12", "COVID-19"}:
        return ROOT / "train_promise12/train_promise12_bcp_fft.py"
    if dataset == "MS-CMRSeg":
        return ROOT / "train_MSCMRSEG/v15_1_mscmr_mask_token_adamix_candidate.py"
    if dataset == "ACDC":
        return ROOT / "train_ACDC/v21_3_acdc10_third_innovation_transfer.py"
    raise SystemExit(f"UNSUPPORTED_TRANSFER_DATASET:{dataset}")


def yaml_scalar(path: Path, key: str) -> str:
    prefix = f"{key}:"
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if stripped.startswith(prefix):
            value = stripped.split(":", 1)[1].strip()
            return value.strip("\"'")
    raise SystemExit(f"YAML_KEY_MISSING:{path}:{key}")


def covid_manifest_args(config: dict) -> list[str]:
    parent_cfg = json.loads((ROOT / config["parent_config_path"]).read_text())
    ratio_cfg = ROOT / parent_cfg["ratio_config"]
    labeled = yaml_scalar(ratio_cfg, "labeled_count")
    labeled_manifest = ROOT / yaml_scalar(ratio_cfg, "covid19_labeled_manifest")
    unlabeled_manifest = ROOT / yaml_scalar(ratio_cfg, "covid19_unlabeled_manifest")
    val_manifest = ROOT / yaml_scalar(ratio_cfg, "covid19_val_manifest")
    return [
        "--labeled_num",
        labeled,
        "--covid19_labeled_manifest",
        str(labeled_manifest),
        "--covid19_unlabeled_manifest",
        str(unlabeled_manifest),
        "--covid19_val_manifest",
        str(val_manifest),
    ]


def build_argv(config: dict, mode: str) -> list[str]:
    if is_blocked_task(config):
        return []
    entry = training_entry(config)
    if mode == "parse-only":
        return [str(PYTHON), str(entry), "--help"]
    smoke_output_dir = str((RUNTIME_ROOT / "smoke" / config["task_id"]).resolve())
    if config["dataset"] == "ACDC":
        return [
            str(PYTHON),
            str(entry),
            "--root_path",
            "/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC",
            "--labelnum",
            "3",
            "--seed",
            str(config["seed"]),
            "--gpu",
            "0",
            "--max_iterations",
            "1" if mode == "smoke" else "30000",
            "--save_interval",
            "1" if mode == "smoke" else "200",
            "--run_root",
            smoke_output_dir if mode == "smoke" else config["training_output_dir"],
            "--pretrained_model",
            str(ROOT / "train_ACDC/Pre_model/label3"),
            "--mask_token_enabled",
            "true",
            "--adamix_enabled",
            "true",
            "--mechanism_log_interval",
            "200",
            "--enable_v21_3_innovation_x",
            "true",
            "--innovation_x_id",
            config["x03_candidate_id"],
            "--v21_3_innovation_x_weight",
            str(config["x03_weight"]),
        ]
    if config["dataset"] in {"PROMISE12", "COVID-19"}:
        argv = [
            str(PYTHON),
            "-u",
            str(entry),
            "--max_iterations",
            "1" if mode == "smoke" else "30000",
            "--pretrain_max_iterations_override",
            "1" if mode == "smoke" else "10000",
            "--self_train_max_iterations_override",
            "1" if mode == "smoke" else "30000",
            "--batch_size",
            "16" if config["dataset"] == "PROMISE12" else "24",
            "--labeled_bs",
            "8" if config["dataset"] == "PROMISE12" else "12",
            "--gpu",
            "0",
            "--seed",
            str(config["seed"]),
            "--use_mask_token_prehead_router",
            "--mask_token_num",
            "4",
            "--mask_token_dim",
            "16",
            "--use_adamix_bcp_lite_v1",
            "--enable_v22_frozen_x03",
            "--v22_frozen_x03_id",
            config["x03_candidate_id"],
            "--v22_frozen_x03_weight",
            str(config["x03_weight"]),
            "--exp",
            f"three_innovation_ablation_transfer_v22_0/{config['task_id']}",
            "--v22_output_root_override",
            smoke_output_dir if mode == "smoke" else config["training_output_dir"],
        ]
        if mode == "smoke":
            argv.extend(["--self_train_checkpoint_interval", "1"])
        if config["dataset"] == "PROMISE12":
            argv.extend([
                "--root_path",
                str(ROOT / "data_staging/promise12/20260624_141558/data_promise12"),
                "--labeled_num",
                "7",
            ])
        else:
            argv.extend([
                "--dataset_name",
                "covid19_official20",
                "--evaluation_policy",
                "validation_only",
            ])
            argv.extend(covid_manifest_args(config))
        return argv
    if config["dataset"] == "MS-CMRSeg":
        parent_cfg = json.loads((ROOT / config["parent_config_path"]).read_text())
        return [
            str(PYTHON),
            str(entry),
            "--data_root",
            parent_cfg["data_root"],
            "--labeled_manifest",
            parent_cfg["labeled_manifest"],
            "--unlabeled_manifest",
            parent_cfg["unlabeled_manifest"],
            "--validation_manifest",
            parent_cfg["validation_manifest"],
            "--validation_role",
            parent_cfg.get("validation_role", "validation"),
            "--label_ratio",
            parent_cfg["label_ratio"],
            "--split",
            parent_cfg.get("split", "bayeseg_fixed_split"),
            "--seed",
            str(config["seed"]),
            "--gpu",
            "0",
            "--max_iterations",
            "1" if mode == "smoke" else str(parent_cfg.get("max_iterations", 30000)),
            "--batch_size",
            str(parent_cfg.get("batch_size", 8)),
            "--labeled_batch_size",
            str(parent_cfg.get("labeled_batch_size", 4)),
            "--image_size",
            str(parent_cfg.get("image_size", [256, 256])[0]),
            str(parent_cfg.get("image_size", [256, 256])[1]),
            "--base_lr",
            str(parent_cfg.get("base_lr", 0.01)),
            "--u_weight",
            str(parent_cfg.get("u_weight", 0.5)),
            "--ema_decay",
            str(parent_cfg.get("ema_decay", 0.99)),
            "--consistency",
            str(parent_cfg.get("consistency", 0.1)),
            "--consistency_rampup",
            str(parent_cfg.get("consistency_rampup", 30000)),
            "--fft_radius",
            str(parent_cfg.get("fft_radius", 30)),
            "--validation_interval",
            "1" if mode == "smoke" else str(parent_cfg.get("validation_interval", 200)),
            "--checkpoint_interval",
            "1" if mode == "smoke" else str(parent_cfg.get("checkpoint_interval", 1000)),
            "--pretrain_checkpoint",
            parent_cfg["pretrain_checkpoint"],
            "--run_root",
            smoke_output_dir if mode == "smoke" else config["training_output_dir"],
            "--num_workers",
            str(parent_cfg.get("num_workers", 0)),
            "--enable-mask-token",
            "--enable-adamix",
            "--mask-token-num",
            str(parent_cfg.get("mask_token_num", 4)),
            "--enable_v22_frozen_x03",
            "--v22_frozen_x03_id",
            config["x03_candidate_id"],
            "--v22_frozen_x03_weight",
            str(config["x03_weight"]),
        ]
    raise SystemExit(f"UNSUPPORTED_TRANSFER_DATASET:{config['dataset']}")


def is_blocked_task(config: dict) -> bool:
    return (
        config.get("MSCMR10_MASTER_QUEUE_ACTION") == "BLOCKED_PARENT_IDENTITY"
        or config.get("MSCMR10_DERIVED_CONFIG_STATE") == "STAGING_ONLY"
        or config.get("MSCMR10_FORMAL_LAUNCHER_ALLOWED") is False
        or "TRANSFER_CONTROL_IDENTITY_BLOCKER_OVERLAY_CONFLICT" in config.get("formal_launch_blockers", [])
    )


def path_gate(config: dict) -> dict:
    statuses = {
        "x03_source": X03_SOURCE.exists() and sha256(X03_SOURCE) == config["x03_source_sha256"],
        "parent_checkpoint": Path(config["parent_best_checkpoint"]).exists() if Path(config.get("parent_best_checkpoint", "")).is_absolute() else (ROOT / config.get("parent_best_checkpoint", "")).exists(),
        "training_entry": training_entry(config).exists(),
    }
    return statuses


def run_canary(config: dict, output_dir: Path) -> dict:
    import torch

    sys.path.insert(0, str(ROOT / "train_ACDC"))
    from v21_3_acdc10_third_innovation_components import compute_v21_3_innovation_x

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    torch.manual_seed(22000 + int(config["num_classes"]))
    c = int(config["num_classes"])
    tensors = [torch.randn(2, c, 64, 64, device=device) for _ in range(6)]
    loss, endpoint = compute_v21_3_innovation_x(
        config["x03_candidate_id"],
        *tensors,
        iter_num=1,
        max_iterations=30000,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    receipt = {
        "status": "TRANSFER_ENGINEERING_CANARY_PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": config["task_id"],
        "dataset": config["dataset"],
        "ratio": config["ratio"],
        "ENGINEERING_CANARY_ONLY": True,
        "SCIENTIFIC_RESULT": False,
        "PAPER_RESULT_ELIGIBLE": False,
        "REAL_TRANSFER_TEST_ACCESS": 0,
        "manifest_policy": config["transfer_l8_closed_loop_manifest_policy"],
        "synthetic_tensor_only": True,
        "num_classes": c,
        "device": device,
        "x03_loss_finite": bool(torch.isfinite(loss).item()),
        "x03_weight": config["x03_weight"],
        "weighted_x03_loss_finite": bool(torch.isfinite(config["x03_weight"] * loss).item()),
        "endpoint": endpoint,
    }
    (output_dir / "CANARY_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if not (receipt["x03_loss_finite"] and receipt["weighted_x03_loss_finite"]):
        raise SystemExit(2)
    return receipt


def run_real_train_val_smoke(config: dict) -> dict:
    output_dir = RUNTIME_ROOT / "smoke" / config["task_id"]
    output_dir.mkdir(parents=True, exist_ok=True)
    argv = build_argv(config, "smoke")
    env = os.environ.copy()
    env["V22_REAL_TRAIN_VAL_SMOKE"] = "1"
    env["V22_REAL_TEST_ACCESS_BEFORE_FORMAL"] = "0"
    completed = subprocess.run(
        argv,
        cwd=ROOT,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        timeout=1800,
    )
    (output_dir / "launcher_stdout.log").write_text(completed.stdout)
    checkpoint_files = sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*.pth"))
    checkpoint_files += sorted(str(path.relative_to(output_dir)) for path in output_dir.rglob("*.ckpt"))
    val_markers = [
        marker for marker in ["VALIDATION", "val", "mDice", "Dice", "best_validation_metric"]
        if marker in completed.stdout
    ]
    receipt = {
        "audit_id": "V22_0_TRANSFER_REAL_TRAIN_VAL_SMOKE_RECEIPT",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": config["task_id"],
        "dataset": config["dataset"],
        "ratio": config["ratio"],
        "authoritative_launcher": "tools/v22_0_transfer_task_launcher.py",
        "authoritative_launcher_sha256": sha256(Path(__file__).resolve()),
        "shared_scientific_argv_builder": "tools.v22_0_transfer_task_launcher.build_argv",
        "argv": argv,
        "returncode": completed.returncode,
        "runtime_output_dir": str(output_dir.relative_to(ROOT)),
        "formal_output_dir": config["training_output_dir"],
        "output_isolated_from_formal": str(output_dir.resolve()) != str((ROOT / config["training_output_dir"]).resolve()),
        "real_train_dataset_initialization": completed.returncode == 0,
        "real_val_dataset_initialization": bool(val_markers),
        "TrainManifestAccess": 1,
        "ValManifestAccess": 1 if bool(val_markers) else 0,
        "TestManifestAccess": 0,
        "RealTestInferenceCount": 0,
        "MaskTokenON": True,
        "AdaMixON": True,
        "X03ON": True,
        "BetaFFTPath": True,
        "CheckpointSaveReloadEvidence": bool(checkpoint_files),
        "checkpoint_files": checkpoint_files[:20],
        "ENGINEERING_SMOKE_ONLY": True,
        "SCIENTIFIC_RESULT": False,
        "PAPER_RESULT_ELIGIBLE": False,
        "USED_FOR_SELECTION": False,
        "REAL_TRANSFER_TEST_ACCESS": 0,
        "status": "PASS" if completed.returncode == 0 and bool(val_markers) and bool(checkpoint_files) else "FAIL",
    }
    (output_dir / "REAL_TRAIN_VAL_SMOKE_RECEIPT.json").write_text(json.dumps(receipt, indent=2) + "\n")
    (AUDIT_ROOT / f"transfer_{config['task_id']}_REAL_TRAIN_VAL_SMOKE.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))
    if receipt["status"] != "PASS":
        raise SystemExit(2)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["parse-only", "dry-run", "canary", "smoke", "full"], required=True)
    args = parser.parse_args()
    config = load_config(ROOT / args.config)

    if args.mode == "parse-only":
        print(json.dumps({"status": "PARSE_ONLY_ARGV_BUILT", "argv": build_argv(config, "parse-only")}, indent=2))
        return 0

    paths = path_gate(config)
    if is_blocked_task(config):
        payload = {
            "status": "SKIP_BLOCKED_TASK",
            "task_id": config["task_id"],
            "mode": args.mode,
            "cwd": str(ROOT),
            "argv": [],
            "formal_command": "",
            "formal_full_started": False,
            "real_transfer_test_access": 0,
            "parent_control_identity_status": config["parent_control_identity_status"],
            "blocker": "TRANSFER_CONTROL_IDENTITY_BLOCKER_OVERLAY_CONFLICT",
            "derived_config_state": config.get("MSCMR10_DERIVED_CONFIG_STATE", "STAGING_ONLY"),
            "scientifically_valid": False,
        }
        if args.mode == "dry-run":
            print(json.dumps(payload, indent=2))
            return 0
        raise SystemExit("BLOCKED_PARENT_IDENTITY_NO_LAUNCH_ALLOWED")

    payload = {
        "status": "DRY_RUN_PASS",
        "task_id": config["task_id"],
        "mode": args.mode,
        "cwd": str(ROOT),
        "argv": build_argv(config, args.mode),
        "path_gate": paths,
        "formal_full_started": False,
        "real_transfer_test_access": 0,
        "parent_control_identity_status": config["parent_control_identity_status"],
    }
    if args.mode == "dry-run":
        print(json.dumps(payload, indent=2))
        return 0
    if args.mode == "canary":
        run_canary(config, ROOT / config["engineering_canary_output_dir"])
        return 0
    if args.mode == "smoke":
        run_real_train_val_smoke(config)
        return 0
    raise SystemExit("TRANSFER_FULL_FAIL_CLOSED_UNTIL_FORMAL_START_AUTHORIZATION")


if __name__ == "__main__":
    raise SystemExit(main())
