#!/usr/bin/env python3
"""PROMISE12 10% paper4 frozen BEST checkpoint TEST-once launcher."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]


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
    return json.loads(path.read_text(encoding="utf-8"))


def receipt_path(config: dict[str, Any]) -> Path:
    return ROOT / config["training_output_dir"] / "FORMAL_TRAINING_RECEIPT.json"


def unlock_state(config: dict[str, Any]) -> dict[str, Any]:
    path = receipt_path(config)
    if not path.exists():
        return {
            "FormalFullComplete": False,
            "BestValCheckpointSelected": False,
            "BestCheckpointSHA256Frozen": False,
            "TestProtocolLockPass": False,
            "RealTestAccessBeforeCheckpointFreeze": 0,
            "TestUnlocked": False,
        }
    receipt = json.loads(path.read_text(encoding="utf-8"))
    ckpt = ROOT / receipt["best_checkpoint"]
    protocol_lock = ckpt.exists() and sha256(ckpt) == receipt["best_checkpoint_sha256"]
    unlocked = (
        bool(receipt.get("FormalFullComplete"))
        and bool(receipt.get("BestValCheckpointSelected"))
        and bool(receipt.get("BestCheckpointSHA256Frozen"))
        and protocol_lock
        and int(receipt.get("REAL_TEST_ACCESS_BEFORE_FULL", 1)) == 0
    )
    return {
        "FormalFullComplete": bool(receipt.get("FormalFullComplete")),
        "BestValCheckpointSelected": bool(receipt.get("BestValCheckpointSelected")),
        "BestCheckpointPath": receipt["best_checkpoint"],
        "BestCheckpointSHA256Frozen": bool(receipt.get("BestCheckpointSHA256Frozen")),
        "BestCheckpointSHA256": receipt["best_checkpoint_sha256"],
        "BestValDice": receipt.get("best_val_dice"),
        "BestIter": receipt.get("best_iter"),
        "TestProtocolLockPass": protocol_lock,
        "RealTestAccessBeforeCheckpointFreeze": int(receipt.get("REAL_TEST_ACCESS_BEFORE_FULL", 1)),
        "TestUnlocked": unlocked,
    }


def binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
    from medpy import metric

    pred = pred.astype(bool)
    target = target.astype(bool)
    inter = int(np.logical_and(pred, target).sum())
    union = int(np.logical_or(pred, target).sum())
    denom = int(pred.sum() + target.sum())
    if pred.any() and target.any():
        hd95 = float(metric.binary.hd95(pred, target))
        asd = float(metric.binary.asd(pred, target))
    elif not pred.any() and not target.any():
        hd95 = asd = 0.0
    else:
        hd95 = asd = float(np.linalg.norm(pred.shape))
    return {
        "dice": 0.0 if denom == 0 else float(2.0 * inter / denom),
        "iou": 0.0 if union == 0 else float(inter / union),
        "hd95": hd95,
        "asd": asd,
    }


@torch.inference_mode()
def run_test(config: dict[str, Any], state: dict[str, Any]) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "train_promise12"))
    from networks.net_factory import net_factory  # type: ignore

    data_root = ROOT / config["root_path"]
    cases = [line.strip().split(".")[0] for line in (data_root / "test.list").read_text().splitlines() if line.strip()]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = net_factory(
        net_type="unet",
        in_chns=1,
        class_num=2,
        use_mask_token_prehead_router=bool(config["mask_token_enabled"]),
        mask_token_num=4,
        mask_token_dim=16,
    ).to(device)
    state_dict = torch.load(ROOT / state["BestCheckpointPath"], map_location=device)
    model.load_state_dict(state_dict, strict=True)
    model.eval()
    rows = []
    for case in cases:
        image = np.load(data_root / "npy_image" / f"{case}.npy")
        target = np.load(data_root / "npy_image" / f"{case}_segmentation.npy")
        pred = np.zeros_like(target, dtype=np.uint8)
        for idx in range(image.shape[0]):
            tensor = torch.from_numpy(image[idx]).unsqueeze(0).unsqueeze(0).float().to(device)
            pred[idx] = torch.softmax(model(tensor), dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        rows.append({"case": case, **binary_metrics(pred == 1, target == 1)})
    macro = {key: float(np.mean([row[key] for row in rows])) for key in ("dice", "iou", "hd95", "asd")}
    out_dir = ROOT / config["test_output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "case_metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "dice", "iou", "hd95", "asd"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    receipt = {
        "schema": "PROMISE10_PAPER4_TEST_COMPLETION_RECEIPT_V1",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "TaskID": config["task_id"],
        "Dataset": "PROMISE12",
        "Ratio": "10%",
        "Method": config["method_display"],
        "BoundaryUnit": config["boundary_unit"],
        "BestValDice": state["BestValDice"],
        "BestIter": state["BestIter"],
        "CheckpointPath": state["BestCheckpointPath"],
        "CheckpointSHA": state["BestCheckpointSHA256"],
        "EvaluatorPath": "tools/promise10_paper4_test_launcher.py",
        "EvaluatorSHA": sha256(Path(__file__).resolve()),
        "TestManifest": config["test_manifest"],
        "TestManifestSHA": sha256(ROOT / config["test_manifest"]),
        "Dice": macro["dice"],
        "IoU": macro["iou"],
        "HD95": macro["hd95"],
        "ASD": macro["asd"],
        "PerClassMetrics": {"foreground": macro},
        "TestComplete": True,
        "TEST_ONLY_MODE": True,
        "training_started": False,
        "optimizer_created": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "checkpoint_reselection_after_test": False,
    }
    write_json(out_dir / "TEST_RESULT_FULL_PRECISION.json", receipt)
    write_json(out_dir / "TEST_COMPLETION_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["dry-run", "canary", "run"], default="dry-run")
    args = parser.parse_args()
    config = load_config(ROOT / args.config)
    state = unlock_state(config)
    if args.mode == "dry-run":
        payload = {
            "status": "DRY_RUN_PASS",
            "task_id": config["task_id"],
            "TestUnlocked": state["TestUnlocked"],
            "unlock_state": state,
            "REAL_TEST_ACCESS": 0,
        }
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if args.mode == "canary":
        payload = {
            "status": "CANARY_PASS",
            "task_id": config["task_id"],
            "ENGINEERING_CANARY_ONLY": True,
            "SCIENTIFIC_RESULT": False,
            "PAPER_RESULT_ELIGIBLE": False,
            "REAL_TEST_ACCESS": 0,
            "TestUnlocked": state["TestUnlocked"],
        }
        out = ROOT / "runtime/promise10_paper4/test_canary" / config["task_id"] / "CANARY_RECEIPT.json"
        write_json(out, payload)
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0
    if not state["TestUnlocked"]:
        raise SystemExit(json.dumps({"status": "TEST_LOCKED_BEFORE_FULL", "task_id": config["task_id"], "unlock_state": state}, indent=2, sort_keys=True))
    receipt = run_test(config, state)
    print(json.dumps({"status": "TEST_PASS", "task_id": config["task_id"], "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
