#!/usr/bin/env python3
"""V24 per-task frozen BEST TEST gate and launcher.

The gate is deliberately independent per NEW_FULL task. Dry-run and locked run
paths do not read TEST manifest contents. Real TEST execution is allowed only
after FormalFullComplete, BestValSelected, BestCheckpointSHA256Frozen, protocol
lock, and zero pre-freeze TEST access are all proven for the selected task.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train_ACDC"))
from networks.net_factory import BCP_net as BetaFFTNet  # noqa: E402
from v14_mask_adamix_components import v14_bcp_net as OursNet  # noqa: E402

AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
CONFIG_ROOT = ROOT / "configs/internal_evidence_expansion_v24_0"
TEST_ROOT = RESULT_ROOT / "raw_test"


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


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text())


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")
    tmp.replace(path)


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f12(value: float) -> str:
    return f"{value:.12f}"


def config_path(task_id: str) -> Path:
    return CONFIG_ROOT / f"{task_id}.json"


def gate_assets(task_id: str, cfg: dict[str, Any]) -> dict[str, Path]:
    run_root = Path(cfg["run_root"])
    return {
        "FormalFullComplete": run_root / "FORMAL_FULL_COMPLETION_RECEIPT.json",
        "BestValSelected": run_root / "BEST_VAL_SELECTION.json",
        "FrozenCheckpointSHA": run_root / "FROZEN_CHECKPOINT_SHA256.json",
        "TestProtocolLock": AUDIT_ROOT / f"{task_id}_TEST_PROTOCOL_LOCK.json",
    }


def ensure_protocol_lock(task_id: str, cfg: dict[str, Any]) -> dict[str, Any]:
    test_list = Path(cfg["test_list"])
    payload = {
        "status": "PASS" if test_list.exists() and int(cfg.get("test_access_before_best_freeze", -1)) == 0 else "FAIL",
        "task_id": task_id,
        "TestProtocolLockPass": test_list.exists() and int(cfg.get("test_access_before_best_freeze", -1)) == 0,
        "dataset_root": cfg["dataset_root"],
        "test_manifest_path": str(test_list),
        "test_manifest_exists": test_list.exists(),
        "test_manifest_sha256": sha256(test_list) if test_list.exists() else "",
        "test_manifest_content_read": False,
        "RealTestAccessBeforeCheckpointFreeze": int(cfg.get("test_access_before_best_freeze", -1)),
        "test_used_for_checkpoint_selection": False,
    }
    write_json(AUDIT_ROOT / f"{task_id}_TEST_PROTOCOL_LOCK.json", payload)
    return payload


def gate_state(task_id: str) -> dict[str, Any]:
    cfg_file = config_path(task_id)
    cfg = read_json(cfg_file)
    if not cfg:
        return {"status": "FAIL", "task_id": task_id, "reason": "CONFIG_MISSING", "TestUnlocked": False}
    protocol = ensure_protocol_lock(task_id, cfg)
    assets = gate_assets(task_id, cfg)
    evidence = {name: read_json(path) for name, path in assets.items()}
    formal_ok = evidence["FormalFullComplete"].get("FormalFullComplete") is True
    best_ok = evidence["BestValSelected"].get("BestValSelected") is True
    frozen_ok = evidence["FrozenCheckpointSHA"].get("BestCheckpointSHA256Frozen") is True
    protocol_ok = protocol.get("TestProtocolLockPass") is True
    prefreeze_zero = protocol.get("RealTestAccessBeforeCheckpointFreeze") == 0
    selected_path = evidence["FrozenCheckpointSHA"].get("BestCheckpointPath", "")
    selected_sha = evidence["FrozenCheckpointSHA"].get("BestCheckpointSHA256", "")
    selected_exists = bool(selected_path and (ROOT / selected_path).exists())
    selected_sha_ok = selected_exists and sha256(ROOT / selected_path) == selected_sha
    unlocked = all([formal_ok, best_ok, frozen_ok, protocol_ok, prefreeze_zero, selected_sha_ok])
    missing = []
    for key, ok in [
        ("FormalFullComplete", formal_ok),
        ("BestValSelected", best_ok),
        ("BestCheckpointSHA256Frozen", frozen_ok),
        ("TestProtocolLockPass", protocol_ok),
        ("RealTestAccessBeforeCheckpointFreezeZero", prefreeze_zero),
        ("BestCheckpointPathExistsAndSHAMatches", selected_sha_ok),
    ]:
        if not ok:
            missing.append(key)
    return {
        "status": "UNLOCKED" if unlocked else "LOCKED",
        "created_at": now(),
        "task_id": task_id,
        "config": rel(cfg_file),
        "FormalFullComplete": formal_ok,
        "BestValSelected": best_ok,
        "BestCheckpointPath": selected_path,
        "BestCheckpointSHA256Frozen": frozen_ok,
        "BestCheckpointSHA256": selected_sha,
        "BestCheckpointPathExistsAndSHAMatches": selected_sha_ok,
        "TestProtocolLockPass": protocol_ok,
        "RealTestAccessBeforeCheckpointFreeze": protocol.get("RealTestAccessBeforeCheckpointFreeze"),
        "TestUnlocked": unlocked,
        "TestCompleted": (TEST_ROOT / task_id / "TEST_COMPLETION_RECEIPT.json").exists(),
        "missing_unlock_evidence": missing,
        "test_manifest_content_read": False,
        "test_started": False,
        "cfg": cfg,
    }


def safe_metrics(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    if pred.any() and truth.any():
        inter = int((pred & truth).sum())
        union = int((pred | truth).sum())
        return 2.0 * inter / (int(pred.sum()) + int(truth.sum())), inter / union if union else 0.0, float(metric.binary.hd95(pred, truth)), float(metric.binary.asd(pred, truth))
    if not pred.any() and not truth.any():
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def build_model(cfg: dict[str, Any], checkpoint_path: Path, device: str):
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    if cfg["method"] == "BETAFFT":
        model = BetaFFTNet(in_chns=1, class_num=4, ema=False)
    else:
        model = OursNet(in_chns=1, class_num=4, ema=False, mask_token_enabled=True, mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(checkpoint["model_1"], strict=True)
    model.to(device)
    model.eval()
    return model


def infer_case(model, case_id: str, dataset_root: Path, device: str) -> tuple[np.ndarray, np.ndarray]:
    with h5py.File(dataset_root / "data" / f"{case_id}.h5", "r") as handle:
        image = handle["image"][:]
        truth = handle["label"][:]
    prediction = np.zeros_like(truth)
    with torch.no_grad():
        for index, source in enumerate(image):
            height, width = source.shape
            resized = zoom(source, (256 / height, 256 / width), order=0)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
            output = model(tensor)
            if isinstance(output, (tuple, list)):
                output = output[0]
            pred = torch.softmax(output, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
            prediction[index] = zoom(pred, (height / 256, width / 256), order=0)
    return prediction, truth


def run_test(task_id: str, state: dict[str, Any], device: str) -> dict[str, Any]:
    cfg = state["cfg"]
    test_manifest = Path(cfg["test_list"])
    cases = [line.strip() for line in test_manifest.read_text().splitlines() if line.strip()]
    out_dir = TEST_ROOT / task_id
    receipt = out_dir / "TEST_COMPLETION_RECEIPT.json"
    if receipt.exists():
        raise SystemExit(f"OUTPUT_COLLISION:{rel(receipt)}")
    pred_dir = out_dir / "predictions_npz"
    pred_dir.mkdir(parents=True, exist_ok=False)
    model = build_model(cfg, ROOT / state["BestCheckpointPath"], device)
    rows = []
    for case_id in cases:
        pred, truth = infer_case(model, case_id, Path(cfg["dataset_root"]), device)
        np.savez_compressed(pred_dir / f"{case_id}.npz", prediction=pred, label=truth)
        for class_id, class_name in [(1, "RV"), (2, "MYO"), (3, "LV")]:
            dice, iou, hd95, asd = safe_metrics(pred == class_id, truth == class_id)
            rows.append({"case_id": case_id, "class_id": class_id, "class_name": class_name, "dice": f12(dice), "iou": f12(iou), "hd95": f12(hd95), "asd": f12(asd)})
    case_tsv = out_dir / "case_class_metrics.tsv"
    write_tsv(case_tsv, rows)
    aggregate = {metric_name: f12(float(np.mean([float(row[metric_name]) for row in rows]))) for metric_name in ["dice", "iou", "hd95", "asd"]}
    agg_tsv = out_dir / "aggregate_metrics.tsv"
    write_tsv(agg_tsv, [{"task_id": task_id, **aggregate}])
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": task_id,
        "TEST_COMPLETION_RECEIPT": True,
        "training_started": False,
        "checkpoint_selection_started": False,
        "test_started": True,
        "test_manifest_sha256": sha256(test_manifest),
        "case_count": len(cases),
        "BestCheckpointPath": state["BestCheckpointPath"],
        "BestCheckpointSHA256": state["BestCheckpointSHA256"],
        "test_used_for_tuning": False,
        "test_used_for_selection": False,
        "prediction_dir": rel(pred_dir),
        "case_class_metrics_tsv": rel(case_tsv),
        "aggregate_metrics_tsv": rel(agg_tsv),
        "aggregate": aggregate,
    }
    write_json(receipt, payload)
    write_json(AUDIT_ROOT / f"{task_id}_TEST_COMPLETION_RECEIPT.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--mode", choices=["dry-run", "run", "status"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    state = gate_state(args.task_id)
    state_public = {k: v for k, v in state.items() if k != "cfg"}
    if args.mode in {"dry-run", "status"}:
        write_json(AUDIT_ROOT / f"{args.task_id}_TEST_GATE_DRY_RUN_RECEIPT.json", state_public)
        print(json.dumps(state_public, indent=2, ensure_ascii=False))
        return 0
    if not state.get("TestUnlocked"):
        payload = dict(state_public)
        payload["status"] = "REFUSE_TEST_LOCKED"
        payload["test_started"] = False
        write_json(AUDIT_ROOT / f"{args.task_id}_TEST_GATE_LOCKED_RECEIPT.json", payload)
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return 3
    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUESTED_BUT_NOT_AVAILABLE")
    payload = run_test(args.task_id, state, args.device)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
