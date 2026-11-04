#!/usr/bin/env python3
"""V22.1 frozen-checkpoint TEST-only launcher.

This launcher reads the frozen V22.1 registry and executes exactly one
post-full TEST task.  Dry-run and canary modes never read a real TEST manifest.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
PYTHON = Path("/home/renguanming/.conda/envs/beta_fft/bin/python")
REGISTRY = ROOT / "audits/three_innovation_postfull_test_v22_1/V22_1_POSTFULL_TEST_TASK_REGISTRY.tsv"


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_registry(path: Path) -> dict[str, dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return {row["TaskID"]: row for row in csv.DictReader(handle, delimiter="\t")}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def validate_row(row: dict[str, str], allow_test_manifest: bool) -> dict[str, Any]:
    ckpt = ROOT / row["CheckpointPath"]
    evaluator = ROOT / row["EvaluatorPath"]
    manifest = Path(row["TestManifest"])
    if not manifest.is_absolute():
        manifest = ROOT / manifest
    checks = {
        "checkpoint_exists": ckpt.exists(),
        "checkpoint_sha_matches": ckpt.exists() and sha256(ckpt) == row["CheckpointSHA"],
        "evaluator_exists": evaluator.exists(),
        "evaluator_sha_matches": evaluator.exists() and sha256(evaluator) == row["EvaluatorSHA"],
        "test_manifest_access_allowed": allow_test_manifest,
        "test_manifest_exists": bool(allow_test_manifest and manifest.exists()),
        "test_manifest_sha_matches": bool(allow_test_manifest and manifest.exists() and sha256(manifest) == row["TestManifestSHA"]),
        "TEST_ONLY_MODE": True,
        "training_started": False,
        "optimizer_created": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "checkpoint_reselection_after_test": False,
    }
    checks["pass"] = all(
        checks[key]
        for key in (
            "checkpoint_exists",
            "checkpoint_sha_matches",
            "evaluator_exists",
            "evaluator_sha_matches",
        )
    ) and (not allow_test_manifest or (checks["test_manifest_exists"] and checks["test_manifest_sha_matches"]))
    return checks


def acdc_run(row: dict[str, str], output_dir: Path, mode: str) -> dict[str, Any]:
    config_path = ROOT / row["ConfigPath"]
    config_payload = json.loads(config_path.read_text(encoding="utf-8"))
    if "dataset_root" not in config_payload:
        config_payload = {
            **config_payload,
            "task_id": row["TaskID"],
            "dataset_root": "/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC",
            "test_manifest": row["TestManifest"],
            "mask_token_enabled": True,
        }
        generated = ROOT / "runtime/three_innovation_postfull_test_v22_1/generated_configs" / f"{row['TaskID']}_acdc_test_config.json"
        write_json(generated, config_payload)
        config_arg = str(generated.relative_to(ROOT))
    else:
        config_arg = row["ConfigPath"]
    cmd = [
        str(PYTHON),
        "tools/v22_0_acdc_test_frozen_checkpoint.py",
        "--config",
        config_arg,
        "--checkpoint",
        row["CheckpointPath"],
        "--output-dir",
        str(output_dir.relative_to(ROOT)),
        "--mode",
        "dry-run" if mode == "dry-run" else "run",
        "--device",
        "cuda:0",
    ]
    if mode == "dry-run":
        proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
        return {"delegate_command": cmd, "return_code": proc.returncode, "stdout": proc.stdout[-4000:], "stderr": proc.stderr[-4000:]}
    proc = subprocess.run(cmd, cwd=ROOT, text=True, capture_output=True)
    if proc.returncode != 0:
        raise RuntimeError(f"ACDC_TEST_FAIL:{row['TaskID']}:{proc.returncode}:{proc.stderr[-2000:]}")
    receipt = json.loads((output_dir / "TEST_RECEIPT.json").read_text(encoding="utf-8"))
    return {"delegate_command": cmd, "return_code": proc.returncode, "delegate_receipt": receipt}


def finite_metrics(metrics: dict[str, Any]) -> bool:
    values = []
    for key in ("dice", "iou", "hd95", "asd", "mDice", "mIoU", "HD95", "ASD"):
        if key in metrics:
            values.append(float(metrics[key]))
    return bool(values) and all(np.isfinite(values))


def make_receipt(row: dict[str, str], output_dir: Path, metrics: dict[str, Any], per_class: dict[str, Any] | None = None) -> dict[str, Any]:
    receipt = {
        "schema": "V22_1_POSTFULL_TEST_COMPLETION_RECEIPT",
        "TaskID": row["TaskID"],
        "Dataset": row["Dataset"],
        "Ratio": row["Ratio"],
        "Arm": row["Arm"],
        "CheckpointSelectionEndpoint": row["CheckpointSelectionEndpoint"],
        "BestValDice": float(row["BestValDice"]) if row["BestValDice"] else None,
        "BestValIter": int(row["BestValIter"]) if row["BestValIter"] else None,
        "CheckpointPath": row["CheckpointPath"],
        "CheckpointSHA": row["CheckpointSHA"],
        "TestManifest": row["TestManifest"],
        "TestManifestSHA": row["TestManifestSHA"],
        "EvaluatorPath": row["EvaluatorPath"],
        "EvaluatorSHA": row["EvaluatorSHA"],
        "BoundaryUnit": row["BoundaryUnit"],
        "Dice": float(metrics.get("dice", metrics.get("mDice"))),
        "IoU": float(metrics.get("iou", metrics.get("mIoU"))),
        "HD95": float(metrics.get("hd95", metrics.get("HD95"))),
        "ASD": float(metrics.get("asd", metrics.get("ASD"))),
        "PerClassMetrics": per_class or {},
        "ReturnCode": 0,
        "Finite": finite_metrics(metrics),
        "TestComplete": True,
        "TestTimestamp": datetime.now(timezone.utc).isoformat(),
        "TEST_ONLY_MODE": True,
        "training_started": False,
        "optimizer_created": False,
        "backward_called": False,
        "optimizer_step_called": False,
        "checkpoint_reselection_after_test": False,
    }
    write_json(output_dir / "TEST_RESULT_FULL_PRECISION.json", receipt)
    write_json(output_dir / "TEST_COMPLETION_RECEIPT.json", receipt)
    return receipt


def promise_binary_metrics(pred: np.ndarray, target: np.ndarray) -> dict[str, float]:
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
def promise_run(row: dict[str, str], output_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT / "train_promise12"))
    from networks.net_factory import net_factory  # type: ignore

    data_root = ROOT / "data_staging/promise12/20260624_141558/data_promise12"
    cases = [line.strip().split(".")[0] for line in (data_root / "test.list").read_text().splitlines() if line.strip()]
    model = net_factory(net_type="unet", in_chns=1, class_num=2, use_mask_token_prehead_router=True, mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(torch.load(ROOT / row["CheckpointPath"], map_location="cpu"))
    model.eval()
    rows = []
    for case in cases:
        image = np.load(data_root / "npy_image" / f"{case}.npy")
        target = np.load(data_root / "npy_image" / f"{case}_segmentation.npy")
        pred = np.zeros_like(target, dtype=np.uint8)
        for idx in range(image.shape[0]):
            tensor = torch.from_numpy(image[idx]).unsqueeze(0).unsqueeze(0).float().cuda()
            pred[idx] = torch.softmax(model(tensor), dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        metrics = promise_binary_metrics(pred == 1, target == 1)
        rows.append({"case": case, **metrics})
    macro = {key: float(np.mean([r[key] for r in rows])) for key in ("dice", "iou", "hd95", "asd")}
    with (output_dir / "case_metrics.tsv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["case", "dice", "iou", "hd95", "asd"], delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)
    return make_receipt(row, output_dir, macro, {"foreground": macro})


@torch.inference_mode()
def covid_run(row: dict[str, str], output_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "train_promise12"))
    from networks.net_factory import net_factory  # type: ignore
    from train_COVID19.covid19_dataset_adapter import Covid19Official20PatientVolumeDataset
    from train_COVID19.covid19_volume_evaluator import SlicePrediction, patient_volume_macro

    manifest = ROOT / "data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv"
    dataset = Covid19Official20PatientVolumeDataset(manifest)
    model = net_factory(net_type="unet", in_chns=1, class_num=2, use_mask_token_prehead_router=True, mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(torch.load(ROOT / row["CheckpointPath"], map_location="cpu"))
    model.eval()
    slices = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        image = sample["image"].float()
        target = sample["mask"].cpu().numpy().astype(np.uint8)
        spacing = tuple(float(x) for x in sample["spacing"].cpu().numpy().tolist())
        for z in range(image.shape[0]):
            tensor = image[z].unsqueeze(0).unsqueeze(0).float().cuda()
            pred = torch.softmax(model(tensor), dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
            slices.append(SlicePrediction(str(sample["patient_id"]), int(sample["slice_idx"][z]), pred > 0, target[z] > 0, spacing))
    result = patient_volume_macro(slices)
    macro = result["macro"]
    metrics = {"dice": macro["dice"], "iou": macro["iou"], "hd95": macro["hd95_mm"], "asd": macro["asd_mm"]}
    write_json(output_dir / "patient_metrics.json", result)
    return make_receipt(row, output_dir, metrics, {"foreground": metrics})


@torch.inference_mode()
def mscmr_run(row: dict[str, str], output_dir: Path) -> dict[str, Any]:
    sys.path.insert(0, str(ROOT))
    from train_MSCMRSEG.v15_1_mscmr_candidate_modules import DecoderHeadRouter, MaskTokenAdapter
    from train_MSCMRSEG.v15_1_mscmr_common import make_model
    from train_MSCMRSEG.v15_1_mscmr_eval import evaluate_manifest

    device = torch.device("cuda:0")
    ckpt = torch.load(ROOT / row["CheckpointPath"], map_location=device)
    model = make_model(device)
    token = MaskTokenAdapter(16, 4).to(device)
    hook = DecoderHeadRouter(model, token)
    hook.attach()
    model.load_state_dict(ckpt["model1"], strict=True)
    token.load_state_dict(ckpt["mask_token_model1"], strict=True)
    model.eval()
    try:
        result = evaluate_manifest(
            model,
            "/home/renguanming/jupyterlab/rgm/datasets/MS-CMRSeg2019/raw",
            ROOT / "audit/mscmr_split_manifest.tsv",
            image_size=(256, 256),
            device=device,
            role="formal_test",
        )
    finally:
        hook.remove()
    write_json(output_dir / "case_metrics.json", result)
    metrics = {"dice": result["mDice"], "iou": result["mIoU"], "hd95": result["HD95"], "asd": result["ASD"]}
    return make_receipt(row, output_dir, metrics, result["classes"])


def canary_run(row: dict[str, str], output_dir: Path) -> dict[str, Any]:
    metrics = {"dice": 0.5, "iou": 1.0 / 3.0, "hd95": 1.0, "asd": 0.5}
    return make_receipt(row, output_dir, metrics, {"engineering_canary": metrics})


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", required=True)
    parser.add_argument("--registry", default=str(REGISTRY))
    parser.add_argument("--mode", choices=["dry-run", "canary", "run"], default="run")
    args = parser.parse_args()

    row = read_registry(Path(args.registry))[args.task_id]
    if row["Action"] != "NEW_TEST_READY":
        raise SystemExit(f"TASK_NOT_NEW_TEST_READY:{args.task_id}:{row['Action']}")
    allow_test_manifest = args.mode == "run"
    checks = validate_row(row, allow_test_manifest=allow_test_manifest)
    if args.mode == "canary":
        output_dir = ROOT / "runtime/three_innovation_postfull_test_v22_1/canary" / args.task_id
    else:
        output_dir = ROOT / row["TestOutputDir"]
    if args.mode == "dry-run":
        payload = {"status": "DRY_RUN_PASS" if checks["pass"] else "DRY_RUN_FAIL", "task": row, "checks": checks}
        print(json.dumps(payload, indent=2, sort_keys=True))
        return 0 if checks["pass"] else 2
    if not checks["pass"]:
        raise SystemExit(json.dumps({"status": "IDENTITY_CHECK_FAIL", "task_id": args.task_id, "checks": checks}, indent=2, sort_keys=True))
    output_dir.mkdir(parents=True, exist_ok=args.mode == "canary")
    if args.mode == "canary":
        receipt = canary_run(row, output_dir)
    elif row["Dataset"] == "ACDC":
        delegate = acdc_run(row, output_dir, args.mode)
        metrics = delegate["delegate_receipt"]["metrics"]
        per_class = {name: {k: metrics[f"{name}_{k}"] for k in ("dice", "iou", "hd95", "asd")} for name in ("RV", "MYO", "LV")}
        receipt = make_receipt(row, output_dir, metrics, per_class)
    elif row["Dataset"] == "PROMISE12":
        receipt = promise_run(row, output_dir)
    elif row["Dataset"] == "COVID-19":
        receipt = covid_run(row, output_dir)
    elif row["Dataset"] == "MS-CMRSeg":
        receipt = mscmr_run(row, output_dir)
    else:
        raise SystemExit(f"UNSUPPORTED_DATASET:{row['Dataset']}")
    print(json.dumps({"status": "TEST_PASS", "task_id": args.task_id, "receipt": receipt}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
