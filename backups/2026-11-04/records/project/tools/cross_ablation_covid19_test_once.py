#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import torch


ROOT = Path(__file__).resolve().parents[1]
TRAIN_DIR = ROOT / "train_promise12"
sys.path.insert(0, str(TRAIN_DIR))
sys.path.insert(0, str(ROOT))

from networks.net_factory import net_factory  # noqa: E402
from train_COVID19.covid19_dataset_adapter import Covid19Official20PatientVolumeDataset  # noqa: E402
from train_COVID19.covid19_volume_evaluator import SlicePrediction, patient_volume_macro  # noqa: E402


QUEUE = ROOT / "runtime/cross_dataset_ablation/protocol_bound_queue.json"
TEST_MANIFEST = ROOT / "data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv"
TEST_PATIENTS = ROOT / "data/COVID19_OFFICIAL20/manifests/COVID19_TEST_PATIENTS.tsv"
OUT_ROOT = ROOT / "formal_test/cross_dataset_ablation_v19_8/covid19"
SUMMARY = ROOT / "final_report/CROSS_DATASET_ABLATION_COVID19_SINGLE_MODULE_TEST_RESULTS.tsv"
AUDIT = ROOT / "audits/CROSS_DATASET_ABLATION_COVID19_TEST_ONCE_RECEIPTS.json"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


import hashlib


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, ensure_ascii=False, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_queue() -> dict[str, Any]:
    return json.loads(QUEUE.read_text(encoding="utf-8"))


def select_cell(run_id: str) -> dict[str, Any]:
    cells = [c for c in read_queue().get("cells", []) if c.get("run_id") == run_id]
    if len(cells) != 1:
        raise RuntimeError(f"COVID_CROSS_ABLATION_CELL_NOT_FOUND:{run_id}")
    cell = cells[0]
    if cell.get("dataset") != "COVID-19":
        raise RuntimeError(f"NOT_COVID_CELL:{run_id}")
    return cell


def manifest_identity() -> dict[str, Any]:
    with TEST_PATIENTS.open("r", encoding="utf-8", newline="") as handle:
        patients = [row["patient_id"] for row in csv.DictReader(handle, delimiter="\t")]
    expected = [
        "radiopaedia_40_86625_0",
        "coronacases_002",
        "radiopaedia_27_86410_0",
        "coronacases_009",
    ]
    if patients != expected:
        raise RuntimeError(f"COVID19_TEST_PATIENT_IDENTITY_MISMATCH:{patients}")
    return {
        "test_patients": patients,
        "test_patient_manifest_sha256": sha256_file(TEST_PATIENTS),
        "test_all_slices_manifest_sha256": sha256_file(TEST_MANIFEST),
    }


def parse_validation_curve(run_id: str, labeled_num: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        ROOT / "train_promise12/beta_FFT" / f"PROMISE12_bcp_{run_id}_{labeled_num}" / "pre_train/log.txt",
        ROOT / "train_promise12/beta_FFT" / f"PROMISE12_bcp_fft_test_seed_{run_id}_{labeled_num}" / "self_train/log.txt",
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in candidates if path.exists())
    curve: list[dict[str, Any]] = []
    for match in re.finditer(r"iteration (\d+) : model1_mean_dice : ([0-9.]+)", text):
        iteration = int(match.group(1))
        nearby = text[match.end(): match.end() + 700]
        metrics = re.search(
            rf"iteration {iteration} : val_patient_volume_macro_dice : ([0-9.]+), iou : ([0-9.]+), hd95_mm : ([0-9.]+), asd_mm : ([0-9.]+)",
            nearby,
        )
        row = {"iter": iteration, "dice": float(match.group(2))}
        if metrics:
            row.update({"iou": float(metrics.group(2)), "hd95_mm": float(metrics.group(3)), "asd_mm": float(metrics.group(4))})
        curve.append(row)
    if not curve:
        raise RuntimeError(f"EMPTY_COVID_VALIDATION_CURVE:{run_id}")
    return max(curve, key=lambda item: item["dice"]), curve


def build_model(mask_token: bool) -> torch.nn.Module:
    model = net_factory(
        net_type="unet",
        in_chns=1,
        class_num=2,
        use_mask_token_prehead_router=mask_token,
        mask_token_num=4,
        mask_token_dim=16,
        use_aupp_prehead_lite=False,
        aupp_prehead_strength=0.05,
    )
    for module in model.modules():
        if hasattr(module, "mask_token_router") and getattr(module, "mask_token_router") is not None:
            setattr(module.mask_token_router, "active", True)
    return model


def predict_volume(image: torch.Tensor, model: torch.nn.Module) -> np.ndarray:
    image_np = image.cpu().numpy().astype(np.float32)
    prediction = np.zeros(image_np.shape, dtype=np.uint8)
    model.eval()
    with torch.no_grad():
        for idx in range(image_np.shape[0]):
            tensor = torch.from_numpy(image_np[idx]).unsqueeze(0).unsqueeze(0).float().cuda()
            logits = model(tensor)
            pred = torch.argmax(torch.softmax(logits, dim=1), dim=1).squeeze(0)
            prediction[idx] = pred.cpu().numpy().astype(np.uint8)
    return prediction


def labeled_num(cell: dict[str, Any]) -> int:
    return 120 if cell["setting"] == "COVID5" else 241


def load_existing_receipts() -> list[dict[str, Any]]:
    if not AUDIT.exists():
        return []
    return json.loads(AUDIT.read_text(encoding="utf-8")).get("receipts", [])


def write_summary(receipts: list[dict[str, Any]]) -> None:
    fields = ["run_id", "setting", "arm", "mask_token", "adamix", "best_iter", "best_val_dice", "dice", "iou", "hd95_mm", "asd_mm", "unit", "receipt"]
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    with SUMMARY.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        for receipt in receipts:
            macro = receipt["metric_result"]["macro"]
            writer.writerow(
                {
                    "run_id": receipt["source_training_run"],
                    "setting": receipt["setting"],
                    "arm": receipt["arm"],
                    "mask_token": "ON" if receipt["mask_token"] else "OFF",
                    "adamix": "ON" if receipt["adamix"] else "OFF",
                    "best_iter": receipt["best_iter"],
                    "best_val_dice": receipt["best_val_dice"],
                    "dice": macro["dice"],
                    "iou": macro["iou"],
                    "hd95_mm": macro["hd95_mm"],
                    "asd_mm": macro["asd_mm"],
                    "unit": "mm",
                    "receipt": receipt["receipt_path"],
                }
            )
    atomic_json(AUDIT, {"schema": "V19_8_CROSS_ABLATION_COVID19_TEST_ONCE_RECEIPTS", "receipts": receipts})


def run_test(cell: dict[str, Any]) -> dict[str, Any]:
    labeled = labeled_num(cell)
    ckpt = ROOT / cell["expected_best_checkpoint"]
    if not ckpt.exists():
        raise RuntimeError(f"MISSING_COVID_BEST_CHECKPOINT:{ckpt}")
    best, curve = parse_validation_curve(cell["run_id"], labeled)
    test_run_id = f"{cell['run_id']}_CONFIRMATORY_TEST_ONCE"
    test_dir = OUT_ROOT / test_run_id
    receipt_path = test_dir / "FORMAL_TEST_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError(f"FORMAL_TEST_ONCE_ALREADY_HAS_RECEIPT:{receipt_path}")
    manifest = manifest_identity()
    dataset = Covid19Official20PatientVolumeDataset(TEST_MANIFEST)
    model = build_model(bool(cell["mask_token"]))
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.cuda()
    model.eval()
    prediction_paths = []
    slices: list[SlicePrediction] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        patient_id = str(sample["patient_id"])
        image = sample["image"].float()
        gt = sample["mask"].cpu().numpy().astype(np.uint8)
        spacing = tuple(float(x) for x in sample["spacing"].cpu().numpy().tolist())
        pred = predict_volume(image, model)
        pred_path = test_dir / "predictions" / f"{patient_id}_prediction.npz"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_path, prediction=pred.astype(np.uint8), patient_id=patient_id, spacing=np.asarray(spacing, dtype=np.float32))
        prediction_paths.append({"patient_id": patient_id, "path": str(pred_path.relative_to(ROOT)), "output_sha256": sha256_file(pred_path)})
        for local_idx, slice_idx in enumerate(sample["slice_idx"].cpu().numpy().tolist()):
            slices.append(SlicePrediction(patient_id=patient_id, slice_idx=int(slice_idx), pred=pred[local_idx] > 0, gt=gt[local_idx] > 0, spacing=spacing))
    receipt = {
        "schema": "V19_8_CROSS_ABLATION_COVID19_FORMAL_TEST_ONCE_RECEIPT",
        "test_run_id": test_run_id,
        "source_training_run": cell["run_id"],
        "setting": cell["setting"],
        "arm": cell["arm"],
        "formal_training_seed": 1337,
        "split_provenance_seed": 42,
        "checkpoint_path": str(ckpt.relative_to(ROOT)),
        "checkpoint_sha256": sha256_file(ckpt),
        "best_iter": best["iter"],
        "best_val_dice": best["dice"],
        "validation_curve_points": len(curve),
        "mask_token": bool(cell["mask_token"]),
        "adamix": bool(cell["adamix"]),
        "test_patient_sha256": manifest["test_patient_manifest_sha256"],
        "test_all_slices_sha256": manifest["test_all_slices_manifest_sha256"],
        "prediction_paths": prediction_paths,
        "metric_result": patient_volume_macro(slices),
        "receipt_path": str(receipt_path.relative_to(ROOT)),
        "created_utc": utc_now(),
        "test_used_for_selection": False,
        "training_executed": False,
    }
    atomic_json(receipt_path, receipt)
    receipts = [r for r in load_existing_receipts() if r.get("source_training_run") != cell["run_id"]]
    receipts.append(receipt)
    write_summary(receipts)
    return receipt


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    cell = select_cell(args.run_id)
    ckpt = ROOT / cell["expected_best_checkpoint"]
    if args.dry_run:
        print(json.dumps({"run_id": cell["run_id"], "checkpoint": str(ckpt), "checkpoint_exists": ckpt.exists(), "dry_run": True}, indent=2, sort_keys=True))
        return 0
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_COVID_CROSS_ABLATION_TEST_ONCE")
    receipt = run_test(cell)
    print(json.dumps({"status": "COMPLETE", "receipt": receipt["receipt_path"]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
