#!/usr/bin/env python3
"""COVID-19 confirmatory formal test-once runner for V19.1 seed1337 runs."""

from __future__ import annotations

import csv
import hashlib
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
if str(TRAIN_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_DIR))
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from networks.net_factory import net_factory  # noqa: E402
from train_COVID19.covid19_dataset_adapter import Covid19Official20PatientVolumeDataset  # noqa: E402
from train_COVID19.covid19_volume_evaluator import SlicePrediction, patient_volume_macro  # noqa: E402


PROTOCOL_IDENTITY = "OFFICIAL20_PATIENTLEVEL_ALLSLICE_V2"
QUEUE = ROOT / "manifests/PROMISE10_COVID1337_FORMAL_QUEUE.json"
TEST_MANIFEST = ROOT / "data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv"
TEST_PATIENTS = ROOT / "data/COVID19_OFFICIAL20/manifests/COVID19_TEST_PATIENTS.tsv"
OUT_ROOT = ROOT / "formal_test/covid19_seed1337"
FINAL_REPORT = ROOT / "final_report"
AUDITS = ROOT / "audits"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def queue_runs() -> list[dict[str, Any]]:
    queue = read_json(QUEUE)
    cells = [cell for cell in queue["cells"] if cell["dataset"] == "COVID19"]
    expected = [
        "COVID19_5P_BASELINE_CANON1337_R1",
        "COVID19_5P_MASK_TOKEN_ADAMIX_CANON1337_R1",
        "COVID19_10P_BASELINE_CANON1337_R1",
        "COVID19_10P_MASK_TOKEN_ADAMIX_CANON1337_R1",
    ]
    if [cell["run_id"] for cell in cells] != expected:
        raise RuntimeError("SEED1337_COVID_TEST_QUEUE_MISMATCH")
    runs = []
    for cell in cells:
        runs.append(
            {
                "test_run_id": f"{cell['run_id']}_CONFIRMATORY_TEST_ONCE",
                "source_run_id": cell["run_id"],
                "percent": cell["ratio"],
                "method_identity": cell["role"],
                "mask_token": cell["role"] == "OURS",
                "adamix": cell["role"] == "OURS",
                "config": cell["config_path"],
                "labeled_num": cell["labeled_count"],
            }
        )
    return runs


def manifest_identity() -> dict[str, Any]:
    with TEST_PATIENTS.open("r", encoding="utf-8", newline="") as handle:
        patients = [row["patient_id"] for row in csv.DictReader(handle, delimiter="\t")]
    if patients != [
        "radiopaedia_40_86625_0",
        "coronacases_002",
        "radiopaedia_27_86410_0",
        "coronacases_009",
    ]:
        raise RuntimeError(f"COVID19_TEST_PATIENT_IDENTITY_MISMATCH:{patients}")
    with TEST_MANIFEST.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    grouped: dict[str, list[int]] = {}
    for row in rows:
        grouped.setdefault(row["patient_id"], []).append(int(row["slice_idx"]))
    if sorted(grouped) != sorted(patients):
        raise RuntimeError("COVID19_TEST_MANIFEST_PATIENT_SET_MISMATCH")
    for patient_id, indices in grouped.items():
        ordered = sorted(indices)
        if ordered != list(range(len(ordered))):
            raise RuntimeError(f"COVID19_TEST_SLICES_NOT_CONTIGUOUS:{patient_id}")
    return {
        "test_patient_count": len(patients),
        "test_patients": patients,
        "test_patient_manifest": str(TEST_PATIENTS),
        "test_patient_manifest_sha256": sha256_file(TEST_PATIENTS),
        "test_all_slices_manifest": str(TEST_MANIFEST),
        "test_all_slices_manifest_sha256": sha256_file(TEST_MANIFEST),
        "slice_counts": {patient_id: len(grouped[patient_id]) for patient_id in patients},
    }


def parse_validation_curve(run_id: str, labeled_num: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    candidates = [
        ROOT / "train_promise12/beta_FFT" / f"PROMISE12_bcp_{run_id}_{labeled_num}" / "pre_train/log.txt",
        ROOT / "train_promise12/beta_FFT" / f"PROMISE12_bcp_fft_test_seed_{run_id}_{labeled_num}" / "self_train/log.txt",
    ]
    text = "\n".join(path.read_text(encoding="utf-8", errors="replace") for path in candidates if path.exists())
    curve: list[dict[str, Any]] = []
    for m in re.finditer(r"iteration (\d+) : model1_mean_dice : ([0-9.]+)", text):
        it = int(m.group(1))
        mm = re.search(
            rf"iteration {it} : val_patient_volume_macro_dice : ([0-9.]+), iou : ([0-9.]+), hd95_mm : ([0-9.]+), asd_mm : ([0-9.]+)",
            text[m.end() : m.end() + 700],
        )
        row = {"iter": it, "dice": float(m.group(2))}
        if mm:
            row.update({"iou": float(mm.group(2)), "hd95_mm": float(mm.group(3)), "asd_mm": float(mm.group(4))})
        curve.append(row)
    if not curve:
        raise RuntimeError(f"EMPTY_SEED1337_COVID_VALIDATION_CURVE:{run_id}")
    return max(curve, key=lambda item: item["dice"]), curve


def checkpoint_path(run_id: str, labeled_num: int) -> Path:
    return ROOT / "train_promise12/beta_FFT" / f"PROMISE12_bcp_fft_test_seed_{run_id}_{labeled_num}" / "self_train/unet_best_model.pth"


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


def run_cell(item: dict[str, Any], manifest: dict[str, Any], evaluator_sha: str) -> dict[str, Any]:
    test_run_dir = OUT_ROOT / item["test_run_id"]
    receipt_path = test_run_dir / "FORMAL_TEST_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError(f"FORMAL_TEST_ONCE_ALREADY_HAS_RECEIPT:{receipt_path}")
    best, curve = parse_validation_curve(item["source_run_id"], item["labeled_num"])
    ckpt = checkpoint_path(item["source_run_id"], item["labeled_num"])
    if not ckpt.exists():
        raise RuntimeError(f"MISSING_SEED1337_COVID_BEST_CHECKPOINT:{ckpt}")
    dataset = Covid19Official20PatientVolumeDataset(TEST_MANIFEST)
    model = build_model(item["mask_token"])
    model.load_state_dict(torch.load(ckpt, map_location="cpu"))
    model.cuda()
    model.eval()
    prediction_paths = []
    slices: list[SlicePrediction] = []
    start = utc_now()
    for idx in range(len(dataset)):
        sample = dataset[idx]
        patient_id = str(sample["patient_id"])
        image = sample["image"].float()
        gt = sample["mask"].cpu().numpy().astype(np.uint8)
        spacing = tuple(float(x) for x in sample["spacing"].cpu().numpy().tolist())
        pred = predict_volume(image, model)
        pred_path = test_run_dir / "predictions" / f"{patient_id}_prediction.npz"
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(pred_path, prediction=pred.astype(np.uint8), patient_id=patient_id, spacing=np.asarray(spacing, dtype=np.float32))
        prediction_paths.append(
            {
                "patient_id": patient_id,
                "path": str(pred_path),
                "source_checkpoint_sha256": sha256_file(ckpt),
                "config_sha256": sha256_file(ROOT / item["config"]),
                "shape": list(pred.shape),
                "spacing": list(spacing),
                "output_sha256": sha256_file(pred_path),
            }
        )
        slice_indices = sample["slice_idx"].cpu().numpy().tolist()
        for local_idx, slice_idx in enumerate(slice_indices):
            slices.append(SlicePrediction(patient_id=patient_id, slice_idx=int(slice_idx), pred=pred[local_idx] > 0, gt=gt[local_idx] > 0, spacing=spacing))
    metrics = patient_volume_macro(slices)
    receipt = {
        "schema": "V19_1_COVID19_SEED1337_FORMAL_TEST_RECEIPT",
        "test_run_id": item["test_run_id"],
        "source_training_run": item["source_run_id"],
        "formal_training_seed": 1337,
        "split_provenance_seed": 42,
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": sha256_file(ckpt),
        "best_iter": best["iter"],
        "best_val_dice": best["dice"],
        "validation_curve_points": len(curve),
        "method_identity": item["method_identity"],
        "mask_token": item["mask_token"],
        "adamix": item["adamix"],
        "protocol_identity": PROTOCOL_IDENTITY,
        "historical_test_set_exposure": True,
        "test_stage_label": "CANONICAL_SEED1337_CONFIRMATORY_FORMAL_TEST",
        "test_patient_sha256": manifest["test_patient_manifest_sha256"],
        "test_all_slices_sha256": manifest["test_all_slices_manifest_sha256"],
        "evaluator_path": "train_COVID19/covid19_volume_evaluator.py",
        "evaluator_sha256": evaluator_sha,
        "inference_mode": {"model_eval": True, "torch_no_grad": True, "tta": False, "threshold_search": False, "ensemble": False},
        "start_time_utc": start,
        "end_time_utc": utc_now(),
        "prediction_paths": prediction_paths,
        "metric_result": metrics,
        "exit_code": 0,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(receipts: list[dict[str, Any]]) -> None:
    rows = []
    for receipt in receipts:
        macro = receipt["metric_result"]["macro"]
        rows.append(
            {
                "test_run_id": receipt["test_run_id"],
                "source_run_id": receipt["source_training_run"],
                "method_identity": receipt["method_identity"],
                "best_val_iter": receipt["best_iter"],
                "best_val_dice": receipt["best_val_dice"],
                "test_dice": macro["dice"],
                "test_iou": macro["iou"],
                "test_hd95_mm": macro["hd95_mm"],
                "test_asd_mm": macro["asd_mm"],
            }
        )
    write_tsv(
        FINAL_REPORT / "COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv",
        rows,
        ["test_run_id", "source_run_id", "method_identity", "best_val_iter", "best_val_dice", "test_dice", "test_iou", "test_hd95_mm", "test_asd_mm"],
    )
    atomic_json(AUDITS / "COVID19_SEED1337_FORMAL_TEST_RECEIPTS.json", {"schema": "V19_1_COVID19_SEED1337_FORMAL_TEST_RECEIPTS", "receipts": receipts})


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_COVID19_SEED1337_FORMAL_TEST")
    torch.manual_seed(1337)
    np.random.seed(1337)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    manifest = manifest_identity()
    evaluator_sha = sha256_file(ROOT / "train_COVID19/covid19_volume_evaluator.py")
    receipts = [run_cell(item, manifest, evaluator_sha) for item in queue_runs()]
    summarize(receipts)
    print(json.dumps({"COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST": "COMPLETE", "test_runs": [r["test_run_id"] for r in receipts]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
