#!/usr/bin/env python3
"""Formal COVID-19 test-once runner for frozen V18.5 best-val checkpoints."""

from __future__ import annotations

import csv
import hashlib
import json
import os
import re
import sys
import time
from dataclasses import asdict
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
TEST_MANIFEST = ROOT / "data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv"
TEST_PATIENTS = ROOT / "data/COVID19_OFFICIAL20/manifests/COVID19_TEST_PATIENTS.tsv"
IDENTITY_JSON = ROOT / "audits/COVID19_FORMAL_RESULT_IDENTITY_FINAL.json"
OUT_ROOT = ROOT / "formal_test/covid19"
FINAL_REPORT = ROOT / "final_report"
AUDITS = ROOT / "audits"

RUNS = [
    {
        "test_run_id": "COVID19_5P_BASELINE_TEST_ONCE",
        "source_run_id": "COVID19_5P_BASELINE_R2",
        "percent": "5P",
        "method_identity": "BASELINE",
        "mask_token": False,
        "adamix": False,
        "expected_best_iter": 11000,
        "expected_best_val_dice": 0.833491,
        "config": "configs/covid19_official20/COVID19_5P_BASELINE_R2.yaml",
        "labeled_num": 120,
    },
    {
        "test_run_id": "COVID19_5P_MASK_TOKEN_ADAMIX_TEST_ONCE",
        "source_run_id": "COVID19_5P_MASK_TOKEN_ADAMIX_R1",
        "percent": "5P",
        "method_identity": "OURS",
        "mask_token": True,
        "adamix": True,
        "expected_best_iter": 3500,
        "expected_best_val_dice": 0.835780,
        "config": "configs/covid19_official20/COVID19_5P_MASK_TOKEN_ADAMIX_R1.yaml",
        "labeled_num": 120,
    },
    {
        "test_run_id": "COVID19_10P_BASELINE_TEST_ONCE",
        "source_run_id": "COVID19_10P_BASELINE_R1",
        "percent": "10P",
        "method_identity": "BASELINE",
        "mask_token": False,
        "adamix": False,
        "expected_best_iter": 16500,
        "expected_best_val_dice": 0.852598,
        "config": "configs/covid19_official20/COVID19_10P_BASELINE_R1.yaml",
        "labeled_num": 241,
    },
    {
        "test_run_id": "COVID19_10P_MASK_TOKEN_ADAMIX_TEST_ONCE",
        "source_run_id": "COVID19_10P_MASK_TOKEN_ADAMIX_R1",
        "percent": "10P",
        "method_identity": "OURS",
        "mask_token": True,
        "adamix": True,
        "expected_best_iter": 25000,
        "expected_best_val_dice": 0.847466,
        "config": "configs/covid19_official20/COVID19_10P_MASK_TOKEN_ADAMIX_R1.yaml",
        "labeled_num": 241,
    },
]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def atomic_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def simple_yaml_bool(text: str, key: str) -> bool | None:
    m = re.search(rf"^\s*{re.escape(key)}\s*:\s*(true|false)\s*$", text, re.M | re.I)
    return None if not m else m.group(1).lower() == "true"


def parse_validation_curve(run_id: str, labeled_num: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    log_path = ROOT / "train_promise12/beta_FFT" / f"PROMISE12_bcp_{run_id}_{labeled_num}" / "pre_train/log.txt"
    text = log_path.read_text(encoding="utf-8", errors="replace")
    curve: list[dict[str, Any]] = []
    for m in re.finditer(r"iteration (\d+) : model1_mean_dice : ([0-9.]+)", text):
        it = int(m.group(1))
        mm = re.search(
            rf"iteration {it} : val_patient_volume_macro_dice : ([0-9.]+), iou : ([0-9.]+), hd95_mm : ([0-9.]+), asd_mm : ([0-9.]+)",
            text[m.end() : m.end() + 700],
        )
        if not mm:
            raise RuntimeError(f"missing COVID validation metrics for {run_id}@{it}")
        curve.append(
            {
                "iter": it,
                "dice": float(mm.group(1)),
                "iou": float(mm.group(2)),
                "hd95_mm": float(mm.group(3)),
                "asd_mm": float(mm.group(4)),
            }
        )
    if not curve:
        raise RuntimeError(f"empty self-train validation curve: {run_id}")
    best = max(curve, key=lambda item: item["dice"])
    return best, curve


def test_patients() -> list[str]:
    with TEST_PATIENTS.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return [row["patient_id"] for row in rows]


def manifest_identity() -> dict[str, Any]:
    patients = test_patients()
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


def checkpoint_path(run_id: str, labeled_num: int) -> Path:
    return (
        ROOT
        / "train_promise12/beta_FFT"
        / f"PROMISE12_bcp_fft_test_seed_{run_id}_{labeled_num}"
        / "self_train/unet_best_model.pth"
    )


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


def save_prediction(path: Path, pred_zyx: np.ndarray, patient_id: str, spacing: tuple[float, float, float]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, prediction=pred_zyx.astype(np.uint8), patient_id=patient_id, spacing=np.asarray(spacing, dtype=np.float32))
    return sha256_file(path)


def evaluate_saved_predictions(prediction_paths: list[dict[str, Any]], dataset: Covid19Official20PatientVolumeDataset) -> dict[str, Any]:
    by_patient = {item["patient_id"]: item for item in prediction_paths}
    slices: list[SlicePrediction] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        patient_id = str(sample["patient_id"])
        pred_payload = np.load(by_patient[patient_id]["path"])
        pred_zyx = pred_payload["prediction"].astype(np.uint8)
        gt_zyx = sample["mask"].cpu().numpy().astype(np.uint8)
        spacing = tuple(float(x) for x in sample["spacing"].cpu().numpy().tolist())
        if pred_zyx.shape != gt_zyx.shape:
            raise RuntimeError(f"PRED_GT_SHAPE_MISMATCH:{patient_id}:{pred_zyx.shape}:{gt_zyx.shape}")
        slice_indices = sample["slice_idx"].cpu().numpy().tolist()
        for local_idx, slice_idx in enumerate(slice_indices):
            slices.append(
                SlicePrediction(
                    patient_id=patient_id,
                    slice_idx=int(slice_idx),
                    pred=pred_zyx[local_idx] > 0,
                    gt=gt_zyx[local_idx] > 0,
                    spacing=spacing,
                )
            )
    return patient_volume_macro(slices)


def run_cell(item: dict[str, Any], checkpoint_sha: str, config_sha: str, evaluator_sha: str, manifest: dict[str, Any]) -> dict[str, Any]:
    test_run_dir = OUT_ROOT / item["test_run_id"]
    receipt_path = test_run_dir / "FORMAL_TEST_RECEIPT.json"
    if receipt_path.exists():
        raise RuntimeError(f"FORMAL_TEST_ONCE_ALREADY_HAS_RECEIPT:{receipt_path}")

    start = utc_now()
    ckpt = checkpoint_path(item["source_run_id"], item["labeled_num"])
    dataset = Covid19Official20PatientVolumeDataset(TEST_MANIFEST)
    model = build_model(item["mask_token"])
    state = torch.load(ckpt, map_location="cpu")
    model.load_state_dict(state)
    model.eval()

    prediction_paths: list[dict[str, Any]] = []
    slices: list[SlicePrediction] = []
    for idx in range(len(dataset)):
        sample = dataset[idx]
        patient_id = str(sample["patient_id"])
        image = sample["image"].float()
        gt = sample["mask"].cpu().numpy().astype(np.uint8)
        spacing = tuple(float(x) for x in sample["spacing"].cpu().numpy().tolist())
        pred = predict_volume(image, model)
        pred_path = test_run_dir / "predictions" / f"{patient_id}_prediction.npz"
        pred_sha = save_prediction(pred_path, pred, patient_id, spacing)
        prediction_paths.append(
            {
                "patient_id": patient_id,
                "path": str(pred_path),
                "source_checkpoint_sha256": checkpoint_sha,
                "config_sha256": config_sha,
                "prediction_source": "model1_argmax_softmax_best_validation_checkpoint",
                "shape": list(pred.shape),
                "spacing": list(spacing),
                "orientation": "RAS",
                "output_sha256": pred_sha,
            }
        )
        slice_indices = sample["slice_idx"].cpu().numpy().tolist()
        for local_idx, slice_idx in enumerate(slice_indices):
            slices.append(
                SlicePrediction(
                    patient_id=patient_id,
                    slice_idx=int(slice_idx),
                    pred=pred[local_idx] > 0,
                    gt=gt[local_idx] > 0,
                    spacing=spacing,
                )
            )

    metrics = patient_volume_macro(slices)
    recomputed = evaluate_saved_predictions(prediction_paths, dataset)
    if json.dumps(metrics, sort_keys=True) != json.dumps(recomputed, sort_keys=True):
        raise RuntimeError(f"COVID19_TEST_METRIC_RECOMPUTATION_MISMATCH:{item['test_run_id']}")

    receipt = {
        "schema": "V18.5_COVID19_FORMAL_TEST_RECEIPT",
        "test_run_id": item["test_run_id"],
        "source_training_run": item["source_run_id"],
        "checkpoint_path": str(ckpt),
        "checkpoint_sha256": checkpoint_sha,
        "best_iter": item["expected_best_iter"],
        "best_val_dice": item["expected_best_val_dice"],
        "method_identity": item["method_identity"],
        "mask_token": item["mask_token"],
        "adamix": item["adamix"],
        "protocol_identity": PROTOCOL_IDENTITY,
        "test_patient_sha256": manifest["test_patient_manifest_sha256"],
        "test_all_slices_sha256": manifest["test_all_slices_manifest_sha256"],
        "evaluator_path": "train_COVID19/covid19_volume_evaluator.py",
        "evaluator_sha256": evaluator_sha,
        "prediction_source": "authoritative_beta_fft_primary_model1_argmax_softmax",
        "inference_mode": {"model_eval": True, "torch_no_grad": True, "tta": False, "threshold_search": False, "ensemble": False},
        "start_time_utc": start,
        "end_time_utc": utc_now(),
        "prediction_paths": prediction_paths,
        "metric_result": metrics,
        "prediction_only_recomputed_metric_result": recomputed,
        "metric_recomputation_match": True,
        "exit_code": 0,
    }
    atomic_json(receipt_path, receipt)
    return receipt


def freeze_checkpoints() -> dict[str, Any]:
    identity = read_json(IDENTITY_JSON)
    if identity.get("COVID19_10P_METHOD_IDENTITY_GATE") != "PASS":
        raise RuntimeError("COVID19_FORMAL_METHOD_IDENTITY_GATE_NOT_PASS")
    manifest = manifest_identity()
    freeze_items = []
    for item in RUNS:
        config_path = ROOT / item["config"]
        config_text = config_path.read_text(encoding="utf-8")
        mask = simple_yaml_bool(config_text, "mask_token_enabled")
        adamix = simple_yaml_bool(config_text, "adamix_enabled")
        if mask != item["mask_token"] or adamix != item["adamix"]:
            raise RuntimeError(f"METHOD_FLAG_FREEZE_MISMATCH:{item['source_run_id']}")
        best, _ = parse_validation_curve(item["source_run_id"], item["labeled_num"])
        if best["iter"] != item["expected_best_iter"] or abs(best["dice"] - item["expected_best_val_dice"]) > 1e-9:
            raise RuntimeError(f"BEST_CHECKPOINT_BINDING_MISMATCH:{item['source_run_id']}:{best}")
        term = read_json(AUDITS / "covid19_formal_terminal" / f"{item['source_run_id']}.json")
        if term.get("COVID19_TERMINAL_VALIDATOR") != "PASS" or term.get("best_iteration") != item["expected_best_iter"]:
            raise RuntimeError(f"TERMINAL_BEST_BINDING_MISMATCH:{item['source_run_id']}")
        ckpt = checkpoint_path(item["source_run_id"], item["labeled_num"])
        if not ckpt.exists():
            raise RuntimeError(f"MISSING_BEST_CHECKPOINT:{ckpt}")
        freeze_items.append(
            {
                **item,
                "checkpoint_path": str(ckpt),
                "checkpoint_size": ckpt.stat().st_size,
                "checkpoint_sha256": sha256_file(ckpt),
                "config_sha256": sha256_file(config_path),
                "terminal_validator": "PASS",
                "protocol_identity": PROTOCOL_IDENTITY,
                "test_checkpoint_selection_source": "BEST_VALIDATION_DICE_ONLY",
            }
        )
    evaluator_sha = sha256_file(ROOT / "train_COVID19/covid19_volume_evaluator.py")
    payload = {
        "schema": "V18.5_COVID19_FORMAL_TEST_CHECKPOINT_FREEZE",
        "created_utc": utc_now(),
        "protocol_identity": PROTOCOL_IDENTITY,
        "global_experiment_seed": 42,
        "test_manifest_identity": manifest,
        "evaluator": {
            "path": "train_COVID19/covid19_volume_evaluator.py",
            "sha256": evaluator_sha,
            "metrics": ["Dice", "IoU", "HD95_mm", "ASD_mm"],
            "aggregation": "PATIENT_VOLUME_MACRO",
        },
        "checkpoints": freeze_items,
    }
    atomic_json(AUDITS / "COVID19_FORMAL_TEST_CHECKPOINT_FREEZE.json", payload)
    return payload


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in fields})


def summarize(receipts: list[dict[str, Any]], freeze: dict[str, Any]) -> None:
    by_test = {r["test_run_id"]: r for r in receipts}
    rows_by_pct = {"5P": [], "10P": []}
    per_patient_rows = []
    for item in RUNS:
        receipt = by_test[item["test_run_id"]]
        macro = receipt["metric_result"]["macro"]
        rows_by_pct[item["percent"]].append(
            {
                "percent": item["percent"],
                "method_identity": item["method_identity"],
                "test_run_id": item["test_run_id"],
                "source_run_id": item["source_run_id"],
                "best_val_iter": item["expected_best_iter"],
                "best_val_dice": item["expected_best_val_dice"],
                "test_dice": macro["dice"],
                "test_iou": macro["iou"],
                "test_hd95_mm": macro["hd95_mm"],
                "test_asd_mm": macro["asd_mm"],
            }
        )
        for patient_id, metrics in receipt["metric_result"]["per_patient"].items():
            per_patient_rows.append(
                {
                    "percent": item["percent"],
                    "method_identity": item["method_identity"],
                    "test_run_id": item["test_run_id"],
                    "source_run_id": item["source_run_id"],
                    "patient_id": patient_id,
                    "dice": metrics["dice"],
                    "iou": metrics["iou"],
                    "hd95_mm": metrics["hd95_mm"],
                    "asd_mm": metrics["asd_mm"],
                }
            )

    for pct, path in [("5P", FINAL_REPORT / "COVID19_5P_FORMAL_TEST.tsv"), ("10P", FINAL_REPORT / "COVID19_10P_FORMAL_TEST.tsv")]:
        write_tsv(
            path,
            rows_by_pct[pct],
            [
                "percent",
                "method_identity",
                "test_run_id",
                "source_run_id",
                "best_val_iter",
                "best_val_dice",
                "test_dice",
                "test_iou",
                "test_hd95_mm",
                "test_asd_mm",
            ],
        )

    vvt_rows = []
    for pct, rows in rows_by_pct.items():
        baseline = next(row for row in rows if row["method_identity"] == "BASELINE")
        ours = next(row for row in rows if row["method_identity"] == "OURS")
        vvt_rows.append(
            {
                "percent": pct,
                "best_val_baseline_dice": baseline["best_val_dice"],
                "best_val_ours_dice": ours["best_val_dice"],
                "best_val_delta_ours_minus_baseline_pp": (ours["best_val_dice"] - baseline["best_val_dice"]) * 100.0,
                "test_baseline_dice": baseline["test_dice"],
                "test_ours_dice": ours["test_dice"],
                "test_delta_ours_minus_baseline_dice_pp": (ours["test_dice"] - baseline["test_dice"]) * 100.0,
                "test_delta_ours_minus_baseline_iou_pp": (ours["test_iou"] - baseline["test_iou"]) * 100.0,
                "test_delta_ours_minus_baseline_hd95_mm": ours["test_hd95_mm"] - baseline["test_hd95_mm"],
                "test_delta_ours_minus_baseline_asd_mm": ours["test_asd_mm"] - baseline["test_asd_mm"],
            }
        )
    write_tsv(
        FINAL_REPORT / "COVID19_VALIDATION_VS_TEST.tsv",
        vvt_rows,
        [
            "percent",
            "best_val_baseline_dice",
            "best_val_ours_dice",
            "best_val_delta_ours_minus_baseline_pp",
            "test_baseline_dice",
            "test_ours_dice",
            "test_delta_ours_minus_baseline_dice_pp",
            "test_delta_ours_minus_baseline_iou_pp",
            "test_delta_ours_minus_baseline_hd95_mm",
            "test_delta_ours_minus_baseline_asd_mm",
        ],
    )
    write_tsv(
        FINAL_REPORT / "COVID19_FORMAL_TEST_PER_PATIENT.tsv",
        per_patient_rows,
        ["percent", "method_identity", "test_run_id", "source_run_id", "patient_id", "dice", "iou", "hd95_mm", "asd_mm"],
    )

    recompute_gate = {
        "schema": "V18.5_COVID19_TEST_METRIC_RECOMPUTATION_GATE",
        "COVID19_TEST_METRIC_RECOMPUTATION_GATE": "PASS",
        "checked_test_runs": [r["test_run_id"] for r in receipts],
        "all_metric_recomputation_match": all(r["metric_recomputation_match"] for r in receipts),
    }
    atomic_json(AUDITS / "COVID19_TEST_METRIC_RECOMPUTATION_GATE.json", recompute_gate)

    identity_gate = {
        "schema": "V18.5_COVID19_FORMAL_TEST_IDENTITY_GATE",
        "COVID19_FORMAL_METHOD_IDENTITY_GATE": "PASS",
        "COVID19_FORMAL_TEST_IDENTITY_GATE": "PASS",
        "source_identity_audit": str(IDENTITY_JSON),
        "formal_test_once": "COMPLETE",
    }
    atomic_json(AUDITS / "COVID19_FORMAL_TEST_IDENTITY_GATE.json", identity_gate)
    atomic_json(AUDITS / "COVID19_FORMAL_TEST_RECEIPTS.json", {"schema": "V18.5_COVID19_FORMAL_TEST_RECEIPTS", "receipts": receipts})

    md = ["# COVID19 Formal Test Summary", "", "`COVID19_FORMAL_TEST_ONCE = COMPLETE`", "`COVID19_FORMAL_EXPERIMENT = COMPLETE`", "`TRAINING_FROZEN = true`", "`TEST_FROZEN = true`", ""]
    for pct in ["5P", "10P"]:
        rows = rows_by_pct[pct]
        baseline = next(row for row in rows if row["method_identity"] == "BASELINE")
        ours = next(row for row in rows if row["method_identity"] == "OURS")
        md.append(f"## COVID-19 {pct} - Formal Test")
        for label, row in [("Baseline", baseline), ("Ours", ours)]:
            md.append(
                f"- {label}: Dice {row['test_dice']*100:.4f}%, IoU {row['test_iou']*100:.4f}%, "
                f"HD95 {row['test_hd95_mm']:.4f} mm, ASD {row['test_asd_mm']:.4f} mm"
            )
        md.append(
            f"- Delta Ours-Baseline: Dice {(ours['test_dice']-baseline['test_dice'])*100:+.4f} pp, "
            f"IoU {(ours['test_iou']-baseline['test_iou'])*100:+.4f} pp, "
            f"HD95 {ours['test_hd95_mm']-baseline['test_hd95_mm']:+.4f} mm, "
            f"ASD {ours['test_asd_mm']-baseline['test_asd_mm']:+.4f} mm"
        )
        md.append("")
    (FINAL_REPORT / "COVID19_FORMAL_TEST_SUMMARY.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def main() -> int:
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA_REQUIRED_FOR_FORMAL_TEST_ONCE")
    torch.manual_seed(42)
    np.random.seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    freeze = freeze_checkpoints()
    manifest = freeze["test_manifest_identity"]
    evaluator_sha = freeze["evaluator"]["sha256"]
    by_run = {item["source_run_id"]: item for item in freeze["checkpoints"]}
    receipts = []
    for item in RUNS:
        frozen = by_run[item["source_run_id"]]
        receipts.append(run_cell(item, frozen["checkpoint_sha256"], frozen["config_sha256"], evaluator_sha, manifest))
    summarize(receipts, freeze)
    print(json.dumps({"COVID19_FORMAL_TEST_ONCE": "COMPLETE", "test_runs": [r["test_run_id"] for r in receipts]}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
