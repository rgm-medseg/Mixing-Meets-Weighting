#!/usr/bin/env python3
"""V19.4 test-only evaluation for the frozen ACDC 5% reproduced baseline.

This script performs no training and never chooses a checkpoint. It only loads
the already frozen validation-best checkpoint and evaluates it on the ACDC
test.list split once, saving raw predictions and metric receipts.
"""

import csv
import hashlib
import json
import sys
from datetime import datetime
from pathlib import Path

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_ACDC"))
from networks.net_factory import BCP_net  # noqa: E402

DATA = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC")
RUN = ROOT / "runs/acdc_5pct_2d_matched_control_seed1337_valonly_20260628_094308"
CHECKPOINT = RUN / "checkpoints/best.ckpt"
OUT = ROOT / "formal_test/acdc_v19_4/ACDC_5P_BASELINE_CONFIRMATORY_TEST"
PRED_DIR = OUT / "predictions_npz"
CASE_TSV = OUT / "case_class_metrics.tsv"
CLASS_TSV = OUT / "class_metrics.tsv"
AGG_TSV = OUT / "aggregate_metrics.tsv"
RECEIPT = OUT / "FORMAL_TEST_RECEIPT.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def safe_metrics(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    pred_sum = int(pred.sum())
    truth_sum = int(truth.sum())
    if pred_sum and truth_sum:
        inter = int((pred & truth).sum())
        union = int((pred | truth).sum())
        return (
            2.0 * inter / (pred_sum + truth_sum),
            inter / union if union else 0.0,
            float(metric.binary.hd95(pred, truth)),
            float(metric.binary.asd(pred, truth)),
        )
    if pred_sum == 0 and truth_sum == 0:
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def load_cases(split_name: str) -> list[str]:
    return [line.strip() for line in (DATA / f"{split_name}.list").read_text().splitlines() if line.strip()]


def infer_case(model: torch.nn.Module, case_id: str) -> tuple[np.ndarray, np.ndarray, list[dict[str, object]]]:
    with h5py.File(DATA / "data" / f"{case_id}.h5", "r") as handle:
        image = handle["image"][:]
        truth = handle["label"][:]
    prediction = np.zeros_like(truth)
    model.eval()
    with torch.no_grad():
        for index, source in enumerate(image):
            height, width = source.shape
            resized = zoom(source, (256 / height, 256 / width), order=0)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().cuda()
            output = model(tensor).argmax(dim=1).squeeze(0).cpu().numpy()
            prediction[index] = zoom(output, (height / 256, width / 256), order=0)

    rows: list[dict[str, object]] = []
    for class_id, class_name in ((1, "RV"), (2, "MYO"), (3, "LV")):
        dice, iou, hd95, asd = safe_metrics(prediction == class_id, truth == class_id)
        rows.append(
            {
                "case_id": case_id,
                "class_id": class_id,
                "class_name": class_name,
                "dice": dice,
                "iou": iou,
                "hd95": hd95,
                "asd": asd,
            }
        )
    return prediction, truth, rows


def mean(rows: list[dict[str, object]], key: str) -> float:
    return float(np.mean([float(row[key]) for row in rows]))


def main() -> None:
    if RECEIPT.exists():
        raise SystemExit(f"Refusing to rerun: receipt already exists at {RECEIPT}")
    OUT.mkdir(parents=True, exist_ok=True)
    PRED_DIR.mkdir(parents=True, exist_ok=True)

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    if "model_1" not in checkpoint:
        raise KeyError(f"Expected model_1 in {CHECKPOINT}")
    model = BCP_net(in_chns=1, class_num=4, ema=False)
    model.load_state_dict(checkpoint["model_1"], strict=True)
    model.cuda()

    all_rows: list[dict[str, object]] = []
    for case_id in load_cases("test"):
        prediction, truth, rows = infer_case(model, case_id)
        np.savez_compressed(PRED_DIR / f"{case_id}.npz", prediction=prediction, label=truth)
        all_rows.extend(rows)

    with CASE_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(all_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(all_rows)

    class_rows: list[dict[str, object]] = []
    for class_id, class_name in ((1, "RV"), (2, "MYO"), (3, "LV")):
        subset = [row for row in all_rows if row["class_id"] == class_id]
        class_rows.append(
            {
                "class_id": class_id,
                "class_name": class_name,
                "dice": mean(subset, "dice"),
                "iou": mean(subset, "iou"),
                "hd95": mean(subset, "hd95"),
                "asd": mean(subset, "asd"),
            }
        )
    with CLASS_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(class_rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(class_rows)

    aggregate = {
        "method_id": "ACDC_5P_BASELINE",
        "dataset": "ACDC",
        "ratio": "5%",
        "identity": "beta-FFT reproduced baseline",
        "mask_token": "OFF",
        "adamix": "OFF",
        "selection_endpoint": "BEST_VAL",
        "reporting_endpoint": "ACDC_CONFIRMATORY_TEST",
        "dice": mean(all_rows, "dice"),
        "iou": mean(all_rows, "iou"),
        "hd95": mean(all_rows, "hd95"),
        "asd": mean(all_rows, "asd"),
        "RV_Dice": class_rows[0]["dice"],
        "MYO_Dice": class_rows[1]["dice"],
        "LV_Dice": class_rows[2]["dice"],
        "boundary_unit": "pixel-unit",
    }
    with AGG_TSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(aggregate), delimiter="\t")
        writer.writeheader()
        writer.writerow(aggregate)

    receipt = {
        "status": "ACDC_5P_BASELINE_CONFIRMATORY_TEST_COMPLETE",
        "created_at": datetime.now().astimezone().isoformat(),
        "training_executed": False,
        "checkpoint_selection_endpoint": "BEST_VAL",
        "test_used_for_checkpoint_selection": False,
        "run_id": str(RUN.relative_to(ROOT)),
        "checkpoint_path": str(CHECKPOINT.relative_to(ROOT)),
        "checkpoint_sha256": sha256(CHECKPOINT),
        "checkpoint_global_iteration": int(checkpoint.get("global_step", -1)),
        "checkpoint_epoch": int(checkpoint.get("epoch", -1)),
        "branch": "model_1",
        "data_root": str(DATA),
        "test_list": str((DATA / "test.list")),
        "test_list_sha256": sha256(DATA / "test.list"),
        "evaluator": str(Path(__file__).relative_to(ROOT)),
        "evaluator_sha256": sha256(Path(__file__)),
        "prediction_directory": str(PRED_DIR.relative_to(ROOT)),
        "case_class_metrics_tsv": str(CASE_TSV.relative_to(ROOT)),
        "class_metrics_tsv": str(CLASS_TSV.relative_to(ROOT)),
        "aggregate_metrics_tsv": str(AGG_TSV.relative_to(ROOT)),
        "aggregate": aggregate,
    }
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt, indent=2))


if __name__ == "__main__":
    main()
