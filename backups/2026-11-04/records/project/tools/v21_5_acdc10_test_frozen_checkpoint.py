#!/usr/bin/env python3
"""Strict TEST-only evaluation for V21.5 ACDC10 frozen checkpoints."""

import argparse
import csv
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "train_ACDC"))
from v14_mask_adamix_components import v14_bcp_net  # noqa: E402

CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_binding(path: Path):
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle, delimiter="\t"))
    return {row["MethodID"]: row for row in rows}


def write_tsv(path: Path, rows, fieldnames):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def safe_metrics(pred, truth):
    pred = pred.astype(bool)
    truth = truth.astype(bool)
    if pred.any() and truth.any():
        intersection = int((pred & truth).sum())
        union = int((pred | truth).sum())
        dice = 2.0 * intersection / (int(pred.sum()) + int(truth.sum()))
        iou = intersection / union if union else 0.0
        return dice, iou, float(metric.binary.hd95(pred, truth)), float(metric.binary.asd(pred, truth))
    if not pred.any() and not truth.any():
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def tensor_finite(state):
    for value in state.values():
        if torch.is_tensor(value) and not torch.isfinite(value).all():
            return False
    return True


def predict_case(model, data_root: Path, case: str, device: str):
    with h5py.File(data_root / "data" / f"{case}.h5", "r") as handle:
        image = handle["image"][:]
        truth = handle["label"][:]

    prediction = np.zeros_like(truth)
    with torch.no_grad():
        for index, source in enumerate(image):
            height, width = source.shape
            resized = zoom(source, (256 / height, 256 / width), order=0)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.softmax(logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
            prediction[index] = zoom(pred, (height / 256, width / 256), order=0)

    rows = []
    for class_id, class_name in CLASS_NAMES.items():
        dice, iou, hd95, asd = safe_metrics(prediction == class_id, truth == class_id)
        rows.append({
            "case": case,
            "class_id": class_id,
            "class_name": class_name,
            "dice": dice,
            "iou": iou,
            "hd95": hd95,
            "asd": asd,
        })
    return rows


def summarize(rows):
    summary = {metric_name: float(np.mean([float(row[metric_name]) for row in rows])) for metric_name in ("dice", "iou", "hd95", "asd")}
    for class_name in ("RV", "MYO", "LV"):
        class_rows = [row for row in rows if row["class_name"] == class_name]
        for metric_name in ("dice", "iou", "hd95", "asd"):
            summary[f"{class_name}_{metric_name}"] = float(np.mean([float(row[metric_name]) for row in class_rows]))
    return summary


def build_dry_payload(args, row, protocol):
    ckpt = ROOT / row["BestCheckpointPath"]
    manifest = Path(protocol["test_manifest"])
    evaluator = Path(__file__).resolve()
    outdir = ROOT / args.output_dir
    actual_ckpt_sha = sha256(ckpt)
    if actual_ckpt_sha != row["BestCheckpointSHA256"]:
        raise RuntimeError(f"checkpoint SHA mismatch for {args.method_id}: {actual_ckpt_sha} != {row['BestCheckpointSHA256']}")
    return {
        "DryRunStatus": "PASS",
        "TEST_ONLY_MODE": True,
        "MethodID": args.method_id,
        "CheckpointPath": row["BestCheckpointPath"],
        "CheckpointSHA": actual_ckpt_sha,
        "ValidationBestDice": row["ValidationBestDice"],
        "ValidationBestIteration": row["ValidationBestIteration"],
        "TestManifest": str(manifest),
        "TestManifestSHA": sha256(manifest),
        "EvaluatorPath": str(evaluator.relative_to(ROOT)),
        "EvaluatorSHA": sha256(evaluator),
        "BoundaryUnit": protocol["boundary_unit"],
        "OutputDir": str(outdir.relative_to(ROOT)),
        "ReceiptPath": str((outdir / "TEST_RECEIPT.json").relative_to(ROOT)),
        "SelectionEndpoint": row["SelectionEndpoint"],
        "CheckpointFrozen": row["CheckpointFrozen"],
        "TrainingEntryStarted": False,
        "OptimizerCreated": False,
        "BackwardCalled": False,
        "TrainingCheckpointWritten": False,
    }


def run_test(args, row, protocol):
    if protocol.get("checkpoint_selection_endpoint") != "VALIDATION_BEST":
        raise RuntimeError("protocol endpoint is not VALIDATION_BEST")
    if row["SelectionEndpoint"] != "VALIDATION_BEST" or row["CheckpointFrozen"] != "true":
        raise RuntimeError(f"unfrozen or non-best checkpoint row: {args.method_id}")

    outdir = ROOT / args.output_dir
    outdir.mkdir(parents=True, exist_ok=True)
    receipt = outdir / "TEST_RECEIPT.json"
    if receipt.exists() and not args.force:
        existing = json.loads(receipt.read_text())
        if existing.get("status") == "TEST_PASS":
            print(json.dumps({"status": "EXISTING_TEST_RECEIPT_REUSED", "method_id": args.method_id, "receipt": str(receipt.relative_to(ROOT))}, indent=2))
            return existing
        raise RuntimeError(f"existing non-pass receipt blocks overwrite: {receipt}")

    ckpt_path = ROOT / row["BestCheckpointPath"]
    if sha256(ckpt_path) != row["BestCheckpointSHA256"]:
        raise RuntimeError(f"checkpoint SHA mismatch before TEST: {ckpt_path}")
    checkpoint = torch.load(str(ckpt_path), map_location="cpu")
    if not tensor_finite(checkpoint["model_1"]):
        raise RuntimeError(f"non-finite tensor in checkpoint: {ckpt_path}")

    device = args.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but not available")
    model = v14_bcp_net(in_chns=1, class_num=4, ema=False, mask_token_enabled=True, mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(checkpoint["model_1"], strict=True)
    model.to(device)
    model.eval()

    data_root = Path(protocol["dataset_root"])
    cases = [line.strip() for line in Path(protocol["test_manifest"]).read_text().splitlines() if line.strip()]
    rows = []
    for case in cases:
        rows.extend(predict_case(model, data_root, case, device))
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    summary = summarize(rows)
    case_fields = ["case", "class_id", "class_name", "dice", "iou", "hd95", "asd"]
    write_tsv(outdir / "case_class_metrics.tsv", rows, case_fields)
    class_rows = []
    for class_name in ("RV", "MYO", "LV"):
        class_rows.append({
            "class_name": class_name,
            "dice": summary[f"{class_name}_dice"],
            "iou": summary[f"{class_name}_iou"],
            "hd95": summary[f"{class_name}_hd95"],
            "asd": summary[f"{class_name}_asd"],
        })
    write_tsv(outdir / "class_metrics.tsv", class_rows, ["class_name", "dice", "iou", "hd95", "asd"])
    (outdir / "summary_metrics.json").write_text(json.dumps(summary, indent=2) + "\n")

    payload = {
        "status": "TEST_PASS",
        "method_id": args.method_id,
        "method_name": row["MethodName"],
        "created_at": datetime.now(timezone.utc).isoformat(),
        "TEST_ONLY_MODE": True,
        "TEST_SELECTION_EXPOSURE": True,
        "BLIND_TEST_CLAIM_ALLOWED": False,
        "UNTOUCHED_TEST_CLAIM_ALLOWED": False,
        "FIRST_EVER_TEST_CLAIM_ALLOWED": False,
        "SelectionEndpoint": "VALIDATION_BEST",
        "CheckpointPath": row["BestCheckpointPath"],
        "CheckpointSHA": row["BestCheckpointSHA256"],
        "CheckpointGlobalStep": int(checkpoint.get("global_step", -1)),
        "ValidationBestDice": float(row["ValidationBestDice"]),
        "ValidationBestIteration": int(row["ValidationBestIteration"]),
        "TestManifest": protocol["test_manifest"],
        "TestManifestSHA": sha256(Path(protocol["test_manifest"])),
        "EvaluatorPath": str(Path(__file__).resolve().relative_to(ROOT)),
        "EvaluatorSHA": sha256(Path(__file__).resolve()),
        "BoundaryUnit": protocol["boundary_unit"],
        "Metrics": summary,
        "CaseClassMetrics": str((outdir / "case_class_metrics.tsv").relative_to(ROOT)),
        "ClassMetrics": str((outdir / "class_metrics.tsv").relative_to(ROOT)),
        "TrainingEntryStarted": False,
        "OptimizerCreated": False,
        "OptimizerStepCalled": False,
        "BackwardCalled": False,
        "TrainingCheckpointWritten": False,
        "RetrainingAfterTest": False,
        "ParameterChangeAfterTest": False,
    }
    receipt.write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--method-id", required=True, choices=["OURS", "X01", "X02", "X03", "X04", "X05", "X06"])
    parser.add_argument("--binding", required=True)
    parser.add_argument("--protocol", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mode", choices=["dry-run", "run"], default="run")
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--force", action="store_true")
    args = parser.parse_args()

    os.environ.setdefault("PYTHONHASHSEED", "0")
    binding = read_binding(ROOT / args.binding)
    protocol = json.loads((ROOT / args.protocol).read_text())
    row = binding[args.method_id]
    payload = build_dry_payload(args, row, protocol)
    if args.mode == "dry-run":
        print(json.dumps(payload, indent=2))
        return
    run_test(args, row, protocol)


if __name__ == "__main__":
    main()
