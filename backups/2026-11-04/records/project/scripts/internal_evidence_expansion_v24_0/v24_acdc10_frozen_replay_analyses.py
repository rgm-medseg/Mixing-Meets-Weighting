#!/usr/bin/env python3
"""V24 ACDC10 frozen-checkpoint replay analyses.

Authoritative script for:
- A6 Boundary F1 at 1/2/3 px on ACDC10 TEST.
- A9 Corruption robustness on ACDC10 TEST.

Modes:
- dry-run: validate frozen identities and output-collision gates without
  reading TEST manifest contents or running inference.
- smoke: run a one-case engineering canary and write engineering-only receipts.
- run: run the full frozen analysis. No training, checkpoint selection, tuning,
  or source/config edits are performed.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import os
import random
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, distance_transform_edt, gaussian_filter, zoom

ROOT = Path(__file__).resolve().parents[2]

AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
TABLE_ROOT = RESULT_ROOT / "tables"
FIGURE_ROOT = RESULT_ROOT / "figures"
RUNTIME_ROOT = ROOT / "runtime/internal_evidence_expansion_v24_0/frozen_replay"

CLASS_NAMES = {1: "RV", 2: "MYO", 3: "LV"}
TOLERANCES = [1, 2, 3]

METHODS = {
    "ACDC10_ARM_000": {
        "display": "BetaFFT",
        "checkpoint": ROOT / "runs/beta_fft_acdc_official_full_seed42_fresh_ckpt/checkpoints/best.ckpt",
        "checkpoint_sha256": "825da2e9a944c930fa54cb9b7abc843345e4a7cde851d34c390739acaa0d5dbf",
        "mask_token_enabled": False,
        "source_receipt": ROOT / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/TEST_REPLAY_RECEIPT.json",
        "clean_metrics": ROOT / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/case_class_metrics.tsv",
    },
    "ACDC10_ARM_110": {
        "display": "MaskTokenAdaMix",
        "checkpoint": ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/checkpoints/best.ckpt",
        "checkpoint_sha256": "f10f69989883a423117a53695311b47e01dc65fbc27fbce2d0c666d6fd733514",
        "mask_token_enabled": True,
        "source_receipt": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/OURS/TEST_RECEIPT.json",
        "clean_metrics": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/OURS/case_class_metrics.tsv",
    },
    "ACDC10_ARM_111": {
        "display": "Ours",
        "checkpoint": ROOT
        / "runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt",
        "checkpoint_sha256": "c447a0a6d643405627cebf4718424b7d102fffd677b23f10ab5ef1e5227c3086",
        "mask_token_enabled": True,
        "source_receipt": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/X03/TEST_RECEIPT.json",
        "clean_metrics": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/X03/case_class_metrics.tsv",
    },
}

DATASET_ROOT = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC")
TEST_MANIFEST = DATASET_ROOT / "test.list"
EXPECTED_TEST_MANIFEST_SHA = "98161bdfaad072bf5713d7345751ab5a2bf3e5468b8edb03fd9ebfedc4503671"
BOUNDARY_SPEC = AUDIT_ROOT / "V24_0_BOUNDARY_SPEC.json"
CORRUPTION_SPEC = AUDIT_ROOT / "V24_0_CORRUPTION_SPEC.json"


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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fields is None:
        path.write_text("")
        return
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_md_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_tex_table(path: Path, rows: list[dict[str, Any]], columns: list[str]) -> None:
    lines = [f"\\begin{{tabular}}{{{'l' * len(columns)}}}", "\\toprule", " & ".join(columns) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(str(row.get(c, "")) for c in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def f12(value: float) -> str:
    return f"{value:.12f}"


def dry_validate() -> dict[str, Any]:
    checks = []
    failures = []
    for method_id, spec in METHODS.items():
        checkpoint_exists = spec["checkpoint"].exists()
        checkpoint_sha = sha256(spec["checkpoint"]) if checkpoint_exists else ""
        checkpoint_ok = checkpoint_exists and checkpoint_sha == spec["checkpoint_sha256"]
        receipt_exists = spec["source_receipt"].exists()
        clean_metrics_exists = spec["clean_metrics"].exists()
        if not (checkpoint_ok and receipt_exists and clean_metrics_exists):
            failures.append(method_id)
        checks.append(
            {
                "MethodID": method_id,
                "Checkpoint": rel(spec["checkpoint"]),
                "CheckpointExists": checkpoint_exists,
                "CheckpointSHA256": checkpoint_sha,
                "CheckpointSHAMatch": checkpoint_ok,
                "Receipt": rel(spec["source_receipt"]),
                "ReceiptExists": receipt_exists,
                "CleanMetrics": rel(spec["clean_metrics"]),
                "CleanMetricsExists": clean_metrics_exists,
            }
        )
    spec_checks = {
        "BoundarySpecExists": BOUNDARY_SPEC.exists(),
        "BoundarySpecSHA256": sha256(BOUNDARY_SPEC) if BOUNDARY_SPEC.exists() else "",
        "CorruptionSpecExists": CORRUPTION_SPEC.exists(),
        "CorruptionSpecSHA256": sha256(CORRUPTION_SPEC) if CORRUPTION_SPEC.exists() else "",
        "DatasetRootExists": DATASET_ROOT.is_dir(),
        "TestManifestPathExists": TEST_MANIFEST.exists(),
        "TestManifestContentRead": False,
        "ExpectedTestManifestSHA256": EXPECTED_TEST_MANIFEST_SHA,
    }
    if not (BOUNDARY_SPEC.exists() and CORRUPTION_SPEC.exists() and DATASET_ROOT.is_dir() and TEST_MANIFEST.exists()):
        failures.append("SPEC_OR_DATASET_ROOT")
    payload = {
        "status": "PASS" if not failures else "FAIL",
        "created_at": now(),
        "mode": "dry-run",
        "training_started": False,
        "inference_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_content_read": False,
        "test_manifest_sha_recomputed": False,
        "method_checks": checks,
        "spec_checks": spec_checks,
        "failures": failures,
    }
    write_json(AUDIT_ROOT / "V24_0_ACDC10_FROZEN_REPLAY_ANALYSIS_DRY_RUN_RECEIPT.json", payload)
    return payload


def read_cases(limit: int | None = None) -> list[str]:
    if sha256(TEST_MANIFEST) != EXPECTED_TEST_MANIFEST_SHA:
        raise SystemExit("ACDC10_TEST_MANIFEST_SHA_MISMATCH")
    cases = [line.strip() for line in TEST_MANIFEST.read_text().splitlines() if line.strip()]
    return cases[:limit] if limit else cases


def load_model(method_id: str, device: str):
    import torch

    sys.path.insert(0, str(ROOT / "train_ACDC"))
    from v14_mask_adamix_components import v14_bcp_net  # noqa: E402

    spec = METHODS[method_id]
    if sha256(spec["checkpoint"]) != spec["checkpoint_sha256"]:
        raise SystemExit(f"CHECKPOINT_SHA_MISMATCH:{method_id}")
    checkpoint = torch.load(str(spec["checkpoint"]), map_location="cpu", weights_only=False)
    model = v14_bcp_net(
        in_chns=1,
        class_num=4,
        ema=False,
        mask_token_enabled=bool(spec["mask_token_enabled"]),
        mask_token_num=4,
        mask_token_dim=16,
    )
    model.load_state_dict(checkpoint["model_1"], strict=True)
    model.to(device)
    model.eval()
    return model


def predict_case(model, case: str, device: str, corruption: dict[str, Any] | None = None) -> tuple[np.ndarray, np.ndarray]:
    import h5py
    import torch

    with h5py.File(DATASET_ROOT / "data" / f"{case}.h5", "r") as handle:
        image = handle["image"][:]
        truth = handle["label"][:]
    pred_volume = np.zeros_like(truth)
    with torch.no_grad():
        for index, source in enumerate(image):
            source = apply_corruption(source.astype(np.float32), corruption, case, index)
            height, width = source.shape
            resized = zoom(source, (256 / height, 256 / width), order=0)
            tensor = torch.from_numpy(resized).unsqueeze(0).unsqueeze(0).float().to(device)
            logits = model(tensor)
            if isinstance(logits, (tuple, list)):
                logits = logits[0]
            pred = torch.softmax(logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy()
            pred_volume[index] = zoom(pred, (height / 256, width / 256), order=0)
    return pred_volume, truth


def stable_seed(case: str, slice_index: int, corruption: dict[str, Any]) -> int:
    text = json.dumps(corruption, sort_keys=True) + f":{case}:{slice_index}:2026"
    return int(hashlib.sha256(text.encode("utf-8")).hexdigest()[:8], 16)


def apply_corruption(image: np.ndarray, corruption: dict[str, Any] | None, case: str, slice_index: int) -> np.ndarray:
    if corruption is None:
        return image
    kind = corruption["type"]
    value = corruption["value"]
    if kind == "gaussian_noise":
        rng = np.random.default_rng(stable_seed(case, slice_index, corruption))
        return (image + rng.normal(0.0, float(value), size=image.shape)).astype(np.float32)
    if kind == "gaussian_blur":
        return gaussian_filter(image, sigma=float(value)).astype(np.float32)
    if kind == "gamma_intensity_shift":
        lo = float(np.min(image))
        hi = float(np.max(image))
        if hi - lo <= 1e-8:
            return image
        norm = np.clip((image - lo) / (hi - lo), 0.0, 1.0)
        return (np.power(norm, float(value)) * (hi - lo) + lo).astype(np.float32)
    raise SystemExit(f"UNKNOWN_CORRUPTION:{kind}")


def surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    eroded = binary_erosion(mask)
    return mask ^ eroded


def boundary_f1(pred: np.ndarray, truth: np.ndarray, tolerance: int) -> float:
    pred_s = surface(pred)
    truth_s = surface(truth)
    if not pred_s.any() and not truth_s.any():
        return 1.0
    if not pred_s.any() or not truth_s.any():
        return 0.0
    pred_match = pred_s & binary_dilation(truth_s, iterations=tolerance)
    truth_match = truth_s & binary_dilation(pred_s, iterations=tolerance)
    precision = float(pred_match.sum()) / float(pred_s.sum())
    recall = float(truth_match.sum()) / float(truth_s.sum())
    if precision + recall == 0:
        return 0.0
    return 2.0 * precision * recall / (precision + recall)


def segmentation_metrics(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    from medpy import metric

    pred = pred.astype(bool)
    truth = truth.astype(bool)
    if pred.any() and truth.any():
        inter = int((pred & truth).sum())
        union = int((pred | truth).sum())
        return {
            "dice": 2.0 * inter / (int(pred.sum()) + int(truth.sum())),
            "iou": inter / union if union else 0.0,
            "hd95": float(metric.binary.hd95(pred, truth)),
            "asd": float(metric.binary.asd(pred, truth)),
        }
    if not pred.any() and not truth.any():
        return {"dice": 0.0, "iou": 0.0, "hd95": 0.0, "asd": 0.0}
    penalty = float(np.linalg.norm(pred.shape))
    return {"dice": 0.0, "iou": 0.0, "hd95": penalty, "asd": penalty}


def corruption_grid() -> list[dict[str, Any]]:
    spec = json.loads(CORRUPTION_SPEC.read_text())
    out = [{"type": "clean", "value": 0.0, "label": "clean"}]
    for item in spec["corruptions"]:
        kind = item["type"]
        values = item.get("sigma_normalized") or item.get("sigma_pixels") or item.get("gamma")
        for value in values:
            out.append({"type": kind, "value": float(value), "label": f"{kind}_{value}"})
    return out


def run_boundary(cases: list[str], device: str, smoke: bool) -> dict[str, Any]:
    import torch

    rows = []
    for method_id in ["ACDC10_ARM_000", "ACDC10_ARM_110", "ACDC10_ARM_111"]:
        model = load_model(method_id, device)
        for case in cases:
            pred, truth = predict_case(model, case, device)
            for class_id, class_name in CLASS_NAMES.items():
                for tol in TOLERANCES:
                    rows.append(
                        {
                            "MethodID": method_id,
                            "MethodDisplay": METHODS[method_id]["display"],
                            "CaseID": case,
                            "ClassID": class_id,
                            "ClassName": class_name,
                            "TolerancePx": tol,
                            "BoundaryF1": f12(boundary_f1(pred == class_id, truth == class_id, tol)),
                            "BoundaryUnit": "pixel",
                            "EngineeringCanaryOnly": smoke,
                        }
                    )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    suffix = "SMOKE" if smoke else "FULL_PRECISION"
    out = RESULT_ROOT / f"ACDC10_BOUNDARY_F1_{suffix}.tsv"
    write_tsv(out, rows)
    summary_rows = summarize_boundary(rows)
    summary_out = RESULT_ROOT / f"ACDC10_BOUNDARY_F1_{suffix}_SUMMARY.tsv"
    write_tsv(summary_out, summary_rows)
    if not smoke:
        write_md_table(TABLE_ROOT / "TABLE_S_ACDC10_BOUNDARY_F1.md", summary_rows, ["MethodID", "ClassName", "TolerancePx", "MeanBoundaryF1"])
        write_tex_table(TABLE_ROOT / "TABLE_S_ACDC10_BOUNDARY_F1.tex", summary_rows, ["MethodID", "ClassName", "TolerancePx", "MeanBoundaryF1"])
    return {"rows": rel(out), "rows_sha256": sha256(out), "summary": rel(summary_out), "summary_sha256": sha256(summary_out)}


def summarize_boundary(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    groups: dict[tuple[str, str, int], list[float]] = {}
    for row in rows:
        groups.setdefault((row["MethodID"], row["ClassName"], int(row["TolerancePx"])), []).append(float(row["BoundaryF1"]))
    out = []
    for (method, class_name, tol), values in sorted(groups.items()):
        out.append(
            {
                "MethodID": method,
                "ClassName": class_name,
                "TolerancePx": tol,
                "MeanBoundaryF1": f12(float(np.mean(values))),
                "CaseCount": len(values),
                "BoundaryUnit": "pixel",
            }
        )
    mean_groups: dict[tuple[str, int], list[float]] = {}
    for row in out:
        mean_groups.setdefault((row["MethodID"], int(row["TolerancePx"])), []).append(float(row["MeanBoundaryF1"]))
    for (method, tol), values in sorted(mean_groups.items()):
        out.append(
            {
                "MethodID": method,
                "ClassName": "Mean",
                "TolerancePx": tol,
                "MeanBoundaryF1": f12(float(np.mean(values))),
                "CaseCount": len(values),
                "BoundaryUnit": "pixel",
            }
        )
    return out


def run_corruption(cases: list[str], device: str, smoke: bool) -> dict[str, Any]:
    import torch

    rows = []
    clean_lookup = load_clean_lookup()
    for method_id in ["ACDC10_ARM_000", "ACDC10_ARM_111"]:
        model = load_model(method_id, device)
        for corruption in corruption_grid():
            actual_corruption = None if corruption["type"] == "clean" else corruption
            case_metric_rows = []
            for case in cases:
                pred, truth = predict_case(model, case, device, corruption=actual_corruption)
                for class_id, class_name in CLASS_NAMES.items():
                    metrics = segmentation_metrics(pred == class_id, truth == class_id)
                    case_metric_rows.append({"case": case, "class_name": class_name, **metrics})
            aggregate = {key: float(np.mean([r[key] for r in case_metric_rows])) for key in ["dice", "iou", "hd95", "asd"]}
            clean = clean_lookup[method_id]
            rows.append(
                {
                    "MethodID": method_id,
                    "MethodDisplay": METHODS[method_id]["display"],
                    "CorruptionType": corruption["type"],
                    "CorruptionValue": corruption["value"],
                    "CorruptionLabel": corruption["label"],
                    "Dice": f12(aggregate["dice"]),
                    "IoU": f12(aggregate["iou"]),
                    "HD95": f12(aggregate["hd95"]),
                    "ASD": f12(aggregate["asd"]),
                    "CleanDiceReference": f12(clean["dice"]),
                    "CleanIoUReference": f12(clean["iou"]),
                    "CleanHD95Reference": f12(clean["hd95"]),
                    "CleanASDReference": f12(clean["asd"]),
                    "DiceRetention": f12(aggregate["dice"] / clean["dice"] if clean["dice"] else math.nan),
                    "IoURetention": f12(aggregate["iou"] / clean["iou"] if clean["iou"] else math.nan),
                    "HD95RelativeDegradation": f12(aggregate["hd95"] / clean["hd95"] if clean["hd95"] else math.nan),
                    "ASDRelativeDegradation": f12(aggregate["asd"] / clean["asd"] if clean["asd"] else math.nan),
                    "CaseCount": len(cases),
                    "EngineeringCanaryOnly": smoke,
                }
            )
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    suffix = "SMOKE" if smoke else "FULL_PRECISION"
    out = RESULT_ROOT / f"ACDC10_CORRUPTION_ROBUSTNESS_{suffix}.tsv"
    write_tsv(out, rows)
    if not smoke:
        write_md_table(TABLE_ROOT / "TABLE_S_ACDC10_CORRUPTION.md", rows, ["MethodID", "CorruptionLabel", "Dice", "IoU", "HD95", "ASD", "DiceRetention"])
        write_tex_table(TABLE_ROOT / "TABLE_S_ACDC10_CORRUPTION.tex", rows, ["MethodID", "CorruptionLabel", "Dice", "IoU", "HD95", "ASD", "DiceRetention"])
    return {"rows": rel(out), "rows_sha256": sha256(out)}


def load_clean_lookup() -> dict[str, dict[str, float]]:
    out = {}
    for method_id in ["ACDC10_ARM_000", "ACDC10_ARM_111"]:
        rows = read_tsv(METHODS[method_id]["clean_metrics"])
        out[method_id] = {metric: float(np.mean([float(r[metric]) for r in rows])) for metric in ["dice", "iou", "hd95", "asd"]}
    return out


def run_mode(args: argparse.Namespace) -> dict[str, Any]:
    if args.mode == "dry-run":
        return dry_validate()
    import torch

    if args.device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUESTED_BUT_NOT_AVAILABLE")
    cases = read_cases(limit=1 if args.mode == "smoke" else None)
    payload: dict[str, Any] = {
        "status": "PASS",
        "created_at": now(),
        "mode": args.mode,
        "analysis": args.analysis,
        "training_started": False,
        "checkpoint_selection_started": False,
        "checkpoint_reselection_after_test": False,
        "test_manifest_content_read": True,
        "TEST_DATA_READ_FOR_PRE_REGISTERED_EVALUATION": True,
        "TEST_USED_FOR_TUNING": False,
        "TEST_USED_FOR_SELECTION": False,
        "test_manifest_sha256": sha256(TEST_MANIFEST),
        "case_count": len(cases),
        "engineering_canary_only": args.mode == "smoke",
        "scientific_result": args.mode == "run",
        "paper_result_eligible": args.mode == "run",
    }
    if args.analysis in {"boundary", "all"}:
        payload["boundary_f1"] = run_boundary(cases, args.device, smoke=args.mode == "smoke")
    if args.analysis in {"corruption", "all"}:
        payload["corruption_robustness"] = run_corruption(cases, args.device, smoke=args.mode == "smoke")
    if args.mode == "run":
        payload["postprocess"] = postprocess_full_outputs()
    receipt = AUDIT_ROOT / ("ACDC10_FROZEN_REPLAY_SMOKE_RECEIPT.json" if args.mode == "smoke" else "ACDC10_FROZEN_REPLAY_ANALYSIS_RECEIPT.json")
    write_json(receipt, payload)
    return payload


def postprocess_full_outputs() -> dict[str, Any]:
    boundary_rows_path = RESULT_ROOT / "ACDC10_BOUNDARY_F1_FULL_PRECISION.tsv"
    boundary_summary_path = RESULT_ROOT / "ACDC10_BOUNDARY_F1_FULL_PRECISION_SUMMARY.tsv"
    corruption_rows_path = RESULT_ROOT / "ACDC10_CORRUPTION_ROBUSTNESS_FULL_PRECISION.tsv"
    canonical_corruption_path = RESULT_ROOT / "ACDC10_CORRUPTION_ROBUSTNESS.tsv"
    if not boundary_summary_path.exists() or not corruption_rows_path.exists():
        raise SystemExit("FULL_OUTPUTS_MISSING_FOR_POSTPROCESS")

    boundary_summary = read_tsv(boundary_summary_path)
    corruption_rows = read_tsv(corruption_rows_path)
    write_tsv(canonical_corruption_path, corruption_rows)
    write_md_table(TABLE_ROOT / "TABLE_S_ACDC10_BOUNDARY_F1.md", boundary_summary, ["MethodID", "ClassName", "TolerancePx", "MeanBoundaryF1"])
    write_tex_table(TABLE_ROOT / "TABLE_S_ACDC10_BOUNDARY_F1.tex", boundary_summary, ["MethodID", "ClassName", "TolerancePx", "MeanBoundaryF1"])
    write_md_table(TABLE_ROOT / "TABLE_S_ACDC10_CORRUPTION.md", corruption_rows, ["MethodID", "CorruptionLabel", "Dice", "IoU", "HD95", "ASD", "DiceRetention"])
    write_tex_table(TABLE_ROOT / "TABLE_S_ACDC10_CORRUPTION.tex", corruption_rows, ["MethodID", "CorruptionLabel", "Dice", "IoU", "HD95", "ASD", "DiceRetention"])

    figures = []
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        mean_rows = [r for r in boundary_summary if r["ClassName"] == "Mean"]
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        for method in sorted({r["MethodID"] for r in mean_rows}):
            rows = sorted([r for r in mean_rows if r["MethodID"] == method], key=lambda r: int(r["TolerancePx"]))
            ax.plot([int(r["TolerancePx"]) for r in rows], [float(r["MeanBoundaryF1"]) for r in rows], marker="o", label=method)
        ax.set_xlabel("Tolerance (px)")
        ax.set_ylabel("Mean Boundary F1")
        ax.set_xticks(TOLERANCES)
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            path = FIGURE_ROOT / f"ACDC10_BOUNDARY_F1_TOLERANCE.{suffix}"
            fig.savefig(path)
            figures.append(rel(path))
        plt.close(fig)

        labels = [r["CorruptionLabel"] for r in corruption_rows if r["MethodID"] == "ACDC10_ARM_000"]
        x = np.arange(len(labels))
        width = 0.38
        fig, ax = plt.subplots(figsize=(9.5, 4.5))
        for offset, method in [(-width / 2, "ACDC10_ARM_000"), (width / 2, "ACDC10_ARM_111")]:
            rows = [r for r in corruption_rows if r["MethodID"] == method]
            ax.bar(x + offset, [float(r["DiceRetention"]) for r in rows], width=width, label=method)
        ax.axhline(1.0, color="black", linewidth=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Dice retention")
        ax.legend(fontsize=8)
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            path = FIGURE_ROOT / f"ACDC10_CORRUPTION_DICE_RETENTION.{suffix}"
            fig.savefig(path)
            figures.append(rel(path))
        plt.close(fig)

        fig, axes = plt.subplots(2, 1, figsize=(9.5, 6.0), sharex=True)
        for ax, key, ylabel in [
            (axes[0], "HD95RelativeDegradation", "HD95 relative degradation"),
            (axes[1], "ASDRelativeDegradation", "ASD relative degradation"),
        ]:
            for method in ["ACDC10_ARM_000", "ACDC10_ARM_111"]:
                rows = [r for r in corruption_rows if r["MethodID"] == method]
                ax.plot(x, [float(r[key]) for r in rows], marker="o", label=method)
            ax.axhline(1.0, color="black", linewidth=0.8)
            ax.set_ylabel(ylabel)
            ax.legend(fontsize=8)
        axes[1].set_xticks(x)
        axes[1].set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            path = FIGURE_ROOT / f"ACDC10_CORRUPTION_BOUNDARY_DEGRADATION.{suffix}"
            fig.savefig(path)
            figures.append(rel(path))
        plt.close(fig)
    except Exception as exc:  # pragma: no cover
        figures.append(f"FIGURE_GENERATION_FAILED:{type(exc).__name__}:{exc}")

    payload = {
        "status": "PASS",
        "created_at": now(),
        "training_started": False,
        "inference_started": False,
        "checkpoint_replay_started": False,
        "postprocess_only": True,
        "boundary_rows": rel(boundary_rows_path),
        "boundary_rows_sha256": sha256(boundary_rows_path),
        "boundary_summary": rel(boundary_summary_path),
        "boundary_summary_sha256": sha256(boundary_summary_path),
        "corruption_rows": rel(canonical_corruption_path),
        "corruption_rows_sha256": sha256(canonical_corruption_path),
        "figures": figures,
        "tables": [
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_BOUNDARY_F1.md",
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_BOUNDARY_F1.tex",
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_CORRUPTION.md",
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_CORRUPTION.tex",
        ],
    }
    write_json(AUDIT_ROOT / "ACDC10_FROZEN_REPLAY_POSTPROCESS_RECEIPT.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "smoke", "run", "postprocess"], required=True)
    parser.add_argument("--analysis", choices=["boundary", "corruption", "all"], default="all")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    payload = postprocess_full_outputs() if args.mode == "postprocess" else run_mode(args)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
