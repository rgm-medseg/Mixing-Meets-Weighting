#!/usr/bin/env python3
"""V24 A2 X03 reliability/uncertainty faithfulness on ACDC10 VAL only."""

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
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom
from scipy.stats import spearmanr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "train_ACDC"))

AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
FIGURE_ROOT = RESULT_ROOT / "figures"

DATASET_ROOT = Path("/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC")
VAL_MANIFEST = DATASET_ROOT / "val.list"
ARM110_CKPT = ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/checkpoints/best.ckpt"
ARM111_CKPT = ROOT / "runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt"
ARM110_SHA = "f10f69989883a423117a53695311b47e01dc65fbc27fbce2d0c666d6fd733514"
ARM111_SHA = "c447a0a6d643405627cebf4718424b7d102fffd677b23f10ab5ef1e5227c3086"
SCORE_CONTRACT = AUDIT_ROOT / "X03_SCORE_SEMANTIC_CONTRACT.json"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 2026
CLASS_IDS = [1, 2, 3]


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


def f12(value: float) -> str:
    if value is None or not math.isfinite(value):
        return "nan"
    return f"{value:.12f}"


def dry_run() -> dict[str, Any]:
    contract = json.loads(SCORE_CONTRACT.read_text()) if SCORE_CONTRACT.exists() else {}
    payload = {
        "status": "PASS",
        "created_at": now(),
        "mode": "dry-run",
        "split": "VAL",
        "test_access": 0,
        "training_started": False,
        "inference_started": False,
        "val_manifest_content_read": False,
        "test_manifest_content_read": False,
        "arm110_checkpoint": rel(ARM110_CKPT),
        "arm110_checkpoint_sha256": sha256(ARM110_CKPT) if ARM110_CKPT.exists() else "",
        "arm110_checkpoint_sha_match": ARM110_CKPT.exists() and sha256(ARM110_CKPT) == ARM110_SHA,
        "arm111_checkpoint": rel(ARM111_CKPT),
        "arm111_checkpoint_sha256": sha256(ARM111_CKPT) if ARM111_CKPT.exists() else "",
        "arm111_checkpoint_sha_match": ARM111_CKPT.exists() and sha256(ARM111_CKPT) == ARM111_SHA,
        "score_contract_status": contract.get("status"),
        "higher_score_means_more_reliable": contract.get("higher_score_means_more_reliable"),
        "higher_score_means_less_reliable_or_more_fragile_boundary": contract.get("higher_score_means_less_reliable_or_more_fragile_boundary"),
    }
    if not (payload["arm110_checkpoint_sha_match"] and payload["arm111_checkpoint_sha_match"] and payload["score_contract_status"] == "PASS"):
        payload["status"] = "FAIL"
    write_json(AUDIT_ROOT / "X03_RELIABILITY_FAITHFULNESS_DRY_RUN_RECEIPT.json", payload)
    return payload


def surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    if not mask.any():
        return mask
    return mask ^ binary_erosion(mask)


def x03_score(prob1, prob2):
    import torch

    confidence = 0.5 * (prob1.max(0).values + prob2.max(0).values)
    dx = torch.zeros_like(confidence)
    dy = torch.zeros_like(confidence)
    dx[1:, :] = (confidence[1:, :] - confidence[:-1, :]).abs()
    dy[:, 1:] = (confidence[:, 1:] - confidence[:, :-1]).abs()
    contour = dx + dy
    flat = contour.flatten()
    return ((contour - flat.min()) / (flat.max() - flat.min()).clamp_min(1e-6)).clamp(0.0, 1.0)


def build_model(ckpt_path: Path, state_key: str, device: str):
    import torch
    from v14_mask_adamix_components import v14_bcp_net

    checkpoint = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    model = v14_bcp_net(in_chns=1, class_num=4, ema=False, mask_token_enabled=True, mask_token_num=4, mask_token_dim=16)
    model.load_state_dict(checkpoint[state_key], strict=True)
    model.to(device)
    model.eval()
    return model


def infer_slice(model, image2d: np.ndarray, device: str):
    import torch

    height, width = image2d.shape
    resized = zoom(image2d, (256 / height, 256 / width), order=0)
    tensor = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(tensor)
        if isinstance(logits, (tuple, list)):
            logits = logits[0]
        prob = torch.softmax(logits, dim=1).squeeze(0)
        pred = torch.argmax(prob, dim=0).cpu().numpy()
    pred = zoom(pred, (height / 256, width / 256), order=0).astype("int16")
    return pred, prob.cpu()


def dice_iou_hd_asd(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    from medpy import metric

    vals = []
    for cid in CLASS_IDS:
        p = pred == cid
        t = truth == cid
        if p.any() and t.any():
            inter = int((p & t).sum())
            union = int((p | t).sum())
            vals.append((2.0 * inter / (int(p.sum()) + int(t.sum())), inter / union if union else 0.0, float(metric.binary.hd95(p, t)), float(metric.binary.asd(p, t))))
        elif not p.any() and not t.any():
            vals.append((0.0, 0.0, 0.0, 0.0))
        else:
            penalty = float(np.linalg.norm(pred.shape))
            vals.append((0.0, 0.0, penalty, penalty))
    return {
        "dice": float(np.mean([v[0] for v in vals])),
        "iou": float(np.mean([v[1] for v in vals])),
        "hd95": float(np.mean([v[2] for v in vals])),
        "asd": float(np.mean([v[3] for v in vals])),
    }


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    neg = int((~labels).sum())
    if pos == 0 or neg == 0:
        return math.nan
    order = np.argsort(scores)
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    return float((ranks[labels].sum() - pos * (pos + 1) / 2.0) / (pos * neg))


def auprc(scores: np.ndarray, labels: np.ndarray) -> float:
    labels = labels.astype(bool)
    pos = int(labels.sum())
    if pos == 0:
        return math.nan
    order = np.argsort(-scores)
    sorted_labels = labels[order]
    tp = np.cumsum(sorted_labels)
    precision = tp / np.arange(1, len(labels) + 1)
    return float(precision[sorted_labels].sum() / pos)


def risk_curve(scores: np.ndarray, errors: np.ndarray) -> list[dict[str, float]]:
    order = np.argsort(scores)  # low uncertainty retained first
    curves = []
    for coverage in range(10, 101, 10):
        n = max(1, int(round(len(scores) * coverage / 100.0)))
        kept = order[:n]
        curves.append({"Coverage": coverage / 100.0, "Risk": float(errors[kept].mean())})
    return curves


def bootstrap_corr(case_rows: list[dict[str, Any]], metric: str) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED + len(metric))
    vals = []
    n = len(case_rows)
    for _ in range(BOOTSTRAP_N):
        sample = [case_rows[rng.randrange(n)] for _ in range(n)]
        xs = [float(r["MeanX03Uncertainty"]) for r in sample]
        ys = [float(r[metric]) for r in sample]
        rho = spearmanr(xs, ys).correlation
        if rho is not None and math.isfinite(float(rho)):
            vals.append(float(rho))
    if not vals:
        return math.nan, math.nan
    vals.sort()
    return vals[int(0.025 * len(vals))], vals[int(0.975 * len(vals))]


def run(device: str) -> dict[str, Any]:
    import h5py
    import torch

    if device.startswith("cuda") and not torch.cuda.is_available():
        raise SystemExit("CUDA_REQUESTED_BUT_NOT_AVAILABLE")
    if sha256(ARM110_CKPT) != ARM110_SHA or sha256(ARM111_CKPT) != ARM111_SHA:
        raise SystemExit("CHECKPOINT_SHA_MISMATCH")
    contract = json.loads(SCORE_CONTRACT.read_text())
    if contract.get("higher_score_means_more_reliable") is not False:
        raise SystemExit("X03_SCORE_DIRECTION_NOT_FIXED_AS_UNCERTAINTY")

    cases = [line.strip() for line in VAL_MANIFEST.read_text().splitlines() if line.strip()]
    arm110 = build_model(ARM110_CKPT, "model_1", device)
    arm111_m1 = build_model(ARM111_CKPT, "model_1", device)
    arm111_m2 = build_model(ARM111_CKPT, "model_2", device)

    band_scores = {band: [] for band in [1, 2, 3]}
    band_errors = {band: [] for band in [1, 2, 3]}
    case_rows = []
    for case in cases:
        case_scores = []
        with h5py.File(DATASET_ROOT / "data" / f"{case}.h5", "r") as handle:
            image = handle["image"][:]
            label = handle["label"][:]
        pred110_volume = np.zeros_like(label)
        pred111_volume = np.zeros_like(label)
        for idx, source in enumerate(image):
            pred110, _ = infer_slice(arm110, source, device)
            pred111, prob111_m1 = infer_slice(arm111_m1, source, device)
            _, prob111_m2 = infer_slice(arm111_m2, source, device)
            pred110_volume[idx] = pred110
            pred111_volume[idx] = pred111
            score = x03_score(prob111_m1, prob111_m2).numpy()
            score = zoom(score, (source.shape[0] / 256, source.shape[1] / 256), order=1)
            err = (pred111 != label[idx]).astype("uint8")
            gt_surface = surface(label[idx] > 0)
            if gt_surface.any():
                dist = distance_transform_edt(~gt_surface)
                for band in [1, 2, 3]:
                    mask = dist <= band
                    if mask.any():
                        band_scores[band].append(score[mask].astype("float32"))
                        band_errors[band].append(err[mask].astype("uint8"))
            fg = label[idx] > 0
            if fg.any():
                case_scores.append(float(score[fg].mean()))
        m110 = dice_iou_hd_asd(pred110_volume, label)
        m111 = dice_iou_hd_asd(pred111_volume, label)
        case_rows.append(
            {
                "CaseID": case,
                "Split": "VAL",
                "MeanX03Uncertainty": f12(float(np.mean(case_scores)) if case_scores else math.nan),
                "ARM110_Dice": f12(m110["dice"]),
                "ARM111_Dice": f12(m111["dice"]),
                "DeltaDice_111Minus110": f12(m111["dice"] - m110["dice"]),
                "ARM110_HD95": f12(m110["hd95"]),
                "ARM111_HD95": f12(m111["hd95"]),
                "DeltaHD95_111Minus110": f12(m111["hd95"] - m110["hd95"]),
                "ARM110_ASD": f12(m110["asd"]),
                "ARM111_ASD": f12(m111["asd"]),
                "DeltaASD_111Minus110": f12(m111["asd"] - m110["asd"]),
            }
        )

    pixel_rows = []
    quintile_rows = []
    risk_rows = []
    for band in [1, 2, 3]:
        scores = np.concatenate(band_scores[band])
        errors = np.concatenate(band_errors[band])
        rho = spearmanr(scores, errors).correlation
        pixel_rows.append(
            {
                "BandPx": band,
                "PositiveLabel": "ARM111_prediction_error",
                "ScoreName": "X03_contour_uncertainty",
                "HigherScoreMeans": "more_uncertain_or_fragile_boundary",
                "PixelCount": len(scores),
                "ErrorRate": f12(float(errors.mean())),
                "AUROC": f12(auroc(scores, errors)),
                "AUPRC": f12(auprc(scores, errors)),
                "SpearmanRho": f12(float(rho) if rho is not None else math.nan),
            }
        )
        quantiles = np.quantile(scores, [0, 0.2, 0.4, 0.6, 0.8, 1.0])
        for q in range(5):
            lo, hi = quantiles[q], quantiles[q + 1]
            mask = (scores >= lo) & (scores <= hi if q == 4 else scores < hi)
            quintile_rows.append({"BandPx": band, "UncertaintyQuintile": q + 1, "PixelCount": int(mask.sum()), "ErrorRate": f12(float(errors[mask].mean()) if mask.any() else math.nan)})
        for item in risk_curve(scores, errors):
            risk_rows.append({"BandPx": band, "Coverage": f12(item["Coverage"]), "Risk": f12(item["Risk"])})

    case_summary = []
    for metric in ["ARM111_Dice", "ARM111_HD95", "ARM111_ASD"]:
        xs = [float(r["MeanX03Uncertainty"]) for r in case_rows]
        ys = [float(r[metric]) for r in case_rows]
        rho = spearmanr(xs, ys).correlation
        lo, hi = bootstrap_corr(case_rows, metric)
        case_summary.append({"Metric": metric, "SpearmanRho": f12(float(rho)), "BootstrapCI95Low": f12(lo), "BootstrapCI95High": f12(hi)})

    pixel_out = RESULT_ROOT / "X03_RELIABILITY_PIXEL_LEVEL.tsv"
    case_out = RESULT_ROOT / "X03_RELIABILITY_CASE_LEVEL.tsv"
    quintile_out = RESULT_ROOT / "X03_RELIABILITY_QUINTILE_ERROR.tsv"
    risk_out = RESULT_ROOT / "X03_RELIABILITY_RISK_COVERAGE.tsv"
    write_tsv(pixel_out, pixel_rows)
    write_tsv(case_out, case_rows)
    write_tsv(quintile_out, quintile_rows)
    write_tsv(risk_out, risk_rows)
    make_figures(quintile_rows, risk_rows, case_rows)

    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "X03_RELIABILITY_FAITHFULNESS",
        "split": "VAL",
        "test_access": 0,
        "training_started": False,
        "checkpoint_selection_started": False,
        "gt_used_to_generate_score": False,
        "gt_used_only_for_retrospective_evaluation": True,
        "case_count": len(cases),
        "score_contract": rel(SCORE_CONTRACT),
        "score_contract_sha256": sha256(SCORE_CONTRACT),
        "score_semantics": contract["score_semantics"],
        "higher_score_means_more_reliable": False,
        "higher_score_means_less_reliable_or_more_fragile_boundary": True,
        "pixel_level_tsv": rel(pixel_out),
        "pixel_level_tsv_sha256": sha256(pixel_out),
        "case_level_tsv": rel(case_out),
        "case_level_tsv_sha256": sha256(case_out),
        "quintile_error_tsv": rel(quintile_out),
        "quintile_error_tsv_sha256": sha256(quintile_out),
        "risk_coverage_tsv": rel(risk_out),
        "risk_coverage_tsv_sha256": sha256(risk_out),
        "case_level_correlations": case_summary,
        "bootstrap_iterations": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "figures": [
            "final_report/internal_evidence_expansion_v24_0/figures/X03_RELIABILITY_QUINTILE_ERROR.png",
            "final_report/internal_evidence_expansion_v24_0/figures/X03_RELIABILITY_QUINTILE_ERROR.pdf",
            "final_report/internal_evidence_expansion_v24_0/figures/X03_RELIABILITY_RISK_COVERAGE.png",
            "final_report/internal_evidence_expansion_v24_0/figures/X03_RELIABILITY_RISK_COVERAGE.pdf",
            "final_report/internal_evidence_expansion_v24_0/figures/X03_RELIABILITY_CASE_CORRELATION.png",
            "final_report/internal_evidence_expansion_v24_0/figures/X03_RELIABILITY_CASE_CORRELATION.pdf",
        ],
    }
    write_json(RESULT_ROOT / "X03_RELIABILITY_FAITHFULNESS_SUMMARY.json", payload)
    write_json(AUDIT_ROOT / "X03_RELIABILITY_FAITHFULNESS_RECEIPT.json", payload)
    return payload


def make_figures(quintile_rows: list[dict[str, Any]], risk_rows: list[dict[str, Any]], case_rows: list[dict[str, Any]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for band in [1, 2, 3]:
        rows = [r for r in quintile_rows if int(r["BandPx"]) == band]
        ax.plot([int(r["UncertaintyQuintile"]) for r in rows], [float(r["ErrorRate"]) for r in rows], marker="o", label=f"{band}px")
    ax.set_xlabel("X03 uncertainty quintile")
    ax.set_ylabel("Prediction error rate")
    ax.legend(title="Band")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "X03_RELIABILITY_QUINTILE_ERROR.png")
    fig.savefig(FIGURE_ROOT / "X03_RELIABILITY_QUINTILE_ERROR.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for band in [1, 2, 3]:
        rows = [r for r in risk_rows if int(r["BandPx"]) == band]
        ax.plot([float(r["Coverage"]) for r in rows], [float(r["Risk"]) for r in rows], marker="o", label=f"{band}px")
    ax.set_xlabel("Coverage retained from lowest uncertainty")
    ax.set_ylabel("Prediction error risk")
    ax.legend(title="Band")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "X03_RELIABILITY_RISK_COVERAGE.png")
    fig.savefig(FIGURE_ROOT / "X03_RELIABILITY_RISK_COVERAGE.pdf")
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(5.5, 4.0))
    ax.scatter([float(r["MeanX03Uncertainty"]) for r in case_rows], [float(r["ARM111_Dice"]) for r in case_rows], s=25)
    ax.set_xlabel("Case mean X03 uncertainty")
    ax.set_ylabel("ARM111 VAL Dice")
    fig.tight_layout()
    fig.savefig(FIGURE_ROOT / "X03_RELIABILITY_CASE_CORRELATION.png")
    fig.savefig(FIGURE_ROOT / "X03_RELIABILITY_CASE_CORRELATION.pdf")
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "run"], required=True)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    os.environ.setdefault("PYTHONHASHSEED", "0")
    payload = dry_run() if args.mode == "dry-run" else run(args.device)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
