#!/usr/bin/env python3
"""Parse V23 X03 weight sensitivity validation results.

No TEST endpoint is read.  Weight 0.00 and 0.05 are exact-reuse anchors; the
remaining points require completed V23 full runs.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "final_report/paper_internal_evidence_v23_0"
FIGURE_ROOT = RESULT_ROOT / "figures"
AUDIT_ROOT = ROOT / "audits/paper_internal_evidence_v23_0"

ANCHORS = {
    "0.00": {
        "task_id": "ACDC10_ARM_110",
        "source_log": ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/train.log",
        "source": "EXACT_REUSE_ARM_110",
    },
    "0.05": {
        "task_id": "ACDC10_ARM_111",
        "source_log": ROOT
        / "runs/acdc10_third_innovation_codex_prep_top6_v21_3"
        / "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/train.log",
        "source": "EXACT_REUSE_ARM_111",
    },
}

NEW_POINTS = {
    "0.01": "ACDC10_X03_WEIGHT_001",
    "0.03": "ACDC10_X03_WEIGHT_003",
    "0.07": "ACDC10_X03_WEIGHT_007",
    "0.10": "ACDC10_X03_WEIGHT_010",
}

DICE_RE = re.compile(r"iteration\s+(\d+)\s+:\s+model1_mean_dice\s+:\s+([0-9.]+)")
FINAL_RE = re.compile(r"resume_check_ok .* global_step=(\d+)")


def parse_val_curve(log_path: Path) -> tuple[list[tuple[int, float]], int | None]:
    if not log_path.exists():
        raise FileNotFoundError(log_path)
    points = [(int(m.group(1)), float(m.group(2))) for m in DICE_RE.finditer(log_path.read_text(errors="replace"))]
    final_iter = None
    for m in FINAL_RE.finditer(log_path.read_text(errors="replace")):
        final_iter = int(m.group(1))
    if not points:
        raise RuntimeError(f"NO_VAL_POINTS:{log_path}")
    return points, final_iter


def summarize(weight: str, task_id: str, log_path: Path, source: str) -> dict[str, object]:
    points, final_iter = parse_val_curve(log_path)
    best_iter, best_dice = max(points, key=lambda x: x[1])
    final_point_iter, final_dice = points[-1]
    if final_iter != 30000 or final_point_iter != 30000:
        raise RuntimeError(f"FULL_RUN_NOT_COMPLETE:{task_id}:final_iter={final_iter}:final_point={final_point_iter}")
    return {
        "Weight": weight,
        "TaskID": task_id,
        "Action": source,
        "BestValDice": f"{best_dice:.16f}",
        "BestValIter": best_iter,
        "FinalValDice": f"{final_dice:.16f}",
        "FinalIter": final_point_iter,
        "BestMinusFinalDice": f"{best_dice - final_dice:.16f}",
        "TestAccess": 0,
        "SourceLog": str(log_path.relative_to(ROOT)),
    }


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot(rows: list[dict[str, object]]) -> None:
    weights = [float(row["Weight"]) for row in rows]
    dice = [float(row["BestValDice"]) * 100.0 for row in rows]
    plt.figure(figsize=(5.8, 4.0), dpi=160)
    plt.plot(weights, dice, marker="o", linewidth=1.8)
    plt.axvline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.6)
    plt.xlabel("X03 weight")
    plt.ylabel("Best validation Dice (%)")
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    plt.savefig(FIGURE_ROOT / "FIG_X03_WEIGHT_SENSITIVITY_ACDC10.png")
    plt.savefig(FIGURE_ROOT / "FIG_X03_WEIGHT_SENSITIVITY_ACDC10.pdf")
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "run"], default="run")
    args = parser.parse_args()
    expected_logs = [str(item["source_log"].relative_to(ROOT)) for item in ANCHORS.values()] + [
        f"experiments/paper_internal_evidence_v23_0/{task_id}/train.log" for task_id in NEW_POINTS.values()
    ]
    if args.mode == "dry-run":
        print(json.dumps({"status": "DRY_RUN_PASS", "expected_logs": expected_logs, "test_access": 0}, indent=2))
        return 0

    rows: list[dict[str, object]] = []
    for weight, item in ANCHORS.items():
        rows.append(summarize(weight, item["task_id"], item["source_log"], item["source"]))
    for weight, task_id in NEW_POINTS.items():
        rows.append(
            summarize(
                weight,
                task_id,
                ROOT / "experiments/paper_internal_evidence_v23_0" / task_id / "train.log",
                "NEW_FULL",
            )
        )
    rows.sort(key=lambda r: float(r["Weight"]))
    out = RESULT_ROOT / "X03_WEIGHT_SENSITIVITY_FULL_PRECISION.tsv"
    write_tsv(out, rows)
    plot(rows)
    payload = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_tsv": str(out.relative_to(ROOT)),
        "figure_png": str((FIGURE_ROOT / "FIG_X03_WEIGHT_SENSITIVITY_ACDC10.png").relative_to(ROOT)),
        "figure_pdf": str((FIGURE_ROOT / "FIG_X03_WEIGHT_SENSITIVITY_ACDC10.pdf").relative_to(ROOT)),
        "point_count": len(rows),
        "test_access": 0,
        "final_weight_frozen": 0.05,
        "x03_weight_reselection_allowed": False,
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "X03_WEIGHT_SENSITIVITY_PARSE_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
