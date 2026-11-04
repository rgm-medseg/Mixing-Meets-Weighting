#!/usr/bin/env python3
"""Extract ACDC10 training dynamics for V23 paper evidence.

This is read-only with respect to the source runs.  The validation curve comes
from the canonical train.log lines emitted at every validation interval.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "final_report/paper_internal_evidence_v23_0"
FIGURE_ROOT = RESULT_ROOT / "figures"
AUDIT_ROOT = ROOT / "audits/paper_internal_evidence_v23_0"
VAL_ONLY_GATE = AUDIT_ROOT / "TRAINING_DYNAMICS_VAL_ONLY_GATE.json"


@dataclass(frozen=True)
class RunSpec:
    method_id: str
    display_name: str
    log_path: Path
    source_identity: str


RUNS = [
    RunSpec(
        method_id="ACDC10_ARM_000",
        display_name="beta-FFT",
        log_path=ROOT / "runs/beta_fft_acdc_official_full_seed42_fresh_ckpt/train.log",
        source_identity="ACDC10 beta-FFT canonical full run, Mask Token OFF, AdaMix OFF, X03 OFF",
    ),
    RunSpec(
        method_id="ACDC10_ARM_110",
        display_name="Mask Token + AdaMix",
        log_path=ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/train.log",
        source_identity="ACDC10 Mask Token + AdaMix canonical full run, X03 OFF",
    ),
    RunSpec(
        method_id="ACDC10_ARM_111",
        display_name="Ours",
        log_path=ROOT
        / "runs/acdc10_third_innovation_codex_prep_top6_v21_3"
        / "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/train.log",
        source_identity="ACDC10 Mask Token + AdaMix + X03 canonical full run",
    ),
]


DICE_RE = re.compile(r"iteration\s+(\d+)\s+:\s+(model[12]_mean_dice(?:_test)?)\s+:\s+([0-9.]+)")
BEST_RE = re.compile(r"best_checkpoint_updated .* global_step=(\d+) metric=([0-9.]+)")
FINAL_RE = re.compile(r"resume_check_ok .* global_step=(\d+)")


def parse_log(spec: RunSpec) -> tuple[list[dict[str, object]], dict[str, object]]:
    if not spec.log_path.exists():
        raise FileNotFoundError(spec.log_path)
    by_iter: dict[int, dict[str, object]] = {}
    best_updates: list[dict[str, object]] = []
    final_iter = None
    for line in spec.log_path.read_text(errors="replace").splitlines():
        best = BEST_RE.search(line)
        if best:
            best_updates.append({"iteration": int(best.group(1)), "best_model1_val_dice": float(best.group(2))})
        final = FINAL_RE.search(line)
        if final:
            final_iter = int(final.group(1))
        dice = DICE_RE.search(line)
        if not dice:
            continue
        iteration = int(dice.group(1))
        key = dice.group(2)
        value = float(dice.group(3))
        row = by_iter.setdefault(iteration, {"iteration": iteration})
        row[key] = value

    rows = []
    running_best = None
    running_best_iter = None
    for iteration in sorted(by_iter):
        row = by_iter[iteration]
        model1 = row.get("model1_mean_dice")
        model2 = row.get("model2_mean_dice")
        selected = model1 if isinstance(model1, float) else None
        if selected is not None and (running_best is None or selected > running_best):
            running_best = selected
            running_best_iter = iteration
        rows.append(
            {
                "MethodID": spec.method_id,
                "DisplayName": spec.display_name,
                "Iteration": iteration,
                "Model1ValDice": model1 if model1 is not None else "",
                "Model2ValDice": model2 if model2 is not None else "",
                "SelectedValDice": selected if selected is not None else "",
                "RunningBestValDice": running_best if running_best is not None else "",
                "RunningBestIter": running_best_iter if running_best_iter is not None else "",
                "SourceLog": str(spec.log_path.relative_to(ROOT)),
                "EndpointIdentity": "VALIDATION_LOG_POINT",
                "SplitProof": "db_val split=VAL; consumed log key=model1_mean_dice/model2_mean_dice; historical *_test aliases quarantined",
                "ConsumedTestPoint": 0,
            }
        )
    if not rows:
        raise RuntimeError(f"NO_VALIDATION_POINTS:{spec.log_path}")

    best_row = max(rows, key=lambda r: float(r["SelectedValDice"]) if r["SelectedValDice"] != "" else -1.0)
    final_row = rows[-1]
    summary = {
        "MethodID": spec.method_id,
        "DisplayName": spec.display_name,
        "SourceIdentity": spec.source_identity,
        "SourceLog": str(spec.log_path.relative_to(ROOT)),
        "ValidationPointCount": len(rows),
        "FirstIteration": rows[0]["Iteration"],
        "FinalIteration": final_row["Iteration"],
        "ResumeFinalIteration": final_iter,
        "BestValDice": best_row["SelectedValDice"],
        "BestIter": best_row["Iteration"],
        "FinalValDice": final_row["SelectedValDice"],
        "BestMinusFinalDice": float(best_row["SelectedValDice"]) - float(final_row["SelectedValDice"]),
        "BestCheckpointUpdateCount": len(best_updates),
        "BestCheckpointUpdatesAgree": (
            not best_updates
            or int(best_updates[-1]["iteration"]) == int(best_row["Iteration"])
            and abs(float(best_updates[-1]["best_model1_val_dice"]) - float(best_row["SelectedValDice"])) < 1e-6
        ),
        "ConsumedTestPointCount": 0,
        "HistoricalTestAliasPolicy": "QUARANTINED_NOT_CONSUMED",
    }
    return rows, summary


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def plot_curves(rows: list[dict[str, object]], out_png: Path, out_pdf: Path) -> None:
    plt.figure(figsize=(7.2, 4.4), dpi=160)
    for method in ["ACDC10_ARM_000", "ACDC10_ARM_110", "ACDC10_ARM_111"]:
        subset = [r for r in rows if r["MethodID"] == method]
        plt.plot(
            [int(r["Iteration"]) for r in subset],
            [float(r["SelectedValDice"]) * 100.0 for r in subset],
            linewidth=1.8,
            label=subset[0]["DisplayName"],
        )
    plt.xlabel("Iteration")
    plt.ylabel("Validation Dice (%)")
    plt.grid(True, alpha=0.25)
    plt.legend(frameon=False)
    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png)
    plt.savefig(out_pdf)
    plt.close()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run", "dry-run"], default="run")
    args = parser.parse_args()
    if args.mode == "dry-run":
        payload = {
            "status": "DRY_RUN_PASS",
            "source_logs": [str(spec.log_path.relative_to(ROOT)) for spec in RUNS],
            "outputs": [
                "final_report/paper_internal_evidence_v23_0/ACDC10_TRAINING_DYNAMICS_VALIDATION_CURVES.tsv",
                "final_report/paper_internal_evidence_v23_0/figures/FIG_TRAINING_DYNAMICS_ACDC10.png",
                "final_report/paper_internal_evidence_v23_0/figures/FIG_TRAINING_DYNAMICS_ACDC10.pdf",
            ],
        }
        print(json.dumps(payload, indent=2))
        return 0

    if not VAL_ONLY_GATE.exists():
        raise SystemExit("TRAINING_DYNAMICS_VAL_ONLY_GATE_MISSING")
    gate = json.loads(VAL_ONLY_GATE.read_text())
    if gate.get("status") != "PASS" or gate.get("TEST_POINT_COUNT") != 0:
        raise SystemExit("TRAINING_DYNAMICS_VAL_ONLY_GATE_NOT_PASS")

    all_rows: list[dict[str, object]] = []
    summaries = []
    for spec in RUNS:
        rows, summary = parse_log(spec)
        all_rows.extend(rows)
        summaries.append(summary)
    raw_tsv = RESULT_ROOT / "ACDC10_TRAINING_DYNAMICS_VALIDATION_CURVES.tsv"
    summary_tsv = RESULT_ROOT / "ACDC10_TRAINING_DYNAMICS_SUMMARY.tsv"
    write_tsv(raw_tsv, all_rows)
    write_tsv(summary_tsv, summaries)
    png = FIGURE_ROOT / "FIG_TRAINING_DYNAMICS_ACDC10.png"
    pdf = FIGURE_ROOT / "FIG_TRAINING_DYNAMICS_ACDC10.pdf"
    plot_curves(all_rows, png, pdf)
    audit = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "raw_tsv": str(raw_tsv.relative_to(ROOT)),
        "summary_tsv": str(summary_tsv.relative_to(ROOT)),
        "figure_png": str(png.relative_to(ROOT)),
        "figure_pdf": str(pdf.relative_to(ROOT)),
        "runs": summaries,
        "endpoint": "VALIDATION_LOG_POINT",
        "new_training_started": False,
        "test_access": 0,
        "val_only_gate": str(VAL_ONLY_GATE.relative_to(ROOT)),
        "TEST_POINT_COUNT": 0,
        "historical_test_alias_policy": "QUARANTINED_NOT_CONSUMED",
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "TRAINING_DYNAMICS_EXTRACTION_SUMMARY.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps(audit, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
