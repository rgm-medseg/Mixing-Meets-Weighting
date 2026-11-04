#!/usr/bin/env python3
"""Read-only V24 Plane-A extractor for assets already produced by V22/V23.

This script intentionally performs no training, no checkpoint replay, and no
TEST manifest access. It only consumes frozen full-precision TSV assets and
writes V24-scoped derived tables plus receipts.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - figure generation is receipt-gated.
    plt = None


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
TABLE_ROOT = RESULT_ROOT / "tables"

TABLE4 = ROOT / "final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv"
DYNAMICS = ROOT / "final_report/paper_internal_evidence_v23_0/TRAINING_DYNAMICS_RAW.tsv"
CASE_METRIC_CANDIDATES = {
    "000": ROOT / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/case_class_metrics.tsv",
    "001": ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC10_ARM_001/case_class_metrics.tsv",
    "010": ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC10_ARM_010/case_class_metrics.tsv",
    "011": ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC10_ARM_011/case_class_metrics.tsv",
    "100": ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC10_ARM_100/case_class_metrics.tsv",
    "101": ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC10_ARM_101/case_class_metrics.tsv",
    "110": ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC10_ARM_110/case_class_metrics.tsv",
    "111": ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC10_ARM_111/case_class_metrics.tsv",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_markdown_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(col, "")) for col in columns) + " |")
    path.write_text("\n".join(lines) + "\n")


def write_tex_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    align = "l" * len(columns)
    lines = [f"\\begin{{tabular}}{{{align}}}", "\\toprule", " & ".join(columns) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(str(row.get(col, "")) for col in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.write_text("\n".join(lines) + "\n")


def f(value: str) -> float:
    return float(value)


def arm_bits(row: dict[str, str]) -> tuple[int, int, int]:
    return (
        1 if row["Mask Token"].strip() else 0,
        1 if row["AdaMix"].strip() else 0,
        1 if row["Contour Reliability"].strip() else 0,
    )


def extract_factorial_effects() -> dict:
    rows = read_tsv(TABLE4)
    if len(rows) != 8:
        raise SystemExit(f"TABLE4_EXPECTED_8_ROWS_GOT_{len(rows)}")
    seen = {"".join(map(str, arm_bits(row))) for row in rows}
    if seen != {format(i, "03b") for i in range(8)}:
        raise SystemExit(f"TABLE4_ARM_SET_MISMATCH:{sorted(seen)}")

    metrics = ["Dice", "IoU", "HD95", "ASD"]
    factors = [
        ("Mask Token", 0),
        ("AdaMix", 1),
        ("X03 Contour Reliability Weight", 2),
    ]
    out_rows: list[dict[str, object]] = []
    formula = (
        "For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). "
        "Main-effect signs use the target factor; interaction signs use the product of coded factors "
        "with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD."
    )
    for metric in metrics:
        values = []
        for row in rows:
            bits = arm_bits(row)
            values.append((bits, f(row[metric])))
        for name, idx in factors:
            on = [v for bits, v in values if bits[idx] == 1]
            off = [v for bits, v in values if bits[idx] == 0]
            raw = sum(on) / len(on) - sum(off) / len(off)
            out_rows.append(
                {
                    "EffectType": "MAIN",
                    "Factor": name,
                    "Metric": metric,
                    "GrandMean": f"{sum(v for _, v in values) / len(values):.12f}",
                    "MeanOn": f"{sum(on) / len(on):.12f}",
                    "MeanOff": f"{sum(off) / len(off):.12f}",
                    "RawEffectOnMinusOff": f"{raw:.12f}",
                    "ImprovementOrientedEffect": f"{(raw if metric in {'Dice', 'IoU'} else -raw):.12f}",
                    "Unit": "fraction" if metric in {"Dice", "IoU"} else "pixel",
                    "Formula": formula,
                    "DESCRIPTIVE_FACTORIAL_ONLY": "true",
                    "SourceAsset": str(TABLE4.relative_to(ROOT)),
                }
            )

        coded_names = [
            ("MaskToken_x_AdaMix", lambda b: (1 if b[0] else -1) * (1 if b[1] else -1)),
            ("MaskToken_x_X03", lambda b: (1 if b[0] else -1) * (1 if b[2] else -1)),
            ("AdaMix_x_X03", lambda b: (1 if b[1] else -1) * (1 if b[2] else -1)),
            (
                "MaskToken_x_AdaMix_x_X03",
                lambda b: (1 if b[0] else -1) * (1 if b[1] else -1) * (1 if b[2] else -1),
            ),
        ]
        for name, sign_fn in coded_names:
            pos = [v for bits, v in values if sign_fn(bits) == 1]
            neg = [v for bits, v in values if sign_fn(bits) == -1]
            raw = sum(pos) / len(pos) - sum(neg) / len(neg)
            out_rows.append(
                {
                    "EffectType": "INTERACTION",
                    "Factor": name,
                    "Metric": metric,
                    "GrandMean": f"{sum(v for _, v in values) / len(values):.12f}",
                    "MeanOn": f"{sum(pos) / len(pos):.12f}",
                    "MeanOff": f"{sum(neg) / len(neg):.12f}",
                    "RawEffectOnMinusOff": f"{raw:.12f}",
                    "ImprovementOrientedEffect": f"{(raw if metric in {'Dice', 'IoU'} else -raw):.12f}",
                    "Unit": "fraction" if metric in {"Dice", "IoU"} else "pixel",
                    "Formula": formula,
                    "DESCRIPTIVE_FACTORIAL_ONLY": "true",
                    "SourceAsset": str(TABLE4.relative_to(ROOT)),
                }
            )

    out = RESULT_ROOT / "ACDC10_FACTORIAL_EFFECTS_FULL.tsv"
    write_tsv(out, out_rows)
    table_rows = [
        {
            "Factor": r["Factor"],
            "Metric": r["Metric"],
            "RawEffect": r["RawEffectOnMinusOff"],
            "ImprovementEffect": r["ImprovementOrientedEffect"],
            "DescriptiveOnly": "true",
        }
        for r in out_rows
    ]
    table_cols = ["Factor", "Metric", "RawEffect", "ImprovementEffect", "DescriptiveOnly"]
    write_markdown_table(TABLE_ROOT / "TABLE_S_ACDC10_FACTORIAL_EFFECTS.md", table_rows, table_cols)
    write_tex_table(TABLE_ROOT / "TABLE_S_ACDC10_FACTORIAL_EFFECTS.tex", table_rows, table_cols)

    figure_assets = []
    if plt is not None:
        for effect_type, stem in [
            ("MAIN", "ACDC10_FACTORIAL_MAIN_EFFECTS"),
            ("INTERACTION", "ACDC10_FACTORIAL_INTERACTIONS"),
        ]:
            subset = [r for r in out_rows if r["EffectType"] == effect_type and r["Metric"] == "Dice"]
            labels = [r["Factor"] for r in subset]
            values = [float(r["ImprovementOrientedEffect"]) for r in subset]
            plt.figure(figsize=(max(6, len(labels) * 1.4), 4))
            plt.axhline(0, color="black", linewidth=0.8)
            plt.bar(range(len(labels)), values, color="#3b6ea8")
            plt.xticks(range(len(labels)), labels, rotation=25, ha="right")
            plt.ylabel("Dice improvement effect")
            plt.tight_layout()
            for suffix in ["png", "pdf"]:
                fig_path = RESULT_ROOT / "figures" / f"{stem}.{suffix}"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_path)
                figure_assets.append(str(fig_path.relative_to(ROOT)))
            plt.close()

    receipt = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "FACTORIAL_EFFECTS",
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "input_asset": str(TABLE4.relative_to(ROOT)),
        "input_sha256": sha256(TABLE4),
        "output_asset": str(out.relative_to(ROOT)),
        "output_sha256": sha256(out),
        "markdown_table": "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_FACTORIAL_EFFECTS.md",
        "tex_table": "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_FACTORIAL_EFFECTS.tex",
        "figure_assets": figure_assets,
        "exact_formula": formula,
        "DESCRIPTIVE_FACTORIAL_ONLY": True,
        "anova_or_p_value_computed": False,
        "arm_count": len(rows),
        "effect_row_count": len(out_rows),
    }
    write_json(AUDIT_ROOT / "FACTORIAL_EFFECTS_RECEIPT.json", receipt)
    return receipt


def extract_training_dynamics() -> dict:
    rows = read_tsv(DYNAMICS)
    if not rows:
        raise SystemExit("TRAINING_DYNAMICS_EMPTY")
    bad_endpoint = [r for r in rows if r["EndpointIdentity"] != "VALIDATION_LOG_POINT"]
    consumed_test = [r for r in rows if str(r["ConsumedTestPoint"]) != "0"]
    if bad_endpoint or consumed_test:
        raise SystemExit("TRAINING_DYNAMICS_NOT_VAL_ONLY")

    normalized_rows = []
    for row in rows:
        normalized_rows.append(
            {
                "Iteration": row["Iteration"],
                "MethodID": row["MethodID"],
                "Method": row["DisplayName"],
                "ValDice": row["SelectedValDice"],
                "ValIoU": row.get("SelectedValIoU", "NOT_RECORDED"),
                "ValHD95": row.get("SelectedValHD95", "NOT_RECORDED"),
                "ValASD": row.get("SelectedValASD", "NOT_RECORDED"),
                "Model1ValDice": row["Model1ValDice"],
                "Model2ValDice": row["Model2ValDice"],
                "RunningBestValDice": row["RunningBestValDice"],
                "RunningBestIter": row["RunningBestIter"],
                "EndpointIdentity": row["EndpointIdentity"],
                "ConsumedTestPoint": row["ConsumedTestPoint"],
                "SourceLog": row["SourceLog"],
            }
        )
    out = RESULT_ROOT / "ACDC10_FULL_TRAINING_DYNAMICS.tsv"
    write_tsv(out, normalized_rows)

    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        grouped.setdefault(row["MethodID"], []).append(row)
    summary_rows = []
    for method, mrows in sorted(grouped.items()):
        best = max(mrows, key=lambda r: f(r["SelectedValDice"]))
        last = max(mrows, key=lambda r: int(r["Iteration"]))
        vals = [f(r["SelectedValDice"]) for r in mrows]
        max_iter = max(int(r["Iteration"]) for r in mrows)
        late_cut = max_iter * 0.8
        late = [r for r in mrows if int(r["Iteration"]) >= late_cut]
        late_vals = [f(r["SelectedValDice"]) for r in late]
        late_x = [int(r["Iteration"]) for r in late]
        x_mean = sum(late_x) / len(late_x)
        y_mean = sum(late_vals) / len(late_vals)
        denom = sum((x - x_mean) ** 2 for x in late_x)
        slope = 0.0 if denom == 0 else sum((x - x_mean) * (y - y_mean) for x, y in zip(late_x, late_vals)) / denom
        running_peak = -1.0
        max_drawdown = 0.0
        for r in sorted(mrows, key=lambda x: int(x["Iteration"])):
            val = f(r["SelectedValDice"])
            running_peak = max(running_peak, val)
            max_drawdown = max(max_drawdown, running_peak - val)
        summary_rows.append(
            {
                "MethodID": method,
                "DisplayName": best["DisplayName"],
                "PointCount": len(mrows),
                "MinIter": min(int(r["Iteration"]) for r in mrows),
                "MaxIter": max(int(r["Iteration"]) for r in mrows),
                "BestIter": best["Iteration"],
                "BestSelectedValDice": f"{f(best['SelectedValDice']):.12f}",
                "FinalIter": last["Iteration"],
                "FinalSelectedValDice": f"{f(last['SelectedValDice']):.12f}",
                "BestMinusFinalGap": f"{f(best['SelectedValDice']) - f(last['SelectedValDice']):.12f}",
                "LateStageMeanValDice": f"{sum(late_vals) / len(late_vals):.12f}",
                "LateStageStdValDice": f"{math.sqrt(sum((v - sum(late_vals) / len(late_vals)) ** 2 for v in late_vals) / len(late_vals)):.12f}",
                "LateStageSlopePerIteration": f"{slope:.12f}",
                "MaximumDrawdown": f"{max_drawdown:.12f}",
                "MeanSelectedValDice": f"{sum(vals) / len(vals):.12f}",
                "StdSelectedValDice": f"{math.sqrt(sum((v - sum(vals) / len(vals)) ** 2 for v in vals) / len(vals)):.12f}",
                "EndpointIdentity": "VALIDATION_LOG_POINT",
                "ConsumedTestPointSum": 0,
                "SourceAsset": str(DYNAMICS.relative_to(ROOT)),
            }
        )
    summary = RESULT_ROOT / "ACDC10_FULL_TRAINING_DYNAMICS_SUMMARY.tsv"
    write_tsv(summary, summary_rows)
    stability = RESULT_ROOT / "ACDC10_TRAINING_STABILITY_SUMMARY.tsv"
    write_tsv(stability, summary_rows)
    figure_assets = []
    if plt is not None:
        for method_filter, stem in [
            ({"ACDC10_ARM_000", "ACDC10_ARM_111"}, "ACDC10_VAL_DYNAMICS_BETAFFT_VS_OURS"),
            (set(grouped), "ACDC10_VAL_DYNAMICS_000_110_111"),
        ]:
            plt.figure(figsize=(8, 4.5))
            for method, mrows in sorted(grouped.items()):
                if method not in method_filter:
                    continue
                ordered = sorted(mrows, key=lambda r: int(r["Iteration"]))
                xs = [int(r["Iteration"]) for r in ordered]
                ys = [f(r["SelectedValDice"]) for r in ordered]
                plt.plot(xs, ys, marker="o", markersize=2.5, linewidth=1.0, label=ordered[0]["DisplayName"])
                if len(ys) >= 5:
                    ma = []
                    for idx in range(len(ys)):
                        lo = max(0, idx - 4)
                        ma.append(sum(ys[lo : idx + 1]) / (idx - lo + 1))
                    plt.plot(xs, ma, linewidth=1.6, alpha=0.75)
            plt.xlabel("Iteration")
            plt.ylabel("VAL Dice")
            plt.legend()
            plt.tight_layout()
            for suffix in ["png", "pdf"]:
                fig_path = RESULT_ROOT / "figures" / f"{stem}.{suffix}"
                fig_path.parent.mkdir(parents=True, exist_ok=True)
                plt.savefig(fig_path)
                figure_assets.append(str(fig_path.relative_to(ROOT)))
            plt.close()

    receipt = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "FULL_TRAINING_DYNAMICS",
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "validation_only": True,
        "test_point_count": 0,
        "input_asset": str(DYNAMICS.relative_to(ROOT)),
        "input_sha256": sha256(DYNAMICS),
        "output_asset": str(out.relative_to(ROOT)),
        "output_sha256": sha256(out),
        "summary_asset": str(summary.relative_to(ROOT)),
        "summary_sha256": sha256(summary),
        "stability_summary_asset": str(stability.relative_to(ROOT)),
        "stability_summary_sha256": sha256(stability),
        "figure_assets": figure_assets,
        "curve_row_count": len(rows),
        "method_count": len(grouped),
        "raw_points_visible_in_figures": bool(figure_assets),
        "moving_average_window_validation_points": 5,
    }
    write_json(AUDIT_ROOT / "FULL_TRAINING_DYNAMICS_RECEIPT.json", receipt)
    return receipt


def audit_patient_level_assets() -> dict:
    rows = []
    missing = []
    for arm, path in sorted(CASE_METRIC_CANDIDATES.items()):
        exists = path.exists()
        row_count = 0
        case_count = 0
        if exists:
            data = read_tsv(path)
            row_count = len(data)
            case_count = len({r["case"] for r in data})
        else:
            missing.append(arm)
        rows.append(
            {
                "Arm": arm,
                "ExpectedCaseClassMetrics": str(path.relative_to(ROOT)),
                "Exists": str(exists),
                "RowCount": row_count,
                "CaseCount": case_count,
                "PatientLevelStatsEligible": str(exists and row_count > 0),
            }
        )
    out = AUDIT_ROOT / "PATIENT_LEVEL_STATS_INPUT_ASSET_AUDIT.tsv"
    write_tsv(out, rows)
    receipt = {
        "status": "BLOCKED_MISSING_SOURCE_ASSET" if missing else "PASS",
        "created_at": now(),
        "task_id": "PATIENT_LEVEL_STATS",
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "required_arms": sorted(CASE_METRIC_CANDIDATES),
        "missing_case_class_metric_arms": missing,
        "audit_asset": str(out.relative_to(ROOT)),
        "audit_sha256": sha256(out),
        "next_action": "FROZEN_CHECKPOINT_REPLAY_REQUIRED_FOR_MISSING_ARMS" if missing else "RUN_PATIENT_LEVEL_STATS",
    }
    write_json(AUDIT_ROOT / "PATIENT_LEVEL_STATS_INPUT_ASSET_AUDIT.json", receipt)
    return receipt


def main() -> int:
    receipts = {
        "factorial_effects": extract_factorial_effects(),
        "full_training_dynamics": extract_training_dynamics(),
        "patient_level_stats_input_audit": audit_patient_level_assets(),
    }
    summary = {
        "status": "PASS_WITH_PATIENT_LEVEL_SOURCE_ASSET_BLOCKER"
        if receipts["patient_level_stats_input_audit"]["status"] != "PASS"
        else "PASS",
        "created_at": now(),
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "receipts": receipts,
    }
    write_json(AUDIT_ROOT / "V24_0_READONLY_PLANE_A_EXTRACTION_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
