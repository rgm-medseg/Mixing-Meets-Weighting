#!/usr/bin/env python3
"""Extract V23.0 ACDC10 8-arm class-wise analysis from V22.1 assets."""
from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "final_report/paper_internal_evidence_v23_0"
TABLE_ROOT = RESULT_ROOT / "tables"
AUDIT_ROOT = ROOT / "audits/paper_internal_evidence_v23_0"
SOURCE = ROOT / "final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fmt_pct(v: Any) -> str:
    return f"{float(v) * 100:.2f}"


def md_table(rows: list[dict[str, Any]]) -> str:
    cols = ["Mask Token", "AdaMix", "Contour Reliability", "RV Dice", "MYO Dice", "LV Dice", "Mean Dice"]
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join(["---"] * len(cols)) + " |"]
    for row in rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    row["Mask Token"],
                    row["AdaMix"],
                    row["Contour Reliability"],
                    fmt_pct(row["RV_Dice"]),
                    fmt_pct(row["MYO_Dice"]),
                    fmt_pct(row["LV_Dice"]),
                    fmt_pct(row["Mean_Dice"]),
                ]
            )
            + " |"
        )
    return "\n".join(lines) + "\n"


def tex_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "\\begin{tabular}{lllllll}",
        "\\toprule",
        "Mask Token & AdaMix & Contour Reliability & RV Dice & MYO Dice & LV Dice & Mean Dice \\\\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    row["Mask Token"],
                    row["AdaMix"],
                    row["Contour Reliability"],
                    fmt_pct(row["RV_Dice"]),
                    fmt_pct(row["MYO_Dice"]),
                    fmt_pct(row["LV_Dice"]),
                    fmt_pct(row["Mean_Dice"]),
                ]
            )
            + " \\\\"
        )
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def main() -> int:
    rows = read_tsv(SOURCE)
    dice_rows = []
    all_rows = []
    review = []
    for item in rows:
        rv = float(item["RV_Dice"])
        myo = float(item["MYO_Dice"])
        lv = float(item["LV_Dice"])
        mean_from_classes = (rv + myo + lv) / 3.0
        test_dice = float(item["TestDice"])
        dice_rows.append(
            {
                "Arm": item["Arm"],
                "Mask Token": "✓" if item["MaskToken"] == "ON" else "",
                "AdaMix": "✓" if item["AdaMix"] == "ON" else "",
                "Contour Reliability": "✓" if item["ContourReliabilityWeight"] == "ON" else "",
                "RV_Dice": rv,
                "MYO_Dice": myo,
                "LV_Dice": lv,
                "Mean_Dice": test_dice,
                "MeanDiceFromClassSimpleAverage": mean_from_classes,
                "MeanAggregationMatchesSimpleClassAverage": abs(mean_from_classes - test_dice) < 1e-10,
                "ResultSource": item["ResultSource"],
            }
        )
        for cls, dice in [("RV", rv), ("MYO", myo), ("LV", lv)]:
            all_rows.append(
                {
                    "Arm": item["Arm"],
                    "Class": cls,
                    "Dice": dice,
                    "IoU": "",
                    "HD95": "",
                    "ASD": "",
                    "MetricSource": "V22_1_8ARM_TABLE_CLASS_DICE_ONLY",
                    "SourceAsset": str(SOURCE.relative_to(ROOT)),
                }
            )
        review.append(
            {
                "Arm": item["Arm"],
                "TestDice": test_dice,
                "SimpleClassAverage": mean_from_classes,
                "Matches": abs(mean_from_classes - test_dice) < 1e-10,
                "Interpretation": "macro class mean" if abs(mean_from_classes - test_dice) < 1e-10 else "case-wise or weighted macro; do not recompute mean from class table",
            }
        )
    write_tsv(RESULT_ROOT / "ACDC10_8ARM_CLASSWISE_DICE_FULL_PRECISION.tsv", dice_rows)
    write_tsv(RESULT_ROOT / "ACDC10_8ARM_CLASSWISE_ALL_METRICS.tsv", all_rows)
    write_tsv(AUDIT_ROOT / "ACDC10_CLASSWISE_MEAN_AGGREGATION_REVIEW.tsv", review)
    (TABLE_ROOT / "TABLE5_ACDC10_CLASSWISE.md").write_text(md_table(dice_rows), encoding="utf-8")
    (TABLE_ROOT / "TABLE5_ACDC10_CLASSWISE.tex").write_text(tex_table(dice_rows), encoding="utf-8")
    summary = {
        "ACDC10_TEST_SELECTION_EXPOSED": True,
        "source_asset": str(SOURCE.relative_to(ROOT)),
        "dice_output": "final_report/paper_internal_evidence_v23_0/ACDC10_8ARM_CLASSWISE_DICE_FULL_PRECISION.tsv",
        "all_metrics_output": "final_report/paper_internal_evidence_v23_0/ACDC10_8ARM_CLASSWISE_ALL_METRICS.tsv",
        "mean_aggregation_all_simple_class_average": all(row["Matches"] for row in review),
        "note": "V22.1 full precision table provides class Dice for all arms; IoU/HD95/ASD per class require raw class_metrics files if needed later.",
    }
    (AUDIT_ROOT / "ACDC10_CLASSWISE_EXTRACTION_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
