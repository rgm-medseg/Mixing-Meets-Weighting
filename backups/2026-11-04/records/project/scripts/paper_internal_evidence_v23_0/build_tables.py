#!/usr/bin/env python3
"""Build V23.0 paper tables from full-precision local evidence."""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
TABLE_ROOT = ROOT / "final_report/paper_internal_evidence_v23_0/tables"
AUDIT_ROOT = ROOT / "audits/paper_internal_evidence_v23_0"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def load_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def fmt_pct(value: Any) -> str:
    return f"{float(value) * 100:.2f}"


def fmt_num(value: Any) -> str:
    return f"{float(value):.2f}"


def find(rows: list[dict[str, str]], key: str, value: str) -> dict[str, str]:
    matches = [row for row in rows if row.get(key) == value]
    if len(matches) != 1:
        raise SystemExit(f"EXPECTED_ONE_ROW:{key}={value}:found={len(matches)}")
    return matches[0]


def metric_value(row: dict[str, Any], metric: str) -> float:
    return float(row[metric])


def formatting_audit(table: str, block: str, rows: list[dict[str, Any]], metrics: list[str]) -> list[dict[str, Any]]:
    out = []
    for metric in metrics:
        higher = metric in {"Dice", "IoU", "RV_Dice", "MYO_Dice", "LV_Dice", "Mean_Dice"}
        sorted_rows = sorted(rows, key=lambda row: metric_value(row, metric), reverse=higher)
        best = sorted_rows[0]
        second = sorted_rows[1] if len(sorted_rows) > 1 else None
        for row in rows:
            rank = "BEST" if row is best else "SECOND" if second is not None and row is second else "OTHER"
            out.append(
                {
                    "Table": table,
                    "Block": block,
                    "Metric": metric,
                    "Direction": "HIGHER_IS_BETTER" if higher else "LOWER_IS_BETTER",
                    "Method": row["Method"],
                    "FullPrecisionValue": row[metric],
                    "Rank": rank,
                    "Bold": rank == "BEST",
                    "Underline": rank == "SECOND",
                }
            )
    return out


def row_block(row: dict[str, Any], block_keys: list[str]) -> str:
    return "|".join(str(row[key]) for key in block_keys) if block_keys else "ALL"


def display_audit(rows: list[dict[str, Any]], metrics: list[str], block_keys: list[str]) -> list[dict[str, Any]]:
    out = []
    blocks = sorted({row_block(row, block_keys) for row in rows})
    for block in blocks:
        block_rows = [row for row in rows if row_block(row, block_keys) == block]
        out.extend(formatting_audit("display", block, block_rows, metrics))
    return out


def markdown_table(rows: list[dict[str, Any]], columns: list[str], metrics: list[str], block_keys: list[str] | None = None) -> str:
    block_keys = block_keys or []
    audit = display_audit(rows, metrics, block_keys)
    rank = {(item["Block"], item["Method"], item["Metric"]): item["Rank"] for item in audit}
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if col in metrics:
                rendered = fmt_pct(value) if col in {"Dice", "IoU", "RV_Dice", "MYO_Dice", "LV_Dice", "Mean_Dice"} else fmt_num(value)
                item_rank = rank[(row_block(row, block_keys), row["Method"], col)]
                if item_rank == "BEST":
                    rendered = f"**{rendered}**"
                elif item_rank == "SECOND":
                    rendered = f"<u>{rendered}</u>"
                cells.append(rendered)
            else:
                cells.append(str(value))
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines) + "\n"


def latex_table(rows: list[dict[str, Any]], columns: list[str], metrics: list[str], block_keys: list[str] | None = None) -> str:
    block_keys = block_keys or []
    audit = display_audit(rows, metrics, block_keys)
    rank = {(item["Block"], item["Method"], item["Metric"]): item["Rank"] for item in audit}
    lines = ["\\begin{tabular}{" + "l" * len(columns) + "}", "\\toprule", " & ".join(columns).replace("_", " ") + " \\\\", "\\midrule"]
    for row in rows:
        cells = []
        for col in columns:
            value = row[col]
            if col in metrics:
                rendered = fmt_pct(value) if col in {"Dice", "IoU", "RV_Dice", "MYO_Dice", "LV_Dice", "Mean_Dice"} else fmt_num(value)
                item_rank = rank[(row_block(row, block_keys), row["Method"], col)]
                if item_rank == "BEST":
                    rendered = "\\textbf{" + rendered + "}"
                elif item_rank == "SECOND":
                    rendered = "\\underline{" + rendered + "}"
                cells.append(rendered)
            else:
                cells.append(str(value))
        lines.append(" & ".join(cells) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}", ""])
    return "\n".join(lines)


def write_table(name: str, rows: list[dict[str, Any]], columns: list[str], metrics: list[str], block_keys: list[str] | None = None) -> list[dict[str, Any]]:
    write_tsv(TABLE_ROOT / f"{name}_FULL_PRECISION.tsv", rows)
    (TABLE_ROOT / f"{name}.md").write_text(markdown_table(rows, columns, metrics, block_keys), encoding="utf-8")
    (TABLE_ROOT / f"{name}.tex").write_text(latex_table(rows, columns, metrics, block_keys), encoding="utf-8")
    return formatting_audit(name, "ALL_DISPLAYED_ROWS", rows, metrics)


def build_table1() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    acdc = read_tsv(ROOT / "final_report/ACDC_5P_10P_TEST_RESULTS.tsv")
    arm = read_tsv(ROOT / "final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv")
    acdc5_final3 = load_json(ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/TEST_RESULT_FULL_PRECISION.json")
    b5 = find(acdc, "method_id", "ACDC_5P_BASELINE")
    b10 = find(acdc, "method_id", "ACDC_10P_BASELINE")
    o10 = find(arm, "Arm", "111")
    rows = [
        {"Dataset": "ACDC", "Ratio": "5%", "Method": "β-FFT", "Dice": b5["dice"], "IoU": b5["iou"], "HD95": b5["hd95"], "ASD": b5["asd"], "SourceAsset": "final_report/ACDC_5P_10P_TEST_RESULTS.tsv"},
        {"Dataset": "ACDC", "Ratio": "5%", "Method": "Ours", "Dice": acdc5_final3["Dice"], "IoU": acdc5_final3["IoU"], "HD95": acdc5_final3["HD95"], "ASD": acdc5_final3["ASD"], "SourceAsset": "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/TEST_RESULT_FULL_PRECISION.json"},
        {"Dataset": "ACDC", "Ratio": "10%", "Method": "β-FFT", "Dice": b10["dice"], "IoU": b10["iou"], "HD95": b10["hd95"], "ASD": b10["asd"], "SourceAsset": "final_report/ACDC_5P_10P_TEST_RESULTS.tsv"},
        {"Dataset": "ACDC", "Ratio": "10%", "Method": "Ours", "Dice": o10["TestDice"], "IoU": o10["TestIoU"], "HD95": o10["TestHD95"], "ASD": o10["TestASD"], "SourceAsset": "final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv"},
    ]
    audit = []
    for ratio in ("5%", "10%"):
        audit.extend(formatting_audit("TABLE1_ACDC_5P_10P", ratio, [row for row in rows if row["Ratio"] == ratio], ["Dice", "IoU", "HD95", "ASD"]))
    write_table("TABLE1_ACDC_5P_10P", rows, ["Dataset", "Ratio", "Method", "Dice", "IoU", "HD95", "ASD"], ["Dice", "IoU", "HD95", "ASD"], ["Ratio"])
    return rows, audit


def build_table2() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    promise = read_tsv(ROOT / "final_report/PROMISE20_FINAL_TEST_RESULTS.tsv")
    final3 = load_json(ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/PROMISE20_FINAL3/TEST_RESULT_FULL_PRECISION.json")
    baseline = find(promise, "method_id", "PROMISE20_BASELINE_BETA_FFT_R1")
    rows = [
        {"Dataset": "PROMISE12", "Ratio": "20%", "Method": "β-FFT", "Dice": baseline["dice"], "IoU": baseline["iou"], "HD95": baseline["hd95"], "ASD": baseline["asd"], "SourceAsset": "final_report/PROMISE20_FINAL_TEST_RESULTS.tsv"},
        {"Dataset": "PROMISE12", "Ratio": "20%", "Method": "Ours", "Dice": final3["Dice"], "IoU": final3["IoU"], "HD95": final3["HD95"], "ASD": final3["ASD"], "SourceAsset": "final_report/three_innovation_postfull_test_v22_1/raw_test/PROMISE20_FINAL3/TEST_RESULT_FULL_PRECISION.json"},
    ]
    audit = formatting_audit("TABLE2_PROMISE20", "20%", rows, ["Dice", "IoU", "HD95", "ASD"])
    write_table("TABLE2_PROMISE20", rows, ["Dataset", "Ratio", "Method", "Dice", "IoU", "HD95", "ASD"], ["Dice", "IoU", "HD95", "ASD"], ["Ratio"])
    return rows, audit


def build_table3() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    covid = read_tsv(ROOT / "final_report/COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv")
    c5 = load_json(ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/COVID5_FINAL3/TEST_RESULT_FULL_PRECISION.json")
    c10 = load_json(ROOT / "final_report/three_innovation_postfull_test_v22_1/raw_test/COVID10_FINAL3/TEST_RESULT_FULL_PRECISION.json")
    rows = []
    for ratio, baseline_id, final3 in [
        ("5%", "COVID19_5P_BASELINE_CANON1337_R1", c5),
        ("10%", "COVID19_10P_BASELINE_CANON1337_R1", c10),
    ]:
        baseline = find(covid, "source_run_id", baseline_id)
        rows.append({"Dataset": "COVID-19", "Ratio": ratio, "Method": "β-FFT", "Dice": baseline["test_dice"], "IoU": baseline["test_iou"], "HD95": baseline["test_hd95_mm"], "ASD": baseline["test_asd_mm"], "SourceAsset": "final_report/COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv"})
        rows.append({"Dataset": "COVID-19", "Ratio": ratio, "Method": "Ours", "Dice": final3["Dice"], "IoU": final3["IoU"], "HD95": final3["HD95"], "ASD": final3["ASD"], "SourceAsset": f"final_report/three_innovation_postfull_test_v22_1/raw_test/{final3['TaskID']}/TEST_RESULT_FULL_PRECISION.json"})
    audit = []
    for ratio in ("5%", "10%"):
        audit.extend(formatting_audit("TABLE3_COVID_5P_10P", ratio, [row for row in rows if row["Ratio"] == ratio], ["Dice", "IoU", "HD95"]))
    write_table("TABLE3_COVID_5P_10P", rows, ["Dataset", "Ratio", "Method", "Dice", "IoU", "HD95"], ["Dice", "IoU", "HD95"], ["Ratio"])
    return rows, audit


def build_table4() -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    arm = read_tsv(ROOT / "final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv")
    rows = []
    for item in arm:
        rows.append(
            {
                "Mask Token": "✓" if item["MaskToken"] == "ON" else "",
                "AdaMix": "✓" if item["AdaMix"] == "ON" else "",
                "Contour Reliability": "✓" if item["ContourReliabilityWeight"] == "ON" else "",
                "Method": item["Arm"],
                "Dice": item["TestDice"],
                "IoU": item["TestIoU"],
                "HD95": item["TestHD95"],
                "ASD": item["TestASD"],
                "SourceAsset": "final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv",
            }
        )
    audit = formatting_audit("TABLE4_ACDC10_8ARM", "8ARM", rows, ["Dice", "IoU", "HD95", "ASD"])
    write_table("TABLE4_ACDC10_8ARM", rows, ["Mask Token", "AdaMix", "Contour Reliability", "Dice", "IoU", "HD95", "ASD"], ["Dice", "IoU", "HD95", "ASD"])
    return rows, audit


def main() -> int:
    TABLE_ROOT.mkdir(parents=True, exist_ok=True)
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    all_audit = []
    for builder in (build_table1, build_table2, build_table3, build_table4):
        _, audit = builder()
        all_audit.extend(audit)
    write_tsv(AUDIT_ROOT / "PAPER_TABLE_BOLD_UNDERLINE_AUDIT.tsv", all_audit)
    summary = {
        "TABLES_BUILT": ["TABLE1_ACDC_5P_10P", "TABLE2_PROMISE20", "TABLE3_COVID_5P_10P", "TABLE4_ACDC10_8ARM"],
        "DISPLAY_BETAFFT_REPORTED": False,
        "DISPLAY_MASK_ADAMIX_ROW": False,
        "DISPLAY_PLUS_MINUS": False,
        "ACDC_REMOVED_ROWS": ["Cross Teaching", "SCP-Net", "PLGCL"],
        "PROMISE10_IN_CURRENT_PAPER": False,
        "MSCMR_IN_CURRENT_PAPER": False,
        "FormattingAudit": "audits/paper_internal_evidence_v23_0/PAPER_TABLE_BOLD_UNDERLINE_AUDIT.tsv",
    }
    (AUDIT_ROOT / "TABLE_REBUILD_SUMMARY.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
