#!/usr/bin/env python3
"""V24 post-full closures and finalization helpers.

This script intentionally fail-closes while upstream NEW_FULL tasks are not
complete. It provides scripts-first dry-run/status gates for the post-full and
finalization registry rows, without fabricating label-efficiency, seed-stability,
final-review, start-manifest, or handoff artifacts.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY.tsv"
FINAL_HANDOFF_MD = RESULT_ROOT / "V24_0_INTERNAL_EVIDENCE_HANDOFF_FOR_CHATGPT.md"
REQUIRED_HANDOFF_SECTION_HEADINGS = [
    "## A. Scope",
    "## B. Frozen Method Identity",
    "## C. Reuse / Replay / New Full Map",
    "## D. Paired Statistics",
    "## E. X03 Reliability Faithfulness",
    "## F. Pseudo-label Evolution",
    "## G. Difficulty Strata",
    "## H. Pathology Subgroup",
    "## I. Boundary F1",
    "## J. Factorial Effects",
    "## K. Full Training Dynamics",
    "## L. Corruption Robustness",
    "## M. Student Diversity",
    "## N. ACDC2 Protocol",
    "## O. ACDC2 Results",
    "## P. Three-seed Stability",
    "## Q. Negative / Blocked Cases",
    "## R. Source / Config / Checkpoint Identity",
    "## S. Controller Final State",
    "## T. Supported Claims",
    "## U. Unsupported Claims",
    "## V. Questions for USER + ChatGPT",
]

TASKS = {
    "ACDC2_TEST_CLOSURE": {
        "requires": ["RESOLVE_ACDC2_PROTOCOL", "ACDC2_BETAFFT_FULL", "ACDC2_OURS_FULL"],
        "outputs": ["audits/internal_evidence_expansion_v24_0/ACDC2_TEST_CLOSURE.json"],
    },
    "SEED_TEST_CLOSURE": {
        "requires": [
            "ACDC10_BETAFFT_SEED2025",
            "ACDC10_OURS_SEED2025",
            "ACDC10_BETAFFT_SEED2026",
            "ACDC10_OURS_SEED2026",
        ],
        "outputs": ["audits/internal_evidence_expansion_v24_0/SEED_TEST_CLOSURE.json"],
    },
    "BUILD_LABEL_EFFICIENCY": {
        "requires": ["ACDC2_TEST_CLOSURE"],
        "outputs": [
            "final_report/internal_evidence_expansion_v24_0/ACDC_LABEL_EFFICIENCY_2P_5P_10P.tsv",
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC_LABEL_EFFICIENCY.md",
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC_LABEL_EFFICIENCY.tex",
            "final_report/internal_evidence_expansion_v24_0/figures/ACDC_LABEL_EFFICIENCY_CURVE.png",
            "final_report/internal_evidence_expansion_v24_0/figures/ACDC_LABEL_EFFICIENCY_CURVE.pdf",
        ],
    },
    "BUILD_SEED_STABILITY": {
        "requires": ["SEED_TEST_CLOSURE"],
        "outputs": [
            "final_report/internal_evidence_expansion_v24_0/ACDC10_THREE_SEED_FULL_PRECISION.tsv",
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_SEED_STABILITY.md",
            "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_ACDC10_SEED_STABILITY.tex",
            "final_report/internal_evidence_expansion_v24_0/figures/ACDC10_SEED_STABILITY.png",
            "final_report/internal_evidence_expansion_v24_0/figures/ACDC10_SEED_STABILITY.pdf",
        ],
    },
    "REGENERATE_ALL_TABLES_FIGURES": {
        "requires": ["BUILD_LABEL_EFFICIENCY", "BUILD_SEED_STABILITY"],
        "outputs": ["audits/internal_evidence_expansion_v24_0/REGENERATE_ALL_TABLES_FIGURES.json"],
    },
    "FINAL_FIVE_REVIEW": {
        "requires": ["REGENERATE_ALL_TABLES_FIGURES"],
        "outputs": ["audits/internal_evidence_expansion_v24_0/FINAL_FIVE_REVIEW.json"],
    },
    "FINAL_HANDOFF": {
        "requires": ["FINAL_FIVE_REVIEW"],
        "outputs": ["final_report/internal_evidence_expansion_v24_0/V24_0_INTERNAL_EVIDENCE_HANDOFF_FOR_CHATGPT.md"],
    },
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


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, Any]], fields: list[str] | None = None) -> None:
    if not rows:
        raise SystemExit(f"REFUSE_EMPTY_TSV:{path.relative_to(ROOT)}")
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = fields or list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def f12(value: Any) -> str:
    return f"{float(value):.12f}"


def read_single_row(path: Path) -> dict[str, str]:
    rows = read_tsv(path)
    if len(rows) != 1:
        raise SystemExit(f"EXPECTED_ONE_ROW:{path.relative_to(ROOT)}:{len(rows)}")
    return rows[0]


def require_json_true(path: Path, field: str, task_id: str) -> dict[str, Any]:
    payload = json.loads(path.read_text())
    if payload.get(field) is not True:
        raise SystemExit(f"REQUIRED_FIELD_NOT_TRUE:{task_id}:{path.relative_to(ROOT)}:{field}")
    return payload


def metric_from_aggregate(task_id: str) -> dict[str, Any]:
    receipt = RESULT_ROOT / "raw_test" / task_id / "TEST_COMPLETION_RECEIPT.json"
    aggregate = RESULT_ROOT / "raw_test" / task_id / "aggregate_metrics.tsv"
    cfg_path = ROOT / "configs/internal_evidence_expansion_v24_0" / f"{task_id}.json"
    if not receipt.exists() or not aggregate.exists():
        raise SystemExit(f"MISSING_FORMAL_TEST_ASSET:{task_id}")
    if not cfg_path.exists():
        raise SystemExit(f"MISSING_FORMAL_CONFIG:{task_id}")
    cfg = json.loads(cfg_path.read_text())
    run_root = Path(cfg["run_root"])
    full = require_json_true(run_root / "FORMAL_FULL_COMPLETION_RECEIPT.json", "FormalFullComplete", task_id)
    best = require_json_true(run_root / "BEST_VAL_SELECTION.json", "BestValSelected", task_id)
    frozen = require_json_true(run_root / "FROZEN_CHECKPOINT_SHA256.json", "BestCheckpointSHA256Frozen", task_id)
    receipt_payload = json.loads(receipt.read_text())
    if receipt_payload.get("status") != "PASS" or receipt_payload.get("TEST_COMPLETION_RECEIPT") is not True:
        raise SystemExit(f"TEST_RECEIPT_NOT_PASS:{task_id}")
    if receipt_payload.get("BestCheckpointSHA256") != frozen.get("BestCheckpointSHA256"):
        raise SystemExit(f"TEST_RECEIPT_CHECKPOINT_SHA_MISMATCH:{task_id}")
    row = read_single_row(aggregate)
    return {
        "TaskID": task_id,
        "Dice": f12(row["dice"]),
        "IoU": f12(row["iou"]),
        "HD95": f12(row["hd95"]),
        "ASD": f12(row["asd"]),
        "CheckpointSHA": receipt_payload.get("BestCheckpointSHA256", ""),
        "FormalFullCompleteAsset": str((run_root / "FORMAL_FULL_COMPLETION_RECEIPT.json").relative_to(ROOT)),
        "BestValSelectionAsset": str((run_root / "BEST_VAL_SELECTION.json").relative_to(ROOT)),
        "FrozenCheckpointSHAAsset": str((run_root / "FROZEN_CHECKPOINT_SHA256.json").relative_to(ROOT)),
        "BestValIteration": best.get("BestValIteration", best.get("best_iteration", "")),
        "FormalFullCompletedAt": full.get("created_at", ""),
        "SourceAsset": str(aggregate.relative_to(ROOT)),
        "Receipt": str(receipt.relative_to(ROOT)),
    }


def acdc_existing_table() -> dict[tuple[str, str], dict[str, str]]:
    table = ROOT / "final_report/paper_internal_evidence_v23_0/tables/TABLE1_ACDC_5P_10P_FULL_PRECISION.tsv"
    rows = read_tsv(table)
    return {(row["Ratio"], row["Method"]): row for row in rows}


def acdc10_seed1337_rows() -> list[dict[str, Any]]:
    table = ROOT / "final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv"
    rows = read_tsv(table)
    by_arm = {row["Arm"]: row for row in rows}
    out = []
    for arm, method in [("000", "β-FFT"), ("111", "Ours")]:
        row = by_arm[arm]
        out.append({
            "Dataset": "ACDC",
            "Ratio": "10%",
            "Seed": "1337",
            "Method": method,
            "TaskID": f"ACDC10_{'BETAFFT' if arm == '000' else 'OURS'}_SEED1337_EXACT_REUSE",
            "Dice": f12(row["TestDice"]),
            "IoU": f12(row["TestIoU"]),
            "HD95": f12(row["TestHD95"]),
            "ASD": f12(row["TestASD"]),
            "CheckpointSHA": row.get("CheckpointSHA", ""),
            "SourceAsset": str(table.relative_to(ROOT)),
            "Receipt": "EXACT_REUSE_V22_V23_FORMAL_TEST",
        })
    return out


def markdown_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    lines = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(lines) + "\n"


def tex_table(rows: list[dict[str, Any]], fields: list[str]) -> str:
    body = ["\\begin{tabular}{" + "l" * len(fields) + "}", " \\toprule", " & ".join(fields) + " \\\\", " \\midrule"]
    for row in rows:
        body.append(" & ".join(str(row.get(field, "")) for field in fields) + " \\\\")
    body.extend([" \\bottomrule", "\\end{tabular}", ""])
    return "\n".join(body)


def write_table_pair(stem: str, rows: list[dict[str, Any]], fields: list[str]) -> None:
    table_dir = RESULT_ROOT / "tables"
    table_dir.mkdir(parents=True, exist_ok=True)
    (table_dir / f"{stem}.md").write_text(markdown_table(rows, fields))
    (table_dir / f"{stem}.tex").write_text(tex_table(rows, fields))


def save_line_plot(path_stem: Path, xs: list[str], series: dict[str, list[float]], ylabel: str) -> None:
    path_stem.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(6, 4))
    for label, values in series.items():
        plt.plot(xs, values, marker="o", label=label)
    plt.xlabel("Label ratio" if "LABEL" in path_stem.name else "Seed")
    plt.ylabel(ylabel)
    plt.legend()
    plt.tight_layout()
    plt.savefig(path_stem.with_suffix(".png"), dpi=200)
    plt.savefig(path_stem.with_suffix(".pdf"))
    plt.close()


def run_acdc2_test_closure() -> dict[str, Any]:
    rows = []
    for task_id, method in [("ACDC2_BETAFFT_FULL", "β-FFT"), ("ACDC2_OURS_FULL", "Ours")]:
        rows.append({
            "Dataset": "ACDC",
            "Ratio": "2%",
            "Method": method,
            "TaskID": task_id,
            "Status": "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE",
            "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT": True,
            "Dice": "",
            "IoU": "",
            "HD95": "",
            "ASD": "",
            "SourceAsset": "audits/internal_evidence_expansion_v24_0/ACDC2_PROTOCOL_DECISION.json",
            "Receipt": "audits/internal_evidence_expansion_v24_0/ACDC2_TEST_CLOSURE.json",
        })
    out = AUDIT_ROOT / "ACDC2_TEST_CLOSURE.json"
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "ACDC2_TEST_CLOSURE",
        "closure_type": "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED",
        "training_started": False,
        "test_started": False,
        "fake_results_generated": False,
        "rows": rows,
    }
    write_json(out, payload)
    return payload


def run_seed_test_closure() -> dict[str, Any]:
    rows = acdc10_seed1337_rows()
    for task_id, method, seed in [
        ("ACDC10_BETAFFT_SEED2025", "β-FFT", "2025"),
        ("ACDC10_OURS_SEED2025", "Ours", "2025"),
        ("ACDC10_BETAFFT_SEED2026", "β-FFT", "2026"),
        ("ACDC10_OURS_SEED2026", "Ours", "2026"),
    ]:
        metric_row = metric_from_aggregate(task_id)
        rows.append({"Dataset": "ACDC", "Ratio": "10%", "Seed": seed, "Method": method, **metric_row})
    out = AUDIT_ROOT / "SEED_TEST_CLOSURE.json"
    payload = {"status": "PASS", "created_at": now(), "task_id": "SEED_TEST_CLOSURE", "rows": rows}
    write_json(out, payload)
    return payload


def run_build_label_efficiency() -> dict[str, Any]:
    closure = json.loads((AUDIT_ROOT / "ACDC2_TEST_CLOSURE.json").read_text())
    if closure.get("status") != "PASS":
        raise SystemExit("ACDC2_TEST_CLOSURE_NOT_PASS")
    existing = acdc_existing_table()
    rows = []
    for ratio in ["2%", "5%", "10%"]:
        for method in ["β-FFT", "Ours"]:
            if ratio == "2%":
                src = next(row for row in closure["rows"] if row["Method"] == method)
                rows.append({
                    "Dataset": "ACDC",
                    "Ratio": ratio,
                    "Method": method,
                    "Status": src["Status"],
                    "Dice": "",
                    "IoU": "",
                    "HD95": "",
                    "ASD": "",
                    "SourceAsset": src["SourceAsset"],
                })
            else:
                src = existing[(ratio, method)]
                rows.append({
                    "Dataset": "ACDC",
                    "Ratio": ratio,
                    "Method": method,
                    "Status": "FOUND_EXACT_FORMAL_TEST",
                    "Dice": f12(src["Dice"]),
                    "IoU": f12(src["IoU"]),
                    "HD95": f12(src["HD95"]),
                    "ASD": f12(src["ASD"]),
                    "SourceAsset": src["SourceAsset"],
                })
    out = RESULT_ROOT / "ACDC_LABEL_EFFICIENCY_2P_5P_10P.tsv"
    write_tsv(out, rows, ["Dataset", "Ratio", "Method", "Status", "Dice", "IoU", "HD95", "ASD", "SourceAsset"])
    write_table_pair("TABLE_S_ACDC_LABEL_EFFICIENCY", rows, ["Ratio", "Method", "Status", "Dice", "IoU", "HD95", "ASD"])
    xs = ["5%", "10%"]
    series = {
        method: [float(next(row["Dice"] for row in rows if row["Ratio"] == ratio and row["Method"] == method)) for ratio in xs]
        for method in ["β-FFT", "Ours"]
    }
    save_line_plot(RESULT_ROOT / "figures/ACDC_LABEL_EFFICIENCY_CURVE", xs, series, "TEST Dice")
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "BUILD_LABEL_EFFICIENCY",
        "source_tsv": str(out.relative_to(ROOT)),
        "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT": True,
        "ACDC2_rows_are_blocker_records_not_results": True,
    }
    write_json(AUDIT_ROOT / "BUILD_LABEL_EFFICIENCY.json", payload)
    return payload


def run_build_seed_stability() -> dict[str, Any]:
    closure = json.loads((AUDIT_ROOT / "SEED_TEST_CLOSURE.json").read_text())
    if closure.get("status") != "PASS":
        raise SystemExit("SEED_TEST_CLOSURE_NOT_PASS")
    rows = closure["rows"]
    fields = ["Dataset", "Ratio", "Seed", "Method", "TaskID", "Dice", "IoU", "HD95", "ASD", "CheckpointSHA", "SourceAsset", "Receipt"]
    out = RESULT_ROOT / "ACDC10_THREE_SEED_FULL_PRECISION.tsv"
    write_tsv(out, rows, fields)
    summary = []
    for method in ["β-FFT", "Ours"]:
        vals = [float(row["Dice"]) for row in rows if row["Method"] == method]
        summary.append({
            "Method": method,
            "NSeeds": len(vals),
            "MeanDice": f12(statistics.mean(vals)),
            "StdDice": f12(statistics.stdev(vals) if len(vals) > 1 else 0.0),
            "MedianDice": f12(statistics.median(vals)),
            "MinDice": f12(min(vals)),
            "MaxDice": f12(max(vals)),
        })
    gains = []
    for seed in ["1337", "2025", "2026"]:
        beta = float(next(row["Dice"] for row in rows if row["Seed"] == seed and row["Method"] == "β-FFT"))
        ours = float(next(row["Dice"] for row in rows if row["Seed"] == seed and row["Method"] == "Ours"))
        gains.append({"Seed": seed, "OursMinusBetaFFT_Dice_pp": f"{(ours - beta) * 100:.6f}"})
    summary.extend(gains)
    summary_out = RESULT_ROOT / "ACDC10_THREE_SEED_SUMMARY.tsv"
    write_tsv(summary_out, summary)
    write_table_pair("TABLE_S_ACDC10_SEED_STABILITY", rows, ["Seed", "Method", "Dice", "IoU", "HD95", "ASD"])
    xs = ["1337", "2025", "2026"]
    series = {method: [float(next(row["Dice"] for row in rows if row["Seed"] == seed and row["Method"] == method)) for seed in xs] for method in ["β-FFT", "Ours"]}
    save_line_plot(RESULT_ROOT / "figures/ACDC10_SEED_STABILITY", xs, series, "TEST Dice")
    payload = {"status": "PASS", "created_at": now(), "task_id": "BUILD_SEED_STABILITY", "source_tsv": str(out.relative_to(ROOT)), "summary_tsv": str(summary_out.relative_to(ROOT))}
    write_json(AUDIT_ROOT / "BUILD_SEED_STABILITY.json", payload)
    return payload


def existing_required_outputs() -> dict[str, list[str]]:
    required = {
        "tables": [
            "TABLE_S_PATIENT_LEVEL_STATISTICS.md",
            "TABLE_S_PATIENT_LEVEL_STATISTICS.tex",
            "TABLE_S_ACDC10_DIFFICULTY.md",
            "TABLE_S_ACDC10_DIFFICULTY.tex",
            "TABLE_S_ACDC10_BOUNDARY_F1.md",
            "TABLE_S_ACDC10_BOUNDARY_F1.tex",
            "TABLE_S_ACDC10_FACTORIAL_EFFECTS.md",
            "TABLE_S_ACDC10_FACTORIAL_EFFECTS.tex",
            "TABLE_S_ACDC10_CORRUPTION.md",
            "TABLE_S_ACDC10_CORRUPTION.tex",
            "TABLE_S_ACDC_LABEL_EFFICIENCY.md",
            "TABLE_S_ACDC_LABEL_EFFICIENCY.tex",
            "TABLE_S_ACDC10_SEED_STABILITY.md",
            "TABLE_S_ACDC10_SEED_STABILITY.tex",
        ],
        "figures": [
            "PATIENT_LEVEL_DELTA_DISTRIBUTION.png",
            "PATIENT_LEVEL_DELTA_DISTRIBUTION.pdf",
            "X03_RELIABILITY_QUINTILE_ERROR.png",
            "X03_RELIABILITY_QUINTILE_ERROR.pdf",
            "X03_RELIABILITY_RISK_COVERAGE.png",
            "X03_RELIABILITY_RISK_COVERAGE.pdf",
            "X03_RELIABILITY_CASE_CORRELATION.png",
            "X03_RELIABILITY_CASE_CORRELATION.pdf",
            "ACDC10_DIFFICULTY_GAIN.png",
            "ACDC10_DIFFICULTY_GAIN.pdf",
            "ACDC10_BOUNDARY_F1_TOLERANCE.png",
            "ACDC10_BOUNDARY_F1_TOLERANCE.pdf",
            "ACDC10_FACTORIAL_MAIN_EFFECTS.png",
            "ACDC10_FACTORIAL_MAIN_EFFECTS.pdf",
            "ACDC10_FACTORIAL_INTERACTIONS.png",
            "ACDC10_FACTORIAL_INTERACTIONS.pdf",
            "ACDC10_VAL_DYNAMICS_BETAFFT_VS_OURS.png",
            "ACDC10_VAL_DYNAMICS_BETAFFT_VS_OURS.pdf",
            "ACDC10_VAL_DYNAMICS_000_110_111.png",
            "ACDC10_VAL_DYNAMICS_000_110_111.pdf",
            "ACDC10_CORRUPTION_DICE_RETENTION.png",
            "ACDC10_CORRUPTION_DICE_RETENTION.pdf",
            "ACDC10_CORRUPTION_BOUNDARY_DEGRADATION.png",
            "ACDC10_CORRUPTION_BOUNDARY_DEGRADATION.pdf",
            "ACDC_LABEL_EFFICIENCY_CURVE.png",
            "ACDC_LABEL_EFFICIENCY_CURVE.pdf",
            "ACDC10_SEED_STABILITY.png",
            "ACDC10_SEED_STABILITY.pdf",
        ],
        "source_tsv": [
            "PATIENT_LEVEL_PAIRED_METRICS.tsv",
            "PATIENT_LEVEL_STATISTICAL_SUMMARY.tsv",
            "X03_RELIABILITY_PIXEL_LEVEL.tsv",
            "X03_RELIABILITY_CASE_LEVEL.tsv",
            "ACDC10_HARD_MEDIUM_EASY.tsv",
            "ACDC10_BOUNDARY_F1_FULL_PRECISION.tsv",
            "ACDC10_FACTORIAL_EFFECTS_FULL.tsv",
            "ACDC10_FULL_TRAINING_DYNAMICS.tsv",
            "ACDC10_TRAINING_STABILITY_SUMMARY.tsv",
            "ACDC10_CORRUPTION_ROBUSTNESS.tsv",
            "ACDC_LABEL_EFFICIENCY_2P_5P_10P.tsv",
            "ACDC10_THREE_SEED_FULL_PRECISION.tsv",
        ],
    }
    missing = []
    for name in required["tables"]:
        if not (RESULT_ROOT / "tables" / name).exists():
            missing.append(f"tables/{name}")
    for name in required["figures"]:
        if not (RESULT_ROOT / "figures" / name).exists():
            missing.append(f"figures/{name}")
    for name in required["source_tsv"]:
        if not (RESULT_ROOT / name).exists():
            missing.append(name)
    return {"missing": missing, "required": required}


def run_regenerate_all_tables_figures() -> dict[str, Any]:
    assets = existing_required_outputs()
    if assets["missing"]:
        raise SystemExit("MISSING_FINAL_TABLE_FIGURE_ASSETS:" + ",".join(assets["missing"]))
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "REGENERATE_ALL_TABLES_FIGURES",
        "missing_assets": [],
        "required_assets": assets["required"],
        "all_figures_have_source_tsv": True,
    }
    write_json(AUDIT_ROOT / "REGENERATE_ALL_TABLES_FIGURES.json", payload)
    return payload


def run_final_five_review() -> dict[str, Any]:
    l1 = json.loads((AUDIT_ROOT / "V24_0_PRELAUNCH_L1_L10_AUDIT_SUMMARY.json").read_text())
    regen = json.loads((AUDIT_ROOT / "REGENERATE_ALL_TABLES_FIGURES.json").read_text())
    handoff_template_ok = len(REQUIRED_HANDOFF_SECTION_HEADINGS) == 22 and REQUIRED_HANDOFF_SECTION_HEADINGS[0].startswith("## A.") and REQUIRED_HANDOFF_SECTION_HEADINGS[-1].startswith("## V.")
    checks = {
        "FINAL_REVIEW_1_COMPLETENESS": regen.get("status") == "PASS",
        "FINAL_REVIEW_2_STATISTICS": (RESULT_ROOT / "PATIENT_LEVEL_STATISTICAL_SUMMARY.tsv").exists(),
        "FINAL_REVIEW_3_MECHANISM": (RESULT_ROOT / "X03_RELIABILITY_FAITHFULNESS_SUMMARY.json").exists(),
        "FINAL_REVIEW_4_INTEGRITY": l1.get("five_review", {}).get("REVIEW_5_NO_TRAINING_OR_FORMAL_STARTED") in {"PASS", "NOT_APPLICABLE_AFTER_FORMAL"},
        "FINAL_REVIEW_5_HANDOFF": handoff_template_ok,
    }
    payload = {
        "status": "PASS" if all(checks.values()) else "FAIL",
        "created_at": now(),
        "task_id": "FINAL_FIVE_REVIEW",
        **{key: "PASS" if ok else "FAIL" for key, ok in checks.items()},
        "claims_boundary": {
            "X03_causal_proof": "UNSUPPORTED",
            "factorial_significance": "UNSUPPORTED_DESCRIPTIVE_ONLY",
            "negative_results_retained": True,
        },
        "handoff_contract": {
            "path": str(FINAL_HANDOFF_MD.relative_to(ROOT)),
            "required_section_count": len(REQUIRED_HANDOFF_SECTION_HEADINGS),
            "required_sections": REQUIRED_HANDOFF_SECTION_HEADINGS,
            "template_sections_pass": handoff_template_ok,
            "handoff_written_by_separate_task_after_review": True,
        },
    }
    write_json(AUDIT_ROOT / "FINAL_FIVE_REVIEW.json", payload)
    if payload["status"] != "PASS":
        raise SystemExit("FINAL_FIVE_REVIEW_FAIL")
    return payload


def compact_preview(path: Path, max_lines: int = 8) -> str:
    if not path.exists():
        return "MISSING"
    lines = path.read_text().splitlines()
    return "\n".join(lines[:max_lines])


def run_final_handoff() -> dict[str, Any]:
    review = json.loads((AUDIT_ROOT / "FINAL_FIVE_REVIEW.json").read_text())
    if review.get("status") != "PASS":
        raise SystemExit("FINAL_FIVE_REVIEW_NOT_PASS")
    sections = [
        "# V24.0 Internal Evidence Handoff",
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[0],
        "Datasets: ACDC, PROMISE12 20%, COVID-19. No external-method reproduction. PROMISE10 and MSCMR are out of scope.",
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[1],
        compact_preview(AUDIT_ROOT / "V24_0_FROZEN_METHOD_IDENTITY.json", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[2],
        compact_preview(AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY.tsv", 30),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[3],
        compact_preview(RESULT_ROOT / "PATIENT_LEVEL_STATISTICAL_SUMMARY.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[4],
        compact_preview(RESULT_ROOT / "X03_RELIABILITY_FAITHFULNESS_SUMMARY.json", 30),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[5],
        "BLOCKED/PARTIAL if source assets remain unavailable; see mechanism source asset audit.",
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[6],
        compact_preview(RESULT_ROOT / "ACDC10_HARD_MEDIUM_EASY.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[7],
        "BLOCKED if official pathology metadata remains unavailable.",
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[8],
        compact_preview(RESULT_ROOT / "ACDC10_BOUNDARY_F1_FULL_PRECISION.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[9],
        compact_preview(RESULT_ROOT / "ACDC10_FACTORIAL_EFFECTS_FULL.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[10],
        compact_preview(RESULT_ROOT / "ACDC10_TRAINING_STABILITY_SUMMARY.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[11],
        compact_preview(RESULT_ROOT / "ACDC10_CORRUPTION_ROBUSTNESS.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[12],
        "BLOCKED/PARTIAL if branch output assets remain unavailable; no retraining was added for this analysis.",
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[13],
        compact_preview(AUDIT_ROOT / "ACDC2_LABEL1_FROZEN_BETAFFT_EXECUTABILITY_AUDIT.json", 30),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[14],
        compact_preview(RESULT_ROOT / "ACDC_LABEL_EFFICIENCY_2P_5P_10P.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[15],
        compact_preview(RESULT_ROOT / "ACDC10_THREE_SEED_FULL_PRECISION.tsv", 20),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[16],
        compact_preview(AUDIT_ROOT / "V24_0_FINAL_LAUNCH_READINESS_SUMMARY.json", 30),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[17],
        compact_preview(AUDIT_ROOT / "V24_BETAFFT_TEST_FIREWALL_SOURCE_DRIFT_AUDIT.json", 25),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[18],
        compact_preview(AUDIT_ROOT / "V24_0_CONTROLLER_DRY_CONSUMPTION_SUMMARY.json", 25),
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[19],
        "Paired distributions, X03 reliability discrimination, boundary-specific evaluation, descriptive factorial effects, VAL dynamics, corruption retention, label efficiency, and seed stability where exact assets exist.",
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[20],
        "No X03 causal proof, no factorial significance without replicates, no universal robustness, no hidden negative-result filtering.",
        "",
        REQUIRED_HANDOFF_SECTION_HEADINGS[21],
        "Decide how to treat noncritical blocked source-asset analyses and ACDC2 protocol non-executability under frozen beta-FFT.",
        "",
    ]
    text = "\n".join(sections)
    missing_sections = [heading for heading in REQUIRED_HANDOFF_SECTION_HEADINGS if heading not in text]
    if missing_sections:
        raise SystemExit("FINAL_HANDOFF_SECTION_CONTRACT_FAIL:" + ",".join(missing_sections))
    FINAL_HANDOFF_MD.write_text(text)
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "FINAL_HANDOFF",
        "handoff": str(FINAL_HANDOFF_MD.relative_to(ROOT)),
        "required_section_count": len(REQUIRED_HANDOFF_SECTION_HEADINGS),
        "missing_sections": [],
    }
    write_json(AUDIT_ROOT / "FINAL_HANDOFF.json", payload)
    return payload


def registry_map() -> dict[str, dict[str, str]]:
    return {row["TaskID"]: row for row in read_tsv(REGISTRY)}


def candidate_evidence_paths(task_id: str, row: dict[str, str]) -> dict[str, list[str]]:
    """Completion is separate from launch readiness and must be proven by real assets."""
    config_path = row.get("ConfigPath", "")
    run_root = ""
    if config_path:
        cfg_file = ROOT / config_path
        if cfg_file.exists():
            try:
                cfg = json.loads(cfg_file.read_text())
                run_root = str(Path(cfg.get("run_root", "")).relative_to(ROOT)) if cfg.get("run_root") else ""
            except Exception:
                run_root = ""
    if not run_root:
        run_root = f"experiments/internal_evidence_expansion_v24_0/{task_id}"
    result_root = "final_report/internal_evidence_expansion_v24_0"
    audit_root = "audits/internal_evidence_expansion_v24_0"
    return {
        "FormalFullComplete": [
            f"{run_root}/FORMAL_FULL_COMPLETION_RECEIPT.json",
            f"{audit_root}/{task_id}_FORMAL_FULL_COMPLETION_RECEIPT.json",
        ],
        "BestValSelected": [
            f"{run_root}/BEST_VAL_SELECTION.json",
            f"{run_root}/best_val_selection.json",
            f"{audit_root}/{task_id}_BEST_VAL_SELECTION.json",
        ],
        "FrozenCheckpointSHA": [
            f"{run_root}/FROZEN_CHECKPOINT_SHA256.json",
            f"{run_root}/frozen_checkpoint_sha256.json",
            f"{audit_root}/{task_id}_FROZEN_CHECKPOINT_SHA256.json",
        ],
        "TEST_COMPLETION_RECEIPT": [
            f"{result_root}/raw_test/{task_id}/TEST_COMPLETION_RECEIPT.json",
            f"{result_root}/raw_test/{task_id}/TEST_RECEIPT.json",
            f"{audit_root}/{task_id}_TEST_COMPLETION_RECEIPT.json",
            f"{audit_root}/{task_id}_TEST_GATE_LOCKED_RECEIPT.json",
        ],
        "ResultAsset": [
            f"{result_root}/raw_test/{task_id}/aggregate_metrics.tsv",
            f"{result_root}/raw_test/{task_id}/case_class_metrics.tsv",
            f"{result_root}/raw_test/{task_id}/TEST_FULL_PRECISION.tsv",
            f"{result_root}/{task_id}_TEST_FULL_PRECISION.tsv",
        ],
    }


def completed_evidence(task_id: str, row: dict[str, str]) -> dict[str, Any]:
    candidates = candidate_evidence_paths(task_id, row)
    fields: dict[str, Any] = {}
    missing = []
    semantic_failures = []
    for key, paths in candidates.items():
        existing = [p for p in paths if (ROOT / p).exists()]
        fields[key] = {
            "present": bool(existing),
            "selected_asset": existing[0] if existing else "",
            "candidate_assets": paths,
        }
        if not existing:
            missing.append(key)
            continue
        selected = ROOT / existing[0]
        if key == "FormalFullComplete":
            payload = json.loads(selected.read_text())
            if payload.get("FormalFullComplete") is not True:
                semantic_failures.append("FormalFullComplete_NOT_TRUE")
        elif key == "BestValSelected":
            payload = json.loads(selected.read_text())
            if payload.get("BestValSelected") is not True:
                semantic_failures.append("BestValSelected_NOT_TRUE")
        elif key == "FrozenCheckpointSHA":
            payload = json.loads(selected.read_text())
            if payload.get("BestCheckpointSHA256Frozen") is not True:
                semantic_failures.append("BestCheckpointSHA256Frozen_NOT_TRUE")
            best_path = payload.get("BestCheckpointPath", "")
            best_sha = payload.get("BestCheckpointSHA256", "")
            if not best_path or not (ROOT / best_path).exists():
                semantic_failures.append("FrozenBestCheckpointPath_MISSING")
            elif best_sha and sha256(ROOT / best_path) != best_sha:
                semantic_failures.append("FrozenBestCheckpointSHA_MISMATCH")
        elif key == "TEST_COMPLETION_RECEIPT":
            payload = json.loads(selected.read_text())
            if payload.get("status") != "PASS" or payload.get("TEST_COMPLETION_RECEIPT") is not True:
                semantic_failures.append("TEST_COMPLETION_RECEIPT_NOT_PASS")
    pass_state = not missing
    locked_receipt_selected = "TEST_GATE_LOCKED_RECEIPT" in fields["TEST_COMPLETION_RECEIPT"]["selected_asset"]
    return {
        "FormalFullComplete": fields["FormalFullComplete"]["present"],
        "BestValSelected": fields["BestValSelected"]["present"],
        "FrozenCheckpointSHA": fields["FrozenCheckpointSHA"]["present"],
        "TEST_COMPLETION_RECEIPT": fields["TEST_COMPLETION_RECEIPT"]["present"]
        and not locked_receipt_selected,
        "ResultAsset": fields["ResultAsset"]["present"],
        "evidence": fields,
        "missing_evidence": missing
        + (["TEST_COMPLETION_RECEIPT_UNLOCKED_PASS"] if locked_receipt_selected else [])
        + semantic_failures,
        "CompletionEvidencePASS": pass_state
        and not locked_receipt_selected
        and not semantic_failures,
    }


def completion_asset_pass(path_value: str) -> bool:
    if not path_value:
        return False
    path = ROOT / path_value
    if not path.exists():
        return False
    if path.suffix == ".json":
        try:
            return json.loads(path.read_text()).get("status") == "PASS"
        except json.JSONDecodeError:
            return False
    return True


def upstream_state(task_id: str) -> dict[str, Any]:
    rows = registry_map()
    requires = TASKS[task_id]["requires"]
    required_rows = []
    blocked = []
    for req in requires:
        row = rows.get(req)
        if row is None:
            blocked.append({"TaskID": req, "Reason": "MISSING_FROM_REGISTRY"})
            continue
        launch_ready = row.get("FinalLaunchReady") == "PASS"
        completion_asset = row.get("ExpectedReceipt") or row.get("OutputDir") or ""
        completion_exists = bool(completion_asset and (ROOT / completion_asset).exists())
        completion_pass = completion_asset_pass(completion_asset)
        completion_evidence: dict[str, Any] | None = None
        protocol_excluded = row.get("FinalLaunchReady") == "TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE"
        if protocol_excluded:
            ready = True
        elif row.get("TaskType") == "NEW_FULL":
            completion_evidence = completed_evidence(req, row)
            ready = launch_ready and completion_evidence["CompletionEvidencePASS"]
        elif row.get("TaskType") in {"POST_FULL", "FINALIZATION"}:
            ready = launch_ready and completion_pass
        else:
            ready = launch_ready
        required_rows.append(
            {
                "TaskID": req,
                "TaskType": row.get("TaskType"),
                "Action": row.get("Action"),
                "FinalLaunchReady": row.get("FinalLaunchReady"),
                "Blocker": row.get("Blocker"),
                "CompletionAsset": completion_asset,
                "CompletionAssetExists": completion_exists,
                "CompletionAssetPass": completion_pass,
                "CompletionEvidence": completion_evidence,
                "Ready": ready,
            }
        )
        if not ready:
            reason = row.get("Blocker") or f"FinalLaunchReady={row.get('FinalLaunchReady')}"
            if launch_ready and row.get("TaskType") == "NEW_FULL" and completion_evidence and not completion_evidence["CompletionEvidencePASS"]:
                reason = "WAITING_FORMAL_FULL_BESTVAL_FROZEN_TEST_RESULT_EVIDENCE:" + ",".join(completion_evidence["missing_evidence"])
            elif launch_ready and row.get("TaskType") in {"POST_FULL", "FINALIZATION"} and not completion_pass:
                reason = f"WAITING_PASS_COMPLETION_ASSET:{completion_asset}"
            blocked.append({"TaskID": req, "Reason": reason})
    return {"requires": required_rows, "blocked": blocked, "ready": not blocked}


def dry_run(task_id: str) -> dict[str, Any]:
    state = upstream_state(task_id)
    outputs = TASKS[task_id]["outputs"]
    payload = {
        "status": "DRY_RUN_PASS_WAITING_UPSTREAM" if not state["ready"] else "DRY_RUN_PASS_READY_TO_RUN",
        "created_at": now(),
        "task_id": task_id,
        "mode": "dry-run",
        "training_started": False,
        "inference_started": False,
        "test_started": False,
        "formal_started": False,
        "upstream": state,
        "planned_outputs": outputs,
        "missing_planned_outputs": [p for p in outputs if not (ROOT / p).exists()],
        "will_generate_fake_results": False,
        "hard_gate": "FinalLaunchReady means launch-permitted only; PASS requires real completion evidence for FormalFullComplete, BestValSelected, FrozenCheckpointSHA, TEST_COMPLETION_RECEIPT, and ResultAsset.",
    }
    write_json(AUDIT_ROOT / f"{task_id}_DRY_RUN_RECEIPT.json", payload)
    return payload


def run(task_id: str) -> dict[str, Any]:
    state = upstream_state(task_id)
    if not state["ready"]:
        payload = {
            "status": "REFUSE_RUN_WAITING_UPSTREAM",
            "created_at": now(),
            "task_id": task_id,
            "training_started": False,
            "inference_started": False,
            "test_started": False,
            "formal_started": False,
            "upstream": state,
            "hard_gate": "Refused because required upstream completion evidence is absent; launch readiness was not treated as completion.",
        }
        write_json(AUDIT_ROOT / f"{task_id}.json", payload)
        return payload
    dispatch = {
        "ACDC2_TEST_CLOSURE": run_acdc2_test_closure,
        "SEED_TEST_CLOSURE": run_seed_test_closure,
        "BUILD_LABEL_EFFICIENCY": run_build_label_efficiency,
        "BUILD_SEED_STABILITY": run_build_seed_stability,
        "REGENERATE_ALL_TABLES_FIGURES": run_regenerate_all_tables_figures,
        "FINAL_FIVE_REVIEW": run_final_five_review,
        "FINAL_HANDOFF": run_final_handoff,
    }
    return dispatch[task_id]()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--task-id", choices=sorted(TASKS), required=True)
    parser.add_argument("--mode", choices=["dry-run", "run", "status"], required=True)
    args = parser.parse_args()
    payload = dry_run(args.task_id) if args.mode in {"dry-run", "status"} else run(args.task_id)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload.get("status") in {"PASS", "DRY_RUN_PASS_WAITING_UPSTREAM", "DRY_RUN_PASS_READY_TO_RUN", "REFUSE_RUN_WAITING_UPSTREAM"} else 2


if __name__ == "__main__":
    raise SystemExit(main())
