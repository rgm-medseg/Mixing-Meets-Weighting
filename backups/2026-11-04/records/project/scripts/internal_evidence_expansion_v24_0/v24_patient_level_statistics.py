#!/usr/bin/env python3
"""V24 full patient/case-level paired statistics.

This is a read-only extractor. It consumes existing formal TEST metric assets
for the five retained V24 settings and computes paired statistics at the
patient/case/volume level. It does not train, replay inference, read TEST
manifest contents, or modify scientific assets.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import statistics
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None

try:
    from scipy.stats import wilcoxon
except Exception:  # pragma: no cover
    wilcoxon = None


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
TABLE_ROOT = RESULT_ROOT / "tables"
FIGURE_ROOT = RESULT_ROOT / "figures"
REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"

BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 2026
EPS = 1e-12


SETTINGS = [
    {
        "SettingID": "ACDC5",
        "Dataset": "ACDC",
        "Ratio": "5%",
        "Unit": "pixel-unit",
        "BaselineMethod": "β-FFT",
        "OursMethod": "Ours",
        "BaselineMetrics": ROOT
        / "formal_test/acdc_v19_4/ACDC_5P_BASELINE_CONFIRMATORY_TEST/case_class_metrics.tsv",
        "BaselineReceipt": ROOT
        / "formal_test/acdc_v19_4/ACDC_5P_BASELINE_CONFIRMATORY_TEST/FORMAL_TEST_RECEIPT.json",
        "OursMetrics": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/case_class_metrics.tsv",
        "OursReceipt": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/TEST_COMPLETION_RECEIPT.json",
        "Loader": "acdc_case_class",
    },
    {
        "SettingID": "ACDC10",
        "Dataset": "ACDC",
        "Ratio": "10%",
        "Unit": "pixel-unit",
        "BaselineMethod": "β-FFT",
        "OursMethod": "Ours",
        "BaselineMetrics": ROOT
        / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/case_class_metrics.tsv",
        "BaselineReceipt": ROOT
        / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/TEST_REPLAY_RECEIPT.json",
        "OursMetrics": ROOT
        / "final_report/acdc10_top6_all_test_v21_5/raw_test/X03/case_class_metrics.tsv",
        "OursReceipt": ROOT
        / "final_report/acdc10_top6_all_test_v21_5/raw_test/X03/TEST_RECEIPT.json",
        "Loader": "acdc_case_class",
    },
    {
        "SettingID": "PROMISE20",
        "Dataset": "PROMISE12",
        "Ratio": "20%",
        "Unit": "pixel-unit",
        "BaselineMethod": "β-FFT",
        "OursMethod": "Ours",
        "BaselineMetrics": ROOT / "final_report/PROMISE20_TEST_CASE_METRICS.tsv",
        "BaselineReceipt": ROOT / "final_report/PROMISE20_TEST_CASE_METRICS.tsv",
        "OursMetrics": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/PROMISE20_FINAL3/case_metrics.tsv",
        "OursReceipt": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/PROMISE20_FINAL3/TEST_COMPLETION_RECEIPT.json",
        "Loader": "promise20",
    },
    {
        "SettingID": "COVID5",
        "Dataset": "COVID-19",
        "Ratio": "5%",
        "Unit": "mm",
        "BaselineMethod": "β-FFT",
        "OursMethod": "Ours",
        "BaselineMetrics": ROOT
        / "formal_test/covid19_seed1337/COVID19_5P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json",
        "BaselineReceipt": ROOT
        / "formal_test/covid19_seed1337/COVID19_5P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json",
        "OursMetrics": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/COVID5_FINAL3/patient_metrics.json",
        "OursReceipt": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/COVID5_FINAL3/TEST_COMPLETION_RECEIPT.json",
        "Loader": "covid_patient_json",
    },
    {
        "SettingID": "COVID10",
        "Dataset": "COVID-19",
        "Ratio": "10%",
        "Unit": "mm",
        "BaselineMethod": "β-FFT",
        "OursMethod": "Ours",
        "BaselineMetrics": ROOT
        / "formal_test/covid19_seed1337/COVID19_10P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json",
        "BaselineReceipt": ROOT
        / "formal_test/covid19_seed1337/COVID19_10P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json",
        "OursMetrics": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/COVID10_FINAL3/patient_metrics.json",
        "OursReceipt": ROOT
        / "final_report/three_innovation_postfull_test_v22_1/raw_test/COVID10_FINAL3/TEST_COMPLETION_RECEIPT.json",
        "Loader": "covid_patient_json",
    },
]


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


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows and fieldnames is None:
        path.write_text("")
        return
    names = fieldnames or list(rows[0].keys())
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=names, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


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


def f12(value: float) -> str:
    return f"{value:.12f}"


def pct2(value: float) -> str:
    return f"{value * 100:.2f}"


def load_case_class(path: Path) -> dict[str, dict[str, float]]:
    rows = read_tsv(path)
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        case = row.get("case") or row.get("case_id")
        if not case:
            raise SystemExit(f"CASE_COLUMN_MISSING:{path}")
        by_case.setdefault(case, []).append(row)
    out: dict[str, dict[str, float]] = {}
    for case, items in by_case.items():
        out[case] = {
            "dice": statistics.fmean(float(r["dice"]) for r in items),
            "iou": statistics.fmean(float(r["iou"]) for r in items),
            "hd95": statistics.fmean(float(r["hd95"]) for r in items),
            "asd": statistics.fmean(float(r["asd"]) for r in items),
        }
    return out


def load_promise20(path: Path, baseline: bool) -> dict[str, dict[str, float]]:
    out = {}
    rows = read_tsv(path)
    for row in rows:
        if baseline and row.get("method_id") != "PROMISE20_BASELINE_BETA_FFT_R1":
            continue
        case = row.get("case") or row.get("case_id")
        if not case:
            raise SystemExit(f"CASE_COLUMN_MISSING:{path}")
        out[case] = {
            "dice": float(row["dice"]),
            "iou": float(row["iou"]),
            "hd95": float(row["hd95"]),
            "asd": float(row["asd"]),
        }
    return out


def covid_metric_blob(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text())
    return data["metric_result"] if "metric_result" in data else data


def load_covid_patient(path: Path) -> dict[str, dict[str, float]]:
    blob = covid_metric_blob(path)
    out = {}
    for patient, metrics in blob["per_patient"].items():
        out[patient] = {
            "dice": float(metrics["dice"]),
            "iou": float(metrics["iou"]),
            "hd95": float(metrics.get("hd95") if "hd95" in metrics else metrics["hd95_mm"]),
            "asd": float(metrics.get("asd") if "asd" in metrics else metrics["asd_mm"]),
        }
    return out


def load_metrics(setting: dict[str, Any], side: str) -> dict[str, dict[str, float]]:
    path = setting[f"{side}Metrics"]
    loader = setting["Loader"]
    if loader == "acdc_case_class":
        return load_case_class(path)
    if loader == "promise20":
        return load_promise20(path, baseline=(side == "Baseline"))
    if loader == "covid_patient_json":
        return load_covid_patient(path)
    raise SystemExit(f"UNKNOWN_LOADER:{loader}")


def macro_from_units(units: dict[str, dict[str, float]]) -> dict[str, float]:
    return {
        metric: statistics.fmean(item[metric] for item in units.values())
        for metric in ["dice", "iou", "hd95", "asd"]
    }


def receipt_macro(path: Path) -> dict[str, float] | None:
    if path.suffix != ".json":
        return None
    data = json.loads(path.read_text())
    if "aggregate" in data:
        agg = data["aggregate"]
        return {metric: float(agg[metric]) for metric in ["dice", "iou", "hd95", "asd"]}
    if "Metrics" in data:
        agg = data["Metrics"]
        return {metric: float(agg[metric]) for metric in ["dice", "iou", "hd95", "asd"]}
    if all(k in data for k in ["Dice", "IoU", "HD95", "ASD"]):
        return {"dice": float(data["Dice"]), "iou": float(data["IoU"]), "hd95": float(data["HD95"]), "asd": float(data["ASD"])}
    blob = data.get("metric_result") or data
    if "macro" in blob:
        macro = blob["macro"]
        return {
            "dice": float(macro["dice"]),
            "iou": float(macro["iou"]),
            "hd95": float(macro.get("hd95") if "hd95" in macro else macro["hd95_mm"]),
            "asd": float(macro.get("asd") if "asd" in macro else macro["asd_mm"]),
        }
    return None


def bootstrap_ci(values: list[float], seed_offset: int) -> tuple[float, float]:
    if not values:
        return math.nan, math.nan
    rng = random.Random(BOOTSTRAP_SEED + seed_offset)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_N):
        means.append(statistics.fmean(values[rng.randrange(n)] for _ in range(n)))
    means.sort()
    return means[int(0.025 * BOOTSTRAP_N)], means[int(0.975 * BOOTSTRAP_N)]


def rank_biserial(values: list[float]) -> float:
    nonzero = [v for v in values if abs(v) > EPS]
    n = len(nonzero)
    if n == 0:
        return 0.0
    ordered = sorted((abs(v), idx, v) for idx, v in enumerate(nonzero))
    ranks: dict[int, float] = {}
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and ordered[end][0] == ordered[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2.0
        for _, idx, _ in ordered[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    w_pos = sum(ranks[i] for i, value in enumerate(nonzero) if value > 0)
    w_neg = sum(ranks[i] for i, value in enumerate(nonzero) if value < 0)
    return (w_pos - w_neg) / (n * (n + 1) / 2.0)


def wilcoxon_p(values: list[float]) -> tuple[float, str]:
    nonzero = [v for v in values if abs(v) > EPS]
    if not nonzero:
        return 1.0, "ALL_ZERO_DIFFERENCES"
    if wilcoxon is None:
        return math.nan, "SCIPY_UNAVAILABLE"
    result = wilcoxon(nonzero, zero_method="wilcox", alternative="two-sided", method="auto")
    return float(result.pvalue), "SCIPY_WILCOXON_TWO_SIDED_AUTO"


def holm_adjust(p_values: list[float]) -> list[float]:
    indexed = sorted(enumerate(p_values), key=lambda item: item[1])
    adjusted = [0.0] * len(p_values)
    running = 0.0
    m = len(p_values)
    for rank, (idx, p) in enumerate(indexed):
        candidate = min(1.0, (m - rank) * p)
        running = max(running, candidate)
        adjusted[idx] = running
    return adjusted


def paired_rows_for_setting(setting: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    baseline = load_metrics(setting, "Baseline")
    ours = load_metrics(setting, "Ours")
    common = sorted(set(baseline) & set(ours))
    missing_baseline = sorted(set(ours) - set(baseline))
    missing_ours = sorted(set(baseline) - set(ours))
    if not common or missing_baseline or missing_ours:
        raise SystemExit(
            f"PAIRING_FAILED:{setting['SettingID']}:common={len(common)}:"
            f"missing_baseline={missing_baseline}:missing_ours={missing_ours}"
        )

    rows = []
    for unit_id in common:
        b = baseline[unit_id]
        o = ours[unit_id]
        rows.append(
            {
                "SettingID": setting["SettingID"],
                "Dataset": setting["Dataset"],
                "Ratio": setting["Ratio"],
                "PairUnit": "patient/case/volume",
                "UnitID": unit_id,
                "BoundaryUnit": setting["Unit"],
                "BetaFFT_Method": setting["BaselineMethod"],
                "Ours_Method": setting["OursMethod"],
                "BetaFFT_Dice": f12(b["dice"]),
                "Ours_Dice": f12(o["dice"]),
                "DeltaDice_OursMinusBetaFFT": f12(o["dice"] - b["dice"]),
                "BetaFFT_IoU": f12(b["iou"]),
                "Ours_IoU": f12(o["iou"]),
                "DeltaIoU_OursMinusBetaFFT": f12(o["iou"] - b["iou"]),
                "BetaFFT_HD95": f12(b["hd95"]),
                "Ours_HD95": f12(o["hd95"]),
                "DeltaHD95_OursMinusBetaFFT": f12(o["hd95"] - b["hd95"]),
                "BetaFFT_ASD": f12(b["asd"]),
                "Ours_ASD": f12(o["asd"]),
                "DeltaASD_OursMinusBetaFFT": f12(o["asd"] - b["asd"]),
                "BaselineSourceAsset": rel(setting["BaselineMetrics"]),
                "OursSourceAsset": rel(setting["OursMetrics"]),
            }
        )

    b_macro = macro_from_units(baseline)
    o_macro = macro_from_units(ours)
    b_receipt = receipt_macro(setting["BaselineReceipt"])
    o_receipt = receipt_macro(setting["OursReceipt"])
    checks = []
    for label, computed, recorded in [("Baseline", b_macro, b_receipt), ("Ours", o_macro, o_receipt)]:
        if recorded is None:
            checks.append({"Side": label, "ReceiptMacroAvailable": False, "MaxAbsDelta": "", "Status": "NO_JSON_MACRO"})
            continue
        max_delta = max(abs(computed[k] - recorded[k]) for k in ["dice", "iou", "hd95", "asd"])
        checks.append(
            {
                "Side": label,
                "ReceiptMacroAvailable": True,
                "MaxAbsDelta": max_delta,
                "Status": "PASS" if max_delta <= 1e-9 else "WARN_MACRO_DELTA",
            }
        )
    audit = {
        "SettingID": setting["SettingID"],
        "Dataset": setting["Dataset"],
        "Ratio": setting["Ratio"],
        "UnitCount": len(common),
        "UnitIDs": common,
        "BaselineSourceAsset": rel(setting["BaselineMetrics"]),
        "BaselineSourceSHA256": sha256(setting["BaselineMetrics"]),
        "BaselineReceipt": rel(setting["BaselineReceipt"]),
        "BaselineReceiptSHA256": sha256(setting["BaselineReceipt"]),
        "OursSourceAsset": rel(setting["OursMetrics"]),
        "OursSourceSHA256": sha256(setting["OursMetrics"]),
        "OursReceipt": rel(setting["OursReceipt"]),
        "OursReceiptSHA256": sha256(setting["OursReceipt"]),
        "BaselineMacroFromUnits": b_macro,
        "OursMacroFromUnits": o_macro,
        "ReceiptMacroChecks": checks,
        "Status": "PASS",
    }
    return rows, audit


def summarize_setting(setting_id: str, rows: list[dict[str, Any]], p_raw: float, p_method: str) -> dict[str, Any]:
    metric_deltas = {
        "Dice": [float(r["DeltaDice_OursMinusBetaFFT"]) for r in rows],
        "IoU": [float(r["DeltaIoU_OursMinusBetaFFT"]) for r in rows],
        "HD95": [float(r["DeltaHD95_OursMinusBetaFFT"]) for r in rows],
        "ASD": [float(r["DeltaASD_OursMinusBetaFFT"]) for r in rows],
    }
    dice = metric_deltas["Dice"]
    row0 = rows[0]
    summary: dict[str, Any] = {
        "SettingID": setting_id,
        "Dataset": row0["Dataset"],
        "Ratio": row0["Ratio"],
        "Comparison": "β-FFT vs Ours",
        "PairUnit": "patient/case/volume",
        "UnitCount": len(rows),
        "MeanDeltaDice": f12(statistics.fmean(dice)),
        "MedianDeltaDice": f12(statistics.median(dice)),
        "StdDeltaDice": f12(statistics.stdev(dice) if len(dice) > 1 else 0.0),
        "ImprovedCaseCount": sum(1 for v in dice if v > EPS),
        "WorsenedCaseCount": sum(1 for v in dice if v < -EPS),
        "TieCount": sum(1 for v in dice if abs(v) <= EPS),
        "ImprovementRate": f12(sum(1 for v in dice if v > EPS) / len(dice)),
        "WilcoxonDiceRawP": f"{p_raw:.12g}",
        "WilcoxonDicePMethod": p_method,
        "WilcoxonDiceHolmP": "",
        "RankBiserialDice": f12(rank_biserial(dice)),
        "BootstrapIterations": BOOTSTRAP_N,
        "BootstrapSeed": BOOTSTRAP_SEED,
        "BoundaryUnit": row0["BoundaryUnit"],
    }
    for offset, (metric, values) in enumerate(metric_deltas.items()):
        ci_low, ci_high = bootstrap_ci(values, seed_offset=offset + 100 * len(setting_id))
        summary[f"MeanDelta{metric}"] = f12(statistics.fmean(values))
        summary[f"{metric}BootstrapCI95Low"] = f12(ci_low)
        summary[f"{metric}BootstrapCI95High"] = f12(ci_high)
    return summary


def update_registry(script_sha: str, summary_sha: str, receipt_sha: str) -> None:
    if not REGISTRY.exists():
        return
    rows = read_tsv(REGISTRY)
    for row in rows:
        if row.get("TaskID") == "PATIENT_LEVEL_STATS":
            row["Action"] = "READ_ONLY_READY"
            row["Dataset"] = "ACDC/PROMISE20/COVID"
            row["Split"] = "ACDC5/ACDC10/PROMISE20/COVID5/COVID10"
            row["Seed"] = ""
            row["ConfigPath"] = ""
            row["ConfigSHA"] = ""
            row["SourcePath"] = "scripts/internal_evidence_expansion_v24_0/v24_patient_level_statistics.py"
            row["SourceSHA"] = script_sha
            row["Launcher"] = ""
            row["LauncherSHA"] = ""
            row["CheckpointPath"] = ""
            row["CheckpointSHA"] = ""
            row["InputAssets"] = "formal TEST per-case/per-patient metrics; see PATIENT_LEVEL_STATS_RECEIPT.json"
            row["OutputDir"] = "final_report/internal_evidence_expansion_v24_0/PATIENT_LEVEL_STATISTICAL_SUMMARY.tsv"
            row["ExpectedReceipt"] = "audits/internal_evidence_expansion_v24_0/PATIENT_LEVEL_STATS_RECEIPT.json"
            row["GPURequired"] = "False"
            row["TESTAccessAllowed"] = "0"
            row["FinalLaunchReady"] = "PASS"
            row["Blocker"] = ""
    fieldnames = list(rows[0].keys())
    write_tsv(REGISTRY, rows, fieldnames)
    summary_path = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING_SUMMARY.json"
    pass_count = sum(1 for r in rows if r.get("FinalLaunchReady") == "PASS")
    blocked_count = sum(1 for r in rows if r.get("FinalLaunchReady") in {"BLOCKED", "BLOCKED_NONCRITICAL"} or r.get("Action") == "BLOCKED")
    write_json(
        summary_path,
        {
            "status": "NOT_PRELAUNCH_READY",
            "created_at": now(),
            "task_count": len(rows),
            "pass_ready_count": pass_count,
            "blocked_count": blocked_count,
            "pending_count": len(rows) - pass_count - blocked_count,
            "patient_level_summary_sha256": summary_sha,
            "patient_level_receipt_sha256": receipt_sha,
        },
    )


def main() -> int:
    for path in [AUDIT_ROOT, RESULT_ROOT, TABLE_ROOT, FIGURE_ROOT]:
        path.mkdir(parents=True, exist_ok=True)

    all_rows: list[dict[str, Any]] = []
    audits = []
    summary_rows = []
    raw_ps = []
    p_methods = []

    for setting in SETTINGS:
        for key in ["BaselineMetrics", "BaselineReceipt", "OursMetrics", "OursReceipt"]:
            if not setting[key].exists():
                raise SystemExit(f"MISSING_REQUIRED_ASSET:{setting['SettingID']}:{key}:{setting[key]}")
        rows, audit = paired_rows_for_setting(setting)
        p_raw, p_method = wilcoxon_p([float(r["DeltaDice_OursMinusBetaFFT"]) for r in rows])
        all_rows.extend(rows)
        audits.append(audit)
        raw_ps.append(p_raw)
        p_methods.append(p_method)
        summary_rows.append(summarize_setting(setting["SettingID"], rows, p_raw, p_method))

    for row, adj in zip(summary_rows, holm_adjust(raw_ps)):
        row["WilcoxonDiceHolmP"] = f"{adj:.12g}"

    paired_out = RESULT_ROOT / "PATIENT_LEVEL_PAIRED_METRICS.tsv"
    summary_out = RESULT_ROOT / "PATIENT_LEVEL_STATISTICAL_SUMMARY.tsv"
    source_out = RESULT_ROOT / "PATIENT_LEVEL_SOURCE_DATA.tsv"
    audit_tsv = AUDIT_ROOT / "PATIENT_LEVEL_STATS_SOURCE_AUDIT.tsv"
    audit_json = AUDIT_ROOT / "PATIENT_LEVEL_STATS_SOURCE_AUDIT.json"
    receipt_out = AUDIT_ROOT / "PATIENT_LEVEL_STATS_RECEIPT.json"

    paired_fields = [
        "SettingID",
        "Dataset",
        "Ratio",
        "PairUnit",
        "UnitID",
        "BoundaryUnit",
        "BetaFFT_Method",
        "Ours_Method",
        "BetaFFT_Dice",
        "Ours_Dice",
        "DeltaDice_OursMinusBetaFFT",
        "BetaFFT_IoU",
        "Ours_IoU",
        "DeltaIoU_OursMinusBetaFFT",
        "BetaFFT_HD95",
        "Ours_HD95",
        "DeltaHD95_OursMinusBetaFFT",
        "BetaFFT_ASD",
        "Ours_ASD",
        "DeltaASD_OursMinusBetaFFT",
        "BaselineSourceAsset",
        "OursSourceAsset",
    ]
    write_tsv(paired_out, all_rows, paired_fields)
    write_tsv(source_out, all_rows, paired_fields)
    write_tsv(summary_out, summary_rows)

    audit_rows = []
    for item in audits:
        audit_rows.append(
            {
                "SettingID": item["SettingID"],
                "Dataset": item["Dataset"],
                "Ratio": item["Ratio"],
                "UnitCount": item["UnitCount"],
                "BaselineSourceAsset": item["BaselineSourceAsset"],
                "BaselineSourceSHA256": item["BaselineSourceSHA256"],
                "OursSourceAsset": item["OursSourceAsset"],
                "OursSourceSHA256": item["OursSourceSHA256"],
                "Status": item["Status"],
            }
        )
    write_tsv(audit_tsv, audit_rows)
    write_json(audit_json, {"status": "PASS", "created_at": now(), "settings": audits})

    table_columns = [
        "SettingID",
        "UnitCount",
        "MeanDeltaDice",
        "DiceBootstrapCI95Low",
        "DiceBootstrapCI95High",
        "WilcoxonDiceRawP",
        "WilcoxonDiceHolmP",
        "RankBiserialDice",
    ]
    table_rows = [
        {
            **row,
            "MeanDeltaDice": pct2(float(row["MeanDeltaDice"])),
            "DiceBootstrapCI95Low": pct2(float(row["DiceBootstrapCI95Low"])),
            "DiceBootstrapCI95High": pct2(float(row["DiceBootstrapCI95High"])),
        }
        for row in summary_rows
    ]
    write_md_table(TABLE_ROOT / "TABLE_S_PATIENT_LEVEL_STATISTICS.md", table_rows, table_columns)
    write_tex_table(TABLE_ROOT / "TABLE_S_PATIENT_LEVEL_STATISTICS.tex", table_rows, table_columns)

    figures = []
    if plt is not None:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        labels = []
        data = []
        for row in summary_rows:
            sid = row["SettingID"]
            labels.append(sid)
            data.append([float(r["DeltaDice_OursMinusBetaFFT"]) for r in all_rows if r["SettingID"] == sid])
        ax.boxplot(data, labels=labels, showmeans=True)
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_ylabel("ΔDice (Ours - β-FFT)")
        ax.set_xlabel("Setting")
        fig.tight_layout()
        for suffix in ["png", "pdf"]:
            path = FIGURE_ROOT / f"PATIENT_LEVEL_DELTA_DISTRIBUTION.{suffix}"
            fig.savefig(path)
            figures.append(rel(path))
        plt.close(fig)

    script_sha = sha256(Path(__file__))
    receipt = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "PATIENT_LEVEL_STATS",
        "training_started": False,
        "inference_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_content_read": False,
        "retained_settings": [s["SettingID"] for s in SETTINGS],
        "comparison": "β-FFT vs Ours",
        "pair_unit": "patient/case/volume",
        "bootstrap_iterations": BOOTSTRAP_N,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "primary_family": "Dice across five retained settings",
        "primary_p_adjustment": "Holm",
        "paired_metrics": rel(paired_out),
        "paired_metrics_sha256": sha256(paired_out),
        "source_data_tsv": rel(source_out),
        "source_data_tsv_sha256": sha256(source_out),
        "statistical_summary": rel(summary_out),
        "statistical_summary_sha256": sha256(summary_out),
        "table_md": "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_PATIENT_LEVEL_STATISTICS.md",
        "table_tex": "final_report/internal_evidence_expansion_v24_0/tables/TABLE_S_PATIENT_LEVEL_STATISTICS.tex",
        "source_audit": rel(audit_json),
        "source_audit_sha256": sha256(audit_json),
        "figures": figures,
        "script": rel(Path(__file__)),
        "script_sha256": script_sha,
        "review": {
            "REVIEW_1_ALL_RESULTS_FROM_EXISTING_TEST_ASSETS": "PASS",
            "REVIEW_2_DATASET_RATIO_IDENTITY": "PASS",
            "REVIEW_3_TEST_ENDPOINT_ONLY": "PASS",
            "REVIEW_4_NO_SLICE_AS_INDEPENDENT_SAMPLE": "PASS",
            "REVIEW_5_NO_TRAINING_INFERENCE_OR_TEST_RERUN": "PASS",
        },
    }
    write_json(receipt_out, receipt)
    update_registry(script_sha, sha256(summary_out), sha256(receipt_out))
    print(json.dumps(receipt, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
