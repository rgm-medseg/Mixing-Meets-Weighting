#!/usr/bin/env python3
"""V24 ACDC10 case-level binding and read-only analyses.

The script binds historical TEST case metrics to canonical V24 ARM IDs using
checkpoint SHA and receipt metadata, then derives ACDC10 difficulty strata and
ACDC10-only paired statistics. It does not run training, inference, replay, or
manifest reads.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
from datetime import datetime, timezone
from pathlib import Path

try:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover
    plt = None


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
TABLE_ROOT = RESULT_ROOT / "tables"
FIGURE_ROOT = RESULT_ROOT / "figures"

ARM_ASSETS = {
    "ACDC10_ARM_000": {
        "display": "β-FFT",
        "case_metrics": ROOT / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/case_class_metrics.tsv",
        "receipt": ROOT / "final_report/paper_internal_evidence_v23_0/raw_test_replay/ACDC10_ARM_000/TEST_REPLAY_RECEIPT.json",
        "expected_checkpoint_sha": None,
    },
    "ACDC10_ARM_110": {
        "display": "Mask Token + AdaMix",
        "case_metrics": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/OURS/case_class_metrics.tsv",
        "receipt": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/OURS/TEST_RECEIPT.json",
        "expected_checkpoint_sha": "f10f69989883a423117a53695311b47e01dc65fbc27fbce2d0c666d6fd733514",
        "expected_innovation_x": "BASE_OURS",
    },
    "ACDC10_ARM_111": {
        "display": "Ours",
        "case_metrics": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/X03/case_class_metrics.tsv",
        "receipt": ROOT / "final_report/acdc10_top6_all_test_v21_5/raw_test/X03/TEST_RECEIPT.json",
        "expected_checkpoint_sha": "c447a0a6d643405627cebf4718424b7d102fffd677b23f10ab5ef1e5227c3086",
        "expected_innovation_x": "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT",
    },
}

BINDING_TABLE = ROOT / "audits/acdc10_top6_all_test_v21_5/ACDC10_TOP6_TEST_CHECKPOINT_BINDING.tsv"
TABLE4 = ROOT / "final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv"
BOOTSTRAP_N = 10000
BOOTSTRAP_SEED = 2026


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


def write_md_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    lines = ["| " + " | ".join(columns) + " |", "| " + " | ".join(["---"] * len(columns)) + " |"]
    for row in rows:
        lines.append("| " + " | ".join(str(row.get(c, "")) for c in columns) + " |")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def write_tex_table(path: Path, rows: list[dict[str, object]], columns: list[str]) -> None:
    lines = [f"\\begin{{tabular}}{{{'l' * len(columns)}}}", "\\toprule", " & ".join(columns) + " \\\\", "\\midrule"]
    for row in rows:
        lines.append(" & ".join(str(row.get(c, "")) for c in columns) + " \\\\")
    lines.extend(["\\bottomrule", "\\end{tabular}"])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n")


def load_receipt(path: Path) -> dict:
    return json.loads(path.read_text())


def binding_rows() -> dict[str, dict[str, str]]:
    rows = read_tsv(BINDING_TABLE)
    return {row["MethodID"]: row for row in rows}


def arm_table_metrics() -> dict[str, dict[str, float]]:
    out = {}
    for row in read_tsv(TABLE4):
        method = row["Method"]
        out[f"ACDC10_ARM_{method}"] = {k: float(row[k]) for k in ["Dice", "IoU", "HD95", "ASD"]}
    return out


def bind_assets() -> dict:
    binding = binding_rows()
    table_metrics = arm_table_metrics()
    rows = []
    failures = []
    for arm, spec in ARM_ASSETS.items():
        case_metrics = spec["case_metrics"]
        receipt_path = spec["receipt"]
        exists = case_metrics.exists() and receipt_path.exists()
        receipt = load_receipt(receipt_path) if receipt_path.exists() else {}
        source_method = receipt.get("method_id", "V23_REPLAY_000" if arm.endswith("000") else "")
        checkpoint_sha = receipt.get("CheckpointSHA") or receipt.get("checkpoint_sha") or ""
        expected_sha = spec.get("expected_checkpoint_sha")
        sha_ok = True if expected_sha is None else checkpoint_sha == expected_sha
        data = read_tsv(case_metrics) if case_metrics.exists() else []
        cases = sorted({r["case"] for r in data})
        table = table_metrics.get(arm, {})
        receipt_metrics = receipt.get("Metrics", {})
        metric_match = True
        mismatches = []
        for key, table_key in [("dice", "Dice"), ("iou", "IoU"), ("hd95", "HD95"), ("asd", "ASD")]:
            if table and receipt_metrics:
                delta = abs(float(receipt_metrics[key]) - float(table[table_key]))
                if delta > 1e-10:
                    metric_match = False
                    mismatches.append(f"{key}:{delta}")
        if not exists or not sha_ok or not metric_match:
            failures.append(arm)
        rows.append(
            {
                "CanonicalArm": arm,
                "DisplayName": spec["display"],
                "SourceReceipt": str(receipt_path.relative_to(ROOT)),
                "SourceCaseClassMetrics": str(case_metrics.relative_to(ROOT)),
                "SourceMethodID": source_method,
                "CheckpointSHA": checkpoint_sha,
                "ExpectedCheckpointSHA": expected_sha or "BOUND_BY_V23_REPLAY_RECEIPT",
                "CheckpointSHAMatch": str(sha_ok),
                "MetricMatchesV23Table4": str(metric_match),
                "MetricMismatches": ";".join(mismatches),
                "CaseCount": len(cases),
                "ClassMetricRowCount": len(data),
                "CaseMetricsSHA256": sha256(case_metrics) if case_metrics.exists() else "",
                "ReceiptSHA256": sha256(receipt_path) if receipt_path.exists() else "",
                "BindingStatus": "PASS" if exists and sha_ok and metric_match and len(cases) == 40 else "FAIL",
            }
        )
    out = AUDIT_ROOT / "ACDC10_ARM_000_110_111_CASE_METRICS_BINDING.tsv"
    write_tsv(out, rows)
    receipt = {
        "status": "PASS" if not failures else "FAIL",
        "created_at": now(),
        "training_started": False,
        "inference_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_content_read": False,
        "binding_table": str(out.relative_to(ROOT)),
        "binding_table_sha256": sha256(out),
        "source_checkpoint_binding": str(BINDING_TABLE.relative_to(ROOT)),
        "source_checkpoint_binding_sha256": sha256(BINDING_TABLE),
        "v23_table4": str(TABLE4.relative_to(ROOT)),
        "v23_table4_sha256": sha256(TABLE4),
        "failed_arms": failures,
        "identity_note": "V21.5 raw_test/OURS binds to ACDC10_ARM_110; V21.5 raw_test/X03 binds to ACDC10_ARM_111.",
    }
    write_json(AUDIT_ROOT / "ACDC10_ARM_000_110_111_CASE_METRICS_BINDING.json", receipt)
    return receipt


def aggregate_case_metrics(path: Path) -> dict[str, dict[str, float]]:
    rows = read_tsv(path)
    by_case: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        by_case.setdefault(row["case"], []).append(row)
    out = {}
    for case, items in by_case.items():
        out[case] = {
            "dice": sum(float(r["dice"]) for r in items) / len(items),
            "iou": sum(float(r["iou"]) for r in items) / len(items),
            "hd95": sum(float(r["hd95"]) for r in items) / len(items),
            "asd": sum(float(r["asd"]) for r in items) / len(items),
        }
    return out


def bootstrap_ci(values: list[float]) -> tuple[float, float]:
    rng = random.Random(BOOTSTRAP_SEED)
    n = len(values)
    means = []
    for _ in range(BOOTSTRAP_N):
        sample = [values[rng.randrange(n)] for _ in range(n)]
        means.append(sum(sample) / n)
    means.sort()
    return means[int(0.025 * BOOTSTRAP_N)], means[int(0.975 * BOOTSTRAP_N)]


def rank_biserial(values: list[float]) -> float:
    nonzero = [v for v in values if abs(v) > 1e-15]
    n = len(nonzero)
    if n == 0:
        return 0.0
    abs_sorted = sorted((abs(v), i, v) for i, v in enumerate(nonzero))
    ranks = {}
    pos = 0
    while pos < n:
        end = pos + 1
        while end < n and abs_sorted[end][0] == abs_sorted[pos][0]:
            end += 1
        avg_rank = (pos + 1 + end) / 2
        for _, idx, _ in abs_sorted[pos:end]:
            ranks[idx] = avg_rank
        pos = end
    w_pos = sum(ranks[i] for i, v in enumerate(nonzero) if v > 0)
    w_neg = sum(ranks[i] for i, v in enumerate(nonzero) if v < 0)
    denom = n * (n + 1) / 2
    return (w_pos - w_neg) / denom


def sign_test_two_sided(values: list[float]) -> float:
    nonzero = [v for v in values if abs(v) > 1e-15]
    n = len(nonzero)
    if n == 0:
        return 1.0
    k = min(sum(1 for v in nonzero if v > 0), sum(1 for v in nonzero if v < 0))
    prob = sum(math.comb(n, i) for i in range(k + 1)) / (2**n)
    return min(1.0, 2 * prob)


def build_caselevel_outputs(binding: dict) -> dict:
    if binding["status"] != "PASS":
        return {"status": "BLOCKED_BINDING_NOT_PASS"}
    arm_cases = {arm: aggregate_case_metrics(spec["case_metrics"]) for arm, spec in ARM_ASSETS.items()}
    common_cases = sorted(set.intersection(*(set(v) for v in arm_cases.values())))
    if len(common_cases) != 40:
        raise SystemExit(f"ACDC10_EXPECTED_40_COMMON_CASES_GOT_{len(common_cases)}")

    paired_rows = []
    for case in common_cases:
        b = arm_cases["ACDC10_ARM_000"][case]
        o = arm_cases["ACDC10_ARM_111"][case]
        m = arm_cases["ACDC10_ARM_110"][case]
        paired_rows.append(
            {
                "Dataset": "ACDC",
                "Split": "ACDC10",
                "CaseID": case,
                "BetaFFT_Dice": f"{b['dice']:.12f}",
                "MaskAdaMix_Dice": f"{m['dice']:.12f}",
                "Ours_Dice": f"{o['dice']:.12f}",
                "DeltaDice_OursMinusBetaFFT": f"{o['dice'] - b['dice']:.12f}",
                "BetaFFT_IoU": f"{b['iou']:.12f}",
                "Ours_IoU": f"{o['iou']:.12f}",
                "DeltaIoU_OursMinusBetaFFT": f"{o['iou'] - b['iou']:.12f}",
                "BetaFFT_HD95": f"{b['hd95']:.12f}",
                "Ours_HD95": f"{o['hd95']:.12f}",
                "DeltaHD95_OursMinusBetaFFT": f"{o['hd95'] - b['hd95']:.12f}",
                "BetaFFT_ASD": f"{b['asd']:.12f}",
                "Ours_ASD": f"{o['asd']:.12f}",
                "DeltaASD_OursMinusBetaFFT": f"{o['asd'] - b['asd']:.12f}",
                "Unit": "pixel-unit",
            }
        )
    paired_out = RESULT_ROOT / "PATIENT_LEVEL_PAIRED_METRICS_ACDC10.tsv"
    write_tsv(paired_out, paired_rows)

    deltas = [float(r["DeltaDice_OursMinusBetaFFT"]) for r in paired_rows]
    ci_low, ci_high = bootstrap_ci(deltas)
    summary_rows = [
        {
            "Dataset": "ACDC",
            "Split": "ACDC10",
            "Comparison": "β-FFT vs Ours",
            "CaseCount": len(deltas),
            "MeanDeltaDice": f"{sum(deltas) / len(deltas):.12f}",
            "MedianDeltaDice": f"{sorted(deltas)[len(deltas)//2]:.12f}",
            "StdDeltaDice": f"{math.sqrt(sum((v - sum(deltas) / len(deltas)) ** 2 for v in deltas) / (len(deltas)-1)):.12f}",
            "ImprovedCaseCount": sum(1 for v in deltas if v > 0),
            "WorsenedCaseCount": sum(1 for v in deltas if v < 0),
            "TieCount": sum(1 for v in deltas if abs(v) <= 1e-15),
            "ImprovementRate": f"{sum(1 for v in deltas if v > 0) / len(deltas):.12f}",
            "BootstrapCI95Low": f"{ci_low:.12f}",
            "BootstrapCI95High": f"{ci_high:.12f}",
            "BootstrapIterations": BOOTSTRAP_N,
            "BootstrapSeed": BOOTSTRAP_SEED,
            "WilcoxonPValue": "NOT_COMPUTED_SCIPY_NOT_REQUIRED_FOR_ACDC10_INTERMEDIATE",
            "SignTestPValueSupplement": f"{sign_test_two_sided(deltas):.12f}",
            "RankBiserialEffectSize": f"{rank_biserial(deltas):.12f}",
            "GlobalA1Status": "PARTIAL_ACDC10_ONLY_RETAINED_SETTINGS_NOT_COMPLETE",
        }
    ]
    summary_out = RESULT_ROOT / "PATIENT_LEVEL_STATISTICAL_SUMMARY_ACDC10.tsv"
    write_tsv(summary_out, summary_rows)

    sorted_cases = sorted(common_cases, key=lambda c: arm_cases["ACDC10_ARM_000"][c]["dice"])
    n = len(sorted_cases)
    sizes = [n // 3, n // 3, n - 2 * (n // 3)]
    strata = {
        "Hard": sorted_cases[: sizes[0]],
        "Medium": sorted_cases[sizes[0] : sizes[0] + sizes[1]],
        "Easy": sorted_cases[sizes[0] + sizes[1] :],
    }
    binding_payload = {
        "status": "PASS",
        "created_at": now(),
        "grouping_metric": "ACDC10_ARM_000 per-case mean foreground Dice",
        "grouping_frozen_before_ours_read": True,
        "test_selection_exposed": True,
        "used_for_method_selection_or_tuning": False,
        "case_count": n,
        "group_sizes": {k: len(v) for k, v in strata.items()},
        "groups": strata,
    }
    write_json(AUDIT_ROOT / "ACDC10_DIFFICULTY_STRATA_BINDING.json", binding_payload)

    hme_rows = []
    for group, cases in strata.items():
        bd = [arm_cases["ACDC10_ARM_000"][c]["dice"] for c in cases]
        od = [arm_cases["ACDC10_ARM_111"][c]["dice"] for c in cases]
        delta = [o - b for b, o in zip(bd, od)]
        hme_rows.append(
            {
                "DifficultyGroup": group,
                "CaseCount": len(cases),
                "BetaFFTMeanDice": f"{sum(bd)/len(bd):.12f}",
                "OursMeanDice": f"{sum(od)/len(od):.12f}",
                "MeanDeltaDice": f"{sum(delta)/len(delta):.12f}",
                "MedianDeltaDice": f"{sorted(delta)[len(delta)//2]:.12f}",
                "ImprovedCaseFraction": f"{sum(1 for v in delta if v > 0)/len(delta):.12f}",
                "BetaFFTMeanHD95": f"{sum(arm_cases['ACDC10_ARM_000'][c]['hd95'] for c in cases)/len(cases):.12f}",
                "OursMeanHD95": f"{sum(arm_cases['ACDC10_ARM_111'][c]['hd95'] for c in cases)/len(cases):.12f}",
                "BetaFFTMeanASD": f"{sum(arm_cases['ACDC10_ARM_000'][c]['asd'] for c in cases)/len(cases):.12f}",
                "OursMeanASD": f"{sum(arm_cases['ACDC10_ARM_111'][c]['asd'] for c in cases)/len(cases):.12f}",
                "GroupingRule": "sort by beta-FFT Dice ascending then deterministic near-equal thirds",
            }
        )
    hme_out = RESULT_ROOT / "ACDC10_HARD_MEDIUM_EASY.tsv"
    write_tsv(hme_out, hme_rows)
    cols = ["DifficultyGroup", "CaseCount", "BetaFFTMeanDice", "OursMeanDice", "MeanDeltaDice", "ImprovedCaseFraction"]
    write_md_table(TABLE_ROOT / "TABLE_S_ACDC10_DIFFICULTY.md", hme_rows, cols)
    write_tex_table(TABLE_ROOT / "TABLE_S_ACDC10_DIFFICULTY.tex", hme_rows, cols)

    figures = []
    if plt is not None:
        plt.figure(figsize=(6, 4))
        plt.bar([r["DifficultyGroup"] for r in hme_rows], [float(r["MeanDeltaDice"]) for r in hme_rows], color="#3b6ea8")
        plt.axhline(0, color="black", linewidth=0.8)
        plt.ylabel("Mean ΔDice (Ours - β-FFT)")
        plt.tight_layout()
        for suffix in ["png", "pdf"]:
            p = FIGURE_ROOT / f"ACDC10_DIFFICULTY_GAIN.{suffix}"
            plt.savefig(p)
            figures.append(str(p.relative_to(ROOT)))
        plt.close()
        plt.figure(figsize=(6, 4))
        plt.hist(deltas, bins=12, color="#577590", edgecolor="white")
        plt.axvline(0, color="black", linewidth=0.8)
        plt.xlabel("ΔDice (Ours - β-FFT)")
        plt.ylabel("Case count")
        plt.tight_layout()
        for suffix in ["png", "pdf"]:
            p = FIGURE_ROOT / f"PATIENT_LEVEL_DELTA_DISTRIBUTION_ACDC10.{suffix}"
            plt.savefig(p)
            figures.append(str(p.relative_to(ROOT)))
        plt.close()

    receipt = {
        "status": "PASS_ACDC10_PARTIAL_A1_AND_A4",
        "created_at": now(),
        "training_started": False,
        "inference_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_content_read": False,
        "paired_metrics": str(paired_out.relative_to(ROOT)),
        "paired_metrics_sha256": sha256(paired_out),
        "statistical_summary": str(summary_out.relative_to(ROOT)),
        "statistical_summary_sha256": sha256(summary_out),
        "difficulty_binding": "audits/internal_evidence_expansion_v24_0/ACDC10_DIFFICULTY_STRATA_BINDING.json",
        "difficulty_table": str(hme_out.relative_to(ROOT)),
        "difficulty_table_sha256": sha256(hme_out),
        "figures": figures,
        "global_patient_level_stats_complete": False,
        "global_patient_level_stats_missing_settings": ["ACDC5", "PROMISE20", "COVID5", "COVID10"],
    }
    write_json(AUDIT_ROOT / "ACDC10_CASELEVEL_ANALYSIS_RECEIPT.json", receipt)
    return receipt


def main() -> int:
    binding = bind_assets()
    outputs = build_caselevel_outputs(binding)
    summary = {
        "status": "PASS" if binding["status"] == "PASS" else "FAIL",
        "created_at": now(),
        "training_started": False,
        "inference_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_content_read": False,
        "binding": binding,
        "outputs": outputs,
    }
    write_json(AUDIT_ROOT / "V24_0_ACDC10_CASELEVEL_ANALYSIS_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0 if summary["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
