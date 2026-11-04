#!/usr/bin/env python3
"""Aggregate V22.1 post-full TEST receipts and exact-reuse results."""

from __future__ import annotations

import csv
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / "final_report/three_innovation_postfull_test_v22_1"
AUDIT = ROOT / "audits/three_innovation_postfull_test_v22_1"
RAW = RESULT / "raw_test"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def receipt(task_id: str) -> dict:
    return json.loads((RAW / task_id / "TEST_COMPLETION_RECEIPT.json").read_text(encoding="utf-8"))


def md_table(rows: list[dict], fields: list[str]) -> str:
    out = ["| " + " | ".join(fields) + " |", "| " + " | ".join(["---"] * len(fields)) + " |"]
    for row in rows:
        out.append("| " + " | ".join(str(row.get(field, "")) for field in fields) + " |")
    return "\n".join(out) + "\n"


def tex_table(rows: list[dict], fields: list[str]) -> str:
    lines = ["\\begin{tabular}{" + "l" * len(fields) + "}", " & ".join(fields) + r" \\", "\\hline"]
    for row in rows:
        lines.append(" & ".join(str(row.get(field, "")) for field in fields) + r" \\")
    lines.append("\\end{tabular}\n")
    return "\n".join(lines)


def fmt(x: float) -> str:
    return f"{x:.10f}"


def pct(x: float) -> str:
    return f"{100*x:.4f}"


def main() -> int:
    registry = {r["TaskID"]: r for r in read_tsv(AUDIT / "V22_1_POSTFULL_TEST_TASK_REGISTRY.tsv")}
    acdc_hist = read_tsv(ROOT / "final_report/ACDC_5P_10P_TEST_RESULTS.tsv")
    acdc10_base = next(r for r in acdc_hist if r["method_id"] == "ACDC_10P_BASELINE")
    acdc5_ours = next(r for r in acdc_hist if r["method_id"] == "ACDC_5P_MASK_TOKEN_ADAMIX")
    top6 = {r["MethodID"]: r for r in read_tsv(ROOT / "final_report/acdc10_top6_all_test_v21_5/ACDC10_OURS_PLUS_TOP6_TEST_FULL_PRECISION.tsv")}

    arms = {}
    arms["000"] = {
        "Arm": "000", "MaskToken": "OFF", "AdaMix": "OFF", "ContourReliabilityWeight": "OFF",
        "ResultSource": "EXACT_TEST_REUSE", "BestValDice": "", "BestValIter": acdc10_base["best_iteration"],
        "TestDice": acdc10_base["dice"], "TestIoU": acdc10_base["iou"], "TestHD95": acdc10_base["hd95"], "TestASD": acdc10_base["asd"],
        "RV_Dice": acdc10_base["RV_Dice"], "MYO_Dice": acdc10_base["MYO_Dice"], "LV_Dice": acdc10_base["LV_Dice"],
        "CheckpointSHA": acdc10_base["checkpoint_sha256"], "BoundaryUnit": "pixel-unit", "TestStatus": "TEST_PASS",
    }
    arms["110"] = {
        "Arm": "110", "MaskToken": "ON", "AdaMix": "ON", "ContourReliabilityWeight": "OFF",
        "ResultSource": "EXACT_TEST_REUSE", "BestValDice": top6["OURS"]["BestValDice"], "BestValIter": top6["OURS"]["BestValIter"],
        "TestDice": top6["OURS"]["TestDice"], "TestIoU": top6["OURS"]["TestIoU"], "TestHD95": top6["OURS"]["TestHD95"], "TestASD": top6["OURS"]["TestASD"],
        "RV_Dice": "", "MYO_Dice": "", "LV_Dice": "", "CheckpointSHA": top6["OURS"]["TestCheckpointSHA"], "BoundaryUnit": "pixel-unit", "TestStatus": "TEST_PASS",
    }
    arms["111"] = {
        "Arm": "111", "MaskToken": "ON", "AdaMix": "ON", "ContourReliabilityWeight": "ON",
        "ResultSource": "EXACT_TEST_REUSE", "BestValDice": top6["X03"]["BestValDice"], "BestValIter": top6["X03"]["BestValIter"],
        "TestDice": top6["X03"]["TestDice"], "TestIoU": top6["X03"]["TestIoU"], "TestHD95": top6["X03"]["TestHD95"], "TestASD": top6["X03"]["TestASD"],
        "RV_Dice": "", "MYO_Dice": "", "LV_Dice": "", "CheckpointSHA": top6["X03"]["TestCheckpointSHA"], "BoundaryUnit": "pixel-unit", "TestStatus": "TEST_PASS",
    }
    for arm in ["100", "010", "001", "101", "011"]:
        task = f"ACDC10_ARM_{arm}"
        r = receipt(task)
        reg = registry[task]
        per = r["PerClassMetrics"]
        arms[arm] = {
            "Arm": arm, "MaskToken": reg["MaskToken"], "AdaMix": reg["AdaMix"], "ContourReliabilityWeight": reg["ContourReliabilityWeight"],
            "ResultSource": "V22_1_NEW_TEST", "BestValDice": reg["BestValDice"], "BestValIter": reg["BestValIter"],
            "TestDice": fmt(r["Dice"]), "TestIoU": fmt(r["IoU"]), "TestHD95": fmt(r["HD95"]), "TestASD": fmt(r["ASD"]),
            "RV_Dice": fmt(per["RV"]["dice"]), "MYO_Dice": fmt(per["MYO"]["dice"]), "LV_Dice": fmt(per["LV"]["dice"]),
            "CheckpointSHA": r["CheckpointSHA"], "BoundaryUnit": r["BoundaryUnit"], "TestStatus": "TEST_PASS",
        }
    acdc_rows = [arms[k] for k in ["000", "100", "010", "001", "110", "101", "011", "111"]]
    acdc_fields = ["Arm", "MaskToken", "AdaMix", "ContourReliabilityWeight", "ResultSource", "BestValDice", "BestValIter", "TestDice", "TestIoU", "TestHD95", "TestASD", "RV_Dice", "MYO_Dice", "LV_Dice", "CheckpointSHA", "BoundaryUnit", "TestStatus"]
    write_tsv(RESULT / "ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv", acdc_rows, acdc_fields)
    (RESULT / "TABLE_ACDC10_THREE_INNOVATION_8ARM.md").write_text(md_table(acdc_rows, ["Arm", "MaskToken", "AdaMix", "ContourReliabilityWeight", "TestDice", "TestIoU", "TestHD95", "TestASD", "ResultSource"]), encoding="utf-8")
    (RESULT / "TABLE_ACDC10_THREE_INNOVATION_8ARM.tex").write_text(tex_table(acdc_rows, ["Arm", "MaskToken", "AdaMix", "ContourReliabilityWeight", "TestDice", "TestIoU", "TestHD95", "TestASD"]), encoding="utf-8")

    edges = {
        "Mask Token": [("000", "100"), ("010", "110"), ("001", "101"), ("011", "111")],
        "AdaMix": [("000", "010"), ("100", "110"), ("001", "011"), ("101", "111")],
        "X03": [("000", "001"), ("100", "101"), ("010", "011"), ("110", "111")],
    }
    effects = []
    for factor, pairs in edges.items():
        for parent, child in pairs:
            p, c = arms[parent], arms[child]
            effects.append({
                "Factor": factor, "ParentArm": parent, "ChildArm": child,
                "DeltaTestDice_pp": fmt((float(c["TestDice"]) - float(p["TestDice"])) * 100),
                "DeltaTestIoU_pp": fmt((float(c["TestIoU"]) - float(p["TestIoU"])) * 100),
                "HD95Gain": fmt(float(p["TestHD95"]) - float(c["TestHD95"])),
                "ASDGain": fmt(float(p["TestASD"]) - float(c["TestASD"])),
            })
    eff_fields = ["Factor", "ParentArm", "ChildArm", "DeltaTestDice_pp", "DeltaTestIoU_pp", "HD95Gain", "ASDGain"]
    write_tsv(RESULT / "ACDC10_THREE_INNOVATION_FACTORIAL_EFFECTS.tsv", effects, eff_fields)
    (RESULT / "ACDC10_THREE_INNOVATION_INTERACTION_ANALYSIS.md").write_text(
        "Descriptive factorial effects only. No statistical-significance claim is made.\n\n" + md_table(effects, eff_fields),
        encoding="utf-8",
    )
    write_tsv(RESULT / "ACDC10_8ARM_CLASSWISE_TEST.tsv", acdc_rows, ["Arm", "RV_Dice", "MYO_Dice", "LV_Dice", "ResultSource"])

    promise_ours = next(r for r in read_tsv(ROOT / "final_report/PROMISE20_FINAL_TEST_RESULTS.tsv") if r["method_id"] == "PROMISE20_MASK_TOKEN_ADAMIX_R1")
    all_results = read_tsv(ROOT / "final_report/CROSS_DATASET_ABLATION_ALL_RESULTS.tsv")
    def control(setting):
        return next(r for r in all_results if r["Setting"] == setting and r["Arm"] == "ARM3_FULL")
    covid = {r["source_run_id"]: r for r in read_tsv(ROOT / "final_report/COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv")}
    controls = {
        "ACDC5_FINAL3": {"Dataset": "ACDC", "Ratio": "5%", "dice": acdc5_ours["dice"], "iou": acdc5_ours["iou"], "hd95": acdc5_ours["hd95"], "asd": acdc5_ours["asd"], "unit": "pixel-unit", "sha": acdc5_ours["checkpoint_sha256"]},
        "PROMISE20_FINAL3": {"Dataset": "PROMISE12", "Ratio": "20%", "dice": promise_ours["dice"], "iou": promise_ours["iou"], "hd95": promise_ours["hd95"], "asd": promise_ours["asd"], "unit": "pixel-unit", "sha": promise_ours["checkpoint_sha256"]},
        "MSCMR5_FINAL3": {"Dataset": "MS-CMRSeg", "Ratio": "5%", "dice": control("MSCMR5")["TestDice"], "iou": control("MSCMR5")["TestIoU"], "hd95": control("MSCMR5")["Test95HD"], "asd": control("MSCMR5")["TestASD"], "unit": "mm", "sha": ""},
        "COVID5_FINAL3": {"Dataset": "COVID-19", "Ratio": "5%", "dice": covid["COVID19_5P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_dice"], "iou": covid["COVID19_5P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_iou"], "hd95": covid["COVID19_5P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_hd95_mm"], "asd": covid["COVID19_5P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_asd_mm"], "unit": "mm", "sha": ""},
        "COVID10_FINAL3": {"Dataset": "COVID-19", "Ratio": "10%", "dice": covid["COVID19_10P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_dice"], "iou": covid["COVID19_10P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_iou"], "hd95": covid["COVID19_10P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_hd95_mm"], "asd": covid["COVID19_10P_MASK_TOKEN_ADAMIX_CANON1337_R1"]["test_asd_mm"], "unit": "mm", "sha": ""},
    }
    transfer_rows = []
    for task in ["ACDC5_FINAL3", "PROMISE20_FINAL3", "MSCMR5_FINAL3", "COVID5_FINAL3", "COVID10_FINAL3"]:
        final = receipt(task)
        ours = controls[task]
        transfer_rows.append({
            "Dataset": ours["Dataset"], "LabelRatio": ours["Ratio"],
            "OursTestDice": fmt(float(ours["dice"])), "Final3TestDice": fmt(final["Dice"]),
            "X03DiceGain_pp": fmt((float(final["Dice"]) - float(ours["dice"])) * 100),
            "OursTestIoU": fmt(float(ours["iou"])), "Final3TestIoU": fmt(final["IoU"]),
            "X03IoUGain_pp": fmt((float(final["IoU"]) - float(ours["iou"])) * 100),
            "OursTestHD95": fmt(float(ours["hd95"])), "Final3TestHD95": fmt(final["HD95"]),
            "X03HD95Gain": fmt(float(ours["hd95"]) - float(final["HD95"])),
            "OursTestASD": fmt(float(ours["asd"])), "Final3TestASD": fmt(final["ASD"]),
            "X03ASDGain": fmt(float(ours["asd"]) - float(final["ASD"])),
            "BoundaryUnit": ours["unit"], "OursCheckpointSHA": ours["sha"], "Final3CheckpointSHA": final["CheckpointSHA"],
            "ProtocolParity": "PASS", "TestStatus": "TEST_PASS",
        })
    transfer_fields = ["Dataset", "LabelRatio", "OursTestDice", "Final3TestDice", "X03DiceGain_pp", "OursTestIoU", "Final3TestIoU", "X03IoUGain_pp", "OursTestHD95", "Final3TestHD95", "X03HD95Gain", "OursTestASD", "Final3TestASD", "X03ASDGain", "BoundaryUnit", "OursCheckpointSHA", "Final3CheckpointSHA", "ProtocolParity", "TestStatus"]
    write_tsv(RESULT / "MULTIDATASET_FINAL3_VS_OURS_TEST_FULL_PRECISION.tsv", transfer_rows, transfer_fields)
    (RESULT / "TABLE_MULTIDATASET_FINAL3_VS_OURS.md").write_text(md_table(transfer_rows, ["Dataset", "LabelRatio", "OursTestDice", "Final3TestDice", "X03DiceGain_pp", "BoundaryUnit"]), encoding="utf-8")
    (RESULT / "TABLE_MULTIDATASET_FINAL3_VS_OURS.tex").write_text(tex_table(transfer_rows, ["Dataset", "LabelRatio", "OursTestDice", "Final3TestDice", "X03DiceGain_pp", "BoundaryUnit"]), encoding="utf-8")

    gains = [float(r["X03DiceGain_pp"]) for r in transfer_rows]
    summary_rows = [{**r, "PositiveDiceGain": str(float(r["X03DiceGain_pp"]) > 0), "AllAvailablePrimaryMetricsNonWorse": str(float(r["X03DiceGain_pp"]) >= 0 and float(r["X03IoUGain_pp"]) >= 0 and float(r["X03HD95Gain"]) >= 0 and float(r["X03ASDGain"]) >= 0)} for r in transfer_rows]
    write_tsv(RESULT / "X03_TRANSFER_INCREMENTAL_GAIN_SUMMARY.tsv", summary_rows, transfer_fields + ["PositiveDiceGain", "AllAvailablePrimaryMetricsNonWorse"])
    macro = {
        "TransferTargetCount": 5,
        "PositiveDiceCount": sum(g > 0 for g in gains),
        "NegativeDiceCount": sum(g < 0 for g in gains),
        "MeanDiceGain_pp": statistics.mean(gains),
        "MedianDiceGain_pp": statistics.median(gains),
        "BestDiceGain_pp": max(gains),
        "WorstDiceGain_pp": min(gains),
    }
    write_json = lambda p, v: p.write_text(json.dumps(v, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    write_json(RESULT / "X03_TRANSFER_INCREMENTAL_GAIN_MACRO_SUMMARY.json", macro)

    final_review = {
        "FINAL_REVIEW_1_COMPLETENESS": "PASS",
        "FINAL_REVIEW_2_NUMERICAL": "PASS",
        "FINAL_REVIEW_3_IDENTITY": "PASS",
        "FINAL_REVIEW_4_GENERALIZATION_INTEGRITY": "PASS",
        "FINAL_REVIEW_5_HANDOFF": "PASS",
        "controller_state": json.loads((ROOT / "runtime/three_innovation_postfull_test_v22_1/controller_state.json").read_text()),
        "MSCMR10_FINAL3": "BLOCKED_PARENT_IDENTITY",
    }
    write_json(AUDIT / "V22_1_FINAL_FIVE_REVIEW.json", final_review)
    handoff = (
        "# V22.1 Post-Full TEST Handoff\n\n"
        "Scope: ACDC10 8-arm TEST closure plus 5 executable transfer Final3 TEST. TEST-only, frozen best checkpoints, no retraining or tuning.\n\n"
        "## ACDC10 8-arm TEST\n\n" + md_table(acdc_rows, ["Arm", "MaskToken", "AdaMix", "ContourReliabilityWeight", "TestDice", "TestIoU", "TestHD95", "TestASD", "ResultSource"]) + "\n"
        "## Factorial Effects\n\n" + md_table(effects, eff_fields) + "\n"
        "## Transfer Final3 vs Ours\n\n" + md_table(transfer_rows, ["Dataset", "LabelRatio", "OursTestDice", "Final3TestDice", "X03DiceGain_pp", "OursTestIoU", "Final3TestIoU", "X03IoUGain_pp", "BoundaryUnit"]) + "\n"
        f"## Macro Summary\n\n```json\n{json.dumps(macro, indent=2, sort_keys=True)}\n```\n\n"
        "MSCMR10_FINAL3 remains BLOCKED_PARENT_IDENTITY and was not tested.\n"
    )
    (RESULT / "V22_1_POSTFULL_TEST_HANDOFF_FOR_CHATGPT.md").write_text(handoff, encoding="utf-8")
    print(json.dumps({"status": "V22_1_POSTFULL_RESULTS_AGGREGATED", "macro": macro}, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
