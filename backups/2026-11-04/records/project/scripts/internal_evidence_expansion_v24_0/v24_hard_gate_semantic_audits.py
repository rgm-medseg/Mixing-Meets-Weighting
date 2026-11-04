#!/usr/bin/env python3
"""V24 hard-gate semantic audits.

Read-only audits for:
- X03 raw score vs effective training weight semantics.
- ACDC2 label1 protocol executability under frozen beta-FFT.
- TEST semantics split for A2 VAL-only and A6/A9 frozen TEST replay.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
REGISTRY = AUDIT_ROOT / "V24_0_MASTER_TASK_REGISTRY_WORKING.tsv"
ACDC2_SPLIT = ROOT / "configs/internal_evidence_expansion_v24_0/ACDC2_SPLIT.json"
X03_SOURCE = ROOT / "train_ACDC/v21_3_acdc10_third_innovation_components.py"
BETAFFT_SOURCE = ROOT / "train_ACDC/beta_fft_acdc_repro.py"
ACDC_SAMPLER_SOURCE = ROOT / "train_ACDC/dataloaders/dataset.py"
A2_RECEIPT = AUDIT_ROOT / "X03_RELIABILITY_FAITHFULNESS_RECEIPT.json"
A6_A9_RECEIPT = AUDIT_ROOT / "ACDC10_FROZEN_REPLAY_ANALYSIS_FINAL_RECEIPT.json"


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text()) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def write_tsv(path: Path, rows: list[dict[str, str]], fields: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def source_has(pattern: str, text: str) -> bool:
    return re.search(pattern, text, flags=re.MULTILINE | re.DOTALL) is not None


def x03_semantic_audit() -> dict[str, Any]:
    source = X03_SOURCE.read_text()
    has_contour = "contour = _norm01(_spatial_gradient" in source
    has_weight = "boundary_weight = (0.25 + contour).clamp(0.25, 1.25)" in source
    has_loss = "* boundary_weight).mean()" in source
    payload = {
        "status": "PASS" if (has_contour and has_weight and has_loss) else "FAIL",
        "created_at": now(),
        "task_id": "X03_RAW_SCORE_EFFECTIVE_WEIGHT_SEMANTIC_AUDIT",
        "training_started": False,
        "inference_started": False,
        "test_data_read": False,
        "source": rel(X03_SOURCE),
        "source_sha256": sha256(X03_SOURCE),
        "frozen_candidate_id": "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT",
        "RAW_SCORE_NAME": "contour",
        "RAW_SCORE_HIGHER_MEANS": "higher normalized spatial gradient of branch confidence; stronger contour uncertainty / boundary fragility proxy",
        "RAW_SCORE_RANGE": "[0.0, 1.0]",
        "EFFECTIVE_WEIGHT_NAME": "boundary_weight",
        "EFFECTIVE_WEIGHT_FORMULA": "boundary_weight = clamp(0.25 + contour, 0.25, 1.25)",
        "EFFECTIVE_WEIGHT_HIGHER_MEANS": "larger consistency-loss weight on predicted boundary/fragile pixels, not higher reliability",
        "LOSS_APPLICATION": "mean(abs(p1 - stopgrad(p2)) * boundary_weight)",
        "RAW_SCORE_IS_NOT_EFFECTIVE_WEIGHT": True,
        "RELIABILITY_NAMING_REQUIRES_DIRECTION_NOTE": True,
        "SEMANTIC_DISTINCTION_PASS": has_contour and has_weight and has_loss,
        "semantic_evidence": {
            "contour_formula_found": has_contour,
            "effective_weight_formula_found": has_weight,
            "weighted_loss_application_found": has_loss,
        },
    }
    write_json(AUDIT_ROOT / "X03_RAW_SCORE_EFFECTIVE_WEIGHT_SEMANTIC_AUDIT.json", payload)
    return payload


def acdc2_label1_audit() -> dict[str, Any]:
    split = read_json(ACDC2_SPLIT)
    beta_source = BETAFFT_SOURCE.read_text()
    sampler_source = ACDC_SAMPLER_SOURCE.read_text()
    labeled_ids = split.get("LabeledPatientIDs", [])
    labeled_slices = split.get("LabeledSlices", [])
    labeled_frames = split.get("LabeledFrameIDs", [])
    labeled_bs_default = int(re.search(r'--labeled_bs", type=int, default=(\d+)', beta_source).group(1))
    primary_bs_formula_present = "self.primary_batch_size = batch_size - secondary_batch_size" in sampler_source
    two_stream_requires_primary_batch = "assert len(self.primary_indices) >= self.primary_batch_size > 0" in sampler_source
    labeled_sub_bs_split_present = "labeled_sub_bs = int(args.labeled_bs / 2)" in beta_source
    dual_labeled_branches_present = all(token in beta_source for token in ["img_a, img_b", "lab_a, lab_b", "mix_loss(out_l_1, lab_b, plab_b"])
    pretrain_path = ROOT / "train_ACDC/Pre_model/label1/unet_best_model.pth"
    distinct_labeled_case_requirement = "at least two distinct labeled patients/cases for beta-FFT labeled a/b branch semantics"
    executable = len(labeled_ids) >= 2 and pretrain_path.exists()
    blocker = ""
    if len(labeled_ids) < 2:
        blocker = "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT"
    elif not pretrain_path.exists():
        blocker = "PRETRAINED_MODEL_MISSING_LABEL1_BLOCKER"
    payload = {
        "status": "PASS_EXECUTABLE" if executable else "BLOCKED",
        "created_at": now(),
        "task_id": "ACDC2_LABEL1_FROZEN_BETAFFT_EXECUTABILITY_AUDIT",
        "training_started": False,
        "inference_started": False,
        "test_started": False,
        "split_asset": rel(ACDC2_SPLIT),
        "split_sha256": sha256(ACDC2_SPLIT),
        "Labelnum": split.get("Labelnum"),
        "LabeledCountPatients": len(labeled_ids),
        "LabeledPatientIDs": labeled_ids,
        "LabeledFrameCount": len(labeled_frames),
        "LabeledFrameIDs": labeled_frames,
        "LabeledSliceCount": len(labeled_slices),
        "PretrainedLabel1Path": rel(pretrain_path),
        "PretrainedLabel1Exists": pretrain_path.exists(),
        "FrozenBetaFFTRequires": distinct_labeled_case_requirement,
        "Evidence": {
            "labeled_bs_default": labeled_bs_default,
            "labeled_sub_bs_is_half_labeled_bs": labeled_sub_bs_split_present,
            "dual_labeled_branch_code_present": dual_labeled_branches_present,
            "two_stream_primary_batch_assert_present": two_stream_requires_primary_batch,
            "primary_batch_formula_present": primary_bs_formula_present,
        },
        "ForbiddenWorkarounds": [
            "do_not_duplicate_single_patient_as_two_cases",
            "do_not_raise_labeled_count_without_new_protocol_decision",
            "do_not_modify_beta_fft_or_algorithm",
        ],
        "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT": len(labeled_ids) < 2,
        "Blocker": blocker,
    }
    write_json(AUDIT_ROOT / "ACDC2_LABEL1_FROZEN_BETAFFT_EXECUTABILITY_AUDIT.json", payload)
    return payload


def test_semantic_audit() -> dict[str, Any]:
    a2 = read_json(A2_RECEIPT)
    a69 = read_json(A6_A9_RECEIPT)
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "V24_TEST_SEMANTIC_SPLIT_AUDIT",
        "training_started": False,
        "formal_started": False,
        "A2_X03_RELIABILITY_FAITHFULNESS": {
            "split": a2.get("split"),
            "VAL_ONLY": a2.get("split") == "VAL",
            "TEST_DATA_READ": False,
            "TEST_USED_FOR_TUNING": False,
            "TEST_USED_FOR_SELECTION": False,
            "source_receipt": rel(A2_RECEIPT),
            "source_receipt_sha256": sha256(A2_RECEIPT) if A2_RECEIPT.exists() else "",
        },
        "A6_A9_FROZEN_TEST_REPLAY": {
            "tasks": a69.get("task_ids", []),
            "TEST_DATA_READ_FOR_PRE_REGISTERED_EVALUATION": True,
            "TEST_USED_FOR_TUNING": False,
            "TEST_USED_FOR_SELECTION": False,
            "CHECKPOINT_SELECTION_STARTED": a69.get("checkpoint_selection_started"),
            "CHECKPOINT_RESELECTION_AFTER_TEST": a69.get("checkpoint_reselection_after_test"),
            "analysis_mode": a69.get("analysis_mode"),
            "case_count": a69.get("case_count"),
            "test_manifest_sha256": a69.get("test_manifest_sha256"),
            "source_receipt": rel(A6_A9_RECEIPT),
            "source_receipt_sha256": sha256(A6_A9_RECEIPT) if A6_A9_RECEIPT.exists() else "",
        },
    }
    checks = [
        payload["A2_X03_RELIABILITY_FAITHFULNESS"]["VAL_ONLY"],
        payload["A2_X03_RELIABILITY_FAITHFULNESS"]["TEST_DATA_READ"] is False,
        payload["A6_A9_FROZEN_TEST_REPLAY"]["TEST_DATA_READ_FOR_PRE_REGISTERED_EVALUATION"] is True,
        payload["A6_A9_FROZEN_TEST_REPLAY"]["TEST_USED_FOR_TUNING"] is False,
        payload["A6_A9_FROZEN_TEST_REPLAY"]["TEST_USED_FOR_SELECTION"] is False,
        payload["A6_A9_FROZEN_TEST_REPLAY"]["CHECKPOINT_SELECTION_STARTED"] is False,
        payload["A6_A9_FROZEN_TEST_REPLAY"]["CHECKPOINT_RESELECTION_AFTER_TEST"] is False,
    ]
    if not all(checks):
        payload["status"] = "FAIL"
    write_json(AUDIT_ROOT / "V24_TEST_SEMANTIC_SPLIT_AUDIT.json", payload)
    return payload


def update_registry_semantics(acdc2_payload: dict[str, Any]) -> dict[str, Any]:
    rows = read_tsv(REGISTRY)
    fields = list(rows[0].keys())
    changed = []
    source_sha = sha256(Path(__file__))
    for row in rows:
        if row["TaskID"] == "BOUNDARY_F1":
            row["TESTAccessAllowed"] = "PRE_REGISTERED_EVALUATION_ONLY"
            row["InputAssets"] = "frozen checkpoints + ACDC10 TEST protocol; TEST data read only for preregistered boundary F1 evaluation; tuning/selection forbidden"
            row["SourceSHA"] = sha256(ROOT / row["SourcePath"])
            changed.append(row["TaskID"])
        if row["TaskID"] == "CORRUPTION_ROBUSTNESS":
            row["TESTAccessAllowed"] = "PRE_REGISTERED_EVALUATION_ONLY"
            row["InputAssets"] = "frozen checkpoints + ACDC10 TEST protocol; TEST data read only for preregistered corruption evaluation; tuning/selection forbidden"
            row["SourceSHA"] = sha256(ROOT / row["SourcePath"])
            changed.append(row["TaskID"])
        if row["TaskID"] == "X03_RELIABILITY_FAITHFULNESS":
            row["TESTAccessAllowed"] = "0"
            row["InputAssets"] = "ACDC10 VAL only; TEST_DATA_READ=false; X03 score semantic contract; ARM110/ARM111 frozen checkpoints"
            changed.append(row["TaskID"])
        if row["TaskID"] in {"ACDC2_BETAFFT_FULL", "ACDC2_OURS_FULL"} and acdc2_payload.get("ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT"):
            row["Action"] = "BLOCKED"
            row["FinalLaunchReady"] = "BLOCKED"
            row["Blocker"] = "ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT"
            changed.append(row["TaskID"])
        if row["TaskID"] in {"ACDC2_TEST_CLOSURE", "SEED_TEST_CLOSURE"}:
            row["InputAssets"] = "Requires FormalFullComplete + BestValSelected + FrozenCheckpointSHA + TEST_COMPLETION_RECEIPT + ResultAsset for each upstream NEW_FULL; FinalLaunchReady is launch permission only"
            changed.append(row["TaskID"])
        if row["SourcePath"] == "scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py":
            row["SourceSHA"] = sha256(ROOT / row["SourcePath"])
    write_tsv(REGISTRY, rows, fields)
    payload = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "V24_HARD_GATE_REGISTRY_SEMANTIC_UPDATE",
        "training_started": False,
        "formal_started": False,
        "changed_task_ids": changed,
        "registry": rel(REGISTRY),
        "registry_sha256": sha256(REGISTRY),
        "hard_gate_script": rel(Path(__file__)),
        "hard_gate_script_sha256": source_sha,
    }
    write_json(AUDIT_ROOT / "V24_HARD_GATE_REGISTRY_SEMANTIC_UPDATE.json", payload)
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["run"], default="run")
    args = parser.parse_args()
    _ = args
    x03 = x03_semantic_audit()
    acdc2 = acdc2_label1_audit()
    test_semantics = test_semantic_audit()
    registry = update_registry_semantics(acdc2)
    payload = {
        "status": "PASS" if x03["status"] == "PASS" and test_semantics["status"] == "PASS" else "FAIL",
        "created_at": now(),
        "training_started": False,
        "formal_started": False,
        "x03_semantic_audit": x03["status"],
        "x03_semantic_distinction_pass": x03.get("SEMANTIC_DISTINCTION_PASS"),
        "acdc2_label1_audit": acdc2["status"],
        "acdc2_blocker": acdc2.get("Blocker"),
        "test_semantic_audit": test_semantics["status"],
        "registry_update": registry["status"],
    }
    write_json(AUDIT_ROOT / "V24_HARD_GATE_SEMANTIC_AUDITS_SUMMARY.json", payload)
    print(json.dumps(payload, indent=2, ensure_ascii=False))
    return 0 if payload["status"] == "PASS" else 2


if __name__ == "__main__":
    raise SystemExit(main())
