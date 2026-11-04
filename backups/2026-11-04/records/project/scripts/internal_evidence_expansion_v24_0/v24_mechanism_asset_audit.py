#!/usr/bin/env python3
"""V24 mechanism-analysis asset audit.

This is a read-only gate for A2/A3/A10. It binds X03 score semantics from the
frozen source and inventories whether existing outputs are sufficient for
reliability faithfulness, pseudo-label evolution, and student diversity.
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/internal_evidence_expansion_v24_0"
RESULT_ROOT = ROOT / "final_report/internal_evidence_expansion_v24_0"
X03_SOURCE = ROOT / "train_ACDC/v21_3_acdc10_third_innovation_components.py"
TRANSFER_SOURCE = ROOT / "train_ACDC/v21_3_acdc10_third_innovation_transfer.py"

RUNS = {
    "ACDC10_ARM_000": ROOT / "runs/beta_fft_acdc_official_full_seed42_fresh_ckpt",
    "ACDC10_ARM_110": ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42",
    "ACDC10_ARM_111": ROOT / "runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT",
}


def now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n")


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def x03_semantic_contract() -> dict:
    text = X03_SOURCE.read_text()
    x03_match = re.search(
        r'if candidate_id == "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT":(?P<body>.*?)(?:\n    if candidate_id ==|\n    raise)',
        text,
        flags=re.S,
    )
    if not x03_match:
        raise SystemExit("X03_BRANCH_NOT_FOUND")
    body = x03_match.group("body")
    required_tokens = ["contour", "boundary_weight", "contour_uncertainty_mean", "boundary_weight_mean"]
    missing = [tok for tok in required_tokens if tok not in body]
    if missing:
        raise SystemExit(f"X03_SEMANTIC_TOKENS_MISSING:{missing}")
    contract = {
        "status": "PASS",
        "created_at": now(),
        "task_id": "X03_RELIABILITY_FAITHFULNESS",
        "source": str(X03_SOURCE.relative_to(ROOT)),
        "source_sha256": sha256(X03_SOURCE),
        "frozen_candidate_id": "V21_3_X03_CONTOUR_RELIABILITY_WEIGHT",
        "score_name": "contour_uncertainty_or_boundary_weight",
        "score_semantics": "higher score means stronger contour uncertainty / stronger boundary consistency weight, not lower risk reliability",
        "higher_score_means_more_reliable": False,
        "higher_score_means_less_reliable_or_more_fragile_boundary": True,
        "direction_fixed_before_result_reading": True,
        "gt_used_to_generate_score": False,
        "semantic_evidence_tokens": required_tokens,
        "renaming_to_reliability_without_direction_note_allowed": False,
    }
    write_json(AUDIT_ROOT / "X03_SCORE_SEMANTIC_CONTRACT.json", contract)
    return contract


def list_files(root: Path, patterns: tuple[str, ...]) -> list[Path]:
    if not root.exists():
        return []
    out = []
    for path in root.rglob("*"):
        if path.is_file() and any(path.match(pattern) for pattern in patterns):
            out.append(path)
    return sorted(out)


def checkpoint_iteration(path: Path) -> str:
    name = path.name.lower()
    if name == "best.ckpt":
        return "BEST"
    if name == "last.ckpt":
        return "LAST"
    match = re.search(r"(\d+)", path.stem)
    return match.group(1) if match else "UNKNOWN"


def audit_checkpoint_and_output_assets() -> dict:
    rows = []
    checkpoint_sets: dict[str, set[str]] = {}
    for method, run_root in RUNS.items():
        ckpts = list_files(run_root / "checkpoints", ("*.ckpt", "*.pth", "*.pt"))
        score_maps = list_files(run_root, ("*reliab*.npy", "*reliab*.npz", "*score*.npy", "*score*.npz", "*contour*.npy", "*contour*.npz"))
        pseudo_outputs = list_files(run_root, ("*pseudo*.npy", "*pseudo*.npz", "*pseudo*.h5", "*pseudo*.hdf5", "*pseudo*.pkl"))
        branch_outputs = list_files(run_root, ("*branch*.npy", "*branch*.npz", "*student*.npy", "*student*.npz", "*logit*.npy", "*logit*.npz"))
        iters = {checkpoint_iteration(p) for p in ckpts}
        checkpoint_sets[method] = iters
        rows.append(
            {
                "MethodID": method,
                "RunRoot": str(run_root.relative_to(ROOT)),
                "RunRootExists": str(run_root.exists()),
                "CheckpointCount": len(ckpts),
                "CheckpointIterations": ",".join(sorted(iters)),
                "ScoreMapCount": len(score_maps),
                "PseudoOutputCount": len(pseudo_outputs),
                "BranchOutputCount": len(branch_outputs),
                "ExistingAssetsEnoughForA2PixelFaithfulness": str(len(score_maps) > 0),
                "ExistingAssetsEnoughForA3PseudoEvolution": str(len(pseudo_outputs) > 0),
                "ExistingAssetsEnoughForA10StudentDiversity": str(len(branch_outputs) > 0),
            }
        )
    common = set.intersection(*checkpoint_sets.values()) if checkpoint_sets else set()
    out = AUDIT_ROOT / "V24_0_MECHANISM_SOURCE_ASSET_AUDIT.tsv"
    write_tsv(out, rows)
    receipt = {
        "status": "PASS_SOURCE_GAP_MAPPED",
        "created_at": now(),
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "audit_asset": str(out.relative_to(ROOT)),
        "audit_sha256": sha256(out),
        "common_checkpoint_iterations": sorted(common),
        "common_checkpoint_count": len(common),
        "pseudo_label_evolution_status": "BLOCKED_COMMON_ITERATION_COUNT_LT_4_OR_PSEUDO_OUTPUTS_MISSING",
        "student_diversity_status": "BLOCKED_COMMON_ITERATION_COUNT_LT_4_OR_BRANCH_OUTPUTS_MISSING",
        "x03_pixel_faithfulness_status": "FROZEN_CHECKPOINT_REPLAY_REQUIRED",
        "required_common_checkpoint_count": 4,
        "note": "BEST/LAST checkpoint names are not sufficient to claim exact iteration schedule without stronger metadata.",
    }
    write_json(AUDIT_ROOT / "V24_0_MECHANISM_SOURCE_ASSET_AUDIT.json", receipt)
    return receipt


def main() -> int:
    summary = {
        "status": "PASS_SOURCE_GAPS_MAPPED",
        "created_at": now(),
        "training_started": False,
        "checkpoint_replay_started": False,
        "test_manifest_accessed": False,
        "x03_score_semantic_contract": x03_semantic_contract(),
        "mechanism_source_asset_audit": audit_checkpoint_and_output_assets(),
        "transfer_source": str(TRANSFER_SOURCE.relative_to(ROOT)),
        "transfer_source_sha256": sha256(TRANSFER_SOURCE),
    }
    write_json(AUDIT_ROOT / "V24_0_MECHANISM_ASSET_AUDIT_SUMMARY.json", summary)
    print(json.dumps(summary, indent=2, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
