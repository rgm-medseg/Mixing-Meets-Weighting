#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

CANONICAL = {
    "dataset_root": "/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC",
    "labelnum": 7,
    "seed": 42,
    "max_iterations": 30000,
    "validation_interval": 200,
    "save_interval": 200,
    "mask_token_enabled": True,
    "mask_token_num": 4,
    "mask_token_dim": 16,
    "adamix_enabled": True,
    "adamix_min_side": 0.18,
    "adamix_max_side": 0.38,
}

EXPECTED_X_KEYS = {
    "candidate_id",
    "candidate_name",
    "rank",
    "only_scientific_difference",
    "source3_risk_id",
    "source4_primary_paper",
    "source4_year",
    "innovation_x_enabled",
    "innovation_x_weight",
    "output_dir",
    "schema",
    "created_at",
    "canonical_parent",
    "test_access",
}


def classify(key: str, actual) -> str:
    if key in CANONICAL:
        return "MATCH" if actual == CANONICAL[key] else "UNEXPECTED_SCIENTIFIC_DIFFERENCE"
    if key in EXPECTED_X_KEYS:
        return "EXPECTED_INNOVATION_X"
    return "ENGINEERING_PATH_ONLY" if key.endswith("_path") or "dir" in key else "UNEXPECTED_SCIENTIFIC_DIFFERENCE"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidate-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    cfg_path = (ROOT / args.candidate_config).resolve() if not Path(args.candidate_config).is_absolute() else Path(args.candidate_config)
    out = (ROOT / args.out).resolve() if not Path(args.out).is_absolute() else Path(args.out)
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    fields = ["Key", "CanonicalValue", "CandidateValue", "DifferenceClass"]
    rows = []
    for key in sorted(set(CANONICAL) | set(cfg)):
        rows.append(
            {
                "Key": key,
                "CanonicalValue": json.dumps(CANONICAL.get(key, ""), sort_keys=True),
                "CandidateValue": json.dumps(cfg.get(key, ""), sort_keys=True),
                "DifferenceClass": classify(key, cfg.get(key)),
            }
        )
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, delimiter="\t", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    unexpected = [r for r in rows if r["DifferenceClass"] == "UNEXPECTED_SCIENTIFIC_DIFFERENCE"]
    print(json.dumps({"out": str(out.relative_to(ROOT)), "UnexpectedScientificDifferenceCount": len(unexpected)}, indent=2))


if __name__ == "__main__":
    main()
