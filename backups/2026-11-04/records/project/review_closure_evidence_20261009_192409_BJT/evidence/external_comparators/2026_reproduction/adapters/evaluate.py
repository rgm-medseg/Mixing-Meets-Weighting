from __future__ import annotations

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPRO = HERE.parent
sys.path.insert(0, str(REPRO))

from adapters.common.metrics import evaluate  # noqa: E402
from adapters.common.state import atomic_json, seed_everything, sha256  # noqa: E402


def build_model(config: dict, device: torch.device) -> torch.nn.Module:
    method = config["method"].lower()
    if method == "hien":
        os.environ.setdefault("TORCH_HOME", str(REPRO / "pretrained/torch"))
        sys.path.insert(0, str(REPRO / "adapters/hien/upstream_code"))
        from unet import UNet  # type: ignore
        return UNet(in_chns=3, class_num=int(config["num_classes"])).to(device)
    sys.path.insert(0, str(REPRO / "adapters/pmeul/upstream_code"))
    from unet import MCNet2d_v2  # type: ignore
    return MCNet2d_v2(in_chns=1, class_num=int(config["num_classes"])).to(device)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--split", choices=("val", "test"), default="test")
    parser.add_argument("--smoke", action="store_true")
    args = parser.parse_args()
    config_path, checkpoint_path = Path(args.config).resolve(), Path(args.checkpoint).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    output, predictions = Path(args.output_dir).resolve(), Path(args.predictions_dir).resolve()
    output.mkdir(parents=True, exist_ok=True)
    predictions.mkdir(parents=True, exist_ok=True)
    seed_everything(int(config["seed"]))
    device = torch.device(config.get("device", "cuda:0"))
    model = build_model(config, device)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    result = model.load_state_dict(checkpoint["model"], strict=True)
    missing, unexpected = list(result.missing_keys), list(result.unexpected_keys)
    if missing or unexpected:
        raise RuntimeError(f"strict checkpoint mismatch: missing={missing}, unexpected={unexpected}")
    model.eval()

    def sink(case_id: str, prediction: np.ndarray) -> None:
        np.savez_compressed(predictions / f"{case_id}.npz", prediction=prediction.astype(np.uint8))

    metrics = evaluate(model, config, args.split, device, prediction_sink=sink, max_cases=1 if args.smoke else None)
    fields = ["case", "class", "dice", "iou", "hd95", "asd"]
    with (output / "case_metrics.tsv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, delimiter="\t")
        writer.writeheader()
        writer.writerows(metrics["rows"])
    atomic_json(metrics, output / "metrics.json")
    receipt = {
        "status": "SMOKE_EVAL_PASS" if args.smoke else "EVALUATION_PASS",
        "job_id": config["job_id"], "method": config["method"], "dataset": config["dataset"],
        "split": args.split, "checkpoint": str(checkpoint_path), "checkpoint_sha256": sha256(checkpoint_path),
        "config_sha256": sha256(config_path), "evaluator": config["data"]["evaluator"],
        "evaluator_sha256": config["data"]["evaluator_sha256"], "checkpoint_missing_keys": missing,
        "checkpoint_unexpected_keys": unexpected, "prediction_output": "selected_logits" if config["method"] == "HiEn" else "decoder1_index_0",
        "metrics": metrics["macro"], "test_used_for_selection": False,
    }
    atomic_json(receipt, output / "EVALUATION_RECEIPT.json")
    print(json.dumps(receipt, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
