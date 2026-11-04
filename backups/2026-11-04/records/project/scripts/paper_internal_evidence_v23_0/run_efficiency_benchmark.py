#!/usr/bin/env python3
"""V23 beta-FFT vs Ours efficiency benchmark.

The benchmark is scratch-only: it writes no checkpoints, reads no TEST data,
and uses identical random input tensors for both models.
"""

from __future__ import annotations

import argparse
import csv
import json
import platform
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
RESULT_ROOT = ROOT / "final_report/paper_internal_evidence_v23_0"
AUDIT_ROOT = ROOT / "audits/paper_internal_evidence_v23_0"


def write_tsv(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), delimiter="\t")
        writer.writeheader()
        writer.writerows(rows)


def benchmark(args: argparse.Namespace) -> dict:
    sys.path.insert(0, str(ROOT / "train_ACDC"))
    import torch
    from v14_mask_adamix_components import v14_bcp_net

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")
    precision = "fp32"
    rows = []
    for method_id, display_name, mask_token_enabled in [
        ("ACDC10_ARM_000", "beta-FFT", False),
        ("ACDC10_ARM_111", "Ours", True),
    ]:
        torch.manual_seed(42)
        model = v14_bcp_net(
            in_chns=1,
            class_num=4,
            ema=False,
            mask_token_enabled=mask_token_enabled,
            mask_token_num=4,
            mask_token_dim=16,
        ).to(device)
        model.eval()
        params = sum(p.numel() for p in model.parameters())
        trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
        sample = torch.randn(args.batch_size, 1, args.height, args.width, device=device)
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        with torch.no_grad():
            for _ in range(args.warmup):
                _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            start = time.perf_counter()
            for _ in range(args.repeat):
                _ = model(sample)
            if device.type == "cuda":
                torch.cuda.synchronize(device)
            elapsed = time.perf_counter() - start
        latency_ms = elapsed * 1000.0 / float(args.repeat)
        fps = args.batch_size * 1000.0 / latency_ms
        peak_mem_mb = torch.cuda.max_memory_allocated(device) / (1024.0 * 1024.0) if device.type == "cuda" else ""
        rows.append(
            {
                "MethodID": method_id,
                "DisplayName": display_name,
                "MaskToken": "ON" if mask_token_enabled else "OFF",
                "AdaMix": "ON" if method_id == "ACDC10_ARM_111" else "OFF",
                "X03ContourReliabilityWeight": "ON" if method_id == "ACDC10_ARM_111" else "OFF",
                "ParameterCount": params,
                "TrainableParameterCount": trainable,
                "FLOPs": "NOT_MEASURED_NO_THOP_DEPENDENCY",
                "BatchSize": args.batch_size,
                "InputShape": f"{args.batch_size}x1x{args.height}x{args.width}",
                "Precision": precision,
                "Device": str(device),
                "Warmup": args.warmup,
                "Repeat": args.repeat,
                "LatencyMsPerBatch": f"{latency_ms:.6f}",
                "FPS": f"{fps:.6f}",
                "PeakInferenceGpuMemoryMB": f"{peak_mem_mb:.6f}" if peak_mem_mb != "" else "",
                "CheckpointWritten": False,
                "TestAccess": 0,
            }
        )
    out = RESULT_ROOT / "EFFICIENCY_BENCHMARK_BETAFFT_VS_OURS.tsv"
    write_tsv(out, rows)
    payload = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "output_tsv": str(out.relative_to(ROOT)),
        "python": sys.executable,
        "platform": platform.platform(),
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "new_training_started": False,
        "checkpoint_written": False,
        "test_access": 0,
        "rows": rows,
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "EFFICIENCY_BENCHMARK_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "run"], default="run")
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--height", type=int, default=256)
    parser.add_argument("--width", type=int, default=256)
    parser.add_argument("--warmup", type=int, default=20)
    parser.add_argument("--repeat", type=int, default=100)
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "dry-run":
        payload = {
            "status": "DRY_RUN_PASS",
            "methods": ["ACDC10_ARM_000", "ACDC10_ARM_111"],
            "input_shape": f"{args.batch_size}x1x{args.height}x{args.width}",
            "checkpoint_written": False,
            "test_access": 0,
        }
        print(json.dumps(payload, indent=2))
        return 0
    payload = benchmark(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
