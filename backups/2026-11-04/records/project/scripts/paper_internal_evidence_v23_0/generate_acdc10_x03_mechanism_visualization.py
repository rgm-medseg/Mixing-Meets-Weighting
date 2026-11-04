#!/usr/bin/env python3
"""Generate VAL-only X03 mechanism visualization for V23."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from datetime import datetime, timezone
from pathlib import Path


ROOT = Path(__file__).resolve().parents[2]
AUDIT_ROOT = ROOT / "audits/paper_internal_evidence_v23_0"
RESULT_ROOT = ROOT / "final_report/paper_internal_evidence_v23_0"
FIGURE_ROOT = RESULT_ROOT / "figures"
SELECTION = RESULT_ROOT / "ACDC10_VAL_MECHANISM_VIS_CASE_SELECTION.tsv"
ARM110_CKPT = ROOT / "runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/checkpoints/best.ckpt"
ARM111_CKPT = ROOT / "runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt"


def read_tsv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle, delimiter="\t"))


def run_visualization(args: argparse.Namespace) -> dict:
    if not SELECTION.exists():
        raise SystemExit("ACDC10_VAL_MECHANISM_VIS_CASE_SELECTION_MISSING")

    sys.path.insert(0, str(ROOT / "train_ACDC"))
    import h5py
    import matplotlib.pyplot as plt
    import numpy as np
    import torch
    from scipy.ndimage import zoom
    from v14_mask_adamix_components import v14_bcp_net

    device = torch.device("cuda:0" if torch.cuda.is_available() and not args.cpu else "cpu")

    def build_model(ckpt_path: Path, state_key: str = "model_1"):
        ckpt = torch.load(ckpt_path, map_location="cpu")
        model = v14_bcp_net(
            in_chns=1,
            class_num=4,
            ema=False,
            mask_token_enabled=True,
            mask_token_num=4,
            mask_token_dim=16,
        )
        model.load_state_dict(ckpt[state_key])
        model.to(device)
        model.eval()
        return model, ckpt

    arm110, ckpt110 = build_model(ARM110_CKPT, "model_1")
    arm111_m1, ckpt111 = build_model(ARM111_CKPT, "model_1")
    arm111_m2, _ = build_model(ARM111_CKPT, "model_2")

    def infer_slice(model, image2d):
        h, w = image2d.shape
        resized = zoom(image2d, (256 / h, 256 / w), order=0)
        inp = torch.from_numpy(resized).float().unsqueeze(0).unsqueeze(0).to(device)
        with torch.no_grad():
            logits = model(inp)
            prob = torch.softmax(logits, dim=1)
            pred = torch.argmax(prob, dim=1).squeeze(0).cpu().numpy()
        pred = zoom(pred, (h / 256, w / 256), order=0)
        return pred.astype("int16"), prob.squeeze(0).cpu()

    def x03_map(prob1, prob2):
        import torch

        confidence = 0.5 * (prob1.max(0).values + prob2.max(0).values)
        dx = torch.zeros_like(confidence)
        dy = torch.zeros_like(confidence)
        dx[1:, :] = (confidence[1:, :] - confidence[:-1, :]).abs()
        dy[:, 1:] = (confidence[:, 1:] - confidence[:, :-1]).abs()
        contour = dx + dy
        flat = contour.flatten()
        contour = ((contour - flat.min()) / (flat.max() - flat.min()).clamp_min(1e-6)).clamp(0.0, 1.0)
        return contour.numpy()

    rows = read_tsv(SELECTION)
    FIGURE_ROOT.mkdir(parents=True, exist_ok=True)
    produced = []
    for row in rows:
        with h5py.File(row["VolumePath"], "r") as handle:
            image = handle["image"][int(row["SelectedSliceIndex0Based"])]
            label = handle["label"][int(row["SelectedSliceIndex0Based"])]
        pred110, _ = infer_slice(arm110, image)
        pred111, prob111_m1 = infer_slice(arm111_m1, image)
        _, prob111_m2 = infer_slice(arm111_m2, image)
        reliability = x03_map(prob111_m1, prob111_m2)
        err110 = (pred110 != label).astype("float32") * (label > 0)
        err111 = (pred111 != label).astype("float32") * (label > 0)

        fig, axes = plt.subplots(1, 6, figsize=(14.5, 2.8), dpi=160)
        panels = [
            (image, "Input", "gray"),
            (label, "GT", "viridis"),
            (pred110, "Mask+AdaMix", "viridis"),
            (pred111, "Ours", "viridis"),
            (reliability, "X03 map", "magma"),
            (err110 - err111, "Error delta", "coolwarm"),
        ]
        for ax, (data, title, cmap) in zip(axes, panels):
            ax.imshow(data, cmap=cmap)
            ax.set_title(title, fontsize=8)
            ax.axis("off")
        fig.tight_layout()
        out = FIGURE_ROOT / f"FIG_X03_MECHANISM_{row['CaseID']}_slice{row['SelectedSliceIndex0Based']}.png"
        fig.savefig(out)
        plt.close(fig)
        produced.append(str(out.relative_to(ROOT)))

    payload = {
        "status": "PASS",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "selection": str(SELECTION.relative_to(ROOT)),
        "split": "VAL",
        "case_count": len(rows),
        "figures": produced,
        "arm110_checkpoint": str(ARM110_CKPT.relative_to(ROOT)),
        "arm110_global_step": ckpt110.get("global_step"),
        "arm111_checkpoint": str(ARM111_CKPT.relative_to(ROOT)),
        "arm111_global_step": ckpt111.get("global_step"),
        "x03_formula": "contour=norm01(spatial_gradient(0.5*(maxprob_model1+maxprob_model2))); boundary_weight=0.25+contour",
        "test_access": 0,
        "new_training_started": False,
    }
    AUDIT_ROOT.mkdir(parents=True, exist_ok=True)
    (AUDIT_ROOT / "ACDC10_X03_MECHANISM_VISUALIZATION_SUMMARY.json").write_text(json.dumps(payload, indent=2) + "\n")
    return payload


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=["dry-run", "run"], default="run")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()
    if args.mode == "dry-run":
        payload = {
            "status": "DRY_RUN_PASS",
            "selection": str(SELECTION.relative_to(ROOT)),
            "split": "VAL",
            "checkpoints": [str(ARM110_CKPT.relative_to(ROOT)), str(ARM111_CKPT.relative_to(ROOT))],
            "test_access": 0,
            "new_training_started": False,
        }
        print(json.dumps(payload, indent=2))
        return 0
    payload = run_visualization(args)
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
