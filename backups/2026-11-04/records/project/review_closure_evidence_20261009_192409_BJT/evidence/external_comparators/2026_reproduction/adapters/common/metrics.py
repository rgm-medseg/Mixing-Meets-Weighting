from __future__ import annotations

from typing import Any, Callable

import numpy as np
import torch
from medpy import metric
from scipy.ndimage import binary_erosion, distance_transform_edt, zoom

from .data import iter_volumes


def _acdc_safe(pred: np.ndarray, truth: np.ndarray) -> tuple[float, float, float, float]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    if pred.any() and truth.any():
        intersection = int((pred & truth).sum())
        union = int((pred | truth).sum())
        return (
            2.0 * intersection / (int(pred.sum()) + int(truth.sum())),
            intersection / union if union else 0.0,
            float(metric.binary.hd95(pred, truth)),
            float(metric.binary.asd(pred, truth)),
        )
    if not pred.any() and not truth.any():
        return 0.0, 0.0, 0.0, 0.0
    penalty = float(np.linalg.norm(pred.shape))
    return 0.0, 0.0, penalty, penalty


def _binary_pixel(pred: np.ndarray, truth: np.ndarray) -> dict[str, float]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    denominator = int(pred.sum() + truth.sum())
    if pred.any() and truth.any():
        hd95 = float(metric.binary.hd95(pred, truth))
        asd = float(metric.binary.asd(pred, truth))
    elif not pred.any() and not truth.any():
        hd95 = asd = 0.0
    else:
        hd95 = asd = float(np.linalg.norm(pred.shape))
    return {
        "dice": 0.0 if denominator == 0 else 2.0 * intersection / denominator,
        "iou": 0.0 if union == 0 else intersection / union,
        "hd95": hd95,
        "asd": asd,
    }


def _surface(mask: np.ndarray) -> np.ndarray:
    mask = mask.astype(bool)
    return mask ^ binary_erosion(mask) if mask.any() else mask


def _binary_mm(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, ...]) -> dict[str, float]:
    pred, truth = pred.astype(bool), truth.astype(bool)
    intersection = int((pred & truth).sum())
    union = int((pred | truth).sum())
    denominator = int(pred.sum() + truth.sum())
    pred_surface, truth_surface = _surface(pred), _surface(truth)
    if not pred_surface.any() and not truth_surface.any():
        distances = np.asarray([0.0])
    elif not pred_surface.any() or not truth_surface.any():
        distances = np.asarray([np.inf])
    else:
        dt_truth = distance_transform_edt(~truth_surface, sampling=spacing)
        dt_pred = distance_transform_edt(~pred_surface, sampling=spacing)
        distances = np.concatenate([dt_truth[pred_surface], dt_pred[truth_surface]])
    return {
        "dice": 1.0 if denominator == 0 else 2.0 * intersection / denominator,
        "iou": 1.0 if union == 0 else intersection / union,
        "hd95": float(np.percentile(distances, 95)),
        "asd": float(np.mean(distances)),
    }


@torch.inference_mode()
def predict_volume(model: torch.nn.Module, image: np.ndarray, config: dict[str, Any], device: torch.device) -> np.ndarray:
    target_size = tuple(int(value) for value in config["inference_size"])
    prediction = np.zeros(image.shape, dtype=np.uint8)
    model.eval()
    for index, source in enumerate(image):
        original_shape = source.shape
        if tuple(original_shape) != target_size:
            source = zoom(source, (target_size[0] / original_shape[0], target_size[1] / original_shape[1]), order=0)
        tensor = torch.from_numpy(source).unsqueeze(0).unsqueeze(0).float().to(device)
        if config["method"].lower() == "hien":
            tensor = tensor.repeat(1, 3, 1, 1)
            logits = model(tensor)[0]
        else:
            logits = model(tensor)[0]
        pred = torch.softmax(logits, dim=1).argmax(dim=1).squeeze(0).cpu().numpy().astype(np.uint8)
        if tuple(original_shape) != target_size:
            pred = zoom(pred, (original_shape[0] / pred.shape[0], original_shape[1] / pred.shape[1]), order=0)
        prediction[index] = pred
    return prediction


def evaluate(model: torch.nn.Module, config: dict[str, Any], split: str, device: torch.device,
             prediction_sink: Callable[[str, np.ndarray], None] | None = None,
             max_cases: int | None = None) -> dict[str, Any]:
    dataset = config["dataset"]
    rows: list[dict[str, Any]] = []
    for volume_index, volume in enumerate(iter_volumes(config, split)):
        if max_cases is not None and volume_index >= max_cases:
            break
        pred = predict_volume(model, volume["image"], config, device)
        truth = volume["label"]
        if prediction_sink:
            prediction_sink(volume["case_id"], pred)
        if dataset == "ACDC":
            for class_id, class_name in ((1, "RV"), (2, "MYO"), (3, "LV")):
                dice, iou, hd95, asd = _acdc_safe(pred == class_id, truth == class_id)
                rows.append({"case": volume["case_id"], "class": class_name, "dice": dice, "iou": iou, "hd95": hd95, "asd": asd})
        elif dataset == "COVID19":
            metric_row = _binary_mm(np.moveaxis(pred > 0, 0, -1), np.moveaxis(truth > 0, 0, -1), tuple(volume["spacing"]))
            rows.append({"case": volume["case_id"], "class": "foreground", **metric_row})
        else:
            rows.append({"case": volume["case_id"], "class": "foreground", **_binary_pixel(pred > 0, truth > 0)})
    keys = ("dice", "iou", "hd95", "asd")
    macro = {key: float(np.mean([float(row[key]) for row in rows])) for key in keys}
    return {"split": split, "dataset": dataset, "aggregation": "case_class_macro" if dataset == "ACDC" else "case_macro", "macro": macro, "rows": rows}
