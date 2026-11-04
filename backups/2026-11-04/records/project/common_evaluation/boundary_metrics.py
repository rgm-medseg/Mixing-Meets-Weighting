"""Frozen V17.3 reference metrics for discrete 3D segmentation masks."""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree


CONNECTIVITY_3D_6 = ndimage.generate_binary_structure(3, 1)


@dataclass(frozen=True)
class RegionMetrics:
    dice: float
    iou: float
    tp: int
    fp: int
    fn: int


@dataclass(frozen=True)
class BoundaryMetrics:
    hd95_mm: float
    asd_mm: float
    both_empty_excluded: bool
    one_empty_penalty: bool
    penalty_mm: float | None


def _as_bool(mask: np.ndarray) -> np.ndarray:
    array = np.asarray(mask)
    if array.ndim != 3:
        raise ValueError(f"mask must be 3D after reconstruction, got shape={array.shape}")
    return array.astype(bool, copy=False)


def _spacing(spacing: tuple[float, float, float] | list[float] | np.ndarray) -> np.ndarray:
    values = np.asarray(spacing, dtype=np.float64)
    if values.shape != (3,) or np.any(values <= 0):
        raise ValueError(f"spacing must contain three positive mm values, got {spacing!r}")
    return values


def region_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> RegionMetrics:
    pred = _as_bool(prediction)
    gt = _as_bool(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, ground_truth={gt.shape}")

    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    denominator = 2 * tp + fp + fn
    union = tp + fp + fn

    if union == 0:
        return RegionMetrics(dice=1.0, iou=1.0, tp=tp, fp=fp, fn=fn)
    dice = 0.0 if denominator == 0 else (2.0 * tp) / float(denominator)
    iou = float(tp) / float(union)
    return RegionMetrics(dice=dice, iou=iou, tp=tp, fp=fp, fn=fn)


def surface(mask: np.ndarray) -> np.ndarray:
    values = _as_bool(mask)
    if not values.any():
        return np.zeros_like(values, dtype=bool)
    eroded = ndimage.binary_erosion(values, structure=CONNECTIVITY_3D_6, border_value=0)
    return np.logical_xor(values, eroded)


def _surface_points_mm(mask: np.ndarray, spacing: np.ndarray) -> np.ndarray:
    points = np.argwhere(surface(mask))
    return points.astype(np.float64) * spacing


def _domain_diagonal_mm(shape: tuple[int, int, int], spacing: np.ndarray) -> float:
    extents = np.maximum(np.asarray(shape, dtype=np.float64) - 1.0, 0.0) * spacing
    return float(np.linalg.norm(extents))


def boundary_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing_mm: tuple[float, float, float] | list[float] | np.ndarray,
) -> BoundaryMetrics:
    pred = _as_bool(prediction)
    gt = _as_bool(ground_truth)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: prediction={pred.shape}, ground_truth={gt.shape}")
    spacing = _spacing(spacing_mm)

    pred_empty = not bool(pred.any())
    gt_empty = not bool(gt.any())
    if pred_empty and gt_empty:
        return BoundaryMetrics(
            hd95_mm=math.nan,
            asd_mm=math.nan,
            both_empty_excluded=True,
            one_empty_penalty=False,
            penalty_mm=None,
        )
    if pred_empty != gt_empty:
        penalty = _domain_diagonal_mm(pred.shape, spacing)
        return BoundaryMetrics(
            hd95_mm=penalty,
            asd_mm=penalty,
            both_empty_excluded=False,
            one_empty_penalty=True,
            penalty_mm=penalty,
        )

    pred_points = _surface_points_mm(pred, spacing)
    gt_points = _surface_points_mm(gt, spacing)
    pred_to_gt = cKDTree(gt_points).query(pred_points, k=1)[0]
    gt_to_pred = cKDTree(pred_points).query(gt_points, k=1)[0]
    distances = np.concatenate([pred_to_gt, gt_to_pred]).astype(np.float64)
    return BoundaryMetrics(
        hd95_mm=float(np.percentile(distances, 95)),
        asd_mm=float(np.mean(distances)),
        both_empty_excluded=False,
        one_empty_penalty=False,
        penalty_mm=None,
    )
