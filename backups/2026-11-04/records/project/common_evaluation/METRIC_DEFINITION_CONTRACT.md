# METRIC_DEFINITION_CONTRACT

protocol: V17.3-E2E-R8
schema_version: V17_3_METRIC_DEFINITION_CONTRACT_V1

## Required Metrics

All formal external methods, auxiliary controls, internal baselines, ablations, and final combinations must report:

- Dice (%) higher is better
- IoU (%) higher is better
- HD95 (mm) lower is better
- ASD (mm) lower is better

Dice and IoU must be computed directly from the same frozen discrete prediction and GT. IoU must not be derived from an averaged Dice value.

## Region Metrics

For each case and foreground class:

- Dice = `2TP / (2TP + FP + FN)`
- IoU = `TP / (TP + FP + FN)`

No smoothing epsilon is used for formal values.

Empty-set policy:

- GT empty and prediction empty: Dice = 1, IoU = 1
- GT nonempty and prediction empty: Dice = 0, IoU = 0
- GT empty and prediction nonempty: Dice = 0, IoU = 0

Aggregation:

- compute per-case, per-class values first
- per-class dataset metric is arithmetic mean over cases
- foreground macro is equal mean over foreground classes
- background is excluded from foreground macro
- global pixel micro statistics may be saved for audit, but are not the default main-table metric

## Boundary Metrics

Reference implementation: `common_evaluation/boundary_metrics.py`

Boundary metrics are computed after 2D-to-3D reconstruction and restoration to the frozen case geometry.

- connectivity: 3D 6-connectivity
- surface: mask XOR binary erosion with the 3D 6-connected structure
- distance: Euclidean nearest-surface distance in physical millimeters using restored spacing
- HD95: 95th percentile of the concatenated symmetric directed distance vector
- ASD: mean of the same concatenated symmetric directed distance vector

Boundary empty-set policy:

- GT empty and prediction empty: HD95/ASD = NaN, excluded from boundary mean/CI, report `excluded_both_empty_count`
- exactly one empty: HD95/ASD = physical evaluation-domain diagonal in mm, included in mean/CI, report `one_empty_penalty_count`

## Statistical Boundary

`TRAINING_SEED_COUNT = 1`

`CASE_BOOTSTRAP_DOES_NOT_ESTIMATE_TRAINING_SEED_VARIANCE`

Case-level bootstrap CIs, when generated, describe case sampling uncertainty only and must not be described as multi-seed training stability.
