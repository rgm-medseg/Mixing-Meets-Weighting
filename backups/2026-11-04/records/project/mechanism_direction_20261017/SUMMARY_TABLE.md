# Quantitative summary: uniform → CTW

One table; panel A is the original Full and panel B is the actual four-model TRC-off factorial. All numbers are source-patient-equal means. L1 shares (%) divide by each patient's full mixed-image consistency-gradient L1 sum. `mean a` uses the supported region's pixel mean, then equal patient mean, in units of 1e-10; positive a locally favours GT cross entropy. Pixel counts are exposures. Each row has 10 supported patients. Spatial rows and teacher rows overlap; do not sum them together. U→C denotes two loss probes at fixed predictions, not before/after training.

| Panel / region or model | Pixels | Total L1 share (%) | GT-aligned L1 share (%) | GT-opposed L1 share (%) | Mean a (×10⁻¹⁰) |
| --- | ---: | ---: | ---: | ---: | ---: |
| A / Clean anatomy, all sources | 152,965 | 46.60 → 56.02 | 24.41 → 28.91 | 22.19 → 27.11 | 0.19 → 0.44 |
| A / Teacher-error source, all eligible | 17,357 | 11.08 → 12.82 | 4.71 → 5.85 | 6.37 → 6.97 | -0.94 → -1.12 |
| A / Anatomy × teacher correct | 67,298 | 16.50 → 19.35 | 9.06 → 10.33 | 7.44 → 9.02 | 0.29 → 0.56 |
| A / Anatomy × teacher error | 11,030 | 7.88 → 10.18 | 3.61 → 4.83 | 4.26 → 5.35 | -0.73 → -0.73 |
| A / Other valid × teacher correct | 2,968,728 | 9.86 → 7.07 | 6.76 → 4.79 | 3.10 → 2.29 | 0.01 → 0.01 |
| A / Other valid × teacher error | 5,089 | 2.39 → 1.73 | 0.74 → 0.63 | 1.65 → 1.10 | -3.84 → -6.50 |
| A / Seam × teacher error | 788 | 0.48 → 0.53 | 0.17 → 0.19 | 0.31 → 0.34 | -1.53 → -3.24 |
| A / Visible cutout edge × teacher error | 450 | 0.33 → 0.38 | 0.18 → 0.20 | 0.15 → 0.18 | -0.89 → -2.13 |
| B / Baseline (000) | 17,357 | 10.50 → 12.24 | 4.30 → 5.24 | 6.20 → 6.99 | -1.11 → -1.74 |
| B / TCSMix (010) | 17,357 | 9.80 → 11.64 | 4.17 → 5.29 | 5.63 → 6.34 | -0.89 → -0.69 |
| B / CTW (001) | 17,357 | 10.91 → 12.56 | 4.41 → 5.37 | 6.50 → 7.19 | -1.15 → -1.46 |
| B / TCSMix+CTW (011) | 17,357 | 10.72 → 12.08 | 4.11 → 5.14 | 6.61 → 6.93 | -2.34 → -2.90 |

The total shares in the first two rows reproduce existing observations; their directional columns and the spatial×correctness intersections are new. The baseline model was trained without CTW; panel B still applies the same diagnostic loss probe to every frozen state. Common teacher correctness comes from the archived main-Full EMA target, not each compared model’s own teacher. Visible cutout-edge support excludes occluded interiors. Neutral mass is retained in CSVs. Mean weights, raw gradients, disagreements, errors, signed magnitudes, paired differences and patient counts are in the complete CSVs.
