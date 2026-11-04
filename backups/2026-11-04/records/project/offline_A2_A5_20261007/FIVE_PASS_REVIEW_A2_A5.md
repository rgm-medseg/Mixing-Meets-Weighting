|round|status|evidence|result|
|---|---|---|---|
|1|PASS|manifests/source_manifest.csv; reports/LOAD_*; SOURCE_UNCHANGED.json|8 real EMA teachers and4 deployed student1 strict loads; no training; old queue paused|
|2|PASS|A2_NUMERIC_REVIEW.json; A2_fixed_patient_recomputation.csv; A2_legacy_replay_comparison.csv|42 predefined slices independently recomputed; sample/batch fields distinct; real GT only; patient equal and zero support explicit|
|3|PARTIAL|A5_OPERATOR_COVERAGE.md; A5_modules_*; A5_operators_*; A5_inference_raw.csv; A5_training_cost.csv|parameters/MACs/timing/timing-boundary checks PASS; complete FLOPs and old complete training costs NA|
|4|PASS|VISUAL_REVIEW.json; figures/A2_*; A2_paper_summary.csv; A5_paper_summary.csv|CSV-derived values and units checked; all5 scientific figure PNGs visually inspected; titles/axes/legends readable, no clipping; SVG/PDF/PNG and captions complete|
|5|PASS|RECOVERY_DUPLICATE_UNIT_RECEIPT.json; SOURCE_UNCHANGED.json; VOLUME_REPEAT_REPRODUCIBILITY.json; SHA256SUMS; ARCHIVE_RECEIPT.json|469 completed files reused byte-identically on resume; duplicate start reused live PID;8 teachers,204 patient units,20 timing blocks,240 volumes; all unfavorable correlations retained; actual extracted archive SHA checked|

整体PARTIAL由真实计数/历史资料缺口导致；实际完成项如上。
