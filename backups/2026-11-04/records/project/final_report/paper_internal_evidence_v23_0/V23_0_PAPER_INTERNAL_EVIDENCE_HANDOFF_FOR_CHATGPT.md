# V23.0 Paper Internal Evidence Handoff for ChatGPT

Generated: 2026-08-24T12:25:42.145532+00:00

## SECTION A — Scope Changes

- Main paper datasets: ACDC, PROMISE12 20%, COVID-19.
- PROMISE12 10% and MS-CMRSeg are excluded from the current paper scope.
- No external-method reproduction was performed in V23.
- ACDC rows Cross Teaching, SCP-Net, PLGCL, β-FFT reported, and Mask Token + AdaMix are not displayed in current rebuilt tables.

## SECTION B — PROMISE10 Removal State

Source: `audits/paper_internal_evidence_v23_0/PROMISE10_SCOPE_REMOVAL_RECEIPT.json`.

```json
{
  "AuditID": "PROMISE10_SCOPE_REMOVAL_RECEIPT",
  "CreatedLocalTime": "2026-08-23 11:12:57 CST",
  "ControllerSession": "promise10_paper4_formal_controller",
  "ObservedControllerState": "QUEUE_COMPLETE",
  "ObservedControllerStatePath": "runtime/promise10_paper4/controller_state.json",
  "ObservedLedgerPath": "runtime/promise10_paper4/controller_ledger.tsv",
  "PROMISE10_ACTIVE_PROCESS_COUNT": 0,
  "PROMISE10_CURRENT_PAPER_IN_SCOPE": false,
  "EXCLUDED_BY_USER_FROM_CURRENT_PAPER": true,
  "PAPER_ELIGIBLE": false,
  "CURRENT_PAPER_TABLE_ELIGIBLE": false,
  "DO_NOT_CONTINUE_PROMISE10": true,
  "PROMISE10_ASSET_ACTION": "RETAIN_HISTORICAL_ASSETS_EXCLUDE_FROM_CURRENT_PAPER",
  "PROMISE10_BETAFFT_STATUS": "COMPLETED_HISTORICAL_EXCLUDED",
  "PROMISE10_OURS_STATUS": "COMPLETED_HISTORICAL_EXCLUDED",
  "SignalSent": "NONE",
  "ReasonSignalNotSent": "controller already naturally completed; no active PROMISE10 process was observed",
  "CurrentPaperMainDatasets": [
    "ACDC",
    "PROMISE12_20P",
    "COVID19"
  ],
  "SafetyLocks": {
    "PROMISE10_IN_CURRENT_PAPER": false,
    "MSCMR_IN_CURRENT_PAPER": false,
    "EXTERNAL_METHOD_REPRODUCTION": false,
    "FAILURE_CASE_SECTION": false
  }
}

```

## SECTION C — Three-Dataset Main Table Structure

- Table 1: ACDC 5% / 10%.
- Table 2: PROMISE12 20%.
- Table 3: COVID-19 5% / 10%.
- Table 4: ACDC10 complete 8-arm ablation.

## SECTION D — Table 1 ACDC

```tsv
Dataset	Ratio	Method	Dice	IoU	HD95	ASD	SourceAsset
ACDC	5%	β-FFT	0.8854931093772528	0.8007279410036044	1.4645282865766618	0.39901075666660246	final_report/ACDC_5P_10P_TEST_RESULTS.tsv
ACDC	5%	Ours	0.8876738495748693	0.8050873304905563	2.299675543911377	0.7381392779338375	final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/TEST_RESULT_FULL_PRECISION.json
ACDC	10%	β-FFT	0.8966063409597648	0.8182007425305878	1.7618789072469767	0.5722597264440701	final_report/ACDC_5P_10P_TEST_RESULTS.tsv
ACDC	10%	Ours	0.9034965765204998	0.8293272953910553	1.4233666940334322	0.4290264993301964	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv

```

## SECTION E — Table 2 PROMISE20

```tsv
Dataset	Ratio	Method	Dice	IoU	HD95	ASD	SourceAsset
PROMISE12	20%	β-FFT	0.8110	0.6844	3.67	1.53	final_report/PROMISE20_FINAL_TEST_RESULTS.tsv
PROMISE12	20%	Ours	0.8386401640018327	0.7237411265148253	2.9863295043764113	1.4106984188286347	final_report/three_innovation_postfull_test_v22_1/raw_test/PROMISE20_FINAL3/TEST_RESULT_FULL_PRECISION.json

```

## SECTION F — Table 3 COVID

```tsv
Dataset	Ratio	Method	Dice	IoU	HD95	ASD	SourceAsset
COVID-19	5%	β-FFT	0.6367743321888475	0.478921765140088	23.51210872051254	4.262073756032654	final_report/COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv
COVID-19	5%	Ours	0.6731436651428727	0.514497054730846	22.791388468075546	4.058622711161191	final_report/three_innovation_postfull_test_v22_1/raw_test/COVID5_FINAL3/TEST_RESULT_FULL_PRECISION.json
COVID-19	10%	β-FFT	0.6550316795817169	0.4964359644862783	13.891287168761059	3.1791635442421144	final_report/COVID19_SEED1337_CONFIRMATORY_FORMAL_TEST.tsv
COVID-19	10%	Ours	0.6794082017690449	0.5237523299362863	11.877109828896515	3.0315494897283286	final_report/three_innovation_postfull_test_v22_1/raw_test/COVID10_FINAL3/TEST_RESULT_FULL_PRECISION.json

```

## SECTION G — Table 4 ACDC 8-arm

```tsv
Mask Token	AdaMix	Contour Reliability	Method	Dice	IoU	HD95	ASD	SourceAsset
			000	0.8966063409597648	0.8182007425305878	1.7618789072469767	0.5722597264440701	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv
✓			100	0.8993844243	0.8225724645	1.4887897770	0.4551023015	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv
	✓		010	0.8944393381	0.8153195059	3.6404427623	0.8369085705	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv
		✓	001	0.8983787034	0.8206322894	2.5549168287	0.6468414189	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv
✓	✓		110	0.8989462872744961	0.8218097590865078	3.4387133736215558	0.7754972915391061	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv
✓		✓	101	0.9015156482	0.8253107331	1.4195291409	0.3939678897	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv
	✓	✓	011	0.9008936846	0.8249883415	3.5541396802	0.9935135085	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv
✓	✓	✓	111	0.9034965765204998	0.8293272953910553	1.4233666940334322	0.4290264993301964	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv

```

## SECTION H — ACDC Class-wise

```tsv
Arm	Mask Token	AdaMix	Contour Reliability	RV_Dice	MYO_Dice	LV_Dice	Mean_Dice	MeanDiceFromClassSimpleAverage	MeanAggregationMatchesSimpleClassAverage	ResultSource
000				0.8937104043708004	0.8752621715821178	0.920846446926376	0.8966063409597648	0.8966063409597647	True	EXACT_TEST_REUSE
100	✓			0.8894626372	0.8814520686	0.9272385672	0.8993844243	0.8993844243333333	True	V22_1_NEW_TEST
010		✓		0.8760605533	0.8763665154	0.9308909454	0.8944393381	0.8944393380333332	True	V22_1_NEW_TEST
001			✓	0.8870247351	0.8784226782	0.9296886968	0.8983787034	0.8983787033666667	True	V22_1_NEW_TEST
110	✓	✓		0.8830198634965505	0.8804392919767	0.9333797063502374	0.8989462872744961	0.8989462872744959	True	EXACT_TEST_REUSE
101	✓		✓	0.8911351221	0.881619302	0.9317925203	0.9015156482	0.9015156481333334	True	V22_1_NEW_TEST
011		✓	✓	0.8891950402	0.8779159194	0.9355700942	0.9008936846	0.9008936846000001	True	V22_1_NEW_TEST
111	✓	✓	✓	0.8903348224029954	0.8829581930922916	0.9371967140662125	0.9034965765204998	0.9034965765204999	True	EXACT_TEST_REUSE

```

Full metric asset: `final_report/paper_internal_evidence_v23_0/ACDC10_8ARM_CLASSWISE_ALL_METRICS.tsv`.
Per-class IoU/HD95/ASD are now available for all eight arms. Arm 000 was recovered by an exact TEST-only replay of the frozen β-FFT checkpoint solely for class-wise metric completion; its macro Dice/IoU/HD95/ASD matches the existing ACDC10 β-FFT table values, and no checkpoint reselection or training occurred.

## SECTION I — X03 Sensitivity

Sensitivity uses ACDC10 VAL only, with TEST access fixed at 0. Final paper X03 weight remains frozen at 0.05; weight reselection is not allowed.

```tsv
Weight	TaskID	Action	BestValDice	BestValIter	FinalValDice	FinalIter	BestMinusFinalDice	TestAccess	SourceLog
0.00	ACDC10_ARM_110	EXACT_REUSE_ARM_110	0.9016550000000000	25200	0.8938050000000000	30000	0.0078500000000000	0	runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/train.log
0.01	ACDC10_X03_WEIGHT_001	NEW_FULL	0.9003280000000000	28800	0.8947230000000000	30000	0.0056050000000000	0	experiments/paper_internal_evidence_v23_0/ACDC10_X03_WEIGHT_001/train.log
0.03	ACDC10_X03_WEIGHT_003	NEW_FULL	0.9004880000000000	23600	0.8943190000000000	30000	0.0061690000000000	0	experiments/paper_internal_evidence_v23_0/ACDC10_X03_WEIGHT_003/train.log
0.05	ACDC10_ARM_111	EXACT_REUSE_ARM_111	0.8999080000000000	29600	0.8969050000000000	30000	0.0030030000000001	0	runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/train.log
0.07	ACDC10_X03_WEIGHT_007	NEW_FULL	0.9008870000000000	27800	0.8965320000000000	30000	0.0043550000000000	0	experiments/paper_internal_evidence_v23_0/ACDC10_X03_WEIGHT_007/train.log
0.10	ACDC10_X03_WEIGHT_010	NEW_FULL	0.9018230000000000	27400	0.8953660000000000	30000	0.0064570000000000	0	experiments/paper_internal_evidence_v23_0/ACDC10_X03_WEIGHT_010/train.log

```

Figure files:

- `figures/X03_WEIGHT_SENSITIVITY_DICE.png`
- `figures/X03_WEIGHT_SENSITIVITY_DICE.pdf`

## SECTION J — Training Dynamics

Raw source-data: `final_report/paper_internal_evidence_v23_0/TRAINING_DYNAMICS_RAW.tsv`.
Summary: `final_report/paper_internal_evidence_v23_0/ACDC10_TRAINING_DYNAMICS_SUMMARY.tsv`.

```tsv
MethodID	DisplayName	SourceIdentity	SourceLog	ValidationPointCount	FirstIteration	FinalIteration	ResumeFinalIteration	BestValDice	BestIter	FinalValDice	BestMinusFinalDice	BestCheckpointUpdateCount	BestCheckpointUpdatesAgree	ConsumedTestPointCount	HistoricalTestAliasPolicy
ACDC10_ARM_000	beta-FFT	ACDC10 beta-FFT canonical full run, Mask Token OFF, AdaMix OFF, X03 OFF	runs/beta_fft_acdc_official_full_seed42_fresh_ckpt/train.log	150	200	30000	30000	0.900923	21200	0.890039	0.010884000000000005	15	True	0	QUARANTINED_NOT_CONSUMED
ACDC10_ARM_110	Mask Token + AdaMix	ACDC10 Mask Token + AdaMix canonical full run, X03 OFF	runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/train.log	150	200	30000	30000	0.901655	25200	0.893805	0.007850000000000024	23	True	0	QUARANTINED_NOT_CONSUMED
ACDC10_ARM_111	Ours	ACDC10 Mask Token + AdaMix + X03 canonical full run	runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/train.log	150	200	30000	30000	0.899908	29600	0.896905	0.003003000000000089	21	True	0	QUARANTINED_NOT_CONSUMED

```

Figure files:

- `figures/TRAINING_DYNAMICS_BETAFFT_VS_OURS.png/.pdf`
- `figures/TRAINING_DYNAMICS_000_110_111.png/.pdf`

## SECTION K — Efficiency

Inference protocol: scratch benchmark with warmup=50, measured iterations=200, repeats=3. Training-step protocol: real ACDC10 TRAIN loader/batch sampler with full BCP mix + CPS + FFT training step, plus Mask Token/AdaMix/X03 enabled for Ours, warmup=20, measured steps=100, repeats=3. No checkpoint was written and TEST access remained 0.

```tsv
MethodID	DisplayName	MaskToken	AdaMix	X03ContourReliabilityWeight	ParameterCount	TrainableParameterCount	FLOPs	BatchSize	InputShape	Precision	Device	CheckpointWritten	TestAccess	MeasuredIterations	InferenceWarmup	InferenceMeasuredIterations	InferenceRepeatCount	LatencyMsMean	LatencyMsMedian	LatencyMsStd	FPSMean	FPSMedian	FPSStd	PeakInferenceGpuMemoryMBMean	TrainingStepLatency	PeakTrainingGpuMemory	TrainingStepBenchmarkType	TrainingStepLatencyMsMean	TrainingStepLatencyMsMedian	TrainingStepLatencyMsStd	PeakTrainingGpuMemoryMBMean	TrainingWarmupSteps	TrainingMeasuredSteps	TrainingRepeatCount	TrainingBenchmarkLimitation
ACDC10_ARM_000	β-FFT	OFF	OFF	OFF	1826828	1826828	NOT_MEASURED_NO_THOP_DEPENDENCY	1	1x1x256x256	fp32	cuda:0	False	0	200	50	200	3	1.613538	1.603713	0.019959	619.819039	623.552968	7.614409	45.062988	NOT_RECORDED_NO_FORMAL_TRAIN_STEP_BENCHMARK	NOT_RECORDED_NO_FORMAL_TRAIN_STEP_BENCHMARK	SCRATCH_REAL_ACDC10_FULL_BCP_FFT_TRAIN_STEP	583.174209	582.709590	18.910608	13322.333008	20	100	3	none_full_BCP_FFT_scratch_step
ACDC10_ARM_111	Ours	ON	ON	ON	1827469	1827469	NOT_MEASURED_NO_THOP_DEPENDENCY	1	1x1x256x256	fp32	cuda:0	False	0	200	50	200	3	2.935293	2.912218	0.062871	340.784631	343.380890	7.226044	54.425293	NOT_RECORDED_NO_FORMAL_TRAIN_STEP_BENCHMARK	NOT_RECORDED_NO_FORMAL_TRAIN_STEP_BENCHMARK	SCRATCH_REAL_ACDC10_FULL_BCP_FFT_TRAIN_STEP	709.435060	722.698841	31.134860	15776.607422	20	100	3	none_full_BCP_FFT_scratch_step

```

Raw full training-step benchmark: `final_report/paper_internal_evidence_v23_0/EFFICIENCY_FULL_TRAINING_STEP_BENCHMARK_RAW.tsv`.

Table files:

- `tables/TABLE6_EFFICIENCY.md`
- `tables/TABLE6_EFFICIENCY.tex`

FLOPs are `NOT_MEASURED_NO_THOP_DEPENDENCY`; no high-risk dependency installation was performed.

## SECTION L — X03 Mechanism Visualization

Fixed validation cases are frozen by deterministic rule in `ACDC10_VAL_MECHANISM_VIS_CASE_SELECTION.tsv`.

Figure files:

- `figures/X03_MECHANISM_FIXED_CASES.png`
- `figures/X03_MECHANISM_FIXED_CASES.pdf`
- `figures/X03_MECHANISM_ALL_VAL_GALLERY.pdf`

All-VAL gallery selection: `ACDC10_VAL_MECHANISM_ALL_GALLERY_SELECTION.tsv`, 20 VAL volumes, max-GT-foreground slice per volume, TEST access 0.

Descriptive statistics:

```tsv
Metric	Status	Reason
X03_MECHANISM_DESCRIPTIVE_STATISTICS	NOT_RECORDED	Current frozen visualization assets do not expose a preregistered numeric reliability statistic beyond fixed-case maps.

```

## SECTION M — Source / Config / Checkpoint Identities

- Formal manifest: `audits/paper_internal_evidence_v23_0/V23_0_FORMAL_START_MANIFEST.json`.
- Final registry: `audits/paper_internal_evidence_v23_0/V23_0_MASTER_TASK_REGISTRY_FINAL.tsv`.
- Controller SHA remained matched after formal start.
- Post-start lock: source/config/launcher/controller edits disabled.

## SECTION N — TEST Exposure Disclosure

- Sensitivity TEST access: 0 for all six points.
- Training dynamics: VAL-only gate PASS, TEST point count 0.
- Mechanism visualization: ACDC10 VAL fixed cases.
- Class-wise table reads existing ACDC10 TEST evidence and does not select or tune checkpoints.

## SECTION O — Blocked / Missing Evidence

- External published rows for Table 1-3 are excluded/deferred by `audits/paper_internal_evidence_v23_0/V23_0_EXTERNAL_PUBLISHED_ROWS_ELIGIBILITY_AUDIT.json`: β-FFT reported conflicts with V23 display rules; HCCP/ACDC is reference-only with endpoint not explicitly stated and boundary unit mismatch; COVID HCCP protocol/pool identity remains unrecovered.
- Class-wise IoU/HD95/ASD are not invented when unavailable from per-class source assets.
- Mechanism descriptive statistics are `NOT_RECORDED` because no preregistered numeric reliability statistic asset is exposed.

## SECTION P — Paper-Supported Conclusions

- ACDC10 8-arm table supports that the three-component Ours arm has the highest full-precision mean Dice/IoU among displayed arms.
- X03 sensitivity shows the frozen default 0.05 is not used for post-hoc reselection; 0.10 has higher VAL Dice in this sensitivity run, but the final paper method remains frozen at 0.05.
- Efficiency benchmark shows Ours adds small parameter overhead but increases measured inference latency in the current benchmark.

## SECTION Q — Claims Not Supported

- Do not claim X03 weight 0.05 is globally optimal.
- Do not claim causal proof from mechanism visualization.
- Do not claim external SOTA reproduction was performed in V23.
- Do not use PROMISE10 or MS-CMRSeg in current main paper tables.

## SECTION R — Questions for USER + ChatGPT

- Whether USER + ChatGPT want a separate reference-only external table remains a paper-format decision; current V23 main tables intentionally include only internally verified β-FFT/Ours rows.

## SECTION S — V23 Closure Addendum

- Objective filename aliases created without content mutation: `audits/paper_internal_evidence_v23_0/V23_0_MASTER_TASK_REGISTRY.tsv` and `audits/paper_internal_evidence_v23_0/MECHANISM_VIS_CASE_SELECTION.json`.
- Per-task readiness receipts: 13/13 generated as `audits/paper_internal_evidence_v23_0/task_<TaskID>_FINAL_LAUNCH_READINESS.json`.
- Terminal registry: `audits/paper_internal_evidence_v23_0/V23_0_MASTER_TASK_REGISTRY_TERMINAL.tsv`.
- External published row eligibility audit: `audits/paper_internal_evidence_v23_0/V23_0_EXTERNAL_PUBLISHED_ROWS_ELIGIBILITY_AUDIT.json`.
- No source/config/launcher/controller edits were made by closure addendum generation.
