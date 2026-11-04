# V24.0 Internal Evidence Handoff

## A. Scope
Datasets: ACDC, PROMISE12 20%, COVID-19. No external-method reproduction. PROMISE10 and MSCMR are out of scope.

## B. Frozen Method Identity
{
  "AuditID": "V24_0_FROZEN_METHOD_IDENTITY",
  "created_at": "2026-08-24T14:18:19.752080+00:00",
  "BASELINE_DISPLAY_NAME": "β-FFT",
  "INTERMEDIATE_DISPLAY_NAME": "Mask Token + AdaMix",
  "FINAL_DISPLAY_NAME": "Ours",
  "INNOVATION_1": "Mask Token",
  "INNOVATION_2": "AdaMix",
  "INNOVATION_3": "X03 Contour Reliability Weight",
  "X03_WEIGHT": 0.05,
  "X03_WEIGHT_RESELECTION_ALLOWED": false,
  "DATASET_SPECIFIC_X03_TUNING": false,
  "TEST_DRIVEN_TUNING": false,
  "assets": {
    "sources": {
      "beta_fft_training_entry": {
        "path": "train_promise12/train_promise12_bcp_fft.py",
        "exists": true,
        "sha256": "e1006ebe9731d7665a886dc5f897f130c5818c132fa575cc16c9ea845cb27386",
        "bytes": 180273

## C. Reuse / Replay / New Full Map
TaskOrder	TaskID	TaskType	Action	Dataset	Split	Seed	ConfigPath	ConfigSHA	SourcePath	SourceSHA	Launcher	LauncherSHA	CheckpointPath	CheckpointSHA	InputAssets	OutputDir	ExpectedReceipt	GPURequired	TESTAccessAllowed	FinalLaunchReady	Blocker
1	ARCHIVE_V23_RUNTIME	SCOPE_GATE	READ_ONLY_READY													final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/V23_RUNTIME_ARCHIVE_RECEIPT.json	False	0	PASS	
2	FROZEN_METHOD_IDENTITY	IDENTITY_GATE	READ_ONLY_READY													final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/V24_0_FROZEN_METHOD_IDENTITY.json	False	0	PASS	
3	INVENTORY_ASSETS	INVENTORY	READ_ONLY_READY													final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/V24_0_REUSE_AND_GAP_MAP.tsv	False	0	PASS	
4	PATIENT_LEVEL_STATS	ANALYSIS	READ_ONLY_READY	ACDC/PROMISE20/COVID	ACDC5/ACDC10/PROMISE20/COVID5/COVID10				scripts/internal_evidence_expansion_v24_0/v24_patient_level_statistics.py	175e24ac64621008829bf9ddf14591715442f66fbc730a3a82b8cc31823f691a					formal TEST per-case/per-patient metrics; see PATIENT_LEVEL_STATS_RECEIPT.json	final_report/internal_evidence_expansion_v24_0/PATIENT_LEVEL_STATISTICAL_SUMMARY.tsv	audits/internal_evidence_expansion_v24_0/PATIENT_LEVEL_STATS_RECEIPT.json	False	0	PASS	
5	X03_RELIABILITY_FAITHFULNESS	FROZEN_REPLAY_ANALYSIS	READ_ONLY_READY	ACDC	ACDC10_VAL		audits/internal_evidence_expansion_v24_0/X03_SCORE_SEMANTIC_CONTRACT.json	7758c6ec04845e46ed692d2a438da681c1868ca804ffadf78001ec5281a8a7e6	scripts/internal_evidence_expansion_v24_0/v24_x03_reliability_faithfulness.py	1d7e016b6a0ce224fd3b43df7ed2be747c3bf8ba0b0ecc00ff7be77479ea81a7	scripts/internal_evidence_expansion_v24_0/run_x03_reliability_faithfulness.sh	0f3bb12e65c58fc79d979588ef0247f60a3cc64d084e038bd18e3a7a102c08b3	ACDC10_ARM_110/111 frozen checkpoints	see X03_RELIABILITY_FAITHFULNESS_RECEIPT.json	ACDC10 VAL only; TEST_DATA_READ=false; X03 score semantic contract; ARM110/ARM111 frozen checkpoints	final_report/internal_evidence_expansion_v24_0/X03_RELIABILITY_FAITHFULNESS_SUMMARY.json	audits/internal_evidence_expansion_v24_0/X03_RELIABILITY_FAITHFULNESS_RECEIPT.json	True	0	PASS	
6	PSEUDO_LABEL_EVOLUTION	FROZEN_REPLAY_ANALYSIS	BLOCKED	ACDC	ACDC10				scripts/internal_evidence_expansion_v24_0/v24_mechanism_asset_audit.py	c01db4d0935875eae5406c82a2d2a9c68708844df5b156e451ed6db1079c9e53						final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/PSEUDO_LABEL_EVOLUTION_TERMINAL_SOURCE_GAP_RECEIPT.json	False	0	BLOCKED_NONCRITICAL	PARTIAL_PSEUDO_LABEL_EVOLUTION_COMMON_CHECKPOINT_COUNT_LT_4_OR_PSEUDO_OUTPUTS_MISSING
7	HARD_MEDIUM_EASY	ANALYSIS	READ_ONLY_READY	ACDC	ACDC10				scripts/internal_evidence_expansion_v24_0/v24_acdc10_caselevel_analysis.py	6cf580f99d1058080671c239bf04e9ab46bd086a2c2deca2b725b698cda76b9e						final_report/internal_evidence_expansion_v24_0/ACDC10_HARD_MEDIUM_EASY.tsv	audits/internal_evidence_expansion_v24_0/ACDC10_CASELEVEL_ANALYSIS_RECEIPT.json	False	0	PASS	
8	PATHOLOGY_SUBGROUP	ANALYSIS	BLOCKED	ACDC	ACDC5/ACDC10											final_report/internal_evidence_expansion_v24_0	final_report/internal_evidence_expansion_v24_0/ACDC_PATHOLOGY_SUBGROUP.tsv	False	0	BLOCKED_NONCRITICAL	BLOCKED_OFFICIAL_PATHOLOGY_METADATA_UNAVAILABLE
9	BOUNDARY_F1	ANALYSIS	READ_ONLY_READY	ACDC	ACDC10		audits/internal_evidence_expansion_v24_0/V24_0_BOUNDARY_SPEC.json	d2d84e8cbdb57efe822f8a63d6f6fc58294c2a5793e2b5d01ebab631fab51b38	scripts/internal_evidence_expansion_v24_0/v24_acdc10_frozen_replay_analyses.py	23d8eb3a5c2ef9dd71e773b197982fd59a6f525c0be78b5948dd42c4e4c2c02b	scripts/internal_evidence_expansion_v24_0/run_acdc10_frozen_replay_analyses.sh	aa3e7af48f202099a681f47c952cd57d3aa1b82295e098457873cd465b9f3939	ACDC10_ARM_000/110/111 frozen checkpoints	see V24_0_ACDC10_FROZEN_REPLAY_ANALYSIS_DRY_RUN_RECEIPT.json	frozen checkpoints + ACDC10 TEST protocol; TEST data read only for preregistered boundary F1 evaluation; tuning/selection forbidden	final_report/internal_evidence_expansion_v24_0/ACDC10_BOUNDARY_F1_FULL_PRECISION.tsv	audits/internal_evidence_expansion_v24_0/ACDC10_FROZEN_REPLAY_ANALYSIS_FINAL_RECEIPT.json	True	PRE_REGISTERED_EVALUATION_ONLY	PASS	
10	FACTORIAL_EFFECTS	ANALYSIS	READ_ONLY_READY	ACDC	ACDC10				scripts/internal_evidence_expansion_v24_0/v24_readonly_plane_a_extractor.py	e9fd3d7149a219547a9ca1844b061311eaa85c9868cc1d915d9ccfb211c9ae64						final_report/internal_evidence_expansion_v24_0/ACDC10_FACTORIAL_EFFECTS_FULL.tsv	audits/internal_evidence_expansion_v24_0/FACTORIAL_EFFECTS_RECEIPT.json	False	0	PASS	
11	FULL_TRAINING_DYNAMICS	ANALYSIS	READ_ONLY_READY	ACDC	ACDC10				scripts/internal_evidence_expansion_v24_0/v24_readonly_plane_a_extractor.py	e9fd3d7149a219547a9ca1844b061311eaa85c9868cc1d915d9ccfb211c9ae64						final_report/internal_evidence_expansion_v24_0/ACDC10_FULL_TRAINING_DYNAMICS.tsv	audits/internal_evidence_expansion_v24_0/FULL_TRAINING_DYNAMICS_RECEIPT.json	False	0	PASS	
12	CORRUPTION_ROBUSTNESS	ANALYSIS	READ_ONLY_READY	ACDC	ACDC10	2026	audits/internal_evidence_expansion_v24_0/V24_0_CORRUPTION_SPEC.json	bcf5aacafcaa8db73f53fa4ce5ea9932becc876a08853850c245b28357d60c8c	scripts/internal_evidence_expansion_v24_0/v24_acdc10_frozen_replay_analyses.py	23d8eb3a5c2ef9dd71e773b197982fd59a6f525c0be78b5948dd42c4e4c2c02b	scripts/internal_evidence_expansion_v24_0/run_acdc10_frozen_replay_analyses.sh	aa3e7af48f202099a681f47c952cd57d3aa1b82295e098457873cd465b9f3939	ACDC10_ARM_000/111 frozen checkpoints	see V24_0_ACDC10_FROZEN_REPLAY_ANALYSIS_DRY_RUN_RECEIPT.json	frozen checkpoints + ACDC10 TEST protocol; TEST data read only for preregistered corruption evaluation; tuning/selection forbidden	final_report/internal_evidence_expansion_v24_0/ACDC10_CORRUPTION_ROBUSTNESS.tsv	audits/internal_evidence_expansion_v24_0/ACDC10_FROZEN_REPLAY_ANALYSIS_FINAL_RECEIPT.json	True	PRE_REGISTERED_EVALUATION_ONLY	PASS	
13	STUDENT_DIVERSITY	ANALYSIS	BLOCKED	ACDC	ACDC10				scripts/internal_evidence_expansion_v24_0/v24_mechanism_asset_audit.py	c01db4d0935875eae5406c82a2d2a9c68708844df5b156e451ed6db1079c9e53						final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/STUDENT_DIVERSITY_TERMINAL_SOURCE_GAP_RECEIPT.json	False	0	BLOCKED_NONCRITICAL	BLOCKED_STUDENT_DIVERSITY_COMMON_CHECKPOINT_COUNT_LT_4_OR_BRANCH_OUTPUTS_MISSING
14	RESOLVE_ACDC2_PROTOCOL	PROTOCOL_GATE	READ_ONLY_READY	ACDC	ACDC2	1337	configs/internal_evidence_expansion_v24_0/ACDC2_SPLIT.json	4aa29e06d3b5bbfcd81331252d3dd7bcd012afb09bf6a88372f659fcf43a124d								final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/ACDC2_PROTOCOL_DECISION.json	False	0	PASS	
15	ACDC2_BETAFFT_FULL	NEW_FULL	TERMINAL_PROTOCOL_BLOCKED_EXCLUDED	ACDC	ACDC2	1337	configs/internal_evidence_expansion_v24_0/ACDC2_BETAFFT_FULL.json	2a18a82c67a4f47f96c9ad9ff608b6cd4e1dbfca3d90d2126fdb34c92922bac9	train_ACDC/beta_fft_acdc_repro.py	18b128da9429211059eed2e36290964439449b52e6e7de621609170a92bed316	scripts/internal_evidence_expansion_v24_0/run_acdc2_betafft_full.sh	27959992e02c140ef71ce199971f414e3c80d36d63ce55aae9bca39669c70bf9	train_ACDC/Pre_model/label1/unet_best_model.pth			final_report/internal_evidence_expansion_v24_0	experiments/internal_evidence_expansion_v24_0/ACDC2_BETAFFT_FULL/train.log	True	0	TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE	ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT
16	ACDC2_OURS_FULL	NEW_FULL	TERMINAL_PROTOCOL_BLOCKED_EXCLUDED	ACDC	ACDC2	1337	configs/internal_evidence_expansion_v24_0/ACDC2_OURS_FULL.json	70e0684729a43d0f8c6fcd6b9158b90f744883f3fb783e9ca6d07b4e644a1d82	train_ACDC/v21_3_acdc10_third_innovation_transfer.py	dbf29b2ae055e88e802e1edd6bc559748fafa64966dcb1557ba64aa302ac5c15	scripts/internal_evidence_expansion_v24_0/run_acdc2_ours_full.sh	2edba4eb4723afba9fbe0adf94a7e257efd16d8fd2052afac78e55097f54fc62	train_ACDC/Pre_model/label1/unet_best_model.pth			final_report/internal_evidence_expansion_v24_0	experiments/internal_evidence_expansion_v24_0/ACDC2_OURS_FULL/train.log	True	0	TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE	ACDC2_PROTOCOL_NOT_EXECUTABLE_UNDER_FROZEN_BETAFFT
17	ACDC10_BETAFFT_SEED2025	NEW_FULL	NEW_FULL_READY	ACDC	ACDC10	2025	configs/internal_evidence_expansion_v24_0/ACDC10_BETAFFT_SEED2025.json	ebe3d5b2df5fb168ae7d6ff9a60d526eb951ecbe00f112b520c01ea54d4c61d9	train_ACDC/beta_fft_acdc_repro.py	18b128da9429211059eed2e36290964439449b52e6e7de621609170a92bed316	scripts/internal_evidence_expansion_v24_0/run_acdc10_betafft_seed2025.sh	b8bed12d332e4c4348af00e973c233e2e7a99dab6ead3db1e70d40883ce07568	train_ACDC/Pre_model/label7/unet_best_model.pth	477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3		final_report/internal_evidence_expansion_v24_0	experiments/internal_evidence_expansion_v24_0/ACDC10_BETAFFT_SEED2025/train.log	True	0	PASS	
18	ACDC10_OURS_SEED2025	NEW_FULL	NEW_FULL_READY	ACDC	ACDC10	2025	configs/internal_evidence_expansion_v24_0/ACDC10_OURS_SEED2025.json	76c3284f4dd28d49cb841c52a83b00fa5904750a6badaa30e69480b73f0bed9f	train_ACDC/v21_3_acdc10_third_innovation_transfer.py	dbf29b2ae055e88e802e1edd6bc559748fafa64966dcb1557ba64aa302ac5c15	scripts/internal_evidence_expansion_v24_0/run_acdc10_ours_seed2025.sh	7a83cdb7c9c64acf4de9a50dfb5763467710a2a72b90b322be322287af8b9d81	train_ACDC/Pre_model/label7/unet_best_model.pth	477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3		final_report/internal_evidence_expansion_v24_0	experiments/internal_evidence_expansion_v24_0/ACDC10_OURS_SEED2025/train.log	True	0	PASS	
19	ACDC10_BETAFFT_SEED2026	NEW_FULL	NEW_FULL_READY	ACDC	ACDC10	2026	configs/internal_evidence_expansion_v24_0/ACDC10_BETAFFT_SEED2026.json	c1672648fd04c9c08a1df4c274813b28d66ee34559f2956204d799ff001035a4	train_ACDC/beta_fft_acdc_repro.py	18b128da9429211059eed2e36290964439449b52e6e7de621609170a92bed316	scripts/internal_evidence_expansion_v24_0/run_acdc10_betafft_seed2026.sh	6407d5006a7dcae659bfe4464ab4d1b54ce4ba4c096897c7570fda91bc232485	train_ACDC/Pre_model/label7/unet_best_model.pth	477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3		final_report/internal_evidence_expansion_v24_0	experiments/internal_evidence_expansion_v24_0/ACDC10_BETAFFT_SEED2026/train.log	True	0	PASS	
20	ACDC10_OURS_SEED2026	NEW_FULL	NEW_FULL_READY	ACDC	ACDC10	2026	configs/internal_evidence_expansion_v24_0/ACDC10_OURS_SEED2026.json	a775951d335febe7e900e1a9d2396da4298d595f365c3376ed73a7bdb9a24be0	train_ACDC/v21_3_acdc10_third_innovation_transfer.py	dbf29b2ae055e88e802e1edd6bc559748fafa64966dcb1557ba64aa302ac5c15	scripts/internal_evidence_expansion_v24_0/run_acdc10_ours_seed2026.sh	9c1ba1013dd28f1e0d386f90a6ee1af6e2f6fa601f3e6ede6a6f805e23982bf4	train_ACDC/Pre_model/label7/unet_best_model.pth	477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3		final_report/internal_evidence_expansion_v24_0	experiments/internal_evidence_expansion_v24_0/ACDC10_OURS_SEED2026/train.log	True	0	PASS	
21	ACDC2_TEST_CLOSURE	POST_FULL	POST_FULL_READY	MULTI					scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py	5b589a5f772b940803eb05fe5b048f0dae5284d515c7d5eb0fdfeed631f20008	scripts/internal_evidence_expansion_v24_0/run_acdc2_test_closure.sh	074fa0b0ebb5520c3fed09825bdc078e898413d7999164ae409601bb330a42a2			Requires FormalFullComplete + BestValSelected + FrozenCheckpointSHA + TEST_COMPLETION_RECEIPT + ResultAsset for each upstream NEW_FULL; FinalLaunchReady is launch permission only	final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/ACDC2_TEST_CLOSURE.json	False	0	PASS	
22	SEED_TEST_CLOSURE	POST_FULL	POST_FULL_READY	MULTI					scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py	5b589a5f772b940803eb05fe5b048f0dae5284d515c7d5eb0fdfeed631f20008	scripts/internal_evidence_expansion_v24_0/run_seed_test_closure.sh	c97f887208ab14d505fc29dd3e608830a77a2949701fd020264bad4fb43abedc			Requires FormalFullComplete + BestValSelected + FrozenCheckpointSHA + TEST_COMPLETION_RECEIPT + ResultAsset for each upstream NEW_FULL; FinalLaunchReady is launch permission only	final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/SEED_TEST_CLOSURE.json	False	0	PASS	
23	BUILD_LABEL_EFFICIENCY	POST_FULL	POST_FULL_READY	MULTI					scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py	5b589a5f772b940803eb05fe5b048f0dae5284d515c7d5eb0fdfeed631f20008	scripts/internal_evidence_expansion_v24_0/run_build_label_efficiency.sh	723b879c3d22bda79507d9f668d388667279d70de581dc7063c30d3c6fcae8f8				final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/BUILD_LABEL_EFFICIENCY.json	False	0	PASS	
24	BUILD_SEED_STABILITY	POST_FULL	POST_FULL_READY	MULTI					scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py	5b589a5f772b940803eb05fe5b048f0dae5284d515c7d5eb0fdfeed631f20008	scripts/internal_evidence_expansion_v24_0/run_build_seed_stability.sh	f1a1358bfcfb6a7e3c8cb2503ad45774a6d0748e45eda8a6eeda2c5b298616cc				final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/BUILD_SEED_STABILITY.json	False	0	PASS	
25	REGENERATE_ALL_TABLES_FIGURES	FINALIZATION	FINALIZATION_READY	MULTI					scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py	5b589a5f772b940803eb05fe5b048f0dae5284d515c7d5eb0fdfeed631f20008	scripts/internal_evidence_expansion_v24_0/run_regenerate_all_tables_figures.sh	b21077ba7e2b51535911fd87290c17328d214928104a33dae3a5a33c8624c2dc				final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/REGENERATE_ALL_TABLES_FIGURES.json	False	0	PASS	
26	FINAL_FIVE_REVIEW	FINALIZATION	FINALIZATION_READY	MULTI					scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py	5b589a5f772b940803eb05fe5b048f0dae5284d515c7d5eb0fdfeed631f20008	scripts/internal_evidence_expansion_v24_0/run_final_five_review.sh	13ccffdb4a746a4a55e9ecba2e6402aba4bb2d6fe5b4248ee6a9252254a67b66				final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/FINAL_FIVE_REVIEW.json	False	0	PASS	
27	FINAL_HANDOFF	FINALIZATION	FINALIZATION_READY	MULTI					scripts/internal_evidence_expansion_v24_0/v24_postfull_and_finalization.py	5b589a5f772b940803eb05fe5b048f0dae5284d515c7d5eb0fdfeed631f20008	scripts/internal_evidence_expansion_v24_0/run_final_handoff.sh	c9ac9e17bf214bb228ebd845caec2f1df68cbde8d921aa86d5d1936b8b521c6f				final_report/internal_evidence_expansion_v24_0	audits/internal_evidence_expansion_v24_0/FINAL_HANDOFF.json	False	0	PASS	

## D. Paired Statistics
SettingID	Dataset	Ratio	Comparison	PairUnit	UnitCount	MeanDeltaDice	MedianDeltaDice	StdDeltaDice	ImprovedCaseCount	WorsenedCaseCount	TieCount	ImprovementRate	WilcoxonDiceRawP	WilcoxonDicePMethod	WilcoxonDiceHolmP	RankBiserialDice	BootstrapIterations	BootstrapSeed	BoundaryUnit	DiceBootstrapCI95Low	DiceBootstrapCI95High	MeanDeltaIoU	IoUBootstrapCI95Low	IoUBootstrapCI95High	MeanDeltaHD95	HD95BootstrapCI95Low	HD95BootstrapCI95High	MeanDeltaASD	ASDBootstrapCI95Low	ASDBootstrapCI95High
ACDC5	ACDC	5%	β-FFT vs Ours	patient/case/volume	40	0.002180740198	0.002669672675	0.032322841818	23	17	0	0.575000000000	0.826382224492	SCIPY_WILCOXON_TWO_SIDED_AUTO	0.826382224492	0.041463414634	10000	2026	pixel-unit	-0.007677955665	0.012256329447	0.004359389487	-0.010243727556	0.019549460876	0.835147257335	0.098629969230	1.777305193407	0.339128521267	0.149372464702	0.566108675738
ACDC10	ACDC	10%	β-FFT vs Ours	patient/case/volume	40	0.006890235561	0.002853619817	0.022815650637	24	16	0	0.600000000000	0.0747265290865	SCIPY_WILCOXON_TWO_SIDED_AUTO	0.298906116346	0.324390243902	10000	2026	pixel-unit	0.000573583243	0.014684169236	0.011126552861	0.001844945489	0.022606875294	-0.338512213214	-0.946225887994	0.093844402676	-0.143233227114	-0.333471896660	0.012315039316
PROMISE20	PROMISE12	20%	β-FFT vs Ours	patient/case/volume	10	0.027608555755	0.028307791670	0.016435924444	9	1	0	0.900000000000	0.00390625	SCIPY_WILCOXON_TWO_SIDED_AUTO	0.01953125	0.963636363636	10000	2026	pixel-unit	0.017246733466	0.036446167510	0.039382899707	0.024226502393	0.053198636179	-0.682812974332	-1.615803356271	-0.098064693202	-0.123593995197	-0.386683303165	0.168967231517
COVID5	COVID-19	5%	β-FFT vs Ours	patient/case/volume	4	0.036369332954	0.043208884812	0.037619246629	3	1	0	0.750000000000	0.25	SCIPY_WILCOXON_TWO_SIDED_AUTO	0.75	0.800000000000	10000	2026	mm	0.006230306852	0.066508359056	0.035575289591	0.004242126638	0.065846288085	-0.720720252437	-6.675785393079	6.264001493047	-0.203451044871	-0.998048469558	0.729640197851
COVID10	COVID-19	10%	β-FFT vs Ours	patient/case/volume	4	0.024376522187	0.024427782522	0.037840106262	3	1	0	0.750000000000	0.375	SCIPY_WILCOXON_TWO_SIDED_AUTO	0.75	0.600000000000	10000	2026	mm	-0.006214757128	0.054916541168	0.027316365450	-0.003937727944	0.058570458843	-2.014177339864	-9.249149872381	3.565865334017	-0.147614054514	-0.745811263546	0.443598110356

## E. X03 Reliability Faithfulness
{
  "status": "PASS",
  "created_at": "2026-08-24T17:19:30.739233+00:00",
  "task_id": "X03_RELIABILITY_FAITHFULNESS",
  "split": "VAL",
  "test_access": 0,
  "training_started": false,
  "checkpoint_selection_started": false,
  "gt_used_to_generate_score": false,
  "gt_used_only_for_retrospective_evaluation": true,
  "case_count": 20,
  "score_contract": "audits/internal_evidence_expansion_v24_0/X03_SCORE_SEMANTIC_CONTRACT.json",
  "score_contract_sha256": "7758c6ec04845e46ed692d2a438da681c1868ca804ffadf78001ec5281a8a7e6",
  "score_semantics": "higher score means stronger contour uncertainty / stronger boundary consistency weight, not lower risk reliability",
  "higher_score_means_more_reliable": false,
  "higher_score_means_less_reliable_or_more_fragile_boundary": true,
  "pixel_level_tsv": "final_report/internal_evidence_expansion_v24_0/X03_RELIABILITY_PIXEL_LEVEL.tsv",
  "pixel_level_tsv_sha256": "3c7c7e23d49c4346800086a6ac3fa2374bd0358654c1bbcaf6c32f9502e763aa",
  "case_level_tsv": "final_report/internal_evidence_expansion_v24_0/X03_RELIABILITY_CASE_LEVEL.tsv",
  "case_level_tsv_sha256": "b86ba497fc2d807553f77951196f250c610e88e3b1ec026c68b79e2aae728a5d",
  "quintile_error_tsv": "final_report/internal_evidence_expansion_v24_0/X03_RELIABILITY_QUINTILE_ERROR.tsv",
  "quintile_error_tsv_sha256": "12cfe5c74b9b232a43929c867111baa148facaa2864473e3ea6ae824c0b8058d",
  "risk_coverage_tsv": "final_report/internal_evidence_expansion_v24_0/X03_RELIABILITY_RISK_COVERAGE.tsv",
  "risk_coverage_tsv_sha256": "1987e4808742b5bc6c1225c83163ecaf933c746e07f8a85bf8b1b1c076e21ba5",
  "case_level_correlations": [
    {
      "Metric": "ARM111_Dice",
      "SpearmanRho": "-0.233082706767",
      "BootstrapCI95Low": "-0.585476550681",
      "BootstrapCI95High": "0.232876712329"

## F. Pseudo-label Evolution
BLOCKED/PARTIAL if source assets remain unavailable; see mechanism source asset audit.

## G. Difficulty Strata
DifficultyGroup	CaseCount	BetaFFTMeanDice	OursMeanDice	MeanDeltaDice	MedianDeltaDice	ImprovedCaseFraction	BetaFFTMeanHD95	OursMeanHD95	BetaFFTMeanASD	OursMeanASD	GroupingRule
Hard	13	0.847021906053	0.868787030363	0.021765124311	0.020014279674	0.769230769231	2.637024703666	2.107450321114	0.871928239957	0.629058023482	sort by beta-FFT Dice ascending then deterministic near-equal thirds
Medium	13	0.903842607157	0.903540509285	-0.000302097872	0.002104131423	0.538461538462	1.707218087863	1.184595569184	0.619085634333	0.475281482786	sort by beta-FFT Dice ascending then deterministic near-equal thirds
Easy	14	0.935929640476	0.935686074671	-0.000243565805	0.001277445465	0.500000000000	1.000000000000	1.009862227676	0.250514906570	0.200331885124	sort by beta-FFT Dice ascending then deterministic near-equal thirds

## H. Pathology Subgroup
BLOCKED if official pathology metadata remains unavailable.

## I. Boundary F1
MethodID	MethodDisplay	CaseID	ClassID	ClassName	TolerancePx	BoundaryF1	BoundaryUnit	EngineeringCanaryOnly
ACDC10_ARM_000	BetaFFT	patient011_frame01	1	RV	1	0.947468675629	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	1	RV	2	0.971993156895	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	1	RV	3	0.981950479806	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	2	MYO	1	0.983585342734	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	2	MYO	2	0.997743936830	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	2	MYO	3	0.999812382739	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	3	LV	1	0.983739410532	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	3	LV	2	0.999868369093	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame01	3	LV	3	1.000000000000	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	1	RV	1	0.964207239697	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	1	RV	2	0.977601732001	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	1	RV	3	0.983800569135	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	2	MYO	1	0.977660650083	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	2	MYO	2	0.992425676955	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	2	MYO	3	0.994520030234	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	3	LV	1	0.977713762756	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	3	LV	2	0.999706830841	pixel	False
ACDC10_ARM_000	BetaFFT	patient011_frame02	3	LV	3	1.000000000000	pixel	False
ACDC10_ARM_000	BetaFFT	patient013_frame01	1	RV	1	0.979290644601	pixel	False

## J. Factorial Effects
EffectType	Factor	Metric	GrandMean	MeanOn	MeanOff	RawEffectOnMinusOff	ImprovementOrientedEffect	Unit	Formula	DESCRIPTIVE_FACTORIAL_ONLY	SourceAsset
MAIN	Mask Token	Dice	0.899207625419	0.900835734074	0.897579516765	0.003256217309	0.003256217309	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	AdaMix	Dice	0.899207625419	0.899443971624	0.898971279215	0.000472692409	0.000472692409	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	X03 Contour Reliability Weight	Dice	0.899207625419	0.901071153180	0.897344097659	0.003727055522	0.003727055522	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_AdaMix	Dice	0.899207625419	0.899356977039	0.899058273800	0.000298703239	0.000298703239	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_X03	Dice	0.899207625419	0.899014475945	0.899400774894	-0.000386298949	-0.000386298949	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	AdaMix_x_X03	Dice	0.899207625419	0.900095256595	0.898319994244	0.001775262351	0.001775262351	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_AdaMix_x_X03	Dice	0.899207625419	0.898924760580	0.899490490259	-0.000565729678	-0.000565729678	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	Mask Token	IoU	0.822270141426	0.824755063019	0.819785219833	0.004969843187	0.004969843187	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	AdaMix	IoU	0.822270141426	0.822861225469	0.821679057383	0.001182168087	0.001182168087	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	X03 Contour Reliability Weight	IoU	0.822270141426	0.825064664848	0.819475618004	0.005589046843	0.005589046843	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_AdaMix	IoU	0.822270141426	0.822492521602	0.822047761250	0.000444760352	0.000444760352	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_X03	IoU	0.822270141426	0.822039569230	0.822500713622	-0.000461144391	-0.000461144391	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	AdaMix_x_X03	IoU	0.822270141426	0.823772210980	0.820768071872	0.003004139109	0.003004139109	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_AdaMix_x_X03	IoU	0.822270141426	0.821962888798	0.822577394054	-0.000614505257	-0.000614505257	fraction	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	Mask Token	HD95	2.410222145500	1.942599746389	2.877844544612	-0.935244798223	0.935244798223	pixel	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	AdaMix	HD95	2.410222145500	3.014165627539	1.806278663462	1.207886964077	-1.207886964077	pixel	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
MAIN	X03 Contour Reliability Weight	HD95	2.410222145500	2.237988085958	2.582456205042	-0.344468119084	0.344468119084	pixel	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_AdaMix	HD95	2.410222145500	2.294718950900	2.525725340100	-0.231006389200	0.231006389200	pixel	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv
INTERACTION	MaskToken_x_X03	HD95	2.410222145500	2.061304376120	2.759139914880	-0.697835538760	0.697835538760	pixel	For contrast sign s in {-1,+1}, RawEffect = mean(y | s=+1) - mean(y | s=-1). Main-effect signs use the target factor; interaction signs use the product of coded factors with OFF=-1 and ON=+1. ImprovementEffect equals RawEffect for Dice/IoU and -RawEffect for HD95/ASD.	true	final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv

## K. Full Training Dynamics
MethodID	DisplayName	PointCount	MinIter	MaxIter	BestIter	BestSelectedValDice	FinalIter	FinalSelectedValDice	BestMinusFinalGap	LateStageMeanValDice	LateStageStdValDice	LateStageSlopePerIteration	MaximumDrawdown	MeanSelectedValDice	StdSelectedValDice	EndpointIdentity	ConsumedTestPointSum	SourceAsset
ACDC10_ARM_000	beta-FFT	150	200	30000	21200	0.900923000000	30000	0.890039000000	0.010884000000	0.893035064516	0.004121390221	0.000000166728	0.023563000000	0.889100180000	0.009284055949	VALIDATION_LOG_POINT	0	final_report/paper_internal_evidence_v23_0/TRAINING_DYNAMICS_RAW.tsv
ACDC10_ARM_110	Mask Token + AdaMix	150	200	30000	25200	0.901655000000	30000	0.893805000000	0.007850000000	0.895319387097	0.003219118860	0.000000458556	0.017760000000	0.887300300000	0.012770770068	VALIDATION_LOG_POINT	0	final_report/paper_internal_evidence_v23_0/TRAINING_DYNAMICS_RAW.tsv
ACDC10_ARM_111	Ours	150	200	30000	29600	0.899908000000	30000	0.896905000000	0.003003000000	0.895876612903	0.002926638639	0.000000542254	0.021667000000	0.887219353333	0.013653440095	VALIDATION_LOG_POINT	0	final_report/paper_internal_evidence_v23_0/TRAINING_DYNAMICS_RAW.tsv

## L. Corruption Robustness
MethodID	MethodDisplay	CorruptionType	CorruptionValue	CorruptionLabel	Dice	IoU	HD95	ASD	CleanDiceReference	CleanIoUReference	CleanHD95Reference	CleanASDReference	DiceRetention	IoURetention	HD95RelativeDegradation	ASDRelativeDegradation	CaseCount	EngineeringCanaryOnly
ACDC10_ARM_000	BetaFFT	clean	0.0	clean	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.896606340960	0.818200742531	1.761878907247	0.572259726444	1.000000000000	1.000000000000	1.000000000000	1.000000000000	40	False
ACDC10_ARM_000	BetaFFT	gaussian_noise	0.02	gaussian_noise_0.02	0.892856293690	0.811836232180	1.876290421091	0.565099322641	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.995817509760	0.992221333934	1.064937217520	0.987487493052	40	False
ACDC10_ARM_000	BetaFFT	gaussian_noise	0.05	gaussian_noise_0.05	0.805450793295	0.696573867242	3.631825281612	0.831253022948	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.898332697975	0.851348368479	2.061336489513	1.452579981669	40	False
ACDC10_ARM_000	BetaFFT	gaussian_noise	0.1	gaussian_noise_0.1	0.372734691854	0.273590325158	56.668007358021	49.522643728562	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.415717215936	0.334380441054	32.163395069283	86.538754065900	40	False
ACDC10_ARM_000	BetaFFT	gaussian_blur	0.5	gaussian_blur_0.5	0.892964379223	0.812567397108	1.801059703793	0.629296496719	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.995938059358	0.993114959288	1.022238075718	1.099669376752	40	False
ACDC10_ARM_000	BetaFFT	gaussian_blur	1.0	gaussian_blur_1.0	0.875716862362	0.786738178352	2.560004902753	0.890630378454	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.976701616258	0.961546644310	1.452997077281	1.556339433474	40	False
ACDC10_ARM_000	BetaFFT	gaussian_blur	1.5	gaussian_blur_1.5	0.855924709523	0.757062719127	5.481247619408	1.349055866595	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.954627098228	0.925277477488	3.111024030574	2.357418850664	40	False
ACDC10_ARM_000	BetaFFT	gamma_intensity_shift	0.7	gamma_intensity_shift_0.7	0.896526490940	0.817946987536	1.711973187269	0.500664124052	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.999910941941	0.999689862180	0.971674716252	0.874889671449	40	False
ACDC10_ARM_000	BetaFFT	gamma_intensity_shift	1.3	gamma_intensity_shift_1.3	0.893612441650	0.813644059887	1.976994092888	0.691889341927	0.896606340960	0.818200742531	1.761878907247	0.572259726444	0.996660854186	0.994430850027	1.122094194304	1.209047762676	40	False
ACDC10_ARM_111	Ours	clean	0.0	clean	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.903496576520	0.829327295391	1.423366694033	0.429026499330	1.000000000000	1.000000000000	1.000000000000	1.000000000000	40	False
ACDC10_ARM_111	Ours	gaussian_noise	0.02	gaussian_noise_0.02	0.896635620867	0.817772609662	1.417235552569	0.479374363336	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.992406218427	0.986067399694	0.995692507426	1.117353739418	40	False
ACDC10_ARM_111	Ours	gaussian_noise	0.05	gaussian_noise_0.05	0.780031833104	0.671643987656	2.581661131495	0.587778606281	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.863347856953	0.809866009944	1.813770929386	1.370028674682	40	False
ACDC10_ARM_111	Ours	gaussian_noise	0.1	gaussian_noise_0.1	0.334168362981	0.240592223121	73.988277882180	66.751776674760	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.369861238731	0.290105275032	51.981178281274	155.588936298746	40	False
ACDC10_ARM_111	Ours	gaussian_blur	0.5	gaussian_blur_0.5	0.901734643333	0.826142172944	1.653907022718	0.493717668420	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.998049872868	0.996159390310	1.161968331598	1.150785951895	40	False
ACDC10_ARM_111	Ours	gaussian_blur	1.0	gaussian_blur_1.0	0.886835900070	0.802399271621	2.705050553740	0.712257510919	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.981559779103	0.967530281567	1.900459358139	1.660171369439	40	False
ACDC10_ARM_111	Ours	gaussian_blur	1.5	gaussian_blur_1.5	0.868997065905	0.775152269091	3.504319006355	0.923186036575	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.961815560222	0.934675939642	2.461993118882	2.151815885537	40	False
ACDC10_ARM_111	Ours	gamma_intensity_shift	0.7	gamma_intensity_shift_0.7	0.902141218201	0.827202276720	1.601670239292	0.456092338775	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.998499874427	0.997437659796	1.125268875551	1.063086637975	40	False
ACDC10_ARM_111	Ours	gamma_intensity_shift	1.3	gamma_intensity_shift_1.3	0.901211732709	0.825137845114	1.490718631273	0.492333099880	0.903496576520	0.829327295391	1.423366694033	0.429026499330	0.997471109608	0.994948375267	1.047318753152	1.147558718748	40	False

## M. Student Diversity
BLOCKED/PARTIAL if branch output assets remain unavailable; no retraining was added for this analysis.

## N. ACDC2 Protocol
{
  "status": "BLOCKED",
  "created_at": "2026-08-24T17:17:43.268613+00:00",
  "task_id": "ACDC2_LABEL1_FROZEN_BETAFFT_EXECUTABILITY_AUDIT",
  "training_started": false,
  "inference_started": false,
  "test_started": false,
  "split_asset": "configs/internal_evidence_expansion_v24_0/ACDC2_SPLIT.json",
  "split_sha256": "4aa29e06d3b5bbfcd81331252d3dd7bcd012afb09bf6a88372f659fcf43a124d",
  "Labelnum": 1,
  "LabeledCountPatients": 1,
  "LabeledPatientIDs": [
    "patient099"
  ],
  "LabeledFrameCount": 2,
  "LabeledFrameIDs": [
    "patient099_frame01",
    "patient099_frame02"
  ],
  "LabeledSliceCount": 32,
  "PretrainedLabel1Path": "train_ACDC/Pre_model/label1/unet_best_model.pth",
  "PretrainedLabel1Exists": false,
  "FrozenBetaFFTRequires": "at least two distinct labeled patients/cases for beta-FFT labeled a/b branch semantics",
  "Evidence": {
    "labeled_bs_default": 12,
    "labeled_sub_bs_is_half_labeled_bs": true,
    "dual_labeled_branch_code_present": true,
    "two_stream_primary_batch_assert_present": true,
    "primary_batch_formula_present": true
  },

## O. ACDC2 Results
Dataset	Ratio	Method	Status	Dice	IoU	HD95	ASD	SourceAsset
ACDC	2%	β-FFT	TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE					audits/internal_evidence_expansion_v24_0/ACDC2_PROTOCOL_DECISION.json
ACDC	2%	Ours	TERMINAL_PROTOCOL_BLOCKED_EXCLUDED_FROM_FORMAL_QUEUE					audits/internal_evidence_expansion_v24_0/ACDC2_PROTOCOL_DECISION.json
ACDC	5%	β-FFT	FOUND_EXACT_FORMAL_TEST	0.885493109377	0.800727941004	1.464528286577	0.399010756667	final_report/ACDC_5P_10P_TEST_RESULTS.tsv
ACDC	5%	Ours	FOUND_EXACT_FORMAL_TEST	0.887673849575	0.805087330491	2.299675543911	0.738139277934	final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/TEST_RESULT_FULL_PRECISION.json
ACDC	10%	β-FFT	FOUND_EXACT_FORMAL_TEST	0.896606340960	0.818200742531	1.761878907247	0.572259726444	final_report/ACDC_5P_10P_TEST_RESULTS.tsv
ACDC	10%	Ours	FOUND_EXACT_FORMAL_TEST	0.903496576520	0.829327295391	1.423366694033	0.429026499330	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv

## P. Three-seed Stability
Dataset	Ratio	Seed	Method	TaskID	Dice	IoU	HD95	ASD	CheckpointSHA	SourceAsset	Receipt
ACDC	10%	1337	β-FFT	ACDC10_BETAFFT_SEED1337_EXACT_REUSE	0.896606340960	0.818200742531	1.761878907247	0.572259726444	825da2e9a944c930fa54cb9b7abc843345e4a7cde851d34c390739acaa0d5dbf	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv	EXACT_REUSE_V22_V23_FORMAL_TEST
ACDC	10%	1337	Ours	ACDC10_OURS_SEED1337_EXACT_REUSE	0.903496576520	0.829327295391	1.423366694033	0.429026499330	c447a0a6d643405627cebf4718424b7d102fffd677b23f10ab5ef1e5227c3086	final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv	EXACT_REUSE_V22_V23_FORMAL_TEST
ACDC	10%	2025	β-FFT	ACDC10_BETAFFT_SEED2025	0.900035248910	0.823419266451	1.610904573530	0.462955647338	18db038e6937975063f6b759884fc470212f2f7d013bf3763236821de6e4b5c9	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_BETAFFT_SEED2025/aggregate_metrics.tsv	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_BETAFFT_SEED2025/TEST_COMPLETION_RECEIPT.json
ACDC	10%	2025	Ours	ACDC10_OURS_SEED2025	0.899003364644	0.822457742469	2.749908138773	0.779080639000	ffa1912640172caa805d4235c6f552a91a649ffad7626357016a2816c84cbde7	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_OURS_SEED2025/aggregate_metrics.tsv	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_OURS_SEED2025/TEST_COMPLETION_RECEIPT.json
ACDC	10%	2026	β-FFT	ACDC10_BETAFFT_SEED2026	0.898270673999	0.821069525602	1.615380977519	0.471190002401	aecbdeb0cb54a098058c7f8d8e67c832c048f2b835313b8a815a3222301bc30c	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_BETAFFT_SEED2026/aggregate_metrics.tsv	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_BETAFFT_SEED2026/TEST_COMPLETION_RECEIPT.json
ACDC	10%	2026	Ours	ACDC10_OURS_SEED2026	0.897029904666	0.818976028477	3.370103202056	0.865228925322	d4b5789ee2693407ba57c02eae3acb3d1947f4bfa8263ac0d5622c038a7af7b2	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_OURS_SEED2026/aggregate_metrics.tsv	final_report/internal_evidence_expansion_v24_0/raw_test/ACDC10_OURS_SEED2026/TEST_COMPLETION_RECEIPT.json

## Q. Negative / Blocked Cases
{
  "status": "PRELAUNCH_READY",
  "created_at": "2026-08-24T17:17:58.772112+00:00",
  "formal_started": false,
  "tmux_started": false,
  "V24_0_PRELAUNCH_READY": true,
  "EXPECTED_NEW_FULL_COUNT": 4,
  "EXPECTED_NEW_FULL_READY": 4,
  "active_new_full_ready_count": 4,
  "task_count": 27,
  "final_launch_ready_count": 22,
  "not_ready_count": 5,
  "not_ready_task_ids": [
    "PSEUDO_LABEL_EVOLUTION",
    "PATHOLOGY_SUBGROUP",
    "STUDENT_DIVERSITY",
    "ACDC2_BETAFFT_FULL",
    "ACDC2_OURS_FULL"
  ],
  "launch_blocking_count": 0,
  "launch_blocking_tasks": [],
  "noncritical_terminal_gap_count": 3,
  "noncritical_terminal_gap_tasks": [
    {
      "TaskID": "PSEUDO_LABEL_EVOLUTION",
      "TaskType": "FROZEN_REPLAY_ANALYSIS",
      "Action": "BLOCKED",
      "FinalLaunchReady": "BLOCKED_NONCRITICAL",
      "Blocker": "PARTIAL_PSEUDO_LABEL_EVOLUTION_COMMON_CHECKPOINT_COUNT_LT_4_OR_PSEUDO_OUTPUTS_MISSING"
    },

## R. Source / Config / Checkpoint Identity
{
  "status": "PASS",
  "created_at": "2026-08-24T16:07:56.391392+00:00",
  "task_id": "V24_BETAFFT_TEST_FIREWALL_SOURCE_DRIFT_AUDIT",
  "training_started": false,
  "formal_started": false,
  "tmux_started": false,
  "change_class": "E1_ENGINEERING_TEST_FIREWALL_ONLY",
  "changed_source": "train_ACDC/beta_fft_acdc_repro.py",
  "current_source_sha256": "18b128da9429211059eed2e36290964439449b52e6e7de621609170a92bed316",
  "semantic_assertions": {
    "algorithm_changed": false,
    "default_behavior_changed": false,
    "new_flag_default": "False",
    "v24_betafft_launcher_sets_disable_intrain_test_eval": "true",
    "purpose": "prevent in-training TEST loader/eval; formal TEST remains per-task frozen BEST gate only",
    "training_parameters_changed": false,
    "loss_changed": false,
    "optimizer_changed": false,
    "scheduler_changed": false,
    "x03_changed": false
  },
  "registry_sha_updates": [
    {
      "TaskID": "ACDC2_BETAFFT_FULL",

## S. Controller Final State
{
  "status": "PASS",
  "created_at": "2026-08-24T17:18:05.760540+00:00",
  "formal_started": false,
  "tmux_started": false,
  "dry_consumption": "audits/internal_evidence_expansion_v24_0/V24_0_CONTROLLER_DRY_CONSUMPTION.tsv",
  "blocked_count": 0,
  "noncritical_blocked_count": 3,
  "protocol_excluded_count": 2,
  "failed_count": 0,
  "tmux_start_allowed": true
}

## T. Supported Claims
Paired distributions, X03 reliability discrimination, boundary-specific evaluation, descriptive factorial effects, VAL dynamics, corruption retention, label efficiency, and seed stability where exact assets exist.

## U. Unsupported Claims
No X03 causal proof, no factorial significance without replicates, no universal robustness, no hidden negative-result filtering.

## V. Questions for USER + ChatGPT
Decide how to treat noncritical blocked source-asset analyses and ACDC2 protocol non-executability under frozen beta-FFT.
