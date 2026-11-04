# ACDC10 mechanism diagnostics and fixed-region control

This package is separate from the current 17-page paper. It does not replace manuscript figures/PDFs or author-confirmed results. Main Full / paper R1 is original X03 / Full_main, test Dice **90.35%**; formal_R1_20261007 / R1_v2_B **89.98%** and its 92×92 control remain independent. E3 and multiple seeds remain cancelled.

## Actual inputs

- Main checkpoint: `runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt`, step29600, SHA256 `c447a0a6d643405627cebf4718424b7d102fffd677b23f10ab5ef1e5227c3086`. Branches `model_1`, `model_2`, `ema_model`, strict loading. Paper prediction branch: `model_1`.
- Pretraining: `train_ACDC/Pre_model/label7/unet_best_model.pth`, SHA256 `477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3`, `net` and `opt` reused. No label3/label7 generation.
- Dataset: `/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC`. Existing `train_slices.list`, `val.list`, `test.list`; their hashes and actual input/source paths are in `receipts/input_identity.json`. The train loop uses the original first136 labeled slices and 12+12 batch sampler.
- Diagnostic seed20261013; training seed42. Diagnostic FP32/eval mode and deterministic cuDNN. Frozen dependencies are in `scripts/frozen`, original Full loop in `scripts/main_full_source.py`.

## Task1: four conditions

The predeclared full validation set contains 20 frame volumes / 10 patients / 210 slices. A hash ordering independent of outcomes makes nine batches of24, with roles img_a6, img_b6, uimg_a6, uimg_b6 simulated for this offline analysis. Six source exposures repeat to complete the last batch; they are disclosed in the plan/manifest, not counted as new patients/slices. Each batch yields twelve complementary mixtures: uimg_a outside/img_a inside and img_b outside/uimg_b inside. Both students see the same mixed content, student1 weak and student2 strong.

Original `WeakStrongAugment` is instrumented with exact output and RNG-after checks against its uninstrumented call. Rotation/flip/nearest resize/ColorJitter/cutout remain literal. Anatomical GT and source boundary masks receive the same geometry; rotation padding and strong-view cutout are tracked. GT never enters the controller, weights, model, or case selection.

Teacher q is the original full-image mean maximum softmax probability over both weak unlabeled sub-batches. Side fraction is `.18 + .20*q`, Python rounded to pixels and clamped1..255. One mask is shared across each batch, both views and complementary directions. Fixed side97 comes from existing main training endpoint records. Positions share a predeclared fractional draw for each strategy: `floor(u*max(1,256-side))`, matching the original exclusive-upper-bound support. The fixed/dynamic images may differ if sides differ. In these actual frozen inputs, every side rounded97; the outputs/gradients were checked bitwise equal.

Main per-class largest connected component teacher pseudo-labels are retained for both weak and strong source views. The canonical pseudo-source accuracy regions use weak teacher pseudo-labels on the portions occupying unlabeled-source roles. Strong pseudo-label arrays are saved separately. Student predictions use the original argmax; no additional student LCC is introduced.

CTW is computed from spatial finite differences of the mean student maximum probability, normalized per image by min/max (denominator clamped1e-6), then `w=.25+normalized`, clamped .25..1.25. The uniform condition uses each whole image's CTW mean, including pixels later excluded from error statistics. Consistency error is `e=mean_class(abs(p1-stopgrad(p2)))`. The actual item is `.05*mean_batch_pixels(w*e)` over twelve mixtures, exactly cross-checked with the original X03 function. Weights and student2 are detached. `torch.autograd.grad` differentiates this item on the real forward logits of student1; the full four-channel gradient arrays are saved. An independent closed-form softmax derivative checks the actual coefficient and batch/class reductions. No optimizer exists in diagnostic scripts; parameters are frozen and all parameter gradients remain None. State hashes are unchanged. This gradient describes the CTW consistency item, not the total loss.

Primary spatial partition, in priority order, on256-grid pixels:

1. **Cutout**: mixed occlusion interior plus Euclidean width2 band around its edge, within valid source geometry.
2. **Seam**: width2 band around the actual rectangle boundary, excluding cutout and invalid geometry.
3. **Anatomy**: width2 band around transformed *source* four-neighbor class boundaries, after composing source masks; excludes seam, cutout and invalid geometry. Artificial class transitions created by pasting are not counted as real anatomy.
4. **Other valid**: remaining valid geometry.
5. **Invalid**: rotation padding. It remains in the original loss; excluded from anatomical/error interpretations.

The five regions partition all pixels. Pseudo-correct/error and RV/MYO/LV/background masks are additional overlapping statistics. Error support excludes actual cutout interior and invalid rotation padding; cutout-edge pixels that remain visible can contribute to its error row. Region masks never alter the loss or weights.

Pixels are assigned to their source patient using the same rectangle composition. For each patient/condition/region, sum pixel statistics across source exposures, divide by that patient's supported pixels, then average patient means equally. Zero-support means are NA. Gradient magnitude is L1 across four logit channels. Regional gradient/loss fractions divide the patient's regional sum by its full source-patient sum across all mixed images, then average across all10 patients (zero region support contributes zero share). Pixel counts are pooled support counts, not equally weighted estimates. Raw image-region and patient CSVs retain denominators. Pseudo-source regions do not form a partition of the entire image.

No complete set of four comparable trained configurations with TRC on and the main signal was found at the known run locations. The partial original-signal E1 is not a completed30000-update result; alternative-signal C1/R1/R2 are not substituted. No new factorial training is launched. Missing trained combinations are recorded; the only new training is Task3.

## Task2: TRC gamma intervention

Main student1 is evaluated slice-wise with original nearest resize to256, argmax, nearest native-grid restoration. Normal inference and temporarily zeroing `decoder.mask_token_router.gamma` use identical inputs/checkpoint. The gate is restored in a `finally` block after every slice; restored logits must be bitwise equal and the final state digest unchanged. Original checkpoint is never written.

Full pre-router feature, raw residual and conditioned feature are saved for every slice in `arrays/trc_features`; full image, GT, both predictions and response maps in `arrays/trc`. Applied residual norm is `||gamma*r||2`; logit response is `||normal-gamma0||2`, without spatial normalization. The feature at the input of TRC must be identical between conditions, and gamma0 output must equal that feature.

Error direction is **gamma0 → normal**: corrected means zero wrong/normal right; introduced means zero right/normal wrong. Changed-but-wrong-in-both pixels are retained separately. Changes total corrected+introduced+bothwrong_changed. Anatomy width2, foreground interior outside the band, and background outside the band partition all pixels. RV/MYO/LV GT-support rows and additional one-vs-rest all-image class rows distinguish organ-local changes from class false positives elsewhere. Cross-sectional diagnostic pixel statistics use the256 grid. Patient Dice uses native frame-volume predictions, mean of each patient's two frames, then equal patient mean. It reproduces the checkpoint's normal validation Dice. No inference LCC is added. PCA/UMAP/linear probes are not needed for this first intervention.

## Task3: F97_main_signal_20261013

No usable old main-signal fixed checkpoint was available at the known E2 location. The sole new condition has97×97 fixed dimensions. Basis:150 actual main-Full log records, steps200..30000 every200, all area9409/65536; full original lines/area record paths are preserved in `configs/fixed_size_basis.json` and `results/fixed_size_historical_records.csv`. This matches recorded endpoints, not a proven full-trajectory average. No offline frozen-teacher calibration or score-based size scan was needed.

The original Full loop is executed literally with a dimension-only replacement in the original mask function. Teacher q is still computed and recorded, the original two NumPy position draws and all downstream mixing are retained. TRC, CTW .05, EMA .99, all original losses/FFT/augmentations, inherited SGD state,16 workers, seed42, batch24, split and30000 updates per student are retained. Exact pretrain base tensors and transplanted optimizer tensors/hyperparameters are checked at initialization. A CPU-thread cap of4 is recorded as execution environment; original FP32 settings are retained. No main-Full/pretraining replay is launched. The new run writes only its independent directory. No existing process is stopped.

Original validation runs every200 and selects student1 best mean three-class native-volume Dice. Test access for this new run waits until30000 updates complete and that selection is frozen in a receipt. The same original test prediction protocol and literal `safe_metrics` function from `tools/v21_5_acdc10_test_frozen_checkpoint.py` are used: nearest256/native resize, argmax; Dice/IoU/grid HD95/directed ASD, original empty-case conventions,40 frame volumes. The existing main-Full test result is retained rather than reevaluated. Main test CSV source is `final_report/acdc10_top6_all_test_v21_5/raw_test/X03/TEST_RECEIPT.json`; no old candidate/import/table audit was reopened.

`scripts/pipeline.py` waits for our full diagnostic receipts and a free GPU, trains the single condition, freezes validation selection, evaluates and refreshes the report. `receipts/pipeline_status.json` and training logs provide live state. A RUNNING result is not PASS or a final numerical claim.

## Figures and writing

Task1 illustration is predeclared batch0/pair0; Task2 is the first sorted validation frame/middle slice. Neither is chosen by score, appearance or intervention response. Full image coordinates and nearest display are retained. GT classes are consistently RV blue, MYO orange, LV purple; gradient maps share limits; error colors have semantic labels. Scalar heatmaps use cividis/magma; data tables and patient points supplement color. Bars start at zero and show raw patient points. No synthetic medical images or generated evidence are used.

SVG retains text, PDF embeds TrueType (Type42), previews are PNG. Files record190mm width and export choices in provenance JSON. These are draft main-text figures; the Neurocomputing journal-specific guide returned403, so final-production journal-format verification remains PENDING. The general official artwork source is [Elsevier artwork instructions](https://www-prod.elsevier.com/about/policies-and-standards/author/artwork-and-media-instructions). Source-data images are not upsampled to claim additional detail;500dpi is a container export setting.

K-Dense skill read: [official scientific-visualization SKILL.md](https://github.com/K-Dense-AI/scientific-agent-skills/blob/main/skills/scientific-visualization/SKILL.md), snapshot saved in `skills/K_Dense_Scientific_Visualization_SKILL.md`, with publication/journal reference files. Workflow reference added to this package's software documentation: Kassis, T., Agarwal, V., He, Y., Patel, D., & Brueckner, A. M. (2026). *Scientific Agent Skills: A Library of Procedural Knowledge for Research Agents*. [doi:10.48550/arXiv.2609.00065](https://doi.org/10.48550/arXiv.2609.00065), current arXiv record verified (v2). It is a workflow/software citation, not evidence for the segmentation conclusions.

English paragraphs and captions are independent main-text suggestions. They explicitly retain geometry saturation, the very small TRC effect, and pending fixed results until actual completion. Figure2 layout foregrounds “Mixed samples → Aligned predictions → Spatial consistency weighting.” No historical manuscript is overwritten.

## Completed fixed-run result and final CPU aggregation

The single F97 run completed30000 updates per student. Validation-selected student1 checkpoint: update29000, SHA256 `a234794d53a3247b9504fd1a1ebc051686eed4495f0417866120941ddcf4b6d3`, validation Dice90.1648222271%. Test Dice89.8548131987%, IoU82.1903108574%, HD952.2512519585 grid and directed ASD0.6997793486 grid. The retained main-Full test Dice is90.3496576520%; fixed-minus-main is−0.4948444533 percentage points. The40 frame volumes represent20 patients, each with2 frames. Equal-patient reaggregation and paired original-Full CSV reuse reproduce the same means; Full was not reevaluated.

The fixed run's own30000 teacher-signal records, remapped offline through the original size formula, imply97×97 at29999 steps and96×96 once (step460), while every actual training mask remains97×97. These signals belong to the new fixed model and cannot recover the original Full trajectory. The score gap is a single-run comparison of prescribed size rules; it does not establish an interaction or a causal benefit from varying area. See `results/fixed_run_signal_geometry_distribution.csv`, `fixed_run_signal_geometry_rare.csv` and `fixed_vs_main_patient_dice.csv`.

`final_control_analysis.py` performs CPU-only aggregation after test completion. `tables.py` exports the one two-panel regional table and the separate fixed-control table as editable SVG/PDF plus PNG previews. Full checkpoint and all diagnostic arrays remain in the workspace; the compact archive includes scripts/configs/results/receipts/vector figures and fixed-case source arrays. A syntax error during development of the CPU aggregation helper was corrected before its execution; no training/evaluation rerun occurred. An unsupported table-header glyph was replaced with ASCII labels, documented in the manual review receipt. Final reviews recheck all new table outputs and numeric aggregation.
