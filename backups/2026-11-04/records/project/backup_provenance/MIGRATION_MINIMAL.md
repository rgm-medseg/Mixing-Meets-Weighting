# Minimal migration paths

- Publication code/config/splits: `/home/renguanming/jupyterlab/rgm/beta_fft_publication_20261010/release`
- Original experiment code: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC`, `/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_promise12`, `/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_COVID19`
- Five portable configs: `/home/renguanming/jupyterlab/rgm/beta_fft_publication_20261010/release/configs/paper/{acdc5,acdc10,promise20,covid5,covid10}.json`
- Original run configs: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/configs/three_innovation_ablation_transfer_v22_0`
- Original ACDC10 Full source/config: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/effective_source.py`; `/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/protocol.json` (original_namespace)

- ACDC current HDF5 and split lists: `/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC` (train_slices.list, train.list, val.list, test.list)
- PROMISE12 current arrays/lists: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/data_staging/promise12/20260624_141558/data_promise12`
- ACDC existing data/geometry location: `/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC/data`; raw physical spacing is not used by the paper grid evaluator
- PROMISE12 original raw/spacing: `/home/renguanming/jupyterlab/rgm/datasets/PROMISE12/extracted/training_data`; `/home/renguanming/jupyterlab/rgm/beta-FFT-main/common_protocols/promise12/raw_data_inventory.tsv`
- PROMISE12 split registry: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/common_protocols/promise12/split_registry.tsv`; actual labeled array indices 0–201, other indices 202–len(X_train)-1
- COVID current derived slices and manifests/spacing: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/data/COVID19_OFFICIAL20/derived/slices_256`; `/home/renguanming/jupyterlab/rgm/beta-FFT-main/data/COVID19_OFFICIAL20/manifests`
- COVID original CT/infection masks: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/data/COVID19_HCCP/derived/stage1_identity_extract/images`; `/home/renguanming/jupyterlab/rgm/beta-FFT-main/data/COVID19_HCCP/derived/stage1_identity_extract/infection_masks`
- COVID original header geometry: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/audits/COVID19_STAGE1_DATASET_IDENTITY.json`

- Final Full checkpoint: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt`
- Final Full checkpoint: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3/checkpoints/best.ckpt`
- Final Full checkpoint: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/PROMISE20_FINAL3/self_train/unet_best_model.pth`
- Actual pretraining initialization: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/PROMISE20_FINAL3/pre_train/unet_best_model.pth`
- Final Full checkpoint: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/unet_best_model.pth`
- Actual pretraining initialization: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/pre_train/unet_best_model.pth`
- Final Full checkpoint: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3/self_train/unet_best_model.pth`
- Actual pretraining initialization: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3/pre_train/unet_best_model.pth`
- Actual pretraining initialization: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label3/unet_best_model.pth`
- Actual pretraining initialization: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label7/unet_best_model.pth`

- Final paper table checklist: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/Table_2_3_4_Row_Checklist_20261009.txt`
- Main Full raw result tables: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/final_report/three_innovation_postfull_test_v22_1/MULTIDATASET_FINAL3_VS_OURS_TEST_FULL_PRECISION.tsv`; `/home/renguanming/jupyterlab/rgm/beta_fft_publication_20261010/release/docs/paper_results.tsv` (five settings only)
- ACDC10 original Full result receipt: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/reports/Historical_Full_EVALUATION_RECEIPT.json` (Historical_Full, not R1_v2_B)
- Author-confirmed current result record: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/PAPER_AUTHOR_CONFIRMED_FACTS.md`
- Current experiment explanation: `/home/renguanming/jupyterlab/rgm/beta-FFT-main/review_ABC_followup_20261009_231554_BJT/REVIEW_CLOSURE_EVIDENCE.md`

Only paths and source records were inspected. No data/weights were moved or loaded. label3/label7 generation commands/logs and actual pretraining update counts are not present in the existing confirmed records.
