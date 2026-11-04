# HiEn Controlled-Reproduction Protocol Audit

Audit date: 2026-09-02

## Identity and frozen recipe

- Formal paper: *HiEn: Hierarchical Ensemble Learning for Semi-Supervised Medical Image Segmentation*, *Neurocomputing* 697 (2026), article 134243, DOI `10.1016/j.neucom.2026.134243`.
- Official repository: `https://github.com/jch3rn/HiEn.git`, frozen at `6dfa06f24c5a80223d14ee07183ced713ae8681e`.
- License: Apache-2.0.
- Official publisher PDF is access restricted. The frozen Crossref DOI record and repository metadata establish the paper/repository identity; no unofficial PDF is represented as the version of record.
- Main network: the released `unet`, whose encoder is ImageNet-pretrained torchvision ResNet34 and whose decoder includes the VAT injection point and projection head.
- Input adaptation for all frozen grayscale datasets: deterministic channel replication `1 -> 3`; no trainable input adapter and no changed pretrained convolution.
- Official defaults are frozen before target results: 30,000 outer iterations, batch 4 (2 labeled + 2 unlabeled), SGD 0.01, momentum 0.9, weight decay 0.0001, seed 1337, consistency 0.1, ramp-up 200.

## Preserved method semantics

- Instantiate student, teacher, and auxiliary teacher with identical full HiEn networks.
- Optimize student and teacher parameters with the released shared optimizer.
- Per outer iteration, run one supervised teacher update with `0.5 * (CE + Dice)`, then EMA teacher into student with alpha 0.98.
- Copy teacher state to auxiliary teacher, then run one unsupervised update.
- Before unsupervised forward, EMA student into teacher and auxiliary teacher with alpha 0.95.
- Preserve weak/strong Gaussian perturbations, decoder-space VAT (`xi=1e-6`, `eps=2.0`), projection-head contrastive loss, and prediction consistency activation only from iteration 1000.
- Preserve the shared validation-only best score: validate student first and teacher second; either may replace the same best checkpoint only if its validation Dice strictly improves. The adapter adds a sidecar `selected_model_role` field so the released overwrite behavior is auditable.
- The selected full model's segmentation logits are the unique test output. No ensemble or test-time model choice is introduced.

## Compatibility and resource ruling

- The released code is 2D and exposes two-, four-, and seven-class commands. ACDC uses four classes; PROMISE12 and COVID-19 use two.
- Dataset-name slice-count shortcuts are replaced by explicit frozen manifests only. Core networks, losses, schedules, perturbations, EMA direction, and validation selection are unchanged.
- The official pretrained weight is frozen locally as `resnet34-b627a593.pth`, SHA256 `b627a593bcbe140c234610266fe4f8ae95ea42fc881d091c9b6052e6b1d0590f`.
- Preflight host: one NVIDIA A100-SXM4-40GB, over 40 GB free VRAM at audit time, 100 GiB RAM, and sufficient project filesystem capacity. Batch 4 is the official default and requires no resource-driven semantic change.

## Gate result

- Publication/repository/commit/license identity: `PASS`.
- Complete official training algorithm: `PASS`.
- Frozen main recipe and pretrained identity: `PASS`.
- Binary/multiclass and three-dataset structural compatibility: `PASS`.
- Frozen split and TEST isolation design: `PASS_PENDING_DATASET_CONTRACT_EXECUTION`.
- Final candidate status: `ELIGIBLE_FOR_CONTROLLED_REPRODUCTION`.
- `TRAINING_STARTED=NO`, `SMOKE_STARTED=NO`, `CONTROLLER_STARTED=NO`.
