# PMEUL Controlled-Reproduction Protocol Audit

Audit date: 2026-09-02

State: source and protocol audit only. No training, smoke test, inference, or controller was started.

## Publication and source identity

- Paper: *Semi-supervised Medical Image Segmentation via Perturbation-Aware Mutual Learning and Edge-Aware Uncertainty Loss for Accurate Anatomical Delineation*.
- Version of record: 21 July 2026, *Interdisciplinary Sciences: Computational Life Sciences*, DOI `10.1007/s12539-026-00862-3`.
- Publisher page: https://link.springer.com/article/10.1007/s12539-026-00862-3
- Publisher-linked official repository: https://github.com/waqasanwaar/SSL4MIS_PMEUL
- Frozen upstream commit: `2232ec0f5f11d8d219604343a427abdd86c25891`.
- License file: none found. This is a provenance and reuse caveat, not a method-semantic gate by itself.

## Five-commit history

| Commit | Date | Content | Audit conclusion |
|---|---|---|---|
| `cc27eb722496cc66e0228cfce46de2a564d58082` | 2026-05-23 | Initial one-line README | No code. |
| `c58eeb8620f31909075608161f07f772f95a0c82` | 2026-05-23 | Code-availability notice | No code. |
| `9814bcd45ba0e409262737995e34328cb729eb11` | 2026-06-13 | All released code, lists, weights, and result text | Sole algorithm release. |
| `da8e91bb292974c62ac7baa437f4fb136e6cb339` | 2026-07-22 | README procedure and identity | No code/data/model change. |
| `2232ec0f5f11d8d219604343a427abdd86c25891` | 2026-07-27 | README command/citation clarification | No code/data/model change. |

`git diff 9814bcd..2232ec0 -- code data model` is empty. The writer defect, model definitions, checkpoints, and result assets have therefore been unchanged since the sole code release.

## Formal 2D method identity

- Selected paper variant for the controlled reproduction: `m=3`, mapped to network type `mcnet2d_v2` and class `MCNet2d_v2`.
- Selection rule is pre-result and architecture-based: the formal paper figures report both `Ours(m=2)` and `Ours(m=3)`; `m=3` is the complete three-perturbation-decoder configuration, while `m=2` is the reduced variant. The official README lists the `mcnet2d_v2` procedures first. No target-run result is used to choose the variant.
- The parser default `mcnet2d_v3` has four decoders but is absent from the official README procedures and the formal paper comparisons inspected. It is not the frozen recipe.
- Correct training entry: `code/train_mcnet_2d_sdf.py --model mcnet2d_v2`.
- Correct train-time network: one shared custom U-Net encoder plus three decoders using bilinear, transposed-convolution, and nearest-neighbor upsampling perturbations.
- Correct checkpoint: full `MCNet2d_v2.state_dict()` saved as `mcnet2d_v2_best_model.pth` on validation mean Dice.
- Correct prediction output: decoder 1 logits, i.e. `model(x)[0]`. The formal Figure 1 marks decoder 1 as used for training and testing and auxiliary decoders as training-only.
- Controlled evaluator rule: instantiate `MCNet2d_v2`, load the full checkpoint with `strict=True`, and expose only output index 0 to the project's frozen evaluator. Do not average decoders and do not retain silent `strict=False` behavior.

## Released optimization semantics

- Input: one channel; output class count is configurable.
- Backbone: custom small U-Net (`1,813,764` parameters for the single-output form), not ResNet and not externally pretrained.
- `MCNet2d_v2`: `3,210,924` parameters at four classes.
- Defaults: 30,000 iterations, batch 24 with 12 labeled samples, SGD, learning rate 0.01, momentum 0.9, weight decay 0.0001, seed 1337, 256x256 patches.
- Actual optimized supervised term: summed decoder Dice plus `0.3 *` SDF MSE. Cross entropy is computed but not included in the final `loss`; controlled reproduction must preserve this released behavior.
- Actual unsupervised term: ramped pairwise MSE between each decoder probability and every other decoder's sharpened probability, with consistency base weight 0.1 and temperature 0.1.
- SDF implementation treats all nonzero labels as one foreground boundary and compares it with channel-0 logits. This released behavior must be preserved for multiclass ACDC and naturally remains a foreground/background boundary for two-class tasks.
- No pretrained checkpoint, ImageNet initialization, resume model, prompt model, or external labeled data is loaded by the 2D training entry.

## SummaryWriter defect ruling

- `SummaryWriter` import and construction are commented out, but `writer.add_image`, `writer.add_scalar`, and `writer.close` remain active.
- The first failure occurs after the iteration-20 optimizer step at the first image-logging call. The official script therefore cannot complete unchanged.
- Ruling: restoring the two author-present but commented lines is a minimal engineering patch. It only creates the intended TensorBoard sink; it does not alter data, model, loss, gradients, optimizer, schedule, validation rule, checkpoint rule, or prediction output.
- Frozen patch policy for later implementation: uncomment `from tensorboardX import SummaryWriter` and `writer = SummaryWriter(...)` in an adapter copy only; keep upstream read-only; record the exact patch and dependency SHA/version. A no-op writer is not the primary repair because restoring the declared logger is closer to the released author intent.
- This ruling removes the logging-only blocker but does not by itself make PMEUL eligible. Dataset, output, environment, and independent review gates still apply.

## Checkpoint key audit

The key comparison used PyTorch 2.4.1 on CPU with `weights_only=True`; it instantiated modules and compared/loaded state dictionaries only. No forward pass or inference was run.

| Labels | Variant | Checkpoint keys | Exact train-class match | Official UNet missing | Official UNet unexpected |
|---|---:|---:|---|---:|---|
| 7 | v1 / m=2 | 202 | yes | 0 | 66, all `decoder2.*` |
| 7 | v2 / m=3 | 268 | yes | 0 | 132, all `decoder2.*` or `decoder3.*` |
| 14 | v1 / m=2 | 202 | yes | 0 | 66, all `decoder2.*` |
| 14 | v2 / m=3 | 268 | yes | 0 | 132, all `decoder2.*` or `decoder3.*` |

The official `test_2d.py` single-output `UNet` has 136 keys, all of which are exact checkpoint keys with matching shapes. Thus its `strict=False` call fully loads the encoder and decoder 1 and ignores only training-only auxiliary decoders. The controlled adapter will make this explicit by loading the full m=3 model strictly and selecting output 0.

Detailed machine-readable evidence is in `PMEUL_CHECKPOINT_KEY_AUDIT.json`.

## ACDC split and risk comparability

- Upstream `train_slices.list` is byte-identical to the project frozen ACDC train-slice manifest: SHA256 `1b21b2a69d1b28b4dfce6837fdb0572414d337566b51b3dd0e0fda342a6f0648`.
- Its first 68 entries are byte-identical to frozen ACDC 5% labeled data: SHA256 `1ef6badf855ec374a6ef8cfb358c6abaa33d8349a67bf892cba668d6e703eb97`.
- Its first 136 entries are byte-identical to frozen ACDC 10% labeled data: SHA256 `21e086cfa888ebc9f8c7d66b4889e43438030e7fc933e8e9afcdd36a72c6d9d4`.
- Upstream val and test lists are byte-identical to frozen lists: SHA256 `61b935c936d8e12f7889e82d954f421fa25fb95305e92a476096b3242de8777d` and `98161bdfaad072bf5713d7345751ab5a2bf3e5468b8edb03fd9ebfedc4503671`.
- The seven labeled patients are exactly `patient099`, `patient038`, `patient050`, `patient100`, `patient058`, `patient021`, and `patient049`.
- Released m=3/v2 seven-label artifact: Dice `0.86500003`, IoU `0.77088360`, HD95 `4.87421689`, ASD `1.33534103`.
- The Dice risk comparison to frozen Ours ACDC 10% is admissible because dataset, labeled identities, val list, test list, input size, one-channel custom U-Net family, and absence of pretraining match. It supports likely-below-Ours screening, not a substitute for controlled reproduction.
- Released boundary values are not accepted into the paper table because spacing/evaluator provenance differs. Every formal prediction must be reevaluated by the project's frozen evaluator.
- No official three-label result is released. A likely-below-Ours judgment at 5% remains an explicitly uncertain extrapolation from the same method and split family.

## Three-dataset adaptation boundary

- ACDC: replace positional slicing with explicit frozen labeled/unlabeled manifests while retaining the same 68/136 slice identities; four output classes.
- PROMISE12 and COVID-19: two output classes, one input channel, frozen explicit manifests, and the same released loss/decoder semantics.
- Allowed adapter work: manifest reader, path mapping, class count, isolated output path, exact writer restoration, and evaluator output wrapper.
- Forbidden changes: adding CE to the optimized loss, changing SDF semantics, averaging decoder outputs, choosing v1/v2 by observed target results, adding pretraining, changing iterations, or changing frozen samples.

## Current gate state

- Publication identity: `PASS`.
- Five-commit provenance: `PASS`.
- Main variant/network/entry/checkpoint/output identity: `PASS` for `mcnet2d_v2`, output 0.
- `strict=False` audit: `PASS_WITH_EXPLICIT_STRICT_ADAPTER`.
- Writer repair classification: `PASS_MINIMAL_ENGINEERING_PATCH`.
- ACDC frozen split comparability: `PASS`.
- Binary/multiclass structural compatibility: `PASS_PENDING_DATASET_CONTRACT`.
- Environment/resource preflight: `PASS_A100_40GB_OFFICIAL_BATCH24`.
- Independent semantic review: `PASS`; it confirms m=3, no optimized CE, all-foreground SDF, pairwise consistency, no pretraining, and decoder 1 TEST output.
- Final PMEUL status: `ELIGIBLE_FOR_CONTROLLED_REPRODUCTION`.
- `TRAINING_STARTED=NO`.
- `SMOKE_STARTED=NO`.
- `CONTROLLER_STARTED=NO`.
