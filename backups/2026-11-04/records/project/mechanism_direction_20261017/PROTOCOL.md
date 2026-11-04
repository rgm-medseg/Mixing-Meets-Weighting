# Frozen GT-direction allocation diagnostic — 2026-10-17

## Scope and identity

Read `../PAPER_AUTHOR_CONFIRMED_FACTS.md` first. Main Full is original X03 / Full_main / paper R1, selected step 29600, checkpoint SHA256 `c447a0a6d643405627cebf4718424b7d102fffd677b23f10ab5ef1e5227c3086`, retained test Dice 90.35%. The 89.98% foreground-margin controller model remains independent. Retained training evidence: TRC +0.26 pp; original dynamic 90.3497% versus fixed97 89.8548%, unrounded difference 0.4948 pp. These scores were not reevaluated here.

This package only reads existing ACDC validation arrays/checkpoints, performs frozen eval-mode forwards and differentiates independent Student-1 logit leaves. No training, optimizer, EMA update, BN-statistic update, checkpoint selection, test-score evaluation, TRC intervention, or fixed-control replay. Multiple seeds and E3 remain cancelled. PROMISE historical loader remains unarchived/unverified; a runnable public entry does not reproduce the historical 83.86% run. No PROMISE computation is performed.

The requested `Mixing_Meets_Weighting_Narrative_Revised_20261017.pdf` was not found in the workspace filename inventory. The 16-page manuscript identity/page count is supplied by the author; it was not independently checked. Deliver independent English text; do not replace an older PDF. There is no Appendix deliverable.

## Reuse and inputs

Original package: `../mechanism_strengthening_20261013/`. Its 432 joint NPZs contain 108 mixtures for each of four *inference/loss* conditions, all from the same Full. The 216 augmented-source NPZs and full input plan are present. Original pseudo_correct/error weight, disagreement, prediction error, L1 gradient and gradient shares already exist; reproducing them below is a data-integrity check, not new experimental evidence.

The missing field is Student-1 probability. For Full, reuse archived four-channel gradients and all archived masks/teacher targets, and run Student 1 once on the saved weak mixtures. Match the historical two groups of six forward calls; eval mode, FP32, deterministic cuDNN and four CPU threads. Cache every probability batch with checkpoint/input hashes; reaggregation uses caches. The initial precision check stopped after batch 0; the corrected run reused that cache, so no input received an extra Full forward. Its failure log is retained.

Use the original 10 patients, 20 frame volumes, 210 distinct source slices, nine batches of 24 roles, 216 source exposures including six repeats, and 108 complementary mixtures. Pixel counts are mixed-image exposure counts, not independent samples. Do not replace the plan, select cases or search rectangle sizes. All archived dynamic rectangles are 97×97 and coincide with the archived fixed masks; this analysis adds no mixing-geometry contrast.

## Supports and source-patient aggregation

Import the original `regions()` without modification. The mutually exclusive spatial partition is cutout (interior plus Euclidean width-two edge band) > seam (width-two rectangle band) > clean anatomical boundary (transformed source four-neighbour boundaries, width two) > other_valid > invalid rotation padding. All pixels remain in the loss.

Teacher correctness is defined only on `pseudo_source & geom_valid & ~occlusion`, using the actual saved weak-view LCC hard teacher target against the identically transformed/composed anatomical GT. Human-labelled source pixels are excluded. Strong-view hard targets remain archived separately and are not substituted for this pre-existing canonical stratification. Cross each original spatial mask with pseudo_correct/error. `other_valid` is the clean non-boundary comparison, excluding seam, cutout and padding. The cutout intersections contain visible edge-band pixels only because actual occlusion interior is excluded by teacher-correctness support. Invalid×teacher rows have no support and are explicitly retained.

For each image, assign pixels to their original source patient by the saved paste mask; sum numerators and denominators over all appearances of that patient, then average patient estimates equally. Mean weight, gradient L1, probability disagreement `e`, argmax disagreement and prediction error rates use supported patients only; no-support means are NA. Prediction-error support excludes occlusion interior and invalid geometry. Shares use all ten source patients, with unsupported regions contributing zero.

`gradient_fraction`, `aligned_gradient_fraction`, `opposing_gradient_fraction` and `neutral_gradient_fraction` all divide by the **same patient's full mixed-image Student-1 logit-gradient L1 sum**, including padding/cutout in that original loss. Aligned/opposing shares therefore do not have their own denominators. Crossed teacher rows overlap the separate spatial summaries; never add both as a global partition. Absolute raw sums and full denominators are retained in patient/image CSVs. Paired CSVs report CTW minus uniform, the number of increasing/decreasing/tied patients, and agreement with the mean direction. Patients are evaluation units, not independent training seeds; no interval is used to claim training stability.

## Loss and GT direction

For 12 mixed images, four classes, H=W=256:

`L = 0.05 * mean_batch,pixel(w * mean_class(abs(p1 - stopgrad(p2))))`.

CTW uses the original per-image spatial finite differences of mean student maximum probability, original min/max normalization (range denominator clamped at 1e-6), and `w=0.25+normalized`, clamped [0.25,1.25]. Both weights and Student 2 are detached. Uniform uses each **whole image's** CTW mean, before applying any statistical region masks. No regional loss support/renormalization is introduced. Per-image means match up to the original FP32 reduction roundoff; equal-patient mean weights need not match because patients occupy different parts of images.

Let `g=dL/dz1`, `h=p1-onehot(mixed_GT)`, `a=sum_class(g*h)`. An infinitesimal step `z1-epsilon*g` changes pixelwise GT cross entropy by `-epsilon*a`: positive a is GT-aligned, negative a is GT-opposed. This is a local logit-space reference, without the network parameter Jacobian or any real parameter update. GT never enters CTW, teacher processing, mixing control or selection. CTW compares students; the teacher hard target belongs to another mixed-supervision item. Teacher error is a stratification property, not the CTW loss target.

Report two kinds of contribution separately:

- L1 mass on aligned/opposed/neutral pixels, divided by the full patient L1 denominator above. `mean_aligned_grad`/`mean_opposing_grad` divide that mass by all supported region pixels, including zero contributions.
- Signed directional magnitude: `a_positive_sum=sum(max(a,0))`, `a_negative_magnitude_sum=sum(max(-a,0))`, and signed/absolute sums. The `a_*_fraction` fields divide by the patient's **whole-image sum(abs(a))** in each condition. These exact signed sums include tiny neutral values. They are different units/denominators from L1 shares and must not be compared as identical quantities. Raw mean-a columns are also available, avoiding reliance on normalized fractions alone.

Analytic check: at fixed predictions and identical normalization, `g_CTW=(w/mean_image(w))*g_uniform`. All weights are positive; exact arithmetic cannot flip a single-pixel direction. Check every channel against this identity and independently check autograd against the softmax-L1 derivative for the four added models. The original Full gradients already have independent analytic checks in the original receipt.

## Floating-point uncertainty

The first overly strict sign-equality assertion exposed FP32 cancellation on saturated probabilities: raw sign differences at gradient magnitudes around 1e-16 are not meaningful direction changes. Preserve these differences rather than claiming bitwise sign identity. Compute dot products in FP64 from the original FP32 arrays.

Let `r=w_CTW/w_uniform`, `c=0.05*w_uniform/(12*256*256*4)` and `eps=machine_epsilon(float32)`. In uniform-gradient units use

`tau = 1e-6*max(||g_uniform||1, ||g_CTW||1/r) + 8*eps*c*||h||1`.

The second term is a conservative allowance of eight rounding units at the per-class loss coefficient, accounting for subtractive cancellation in softmax backward. Classify both `a_uniform` and `a_CTW/r` against this common band. If either is within it or crosses its edge, label the paired pixel neutral. A positive/negative disagreement outside the band fails the analysis. Exact-zero gradients are counted separately. The relative tolerance is dimensionless; no case/region-specific threshold is chosen. Raw sign differences, their L1 mass, edge disagreements and peak relative scaling residuals are recorded per image. The uncertainty classification only affects L1 directional assignment; raw gradients and signed-a sums remain unchanged.

## Conditional four-trained-model extension

The relevant TRC-off factorial is 000 Baseline, 010 TCSMix, 001 CTW, 011 TCSMix+CTW. This differs from the old fixed97/dynamic × uniform/CTW inventory. All four actual selected checkpoints are available and match the historical eight-arm SHA records. Their last checkpoints completed 30,000 updates; common data/optimization/augmentation/budget arguments match. Selected steps are 21200, 22200, 25200 and 26600 respectively. Both student branches load strictly; state hashes and on-disk checkpoint hashes remain unchanged. The TRC-on group is also complete but is inventoried only, with no score-based substitution.

The static training sources retain LCC pseudo-labels, the same weak/strong processing, base losses and 0.05 CTW coefficient; TCSMix uses the original teacher mean-max controller, while TCSMix-off uses the original deterministic top-left 128×128 mask. The configured toggles are the intended training differences. Source/config hashes, arguments, manifests, selected/checkpoint identities and completed budgets are retained.

Use **the exact archived main-Full weak/strong mixed images** for every model, with **the same archived main-Full EMA teacher reference** for correctness groups. This compares trained model states under common input, not each model's own construction rule or own teacher error set. Within every model probe uniform and CTW on its own fixed predictions; even CTW-off trained models receive this diagnostic loss probe. This does not mean CTW was present in their training. Retain each model's own predictions/weights/gradients, then compare paired patient states and CTW-minus-uniform responses. No own-rule mixing replay or training-causal interpretation is added.

## Delivery and technical reference

One quantitative summary table is delivered as Markdown/CSV; no new figure. Independent English text belongs in 4.4/Discussion. Five reviews cover identity, alignment/supports, numerics, evidence-to-conclusion mapping, and table/English. The original 46.60%→56.02% boundary share remains an existing observation, not a new training claim.

The supplied PyTorch 2.14 URL could not be fetched. The official stable link resolved to [PyTorch autograd.grad documentation](https://docs.pytorch.org/docs/2.13/generated/torch.autograd.grad.html), checked for differentiation of specified inputs without accumulation into parameter `.grad`. The installed runtime version is recorded in `receipts/environment.json`; it is not inferred from that documentation URL.
