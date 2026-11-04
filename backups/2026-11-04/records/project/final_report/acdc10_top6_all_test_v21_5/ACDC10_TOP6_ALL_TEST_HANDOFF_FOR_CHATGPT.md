# ACDC10 Top6 All-Test Handoff For ChatGPT

Created: 2026-08-19T12:57:13.111941+00:00

ACDC10_TEST_USED_TO_COMPARE_TOP6 = true
ACDC10_TEST_SELECTION_EXPOSED = true
ACDC10_BLIND_TEST_CLAIM_ALLOWED = false
ACDC10_UNTOUCHED_TEST_CLAIM_ALLOWED = false
FINAL_THIRD_INNOVATION_SELECTION = DEFERRED_TO_USER_AND_CHATGPT

Checkpoint selection was frozen at validation BEST before TEST. No retraining, parameter change, checkpoint reselection, Round2, or V20.5 was performed.

## Main TEST Ranking

# ACDC10 Ours + Top6 Strict TEST

TEST_SELECTION_EXPOSURE = true; BLIND_TEST_CLAIM_ALLOWED = false; UNTOUCHED_TEST_CLAIM_ALLOWED = false.

| Rank | Method | Dice | IoU | HD95 | ASD | Dice Gain vs Ours | Class |
|---:|---|---:|---:|---:|---:|---:|---|
| 1 | X03 Contour Reliability Weight | 90.3497% | 82.9327% | 1.423367 | 0.429026 | +0.4550 pp | STRONG_TEST_GAIN |
| 2 | X05 Label-Adaptive Boundary Optimization | 90.2693% | 82.7905% | 1.732491 | 0.479648 | +0.3747 pp | STRONG_TEST_GAIN |
| 3 | X02 Uncertainty Cross-Branch Pseudo Label Filter | 90.2127% | 82.6849% | 2.719674 | 0.743033 | +0.3181 pp | STRONG_TEST_GAIN |
| 4 | OURS Mask Token + AdaMix | 89.8946% | 82.1810% | 3.438713 | 0.775497 | +0.0000 pp | REFERENCE_OURS |
| 5 | X04 Region Prototype Semantic Alignment | 89.8284% | 82.1201% | 2.855186 | 0.798738 | -0.0662 pp | TEST_NEGATIVE |
| 6 | X06 Late Stability Reliability Ramp | 89.6774% | 81.8683% | 2.957206 | 0.801183 | -0.2173 pp | TEST_NEGATIVE |
| 7 | X01 Class Distribution Reliability Guard | 89.5761% | 81.7328% | 2.105070 | 0.672598 | -0.3185 pp | TEST_NEGATIVE |

## Required Files

- `ACDC10_OURS_PLUS_TOP6_TEST_FULL_PRECISION.tsv`
- `ACDC10_TOP6_TEST_RANKING.tsv`
- `ACDC10_TOP6_TEST_CLASSWISE.tsv`
- `VAL_TEST_RANK_REVERSAL.tsv`
- `../../audits/acdc10_top6_all_test_v21_5/TEST_EXPOSURE_LEDGER.tsv`
