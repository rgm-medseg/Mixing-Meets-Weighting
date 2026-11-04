# 五项审查

这是本次执行的五项顺序审查，不冒充五位独立审稿人。原始过严断言失败与修正日志均保留。

## 1_identity_configuration

**PASS** — 8 historical arm SHAs and selected steps match; all budgets 30000; actual TRC-off four used; main X03 retained; both student branches strict; frozen state/checkpoint hashes unchanged

## 2_alignment_and_groups

**PASS** — 216 source hashes; all108 GT/image/geometry/occlusion/source-patient compositions exact; teacher strata exclude annotated sources; identical input hashes across four models; original exclusions preserved; 210 unique slices plus6 repeats

## 3_numerics_and_gradients

**PASS** — 108 main and432 factorial scaling/direction checks; no robust flips; explicit FP32 cancellation handling; class/batch/lambda reductions checked; independent GT CE derivative max error 5.960e-08; L1/signed sums reconcile; weight-switch predictions/errors identical; optimizer steps0

## 4_claims_and_evidence

**PASS_WITH_MIXED_FINDINGS** — Both GT-aligned and GT-opposed mass rise in teacher-error boundary support; original existing shares identified; raw mean-a remains negative on teacher-error pixels; no selective-correction, no-error-amplification, training-stability or complete-causal-synergy claim; no PROMISE computation

## 5_table_and_english

**PASS** — One 12-row table in Markdown/CSV with full denominator/support/paired-probe caption; independent 4.4 and Discussion prose retains negative result and scope; no PDF replacement or Appendix

## 结论界限

审查通过指身份、实现和报告与证据一致；结果本身为混合观察，不能证明CTW只强化纠错或完整训练协同。该分析没有训练、重选checkpoint、重评价主表或PROMISE历史运行。
