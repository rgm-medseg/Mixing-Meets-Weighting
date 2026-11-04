# 本包阅读索引

先读 `REVIEW_CLOSURE_EVIDENCE.md`。本包仅增补尚未闭合的 A/B/C，不替代历史证据包；NOT_FOUND 不表示未复现。

| 文件 | 用途 |
| --- | --- |
| TABLE2_3_4_73_ROW_INDEX.csv | 作者清单的73个表位置、20个既有映射及原始引用行 |
| ROW_FIELD_STATUS.csv | 每个表位置各来源/协议字段的 VERIFIED / PARTIAL / NOT_FOUND |
| MISSING_53_ROW_BINDINGS.csv | 53个尚缺绑定的位置，原始精度不以显示值代填 |
| PROMISE_FULL_EVALUATOR_VERSION_CHAIN.csv | ROW06原/新两版评价SHA链 |
| PROMISE_RECEIPT_COMPARISON.csv | 同checkpoint、同split的四项标量差值＝0 |
| COVID_MAIN_TABLE_CORRECTION_COVERAGE.csv | 26个当前主表位置的已有修正覆盖 |
| COVID_EXISTING_SPACING_ONLY_FOUR_ROWS.csv | 已有另版spacing-only记录，保留原source_run_label；不替换论文采用值 |
| COVID_EXTERNAL_ACTUAL_PROTOCOL.csv | UCAD及四条未入表HiEn/PMEUL的实际driver/metric与旧结果 |
| COVID_HIEN_PMEUL_INCLUSION_AUDIT.csv | 当前排除已知；预定纳入原因及既有修正未找到 |
| A_EXISTING_RECORD_CONFLICT.csv | DC-Net ACDC10论文采用值已确认，数值差异待办已撤回；旧值保留历史身份，来源绑定仍缺 |
| B_PRETRAIN_GENERATION_FIELDS.csv | label3/label7实际生成的14字段缺口 |
| B_BUDGET_BOUNDARY.csv | 已确认30000半监督更新与未知预训练/总预算分开 |
| C_HISTORICAL_IMPORT_BINDINGS.csv | 四组件、六目标运行记录的历史import绑定边界 |
| SOURCE_INDEX.csv | 原文件路径、完整SHA、引用行、小型副本/摘录及其独立SHA |
| SEARCH_LEDGER.csv | 项目内限定搜索范围与证据缺失的适用范围 |
| BUILD_VALIDATION.json、PACKAGE_VALIDATION.json | 本轮文本/receipt核对与新包校验 |
| SHA256SUMS | 本包内每个文件的校验清单 |

论文主Full/R1仍是90.35%的原X03/Full_main；89.98%单独叫“更换混合控制信号”。COVID主Full论文采用值仍为22.79/11.88 mm；此前改为33.56/18.96的必须修改建议已撤回。所有比较沿用作者确认“复现”；不由旧助手标签推断ACDC测试参与方法选择。

`evidence/`包含小型既有receipt、配置、清单和带原行号的源码/日志摘录，不包含影像、预测数组或权重。索引的 `original_sha256` 是完整原文件SHA；`exported_sha256` 是实际入包副本/摘录SHA，二者对摘录可以不同。清单里的PDF SHA仅为作者提供的记录，本包未独立校验PDF。

`scripts/build_update.py`只进行stdlib文本解析与四个既有receipt标量差值比较。`scripts/validate_package.py`只校验新包与源码摘录索引。两者均不导入项目模型或执行训练/评价入口。


作者最新确认（2026-10-09T23:58:36.745464+08:00）：**DC-Net / ACDC 10%：HD95＝2.28、ASD＝0.68，单位grid（表2口径）**。旧V19_4的1.28／0.38仅作历史版本；两组数值之间的待核实及替换建议已按作者确认撤回。**论文采用值确认与历史运行来源绑定分开：该行run/checkpoint/split/evaluator绑定仍为NOT_FOUND，未补出新原始精度。**本次只更新记录，不训练、不重评价、不改原始实验产物。

旧压缩包、SHA256SUMS和验证JSON适用于首次归档快照，保留原字节；当前活动报告已更新。最新记录更新包见项目根目录 `dcnet_acdc10_author_confirmation_20261009_235640_BJT.tar.gz`。SOURCE_INDEX中的原SHA为采集时历史快照，不用于声称修改后的作者记录仍与旧SHA相同。
