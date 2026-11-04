# A/B/C 收尾证据增补

生成：2026-10-09T23:35:07.932815+08:00；实际主机：n1。题目：Mixing Meets Weighting: Confidence-Modulated Semi-Supervised Medical Image Segmentation。模块 TRC、TCSMix、CTW；ACDC、PROMISE12、COVID-19。

本轮仅解析既有文本、源码和 receipt，进行四个标量差值复算；没有加载权重、影像或预测数组，没有运行模型或评价器，没有启动/恢复训练、重选 checkpoint、增加 seed、改动原产物或干预进程。E3 保持取消。D、237 文件校验、29 运行/116 均值及已关闭预算/CTW/入口事项没有重复执行。

**本轮新关闭：A 中 ROW06 PROMISE Full 的评价器版本差异。A 整体 PARTIAL；B 生成记录与 C 历史 import 绑定仍未闭合。** 缺失字段逐个列在 CSV；NOT_FOUND 仅表示在本轮限定资料中未取得证据，不表示方法未复现。

| 项目 | 状态 | 本轮结论 | 关闭条件 |
| --- | --- | --- | --- |
| A：当前表 2–4 范围 | VERIFIED | 清单实读 73 个表位置；20 个有既有映射，53 个尚缺；不是 73 个不同训练 | 已关闭范围计数 |
| A：ROW06 评价版本链 | VERIFIED | 原 e338… 与 Sept4 的 2a20… 均保留；同 checkpoint、同测试 manifest、四指标差值均 0 | 该差异缺口关闭 |
| A：其余 53 行绑定 | NOT_FOUND | ACDC22、PROMISE11、COVID20，原精度、协议等留空 | 提供对应 run/receipt、checkpoint与 split SHA、验证选点、实际 driver/metric 和距离规则 |
| A：UCAD 与其他主表行已有 COVID 修正 | NOT_FOUND | UCAD 两条有旧 receipt，未找到既有修正；其余20条先缺行绑定；未新做评价 | 既有修正 receipt/结果及版本身份；本轮不授权新增重评价 |
| A：未入表的 HiEn/PMEUL COVID | PARTIAL | 四条真实评价完成，当前清单排除；既定纳入原因及已有修正均 NOT_FOUND | 既定范围/纳入决策原记录，及已有修正结果；不按成绩改范围 |
| B：label3/label7 实际生成 | NOT_FOUND | 两检查点共14生成字段为空；5%总预算未知 | 精确绑定 checkpoint SHA 的生成命令、日志、seed、冻结标签、更新数、选模步数/分数 |
| C：四项历史 import 绑定 | PARTIAL | 导入声明及现有/后期副本 SHA 已知；目标运行历史绑定6条均 NOT_FOUND | 目标运行当时的依赖副本或 loaded __file__+SHA/commit |

> 作者最新确认（2026-10-09T23:58:36.745464+08:00）：**DC-Net / ACDC 10%：HD95＝2.28、ASD＝0.68，单位grid（表2口径）**。旧V19_4的1.28／0.38仅作历史版本；两组数值之间的待核实及替换建议已按作者确认撤回。**论文采用值确认与历史运行来源绑定分开：该行run/checkpoint/split/evaluator绑定仍为NOT_FOUND，未补出新原始精度。**本次只更新记录，不训练、不重评价、不改原始实验产物。

## A：范围、版本和实际评价链

清单原路径 `Table_2_3_4_Row_Checklist_20261009.txt`，SHA256 `203c1bcaad9b9b659626c96dfaa5ee93f4e09ec4531768bc85aaf3c2a7d00f1e`。头部所列 PDF SHA 是清单作者声明；本轮未取得/重新解析该 PDF，不将其写成独立 PDF 校验。

`TABLE2_3_4_73_ROW_INDEX.csv` 保留清单显示值与行号；`ROW_FIELD_STATUS.csv` 分 run、checkpoint SHA、test/train/val split、验证选点、评价版本/实际 driver/metric、原指标、单位、spacing/轴/网格、聚合、空例/非有限值及后处理登记。20/73 指已有**行映射**，并不意味着20行所有历史协议字段均已闭合。53缺口为22+11+20；4条未入表 COVID HiEn/PMEUL、5条补实验均不计入73的分子。全部比较沿用作者确认“复现”，不创建 local/published 分类，不由取整分数倒推出 run。

ROW06 原 receipt：`final_report/three_innovation_postfull_test_v22_1/raw_test/PROMISE20_FINAL3/TEST_RESULT_FULL_PRECISION.json`，SHA `2c3c1e8e220d1fe8ef2026bcfb2dacc95a89ce9c0352c12fcceb63230a12182b`；原 evaluator `e3380e9c4f1fa2853118a6f7437e61c634f29e2a8b8fd80c110b4e6c3cbc7fa1`。Sept4 receipt：`external_comparators/2026_reproduction/paper_blocker_remediation_20260904/outputs/promise20_final3_canonical_reeval/EVALUATION_RECEIPT.json`，SHA `42c06caaa2187841960ca1e26b8655b5c35598cdd093c5444e17fe87cfaa802d`；Sept7 归档副本字节一致。新 evaluator `2a20e429f54942c5392c57d1469a4edc1d0d93e144de4c3403c4d70407986ad6`；保留旧 SHA，不覆盖成新 SHA。版本链与数值复算见 `PROMISE_FULL_EVALUATOR_VERSION_CHAIN.csv`、`PROMISE_RECEIPT_COMPARISON.csv`，各 receipt 的确切引用行见 CSV 和 `SOURCE_INDEX.csv`。

两版本 checkpoint SHA `89c670b16b9a5b1da3bdd8f8c19094424fceef84398773d0ec27246475a48f2c`，test manifest SHA `c981d3a78c3e4bd817438edd159e77361d5dc7f88493f206ccdbfb388f6e35fb` 均一致；验证选点仍是24400步、Dice0.8912。原评价时间 `2026-08-22T10:22:36.555516+00:00`；既有新评价完成 `2026-09-04T08:52:59.505765+00:00`。四原精度：Dice0.8386401640018327、IoU0.7237411265148253、HD95 2.9863295043764113 grid、ASD1.4106984188286347 grid；逐项新−旧＝0.0。receipt 的 training_started、optimizer_created、backward_called、checkpoint_reselection_after_test 均 false；completion 的 optimizer_step_called 亦 false。**这些是 Sept4 已有记录，本轮没有重新评价。**

实际新 driver 为 `canonical_promise20_reeval.py`，调用 `tools/v22_1_postfull_test_launcher.py::promise_run`。源码172–225行：二值前景、原 staged 网格病例重建，无物理 spacing、10病例普通均值；HD95用 MedPy binary.hd95，ASD用预测到GT单向 binary.asd；双空Dice/IoU为0、距离0，单空为网格对角线距离；无后处理。空例定义来自代码，不能据此声称历史患者出现了空例。

COVID 26个主表位置的修正覆盖见 `COVID_MAIN_TABLE_CORRECTION_COVERAGE.csv`。本轮依据已有 `revision_runs_20261006/covid_metrics_before_after_macro.csv` 仅摘录4个 `spacing_only_derived_center_grid` 原记录（源行6、9、16、18，原标签保留），没有调用距离函数：β-FFT5=35.761468219124396，β-FFT10=23.022861570142222；原标签COVID5_FINAL3=33.56438112527303、COVID10_FINAL3=18.957128805392596 mm。它们是本轮用户指定登记的**另一个既有 spacing-only 结果版本**，固定预测/GT、只修正XY、Z不变；不是 native-grid 重评价，Dice/IoU原记录逐字一致。

**论文采用的主 Full HD95 仍为COVID5% 22.79 mm、10% 11.88 mm。将论文主 Full 改为33.56/18.96的建议仍标记“结果对应关系混淆，已按作者确认撤回”，不再列作必须修改项；上述另版数据不替换或重绑定论文主结果。** 原材料和全部旧版本保留。

UCAD 两条实际 wrapper `reproductions/UCAD_2026/scripts/eval_ucad_with_frozen_metrics.py:113–130`，直接把 derived NPZ spacing传给 `covid19_volume_evaluator.py`；未修正XY。HiEn/PMEUL 的实际 driver 是 `adapters/evaluate.py`，经 `adapters/common/metrics.py` 和 `adapters/common/data.py`；receipt中canonical evaluator是**参考字段**，不能说实际运行了该参考入口。源 SHA、出处/行号与六条旧结果见 `COVID_EXTERNAL_ACTUAL_PROTOCOL.csv`、`SOURCE_INDEX.csv`。

UCAD 用XY切片沿axis2堆叠；HiEn/PMEUL先构成Z,XY，评价前moveaxis(0,-1)转XY,Z，spacing对应三轴。二者旧流程都直接沿用 derived spacing，没有XY resize-spacing修正。COVID前景二值，HD95是拼接双向表面距离的95百分位；ASD是同一拼接距离的均值，四病例等权普通均值，无后处理。双空Dice/IoU=1、距离0；单空Dice/IoU=0、距离数组[inf]，percentile可能非有限；普通mean不剔除非有限值、不改成nanmean。沿用既有 `NONFINITE_RECORDS.csv` 的已审查范围0条非有限记录；它不覆盖53缺失行，也不能由有限距离反推所有掩码非空。仅解析这些定义，本轮不生成新体距离。UCAD及四条外部方法没有找到既有 spacing-only结果，保持空值而不比例推算。既有四患者几何表与Oct6 receipt已随包提供；receipt自身只声明48条病例记录，不强称覆盖后来所有macro行。

当前清单未纳入的四条 COVID HiEn/PMEUL，旧评价receipt已完成；既有协议规定冻结划分和医学适配，但没有找到作者对当前主表的既定纳入/排除原因。`COVID_HIEN_PMEUL_INCLUSION_AUDIT.csv`将“当前排除”VERIFIED、“预定纳入原因”NOT_FOUND分别记录，旧助手判断和分数不用于推出作者选表原因。

DC-Net ACDC10%的论文采用值已由作者最新确认：**HD95=2.28、ASD=0.68，单位grid（表2）**。旧V19_4的1.28/0.38只作历史版本；数值差异的待核实和替换建议**WITHDRAWN，已按作者确认撤回**。`A_EXISTING_RECORD_CONFLICT.csv`保留双方历史路径/SHA/行号，并将论文数值确认记为VERIFIED；该行历史run/checkpoint/split/evaluator来源仍为NOT_FOUND，不能用数值确认关闭来源绑定，也未填入原始未舍入精度。

## B：只缺预训练生成证据

label3 SHA `3486fd3aecf5f30e77e7e15bec0841f56c87db6246a6dc50cc4dc6f2ee2087e0`；label7 SHA `477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3`。沿用既有CPU解析元数据，两者仅net/opt，SGD无step；本轮未加载/复制权重。`B_PRETRAIN_GENERATION_FIELDS.csv` 的14个字段NOT_FOUND：实际生成命令/日志/seed/冻结标注清单/真实更新数/验证选点步数/分数。

June18恢复报告只证明README入口及源码默认10000步，且本地主入口预训练调用曾注释；June24日志只证明后续SSL加载label7及Namespace默认pre_iterations10000。这些不等于这两个检查点真的训练10000步，也不能转移后续SSL seed到预训练。label3后续seed审计同样不是生成证明。两个目录当前只有权重，无生成日志。原始引用行和SHA见索引及小型摘录。

ACDC5/10每学生30000次半监督更新、两学生/教师初始化、SGD继承、恒定LR0.01为**已确认事项**，本轮仅沿用。`B_BUDGET_BOUNDARY.csv` 将未知预训练和已知SSL分开；5%总预算仍空/未知，不用derived config补出总数，不再把30000半监督预算列待办。

## C：声明、入口副本与历史依赖绑定分开

`C_HISTORICAL_IMPORT_BINDINGS.csv`有6条目标运行×依赖记录，归属4个组件：主Full WeakStrongAugment、DiceLoss；COVID_FINAL3 adapter、FFT helper（5%和10%分别记录）。入口导入与调用VERIFIED，现有/Sept7副本或早期家族SHA可读；但本轮没有找到这些依赖针对目标历史运行的副本或loaded __file__+SHA/commit，历史绑定字段NOT_FOUND，整体PARTIAL。

主Full effective_source.py:21/24/74/173–174/276，证明导入、调用及仅保存入口自身；不证明所导入模块当时的字节。COVID归档入口:29/2438/2459，lineage2311–2329只保存训练核心、预训练权重和manifest身份，未覆盖adapter/FFT导入SHA。Sept7 `RUN_SOURCE_MAP.tsv:3–4`明确区分入口已归档与当前依赖副本，不等于FINAL3当时依赖存档。早期COVID家族身份SHA不可自动转移到FINAL3。

CTW启动前SHA链与COVID5/10共同归档实际入口已核验，仍是关闭项；不再报告“找不到COVID训练入口”。闭合C只需目标历史运行对应依赖副本或加载身份记录，不需要运行模型或重训。

## 可采用的论文/作者说明句子

- “论文主Full（论文R1）对应原始X03/Full_main，在ACDC10%取得Dice90.35%；89.98%的独立补实验命名为‘更换混合控制信号’，为E4、C1、R2固定92×92的配对参照。”
- “PROMISE12 Full的既有canonical复核使用相同验证选定检查点及测试清单，Dice、IoU、HD95和ASD与原receipt完全一致；两版评价器SHA分别保留。”
- “ACDC5%和10%半监督阶段每个学生均完成30000次更新，学习率为0.01；现存记录不足以确定label3/label7预训练的实际更新数及5%预训练加半监督总预算。”
- “COVID论文主Full的HD95沿用作者确认的5%22.79mm和10%11.88mm；其他既有spacing-only版本分别登记，不代替论文采用结果。”
- 内部来源说明可写：“当前主表73个位置中20个已绑定既有运行资料，其余53个尚需补来源链；这不改变作者已确认的复现身份。” 此句适合交接/审查记录，未经作者决定不当作论文实验结论。

ACDC测试未参与候选方法选择，沿用作者更正；旧硬编码字段、助手总结和rank标签均不构成新的选择事件。本轮未发现新的原始选择事件记录，也未再次由这些字段推出选择因果。

## 交付与未闭合项

下一步只需要现有历史材料：53行对应receipt/冻结协议；label3/label7生成日志与选模链；四项目标import版本；既有外部COVID修正和预先表格纳入规则（如存在）。未找到的修正保持NOT_FOUND，本轮不授权体距离重评价。具体状态与关闭条件逐字段见CSV；所有来源的完整SHA与引用行见 `SOURCE_INDEX.csv`。小包不含影像、预测数组、权重、完整训练日志、凭据或无关项目资料。
