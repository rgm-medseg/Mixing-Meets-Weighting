# REVIEW_CLOSURE_EVIDENCE

> 作者最新确认（2026-10-09T23:58:36.745464+08:00）：**DC-Net / ACDC 10%：HD95＝2.28、ASD＝0.68，单位grid（表2口径）**。旧V19_4的1.28／0.38仅作历史版本；两组数值之间的待核实及替换建议已按作者确认撤回。**论文采用值确认与历史运行来源绑定分开：该行run/checkpoint/split/evaluator绑定仍为NOT_FOUND，未补出新原始精度。**本次只更新记录，不训练、不重评价、不改原始实验产物。

> 作者确认更正（2026-10-09T23:03:41.867783+08:00）：论文主 Full 的 COVID HD95 为 **5%：22.79 mm；10%：11.88 mm**。此前将其替换为33.56／18.96 mm的建议：**“结果对应关系混淆，已按作者确认撤回”**，不再列为论文必须修改项，也不将另一组数值归入主 Full。原始实验产物及旧证据包保留。本次只更正记录，未训练、未重评价、未重选 checkpoint。论文 R1／主 Full Dice90.35%；“更换混合控制信号”Dice89.98%单独保留。

核查完成时间：2026-10-09T20:05:54.137715+08:00；主机 n1。
项目根目录：`/home/renguanming/jupyterlab/rgm/beta-FFT-main`。本轮仅核对 A–D 四项缺口，CPU 读取既有检查点、解析日志/CSV、复算现有统计与十例 PROMISE 保存预测。未构建分割模型、未做 GPU 前向，未启动/恢复训练、未重选检查点、未修改原源码、数据、检查点或历史结果。E3 保持取消；不新增 seed。
适用 AGENTS.md：已检查项目祖先路径及所给证据子树，未找到；记录于 `AUDIT_SCOPE.json`。

论文题目：**Mixing Meets Weighting: Confidence-Modulated Semi-Supervised Medical Image Segmentation**。统一模块名 TRC — Token-Guided Residual Conditioning、TCSMix — Teacher-Confidence-Scaled Mixing、CTW — Confidence-Transition Weighting。保留 ACDC、PROMISE12、COVID-19。

## 四项关闭结论

| 缺口 | 状态 | 能否关闭 | 仍缺的具体证据 |
| --- | --- | --- | --- |
| A | PARTIAL | NO_ALL_ROWS; YES_LOCATED_METRICS_AND_GEOMETRY | Current manuscript Table2-4 full row roster; some early evaluator/import SHA provenance; corrected distances for HiEn/PMEUL/UCAD existing runs |
| B | PARTIAL | YES_SSL_5_10; NO_PRETRAIN_GENERATION_OR_TOTAL_5 | Actual pretraining generation commands/logs/seed/selection step/frozen label subset/update budget; 5% combined pretrain+SSL total |
| C | PARTIAL | YES_CTW_AND_COVID_ENTRY; NO_ALL_IMPORTED_DEPENDENCIES | Main Full-specific historical augment/DiceLoss runtime hash; FINAL3-specific historical COVID adapter/FFT import receipt/archive |
| D | VERIFIED | YES | 无 |

`VERIFIED` 是列明范围内有现存证据及复核；`PARTIAL` 不代表实验失败；`NOT_FOUND` 仅表示本轮在授权项目记录链中未定位到该项。新的 CPU 复算与历史原始评价分开标注，未把复算当历史执行记录。完整 SHA、原路径、打包路径见 `SOURCE_INDEX.csv`；二进制权重只列 SHA 和 CPU 元数据，不进入压缩包。

## A — 表 2–4 的行映射、真实协议与评价口径：PARTIAL

### A1. 已定位行：VERIFIED；最终逐行覆盖：PARTIAL

`TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv` 一行一个实际结果，共 **24 条主表候选记录**：β-FFT/完整方法的 10 个任务预算行、HiEn/PMEUL 的 10 个行、UCAD 的 4 个行；另列 **5 条补实验**。每行分别登记 run、checkpoint、实际/记录 SHA、split manifest、验证选点、评价来源、真实评价 driver 与 metric implementation、单位、聚合和协议缺口。没有新增 local/published 分类；均沿用作者确认的复现身份。

服务器 v23 文件名是 TABLE1 ACDC、TABLE2 PROMISE、TABLE3 COVID、TABLE4 八组合，并不是当前论文表 2–4 的逐行排版索引。当前论文最终逐行名单未在本轮引用链中定位到，不能用这 24 行宣布“表 2–4 所有行闭合”，也没有凭空添加其他方法。完整八组合原 TSV 同时入包，保留每个真实运行。

主 Full / 论文 R1 绑定原 X03 检查点29600，**不是**服务器 R1_v2_B。原成绩和补实验配对关系如下（距离均 grid）：

| 论文名称 | 服务器历史编号 | Dice% | IoU% | HD95 | ASD | 比较参照/范围 |
| --- | --- | --- | --- | --- | --- | --- |
| 完整方法 / 论文 R1 | V21_3_X03_CONTOUR_RELIABILITY_WEIGHT | 90.35 | 82.93 | 1.423 | 0.429 | 论文主结果/原八组合 |
| 更换混合控制信号 | R1_v2_B | 89.98 | 82.34 | 2.942 | 0.769 | 该配置自身：模块替代配对控制 |
| 固定混合区域（92×92） | R2_fixed92 | 90.20 | 82.71 | 1.880 | 0.509 | 更换混合控制信号，89.98% |
| 均匀权重替换 CTW | C1_CTW_mean_uniform_v2B | 89.65 | 81.85 | 1.548 | 0.495 | 更换混合控制信号，89.98% |
| 普通残差替换 TRC | E4_plain_residual_v2B | 89.81 | 82.08 | 1.675 | 0.507 | 更换混合控制信号，89.98% |
| AdaMix-CT 共同协议比较 | E5_AdaMix_CT_common_protocol | 88.37 | 79.78 | 2.014 | 0.836 | 完整方法（外部方法比较） |

E4、C1、R2 对照均使用 89.98% 更换混合控制信号配置；不能把这条参照换成主 Full 90.35% 后计算模块单因素收益。R2 结果只支持该控制信号配置的动态控制与 fixed92 比较。E5 是整个方法比较，不能将整个差值归因于 TCSMix。

关键原始 receipt 的实际路径、SHA 与引用行：

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/final_report/acdc10_top6_all_test_v21_5/raw_test/X03/TEST_RECEIPT.json`，行 12,13,15,17,18,19；SHA256 `73c1c4ee5b5033d8c24f0e151fc737605d01924f65ed728481057ad02664927c`（S0019）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/final_report/three_innovation_postfull_test_v22_1/raw_test/ACDC5_FINAL3/TEST_RESULT_FULL_PRECISION.json`，行 4,7,8,11,12,42,43；SHA256 `c9c2077bee834581d17257caa74054c99950d99f8086a2d8721b976655540454`（S0010）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/final_report/three_innovation_postfull_test_v22_1/raw_test/COVID5_FINAL3/TEST_COMPLETION_RECEIPT.json`，行 4,7,8,11,12,30,31；SHA256 `2ee30f2e5b7703601e20946af3c164175d7b783c7bf339bff30864bf35fc5a5c`（S0229）

### A2. 已有病例记录复算与空例：VERIFIED

`AGGREGATE_RECOMPUTATION.csv` 对 **116 项指标均值**作复算，全部与相应历史汇总在 1e-12 内一致；没有重新推理或选择 checkpoint。`NONFINITE_RECORDS.csv` 为已检查原始数值的非有限值清单，本轮为 **0 条**。这只覆盖实际列明的病例表，不能泛指论文尚未映射的所有方法。

COVID两条canonical baseline额外登记了八条已存在的2026-10-06病例legacy重放记录，其`before_*`均值与原receipt逐项一致，检查点SHA一致、四例Dice均大于0、数值有限。来源是现有保存预测的历史重评价CSV，**不是新找回的原训练期病例表**；筛选run/spacing-only及`before_*`列条件写入映射表和 `COVID_BASELINE_CASE_REPLAY_VERIFICATION.csv`。

PMEUL ACDC5 的 120 条类别记录中有 **18 条 Dice=0**。读取既有预测 NPZ 与对应 GT label 数组后，确认 **17 条预测为空且 GT 非空**，分布于 9 个帧病例；另1条 patient024_frame02 / LV 双方非空、零重叠。17条的 HD95 与 ASD 均严格等于原数组 `norm(shape)` 惩罚，全部计入普通均值，没有删除或用有限值替换。原始来源、mask SHA、像素数、形状和逐行对照见 `EMPTY_CASE_COUNTS_FROM_SAVED_MASKS.csv` 与 `SAVED_MASK_INPUT_SHA256.csv`；包内不含这些数组。

ACDC 主 Full/β-FFT、模块补实验、HiEn/PMEUL：前景 RV=1、MYO=2、LV=3；每个 ED/ES 体分别算三类，40帧体×3类等权平均；20患者表另作两帧均值。2D nearest/order0 到256²，预测回到 H5 原平面，Z不重采样；不做推理后 LCC。MedPy HD95 为合并双方向表面距离的95百分位；**ASD 是 prediction→GT 单方向表面距离均值**。两者不传 voxel spacing，因此为重建数组 **grid** 距离。双空四指标0，单空 Dice/IoU0、距离 `norm(shape)`。教师训练时 LCC 与推理后处理不是同一流程。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tools/v22_0_acdc_test_frozen_checkpoint.py`，行 35,44,45,47,49,54,62,68,73,84；SHA256 `5958b1a7514415f67beb184676c649867f6aeb15cce8d3ee5ee3fe5517a798f4`（S0009）

### A3. PROMISE12 基线精度和实际规则补回：VERIFIED 数值；PARTIAL 历史依赖绑定

找到 `audits/PROMISE20_FROZEN_BEST_TEST_IDENTITY_AUDIT.json`，并对其对应 Case40–49 的十份既有预测做 **3.353 秒 CPU 复算**：

Dice **0.8110316082469234**；IoU **0.6843582268075924**；HD95 **3.669142478708649 grid**；ASD **1.5342924140259098 grid**。

四项与历史精度记录逐项差 **0.0**。原舍入 TSV 0.8110/0.6844/3.67/1.53 保留；本轮更新新索引中的精度来源，不覆盖旧表。新复算 CSV 和 receipt 明确是当前 CPU 复算，历史精度由现有 audit JSON 支持，未冒充训练期完整 test 行。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/audits/PROMISE20_FROZEN_BEST_TEST_IDENTITY_AUDIT.json`，行 3,12,14,17,19,21,22；SHA256 `fe6e84fa3a94a3067fd00fa73c3da87cf4360bee63656dfc512309d2b001dbf4`（S0025）

该基线实际定位到的评价实现为 `audits/run_v19_5_promise20_and_mscmr_final_r3.py:119–176`：对现存 staged `(Z,224,224)` 预测/GT计算前景，10病例等权宏平均，HD95 同为 MedPy 双方向合并95百分位，ASD单方向，grid，无额外重采样/后处理。**基线双空 Dice/IoU=1，距离0；单空 Dice/IoU=0、距离 inf**。Full 的 v22.1 程序则双空 Dice/IoU=0、单空网格对角线惩罚；HiEn/PMEUL 共用 adapter 为后一规则。不能将所有 PROMISE 行笼统写成完全相同空例协议。十例基线的保存预测与 GT 均非空，故该空例差异未改变这十例结果。历史基线执行时导入依赖 SHA 仍未定位到，原入口/依赖版本绑定不能由当前复算替代。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/audits/run_v19_5_promise20_and_mscmr_final_r3.py`，行 119,124,126,128,136,151,159,164,179；SHA256 `a33541f1a332cd952fa09882ee2ab59a715b18d3fccedc15f38ab939dfe47e68`（S0023）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tools/v22_1_postfull_test_launcher.py`，行 172,183,186,189,193,200,208,219；SHA256 `2a20e429f54942c5392c57d1469a4edc1d0d93e144de4c3403c4d70407986ad6`（S0031）

### A4. COVID spacing、轴顺序、网格、单位：VERIFIED 既有几何/修正范围

原 canonical NIfTI 为 XYZ；slice 存 XY，dataset/保存预测体为 **ZXY**；原评价器按 `axis=2` 将 slice 堆回 **XYZ**，spacing `(sx,sy,sz)` 与 XYZ 对应，轴顺序本身正确。平面512²或630²到256²，但旧 NPZ仍写原 sx/sy，**未给缩小后的 XY 网格修正 spacing**；Z切片逐一保留，不能同时缩放三轴。这一段登记历史实现及几何资料；其结果对应关系不得直接用于替换论文主Full。论文主Full HD95单位与采用值按上方作者确认记录。

| 患者 | canonical XYZ | 评价 XYZ | 原 spacing XYZ | 修正 spacing XYZ |
| --- | --- | --- | --- | --- |
| coronacases_002 | [512, 512, 200] | [256, 256, 200] | [0.68359375, 0.68359375, 1.5] | [1.3698682598039216, 1.3698682598039216, 1.5] |
| coronacases_009 | [512, 512, 256] | [256, 256, 256] | [0.6835939884185791, 0.6835939884185791, 1.0] | [1.3698687375760545, 1.3698687375760545, 1.0] |
| radiopaedia_27_86410_0 | [630, 630, 66] | [256, 256, 66] | [0.6835939884185791, 0.6835939884185791, 6.0] | [1.6861985047658286, 1.6861985047658286, 6.0] |
| radiopaedia_40_86625_0 | [630, 630, 93] | [256, 256, 93] | [0.6835939884185791, 0.6835939884185791, 4.0] | [1.6861985047658286, 1.6861985047658286, 4.0] |

固定原预测、GT和后处理，按 SciPy zoom 默认 `grid_mode=False` 的首末像素中心跨度，sx乘 `(Nx−1)/255`、sy乘 `(Ny−1)/255`，**sz保持原值**。`COVID_GEOMETRY_RECOMPUTATION.csv` 本轮只对四例 spacing 公式复算，误差0；距离值取现有 EDT 实算结果，未通过比例推测。

上表使用XYZ表达同一个物理网格。既有修正CSV同时登记实际函数调用：原评价用XYZ/`(sx,sy,sz)`，部分修正执行直接用保存的ZXY数组/`(sz,sx',sy')`；这只是数组与spacing同时对应的轴重排，未改变预测、GT或重采样。不能将XYZ spacing直接传给ZXY数组；逐例`before_call_array_order`、`after_call_array_order`和`*_call_spacing`原样保留。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_COVID19/covid19_official20_v2.py`，行 80,82,149,154,156,159,162,163,172；SHA256 `c48db1e1f4377dcbae4782a8adc48684b01f05569a03305d63ddb5d3702b1dd5`（S0195）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_COVID19/covid19_volume_evaluator.py`，行 31,40,42,45,53,60,61,65,83,97,102；SHA256 `ff20153cdbbe9e164bbdbd2348f6799daf0180a5503291177a86a5f9034f560b`（S0037）

论文主Full采用值按作者明确确认：

| 标注预算 | 论文主Full HD95 | 单位 |
| --- | --- | --- |
| COVID 5% | 22.79 | mm |
| COVID 10% | 11.88 | mm |

先前把33.56438112527303／18.957128805392596（显示33.56／18.96）关联为主Full必须替换值的结论：**“结果对应关系混淆，已按作者确认撤回”**。这些数值仅保留于历史材料，不再归入论文主Full；本轮不另行指定其真实运行身份。原始CSV及此前证据包不删除、不改写。当前 `COVID_RESULT_VERSION_COVERAGE.csv` 给原误关联的两行加撤回状态和作者采用值，避免后续自动取用。

历史spacing-only统计中的Dice/IoU不变仅描述那些历史记录，不构成主Full成绩替换依据。`native_grid_separate_reevaluation` 另外改变预测重采样及 GT 网格，Dice/IoU会变化，**不能称纯 spacing 修正，也不混入 spacing-only 比较**。逐患者预测/GT来源、数组/hash和旧/新 spacing 见入包的 `covid_metrics_before_after.csv`。现有重评价 receipt 的 legacy最大逐指标差0.0，时间2026-10-06T21:14:54.761639+08:00，geometry修正记录2026-10-06T21:19:54.932521+08:00。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/reports/covid_reevaluation_receipt.json`，行 7,9,10,11,12,13,14；SHA256 `8781df2c4d86d896209013ecabf0ec5ec7e224f5e23c804c437ecc91059e92d4`（S0205）

HiEn/PMEUL 的 receipt 中 `evaluator` 是 canonical参考路径；真实执行链是 `adapters/evaluate.py → common/metrics.py → common/data.py`。这三个文件的当前 SHA 全部与预启动 `FINAL_SOURCE_FREEZE.sha256` 匹配。COVID 路径从 NPZ读取旧 spacing、`moveaxis(ZXY,0,-1)` 回XYZ，直接算双方向 EDT，与上述旧几何问题相同。**HiEn/PMEUL/UCAD 精确运行的已完成 spacing 修正结果未定位到**，`COVID_RESULT_VERSION_COVERAGE.csv` 为其记 NOT_FOUND；本轮没有启动体距离重评价，也没用比例补值。这些历史资料及外部行缺口保持原范围；论文主Full的22.79／11.88 mm已按作者确认，不再列为必须替换或必须重评价事项。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/external_comparators/2026_reproduction/adapters/common/data.py`，行 202,206,210,218；SHA256 `4634f60a6e1267d5d045923c07047c90a869947921c4e308d168861670cd425a`（S0190）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/external_comparators/2026_reproduction/adapters/common/metrics.py`，行 60,70,79,117；SHA256 `7d3402a95f857bb018798c774319a45b4ea0e26018c502b450538d0694f73a71`（S0191）

COVID 为4患者体等权平均、单前景；HD95为双方向表面距离合并后95百分位，**ASD为双方向表面样本合并后的均值**，与ACDC/PROMISE单方向ASD不同。双空 Dice/IoU1、HD95/ASD0。单空代码先返回 `[inf]`；当前 NumPy **2.2.6** 的隔离 CPU 检查得到 HD95 **NaN**、ASD **inf**（percentile 的非有限插值），不是两个有限惩罚。程序用普通mean，没有过滤/插补；若发生会向病例/宏平均传播。该结果仅证明当前纯 metric 函数的行为，不证明历史 NumPy 版本；本轮已查现存 COVID病例记录均无单空/非有限值。详见 `METRIC_EMPTY_EDGE_CPU_CHECK.csv`，未修改评价代码。

### A5. 比较真实协议与选择流程：VERIFIED 所列范围

`PER_RUN_PROTOCOLS.csv` 保留每条配置的完整结构；`PROTOCOL_FACTS_BY_ROW.csv` 提取配置/检查点实际证据，不把 derived config 当观测执行。共同使用绑定的划分和对应评价口径，方法特有网络、初始化、优化器和增强按实际记录保留。

HiEn 在所列配置中使用 ImageNet ResNet34 初始化、3通道复制、训练patch224²、自己的监督/特征一致性损失与教师更新。PMEUL使用自己的MCNet双decoder及损失；具体字段逐run入包，不泛称所有行等参数或等初始化。HiEn/PMEUL formal TEST receipts 明确 `test_used_for_selection:false`、严格加载零missing/unexpected，配置SHA匹配。RUN_MATRIX 中早期 PENDING_GATES 标签不作完成证据。

本轮仅CPU读取这些**已选定**的检查点元数据，未重新验证或选模。HiEn的COVID10%/5%测试检查点分别为step3500/5300、角色均为teacher；ACDC10%/5%及PROMISE20%分别为step10500/19800/25500、角色student。PMEUL对应ACDC10%/5%、COVID10%/5%、PROMISE20%为27200/23400/14200/600/8000。UCAD四份已选检查点的`iter`分别为ACDC5%10600、ACDC10%23000、COVID5%3000、COVID10%15600；其旧receipt未记该字段，本轮元数据和当前SHA不能冒充旧receipt绑定。逐份键、SHA、角色及读取边界见 `EXTERNAL_SELECTED_CHECKPOINT_CPU_METADATA.json`。

E5实际初始化 receipt（2026-10-08T20:27:21.020838+08:00）记录 fresh、seed42、随机初始化、预训练0、无教师；官方 commit `85361fb91cf43a69a158d41dfa9e7ee9fce36db9`。AdamW lr1e-4、weight_decay0.01、betas0.9/0.999、无调度；4流×6=24图，12GT，严格63无标注患者；迭代30000，验证200步，student1验证Dice选点27200。官方CT增强保留颜色扰动/自动对比度/直方图均衡以及随机翻转/旋转，适配器只绑定共同数据来源和划分。与 Full 的预训练来源、SGD、配对弱强增强、EMA及目标不同，比较为完整方法层面。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/formal_E4_E5_20261008/tasks/E5/reports/INITIALIZATION_ACTUAL.json`，行 4,5,6,7,8,11,12；SHA256 `037771309f01a35409c0132825f44386abf2e27cd6491c9c35836232cbf39503`（S0217）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/formal_E4_E5_20261008/tasks/E5/protocol.json`，行 8,18,19,20,25,31；SHA256 `df152afe3ae558f84b0a3bf822aee1d844a8de571bc4c6c84074720f417ed729`（S0164）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/formal_E4_E5_20261008/external/AdaMix/exp_acdc/data/dataset.py`，行 76,78,79,80,87,88,89；SHA256 `d2f8ab5765e6798a7bf54d35e51d87f33c97b67899a85f6aa04bdcd7a5a179f8`（S0219）

作者已更正 ACDC 测试未参与候选方法选择。本轮沿用 **验证集选 checkpoint、随后固定 checkpoint测试** 的实际流程，不根据旧rank/成绩排序推断选择因果。保留两个历史文字冲突：X03原TEST receipt（2026-08-19T12:55:05.155768+00:00）写 `"TEST_SELECTION_EXPOSURE": true`；其生成器 `tools/v21_5_acdc10_test_frozen_checkpoint.py:203` **固定写 True**，并非方法选择事件追踪。20260904助手派生汇总写“CLASS_B material TEST contamination and evaluator mismatch”；它也是判断标签，不是新的原始选择事件。相反，正式启动 readiness、Full completion 记录 test_access_count0，入口的 db_test 实际split=val，真正test另由冻结评价入口消费。原字段、身份、时间、原句与生成位置均保留于 `HISTORICAL_ANNOTATION_CONFLICTS.csv`；这些标签不足以推翻作者更正，本轮没有发现新的具体选择事件与其冲突。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/full_completion_report.json`，行 12,13,17；SHA256 `0afd50577a4ad19f721374dae04982b578eb202556864558024f1b84918164d0`（S0214）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tools/v21_5_acdc10_test_frozen_checkpoint.py`，行 145,203,207；SHA256 `f49d5e5e284ffcc5093e337c3826d58a3e27b789f13d398d4588185cfab8734e`（S0018）

## B — ACDC 5%/10% 预训练、初始化与更新预算：PARTIAL

### B1. 现有预训练权重和继承状态：VERIFIED；实际生成历史：NOT_FOUND

label3、label7目录目前各只有 `unet_best_model.pth`。CPU读取均仅含 `net,opt`，无step、seed、args、实际标签清单或选模指标；SGD state都有82个 momentum_buffer、142个参数引用，lr0.01/momentum0.9/weight_decay1e-4；SGD缓冲没有更新计数，不能从内容推回10000步或选模时刻。

label3 SHA256：`3486fd3aecf5f30e77e7e15bec0841f56c87db6246a6dc50cc4dc6f2ee2087e0`。
label7 SHA256：`477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3`。
实际路径分别为 `/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label3/unet_best_model.pth` 与 `/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label7/unet_best_model.pth`；本轮CPU元数据与SHA见 `CHECKPOINT_CPU_METADATA.json`，不导出权重。

`beta_fft_label3.py` / `beta_fft_label7.py::pre_train` 的代码目标pre_iterations10000、验证200步、最佳验证Dice保存net/opt，只能证明**代码定义/目标**；两者当前 `__main__` 中 pre_train 均注释，label7脚本还保留labelnum3及label3路径，不能以脚本文件名证明实际生成协议。没有定位到这两份权重对应的原始启动命令、生成log、实际seed、选模步数/分数、真实更新预算及冻结标签子集。因此这些字段保持 NA/NOT_FOUND，不报40000总预算。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/beta_fft_label3.py`，行 31,192,195,286,288,558,560；SHA256 `1a80f108684333f9c649b95696b8eefbc3303594ced1da61808d7bb33fca9c29`（S0193）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/beta_fft_label7.py`，行 53,54,63,275,278,667,681；SHA256 `cbdc17426993278c71cfa897f2d5377c21fcb6da5e983d3fe1cc5d804e1ee7dc`（S0194）

`audit/beta_fft_clean_supervised_pretrain_final_config.json` 明确 `training_started:false`，输出是另一套 clean_label12_seed42；这是后来准备配置，不是label3/label7生成记录。20260618 clean label7审计针对clean split复用，不能将其“不适用clean复用”扩大为主Full无效。

### B2. 半监督实际预算、学生/教师/SGD、LR：VERIFIED

5% Full：labelnum3、seed1337，last global_step30000，student1验证选点28400、Dice0.8892521589382639。
10%主Full：labelnum7、seed42，last global_step30000，student1验证选点29600、Dice0.8999080952059592。

两份 dynamics CSV各30001行，但只有 step1–30000这 **30000个不同更新步**；最后重复的step30000是 `effective_source.py:549` 写入的零loss检查点终结标记，不是第30001次更新。两条轨迹的loss/grad有限性记录均通过。每个循环各执行optimizer1.step、optimizer2.step，故证据支持 **每学生30000次半监督更新**、30000次student1→EMA更新，不能把两学生计数相加当一个60000-step训练预算；这是“完整轨迹＋入口＋checkpoint”证据，不是独立optimizer调用计数器。

last 含 model_1、model_2、ema_model、两个完整SGD、epoch/global_step、best_metrics、random_state、args；EMA=0.99。新run加载label3或label7 `net` 到两学生和教师，TRC新参数按机制初始化；两学生继承pretrain opt的base momentum/超参数，TRC参数单列组。保存状态各optimizer90 momentum_buffers，group0参数引用142、group1 8。没有丢弃SGD历史状态后假称完全从头初始化。

原Full及ACDC5有效入口**没有LR schedule更新语句**；pretrain opt记录lr0.01，base组继承该值、router组由args.base_lr0.01构建，两份last各组仍lr0.01。该运行证据对应 **恒定0.01**，不能沿用其他入口的poly说法。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/effective_source.py`，行 21,22,276,277,279,291,292,315,316,317,318,320,321,443,444,445,461,549；SHA256 `d3959d5baf54727db740f3ece79608292b396d3f0f49b7b97743398bda42ad00`（S0170）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/v14_mask_adamix_components.py`，行 159,169,176,180,182,185,187,193；SHA256 `75754fa75e6fd6fc01c97682262895998799c7ce3c0bc2dcbe058ef4c63dcaee`（S0175）

ACDC5 derived config中“30000 inferred from run family”原句保留；本轮确认30000来自独立真实log、连续轨迹和CPU last读取，不引用那句推断作为运行预算。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3/train.log`，行 1,3,61219,61224；SHA256 `a6beb2b1262c8750a52721eb2ed6731157190d5f5a3d8f7fe02c4ccb442089b1`（S0174）

**5%总预算仍 NA**：已经确认半监督30000，但label3实际预训练总更新数未找回；写成“pretrain未知＋self-train30000”，不能以默认10000补成总40000。

### B3. 标签子集：VERIFIED 当前SSL重建；PARTIAL 历史绑定；NOT_FOUND 预训练子集

当前实际引用train_slices.list共1312切片/70患者，val20ED/ES帧/10患者，test40帧/20患者。入口patients_to_slices固定5%首68片、10%首136片。按该现存manifest CPU重建：

5%：patient038、patient050、patient099（68片）；10%：patient021、patient038、patient049、patient050、patient058、patient099、patient100（136片）。逐slice索引和当前manifest SHA见 `ACDC_SSL_LABEL_PREFIX_RECONSTRUCTION.csv`。这是**现存SSL入口/manifest的重建**；原Full checkpoint没有保存train manifest hash，不能由此证明最初pretrain也用这套标签清单；历史生成子集仍NOT_FOUND。

## C — 主 Full 增强/DiceLoss/CTW 与 COVID 适配器/FFT 的历史版本：PARTIAL

### C1. 归档入口与CTW机制：VERIFIED

原主Full和ACDC5有效入口当前SHA一致：`d3959d5baf54727db740f3ece79608292b396d3f0f49b7b97743398bda42ad00`。它证明运行目录生成入口、参数和调用位置，**不自动证明所有 import 内容**。CTW组件有20260817预启动冻结SHA，当前文件与该值完全相同，实际training log持续记录X03 loss/weighted loss端点：

CTW：`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/v21_3_acdc10_third_innovation_components.py`。
SHA256 `f28abc48fedff330f9aa7470072ebc16a607c6c59fd14942c2adb111b8ce748a`。

具体计算：q为两学生detach概率的最大类置信度均值；对q作水平/垂直绝对后向差分，首位置0；逐图min–max、分母下限1e-6得到r；w=0.25+r，范围[0.25,1.25]；误差是p1与detach(p2)的类别平均概率L1，空间batch取mean，外系数0.05，只由这项更新student1。不是教师置信度梯度图、不是全边界GT监督。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/v21_3_acdc10_third_innovation_components.py`，行 28,32,35,38,39,67,68,99,100,101；SHA256 `f28abc48fedff330f9aa7470072ebc16a607c6c59fd14942c2adb111b8ce748a`（S0177）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/audits/acdc10_third_innovation_codex_prep_top6_v21_3/ROUND1_TOP6_CONTROLLER_DRY_CONSUMPTION.json`，行 3,40,79,88,89,92,93；SHA256 `e526f294cb02a2ef382e7dddc9985b72c3d0d9b24b5ed5c1033a09255ff4d572`（S0178）

### C2. 增强与DiceLoss静态定义：VERIFIED；原 Full 导入绑定：PARTIAL

主入口import `dataloaders.dataset.WeakStrongAugment` 与 `utils.losses.DiceLoss`。当前二者与20260907已有证据快照逐byte相同，但这份依赖快照晚于原Full运行；未定位到原Full启动时针对两文件的模块SHA/依赖副本/loaded __file__+SHA收据。

现存2D增强：weak共享图/标签随机90°旋转及翻转，或[-20,20)角度order0旋转；图与标签nearest到256²。strong在该共享几何上加cutout(p0.5，面积比例0.02–0.4)和ColorJitter(0.8,0.8,0.8,0.2)；cutout区标签写背景0。不能凭args.magnitude6/s_param6推断该路径用了另一套RandAugment。

现存DiceLoss对batch+空间维求和，按四类含背景等权平均；分母为 `sum(p²)+sum(y²)`，非mask smooth1e-5、mask smooth1e-10。mask先按类别复制，分别施加到交集和两分母项；不能把它写成线性分母Dice或只前景Dice。训练DiceLoss与测试Dice统计定义不同。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/dataloaders/dataset.py`，行 218,229,245,250,254,278,440,443,446,449,450；SHA256 `0ee048bd93cda8cbea35c06755e855d9a1540a57077efa6b541b69945b22bd58`（S0179）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/utils/losses.py`，行 156,169,171,173,174,183,186,188,193,203,208,211；SHA256 `06f3ef39004df0a70e116c783297ae2012ca7cd61de682fcdf964bbd32fcd480`（S0181）

SHA分别 `0ee048bd93cda8cbea35c06755e855d9a1540a57077efa6b541b69945b22bd58` 和 `06f3ef39004df0a70e116c783297ae2012ca7cd61de682fcdf964bbd32fcd480`；`SOURCE_VERSION_BINDING.csv` 区分静态定义、晚期依赖归档和原run绑定。论文可准确描述“可核查实现”，但不要写“所有历史依赖已完全冻结证明”。

### C3. COVID5/10实际入口：VERIFIED；FINAL3 导入依赖：PARTIAL

已读取并核验两份实际归档入口：

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/train_promise12_bcp_fft.py`；
`/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3/self_train/train_promise12_bcp_fft.py`。

二者SHA均 **e1006ebe9731d7665a886dc5f897f130c5818c132fa575cc16c9ea845cb27386**，与20260819正式启动manifest相同。不是“找不到COVID训练入口”。入口导入COVID dataset adapter，并从 `utils.displacement` 调用逐对应样本FFT；归档入口自身self_train lineage只写training_core/pretrain/manifest SHA，没有写所有导入模块SHA。

还找回早期COVID seed42家族的 `audits/COVID19_FORMAL_RUN_BOUND_IDENTITY.json`：adapter SHA `a6031696e9a63888c385cfe735bc010e73f5bad5cdf80e1cb6ed264e812d4ba1`，FFT helper SHA `7e807dfc8944ed0d683110e4077bb3d355021b5f7b77ab08c9b989176836ff96`，均与当前文件一致。20260812 seed1337 critical source清单也包含adapter/hash；但**早期family的绑定不能自动移给COVID5/10_FINAL3**，尚缺FINAL3当时的依赖副本或runtime imported-file/hash收据。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/audits/COVID19_FORMAL_RUN_BOUND_IDENTITY.json`，行 4,5,25,27,35；SHA256 `897a5b8701bc03a8376fb7dd6c40e0b4a1ad364dcc293ce6f553a23b2e87dd45`（S0187）

当前所附adapter强视图为 `image_batch.clone()`、label也clone；在共用BCP mask及配对索引下两份mix输入重复。FFT helper按同一i交换对应样本的低频部分，阈值30；若两输入相同，则交换项相消，不能声称当前这条链产生独立频域扰动。入口随后仍让两学生在该额外输入上做互监督。**静态行为已确认，早期绑定已有SHA；将其确定归给FINAL3全历史仍受上述import版本缺口限制**。不修改算法、不重训、不否定其他已核实模块。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_COVID19/covid19_dataset_adapter.py`，行 167,174,176,177,178,179；SHA256 `a6031696e9a63888c385cfe735bc010e73f5bad5cdf80e1cb6ed264e812d4ba1`（S0186）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_promise12/utils/displacement.py`，行 32,50,52,53,59,60,63,64,73；SHA256 `7e807dfc8944ed0d683110e4077bb3d355021b5f7b77ab08c9b989176836ff96`（S0188）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/experiments/three_innovation_ablation_transfer_v22_0/COVID5_FINAL3/self_train/train_promise12_bcp_fft.py`，行 29,2323,2443,2449,2929,2930,2933,2937,2938；SHA256 `e1006ebe9731d7665a886dc5f897f130c5818c132fa575cc16c9ea845cb27386`（S0183）

## D — R2 fixed92 的120条记录及20患者表：VERIFIED，可关闭

原始文件原样入包：

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/results/case_class_metrics.csv`，行 1,2,121；SHA256 `e63572ebfbfceba8549d9461d6bc64f251b582b270642e330445aa7083c0a0e9`（S0145）

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/results/R2_patient_metrics.csv`，行 1,2,21；SHA256 `9be546367e229e2a1621be87658bd9d411fe9d7efe4e7f29774aeb5f0949cc54`（S0166）

120个独立(case,class)键；40个ED/ES帧体、20患者；每类40行，每患者2帧×3类=6行。20患者表逐项与原120行的分组均值相比，最大误差 **4.440892098500626e-16**；所有数值有限。120行宏平均、3类宏平均、40帧均值及20患者均值在浮点误差内一致（每患者恰好两帧，因此等权患者均值与等权帧均值一致，不推广到不均衡帧数队列）。

原receipt未舍入结果：Dice **0.9020444131484706**、IoU **0.8270690413278287**、HD95 **1.880012540213665 grid**、ASD **0.5087759056993733 grid**。学生1验证选点29000，checkpoint SHA **b56292a270ccf6679b5597e9d556eda686df91778e496e715174aaa367b710de**。论文显示90.20/82.71/1.880/0.509；对89.98控制的Dice差 **+0.219989246764540个百分点**，不归到原主Full动态TCSMix。

`R2_RECORD_VERIFICATION.json`、`R2_PATIENT_RECOMPUTATION.csv`、`R2_CLASS_RECOMPUTATION.csv` 和 `AGGREGATE_RECOMPUTATION.csv` 提供逐行检查。两个原始CSV有完整SHA、包内路径；无需登录服务器或取得影像/权重即可复核这项统计。

## 可直接采用的论文句子（按证据范围）

1. “论文主 Full（R1）为原始动态TCSMix完整方法；其ACDC10%成绩为Dice90.35%、IoU82.93%、HD951.423和ASD0.429，距离为重建网格单位。”
2. “更换混合控制信号的89.98%运行是普通残差替代TRC、均匀权重替代CTW及fixed92实验的共同配对参照；这组结果不用于估计原主Full的单因素模块增益。”
3. “ACDC5%和10%完整方法均完成30000次半监督联合迭代，每学生每迭代更新一次；使用验证集学生1平均前景Dice选定检查点，固定该检查点后报告测试结果。”
4. “ACDC完整方法继承既有监督预训练权重及base SGD动量状态；可核查半监督入口/末尾状态对应lr0.01和EMA0.99。预训练生成日志及其确切更新预算未在本轮定位到，故不将推断的预训练步数计作已核实总预算。”
5. “ACDC和PROMISE12的ASD按prediction到GT的单方向表面距离计算；COVID-19的ASD则为双方向表面样本合并后的均值，跨任务的距离统计定义分别说明。”
6. “COVID-19主Full在5%和10%标注预算下的HD95分别为22.79 mm和11.88 mm。”这是作者确认的论文采用值；历史其他组统计不替换此结果。
7. “AdaMix-CT在共同划分、联合迭代预算和验证/测试口径下比较，保留其随机初始化、AdamW与方法规定的增强；因此结果是完整方法比较。”
8. “R2 fixed92包含20患者的40个ED/ES帧体及120个类别记录；对共同混合控制信号参照的Dice差为+0.219989个百分点，单次实验不证明多随机种子显著性。”
9. “现存COVID适配实现的两视图由clone得到，配对FFT交换不产生独立输入扰动，额外两学生互监督仍保留。”这句表述**可核查实现**；若正文要无条件归于FINAL3历史全过程，依赖绑定缺口需先关闭。

## 剩余材料与收尾边界

仅需补**已有记录**：当前论文表2–4逐行方法/预算名单；label3/label7实际生成与选模日志/命令/label manifest/update budget；原Full增强/DiceLoss及COVID_FINAL3 adapter/FFT的历史导入快照或加载SHA收据；HiEn/PMEUL/UCAD既有spacing修正结果（若不存在，当前记NOT_FOUND）。不建议用新训练补造历史来源。

本轮已关闭D、ACDC5/10半监督预算与状态、PROMISE基线精度复算、所列病例记录均值、实际单空处理、COVID入口及几何定位、CTW组件SHA链。A/B/C总体仍PARTIAL，缺失项不填推断。COVID主Full的HD95采用值已由作者明确为22.79／11.88 mm；相关替换建议已撤回，无需作者再次决定。其他来源绑定缺口保留，未据测试成绩选版本或改参数。

证据包入口为本文件；索引 `SOURCE_INDEX.csv`，四项状态 `CLOSURE_DECISIONS.csv`，完整性清单 `SHA256SUMS.txt`。只包含必要小型源码/片段、配置、receipt、原病例统计、复算和校验记录；不含影像、预测数组、权重、数据集、凭据或其他项目材料。
