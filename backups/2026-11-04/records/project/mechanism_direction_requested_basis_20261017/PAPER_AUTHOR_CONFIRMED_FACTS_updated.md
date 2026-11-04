# 论文结果：作者已确认的固定记录

最后同步：2026-10-09T23:58:36.745464+08:00。来源：作者本轮明确更正；这是记录同步，不是新评价结果。

| 对象 | 最终确认值 | 单位/身份 |
| --- | --- | --- |
| COVID-19 5% 论文主 Full HD95 | 22.79 | mm |
| COVID-19 10% 论文主 Full HD95 | 11.88 | mm |
| ACDC10% 论文 R1／原始主 Full Dice | 90.35 | % |
| ACDC10% 更换混合控制信号 Dice | 89.98 | %，服务器旧formal_R1_20261007／R1_v2_B，独立实验 |
| ACDC10% DC-Net HD95 | 2.28 | grid，表2论文采用值，作者确认 |
| ACDC10% DC-Net ASD | 0.68 | grid，表2论文采用值，作者确认 |

先前建议把COVID主Full HD95改成33.56／18.96 mm，现标记为：**“结果对应关系混淆，已按作者确认撤回”**。该建议与相关必须修改待办已撤回。另一组数值不归入论文主Full，也不猜测它属于哪次运行。原始成绩、源码、配置、检查点、评价数据、旧压缩包及历史证据保留。

论文题目：Mixing Meets Weighting: Confidence-Modulated Semi-Supervised Medical Image Segmentation。模块TRC、TCSMix、CTW；任务ACDC、PROMISE12、COVID-19。其他设定保持既有作者确认。

本次仅更正记录，不启动/恢复训练，不重评价，不重选checkpoint。具体修改索引：[covid_full_hd95_author_correction_20261009_230341_BJT/CHANGE_INDEX.csv](covid_full_hd95_author_correction_20261009_230341_BJT/CHANGE_INDEX.csv)。

## DC-Net ACDC10%：作者最新确认

作者最新确认（2026-10-09T23:58:36.745464+08:00）：**DC-Net / ACDC 10%：HD95＝2.28、ASD＝0.68，单位grid（表2口径）**。旧V19_4的1.28／0.38仅作历史版本；两组数值之间的待核实及替换建议已按作者确认撤回。**论文采用值确认与历史运行来源绑定分开：该行run/checkpoint/split/evaluator绑定仍为NOT_FOUND，未补出新原始精度。**本次只更新记录，不训练、不重评价、不改原始实验产物。

确认仅适用于论文采用数值与表2单位；run、checkpoint SHA、split SHA、实际driver/evaluator及原始未舍入精度仍待已有来源绑定，不因作者数值确认而关闭。后续论文、审查、实验汇总及交接均采用2.28／0.68 grid。

本次修改索引：[dcnet_acdc10_author_confirmation_20261009_235640_BJT/CHANGE_INDEX.csv](dcnet_acdc10_author_confirmation_20261009_235640_BJT/CHANGE_INDEX.csv)。旧论文记录副本及V19_4原文件保留。


## CTW方向份额：本轮用户指定口径（2026-10-17；与计算记录尚未对齐）

用户要求后续讨论以以下比较为基点：**uniform→CTW，真实边界×教师错误区域的GT同向份额3.61%→4.83%，GT反向份额5.35%→4.26%。**

来源身份：**用户指定的汇总口径，非本轮新增计算确认。** 此请求承接此前条件性讨论；没有提供新运行或更正后的来源表。此前已计算CSV的反向对应关系为uniform 4.26%、CTW 5.35%，因此与指定口径存在冲突。必须保留该来源区分，不能把用户指定口径标成原始计算已验证结果。

采用指定口径作条件性解释时，可写“同向增加、反向减少，支持相对偏向GT一致方向”；不能自动沿用原7.88%→10.18%总份额、10/10配对患者数、原始mean a或五审状态作为此口径的新证据。原始数组、统计CSV和原证据归档保持不变；本次不训练、不重评价、不重选checkpoint。

单独修订材料：[mechanism_direction_requested_basis_20261017/SUMMARY.md](mechanism_direction_requested_basis_20261017/SUMMARY.md)。后续处理该组数值须同时读取本段和材料中的来源状态，不将其混入上方已核实固定结果表。
