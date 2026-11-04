# 本轮机制补强 SUMMARY

任务1：DONE。原始X03 / Full_main / 论文R1 best，step29600，两学生和EMA严格加载；主Full测试Dice90.35%保留。验证集20帧、10患者、210不同切片；9批24模拟角色，216来源暴露含6重复，108混合图/条件。四条件均实际计算。所有动态区域97×97，与固定完全重合；不能据此声称动态优势或TCSMix×CTW交互收益。

CTW把真实解剖边界的患者等权一致性梯度贡献从46.60%提高至56.02%；平均真实梯度强度比2.23。这是0.05缩放CTW一致性项对学生1 logits的L1梯度，不是总训练梯度。权重同图均值匹配，student2/权重detach；全图损失未按区域筛选。固定checkpoint的预测不因切换CTW改变。

任务2：DONE。gamma=0.035987895；改变86像素，纠正46、新增36、双方均错误但标签改变4。正常/干预患者等权native-volume Dice=89.99081% / 89.98905%，差0.00176个百分点；RV/MYO/LV及区域明细见CSV。全部210切片恢复gamma后logits逐元素一致，模型/原checkpoint未更新。既有训练消融在TCSMix/CTW保持启用时，无TRC/Full测试Dice=90.0894% / 90.3497%（独立已有训练结果，非本轮新评价）；即时验证干预与训练测试消融分别解释。

任务3：DONE。唯一新增条件F97_main_signal_20261013，97×97，TRC/CTW均开，原mean-max信号；复用原预训练及优化器继承，seed42，每学生30000更新。当前报告读取到30000次已完成更新（训练完成状态以receipt为准）。尺寸来自主Full150个稀疏训练时点，匹配这些时点的实际面积，未标为全程平均面积匹配。旧E3未运行，旧92×92替代信号结果未归入本对照。初始化三模型base tensors及两优化器base状态继承均逐元素核验。

The fixed control completed 30,000 updates per student. Student 1 was selected at update 29000 using the original validation criterion and achieved 89.8548% test Dice, compared with the retained main-Full value of 90.3497%, a difference of -0.4948 percentage points. Its HD95 and ASD were 2.2513 and 0.6998 grid units. This is a single-seed, run-level comparison of the size rules. The fixed run's own teacher signals would yield 97-by-97 rectangles at 29,999 updates and 96-by-96 once under the original rounding rule; these logged signals do not reconstruct the original Full training trajectory.

新增固定run的完整信号日志含30000条，按原q/取整规则离线重映射时，29999条对应97×97、1条（step460）对应96×96；实际训练mask始终固定97×97。这只是该固定run自身的EMA信号分布，不能补成旧Full全程轨迹，也不将0.4948个百分点差解释为已证明的动态面积因果效应或TCSMix×CTW交互。40帧/20测试患者的等权重算与主Full原始病例CSV比较见fixed_vs_main_patient_dice.csv；主Full没有重评价。

交付：figures/Fig_joint_TCSMix_CTW和Fig_TRC_residual_intervention，各SVG/PDF/PNG；results/TABLE_regions.csv、joint_patient_region.csv、trc_patient_dice.csv、trc_patient_region.csv、fixed_control_table.csv、fixed_vs_main_patient_dice.csv；区域及固定对照结果表另有SVG/PDF/PNG预览和LaTeX。完整图像/预测/真实梯度在arrays/joint，TRC特征/残差在arrays/trc_features。scripts、configs、COMMANDS.sh与PROTOCOL.md提供复现。MANUSCRIPT_INSERTIONS.tex是独立英文正文段落与贡献建议，FIGURE2_LAYOUT.txt给布局；不覆盖旧稿，不替换论文图片或PDF。

五审见FIVE_REVIEWS.md和receipts/FIVE_REVIEWS.json。三项实验均DONE，训练、验证选点、40帧测试及复现产物已完成；没有尚需用户提供的初始化或实验配置。未找到可合入的最新17页可编辑稿，按授权交付独立正文插入内容。两种uniform训练条件仍缺少同设定的完整checkpoint，已登记且未新增训练；这不影响本轮四条件冻结诊断的完成。期刊最终生产格式核验仍PENDING。

使用K-Dense Scientific Visualization，实际数据与全部变换保留；skill来源与软件引用见PROTOCOL.md。Neurocomputing指南读取403，最终生产格式核验保留PENDING，不声称自动合规。最新17页正文未定位为可编辑工作稿，因此只交独立插入内容。作者固定记录未改动。
