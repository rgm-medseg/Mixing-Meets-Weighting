# 给写作助手的交接

本轮版本TCSMix v2_B，保留模块TRC/TCSMix/CTW名称及论文标题。准确方法：teacher weak logits→类别轴softmax、detach→原argmax/每类LCC前景支持→各前景类mean(P_c−max其他类P)→有支持类别等权均值；全无前景取0。原s=.18+.20q，整数round/clamp、随机位置、单共享mask/双视图互补混合保持原定义。背景与像素面积不直接进入控制平均；不是概率校准、GT可靠率或轮廓度量。公式与伪代码见TCSMIX_CONTROL_DESIGN_CN.md。

可写事实：受控背景/类别面积变化时A/B均值保持、前景概率提高时控制单调；70个固定离线batch old全97而B86–94，部分stage内A仍固定所以选择B；Full末期一个snapshot仍单尺度。真实short240步B85–90、88最多38.75%、176次变化、实际面积11.0244751%–12.3596191%，fallback0。同一新teacher old回放全97，明确是反事实控制回放，不能称另训baseline收益。仅诊断响应/实现，不是完整版训练提升。

数据与GT说明：原7标注患者14张与10验证患者20个中间frame切片用于quality；另外36张无标注训练/边缘切片不读取GT。每个来源各自保留patient/slice/checkpoint/hash，早期E1为uniform CTW教师，SSL0为原pretrain构建的零门控teacher，不是存下的历史step0；中期Full缺失。切片/重复checkpoint不作独立病例，Spearman只描述；all3宏Dice双空类0，另给GT-present宏Dice/GT类别与像素数。高控制仍可能错误；保留不利例与极小预测区域。

修改论文公式/图注时区分old与v2。旧Full Dice90.35%（统一旧复评精确90.349657652%）仍只属于old。v2正式Dice/IoU/HD95/ASD=NA，R1/R2未启动。原八组合/原E1控制不能自动充当v2消融或交互证据；本阶段没有方法收益、跨seed显著性或新版CTW交互结论。后续R1/R2仅单次、固定面积从R1完整实际轨迹锁定，不据测试指标挑值，也不能证明teacher条件化优于同分布随机尺度。

图：analysis/control_replay.png/SVG为静态同输入teacher回放；short_control_trace为新B真实局部训练、200标记恢复边界，old/A线只是同teacher信号回放；controlled_signal_response是人工概率验证。各图原始CSV与manifest均入包。CSV概率/Dice/面积为fraction0–1，尺寸pixel、step联合更新、计数pixel；short预算严格按两个optimizer调用计，240联合步=480次调用。成本计时repeat不是训练重复。

原CTW空间差分/.25+r/L1/stop-gradient/.05/方向、TRC4×16、EMA.99、loss/数据/增强/优化器保持原Full，逐项证据见SCOPE_DIFF_AUDIT.md。正式阶段仍需用户明确确认，当前WAITING_FOR_FULL_APPROVAL。短程权重不可作为正式初始化。
