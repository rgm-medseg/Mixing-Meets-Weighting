正式名称：TRC — Token-Guided Residual Conditioning；TCSMix — Teacher-Confidence-Scaled Mixing；CTW — Confidence-Transition Weighting。

C1完成30000逻辑联合迭代。两学生成功物理优化器调用：{'optimizer1': 30000, 'optimizer2': 30000}；重复逻辑轨迹0条，额外物理调用0。自身严格验证选择student1 step=27800，SHA=d37c9a8f6f28e3be0ce1b7041b9092044160011de10441f8eaea62b13c6517c7。

C1 Dice=89.65431117%，IoU=81.85149190%，HD95=1.548304888339121 grid，ASD=0.49454106974328754 grid。R1 Dice=89.98445207%。C1−R1 Dice=-0.33014090百分点；空间优势R1−C1=+0.33014090百分点。HD95/ASD越低越好，表中两个代数差值方向明确。

20患者先合并ED/ES两帧；C1改善/持平/下降=7/0/13。40帧×3前景类全部保留。单次seed42描述性比较，性能对照回答v2_B＋TRC设置下CTW空间分配的训练贡献；同图均值匹配不要求两个训练轨迹相同。离线A3边界响应沿用paper_offline_followup_20261007版本，另述其证据范围。

成本见runtime_cost.json/csv和stage_history；验证包含于训练wall，独立值NA。allocated/reserved与全卡NVML分开。16worker随机流/prefetch不能位级恢复；恢复段、回滚和短程开销均保留。历史预训练生成命令和旧训练H5内容清单NA，当前资产哈希锁定。原Full仍NEEDS_MAPPING，90.35%不转给R1或C1。

三个误差患者选择空间Dice优势最大/最接近中位数/最小；帧类选绝对差最大，切片沿用R1单向表面距离贡献规则及显式空例回退。全部配对行保留。完成后停止C1队列，E1保持3976暂停、E2–E5未启动。建议等待本次结论后决定是否需要额外随机性评估，本轮不追加训练。

精确数字与正反差值：results/C1_vs_R1_summary.csv、patient_pairs.csv、frame_pairs.csv、class_pairs.csv、frame_class_pairs.csv。Dice/IoU原值fraction；pp差乘100。距离grid、nearest平面缩放回原H5，z不变、无后处理；ASD预测→GT；双空全0、单空距离体对角线。控制六固定区间及全程见fixed_window_control_weight_summary.csv。逐图全H×W均值见per_image_mean_checks.csv；范围恒0，FP32均值误差容差1e−6。图及坐标见analysis和qualitative_selection.csv。
