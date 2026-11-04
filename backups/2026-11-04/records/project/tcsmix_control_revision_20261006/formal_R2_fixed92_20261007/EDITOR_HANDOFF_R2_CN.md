# 给写作助手：R2固定92面积对照

R2={'dice': 0.9020444131484706, 'iou': 0.8270690413278287, 'hd95': 1.880012540213665, 'asd': 0.5087759056993733}，选中step29000，SHAb56292a270ccf6679b5597e9d556eda686df91778e496e715174aaa367b710de。实际来源results/R2_results.csv，分数fraction×100展示百分比，HD95/ASD为grid不是mm。旧Full90.35%归旧机制，v2_B R1与固定R2独立列明，不能把旧分数移到新版；本轮未修改论文。
唯一方法变化是v2_B动态尺寸变为固定92；公共科学循环、模型/损失/空间CTW/EMA/初始化/增强与评价保持R1。q_B继续记录但不控制掩码，s_dynamic_B参考，s_actual=.359375。R2面积比完整R1平均低约.04295面积pp（相对-.33145%），平均匹配不等于每阶段匹配。

主要对照原始CSV、配对病例和阶段面积图在analysis；全部不利行保留。R1/R2一次对照只比较整体动态和近平均固定策略，不能分离置信度信息、尺度波动/课程贡献，也不能证明与CTW协同、跨种子稳定性。训练结果下降允许报告，不引出自动搜索。
R1_ERROR_ANALYSIS_CN.md和R1_R2_ERROR_ANALYSIS_CN.md只描述冻结测试预测几何；不能用误差slice倒推固定尺度/控制器/最佳checkpoint。HD95需要双向表面距离，ASD单向prediction→GT；孤立域和GT空slice范围延伸有实际mask/连通域/方向距离贡献证据，局部偏移和漏分割可共存，不能统称一个原因。案例选择为真实距离贡献最大的slice，规则/匿名ID/方向/颜色/指标附CSV，不只展示最大GT面积。
R1固定标注/验证q_B质量诊断是另一组证据，未使用测试选择方法；切片或病例不等于独立训练重复。历史生成预训练日志与部分旧import版本缺失如实标NA，不强称全部历史完全可核验。

完整协议、有效代码/工程patch、运行命令、初始化/检查点回执、数据/预测/权重哈希和科学核查附包。snapshot状态是归档前时刻，最终状态与外层校验由完成回执另记。当前分析状态ANALYZED，实际错误[{'stage': 'TRAIN', 'command': ['/bin/bash', '/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/scripts/train_R2.sh'], 'start': '2026-10-07T11:03:10.072726+08:00', 'log': '/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/logs/train_2026-10-07T110310072633+0800.log', 'resume_step': 0, 'pid': 81682, 'end': '2026-10-07T11:03:20.112001+08:00', 'exit_code': 1, 'elapsed_seconds': 10.03934931755066, 'measurement': 'Popen launch to detected subprocess completion; <=5s polling detection lag, imports/validation/checkpoints included'}, {'stage': 'PIPELINE', 'time': '2026-10-07T11:03:20.140066+08:00', 'error': 'Training exited1; no automatic retry or parameter change'}]。
