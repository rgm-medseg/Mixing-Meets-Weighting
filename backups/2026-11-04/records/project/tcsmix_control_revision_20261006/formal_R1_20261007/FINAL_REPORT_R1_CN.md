# R1新版Full实际结果

报告时刻2026-10-07T08:02:06.847382+08:00。仅R1：TRC+锁定TCSMix v2_B+原空间CTW；seed42、ACDC10%、同原pretrain net/opt、30000联合迭代。R2和旧E1–E5未启动/恢复。当前阶段SUMMARY，状态ANALYZED；训练完成不自动等同评价与归档完成。

|指标|实际值（fraction或grid距离）|
|---|---|
|dice|0.8998445206808252|
|iou|0.8234190597056135|
|hd95|2.94213147078008|
|asd|0.7689256378227837|

评价完成：True。选择step=29000，checkpoint SHA=5df2de9b8df16e4d89df9265f104b4b8c7c321245d9eaa0479971672b4d7a1a9；按原严格student1验证前景macroDice，测试只用于冻结后的最终评价。两学生实际调用数/重复更新见reports/TRAIN_POSTCHECK.json、results/optimizer_calls.csv、完整R1_control_trace.csv。初始化实际三分支/零门控/预训练哈希见INITIALIZATION_ACTUAL.json；原始预训练生成日志未知，未补造。

评价协议为原统一ACDC_v22_grid_directed_ASD：输入切片nearest resize256²，输出nearest回原H5切片平面，z轴不重采样；原重建grid单位spacing(1,1,1)，HD95/ASD不能标mm；无预测后处理，ASD为预测→GT方向。双空四指标0，单空Dice/IoU0、距离为体积grid对角。总体等权平均40个ED/ES frame体积×3前景类；病例/患者/类表单独保留，20个患者的两frame平均用于配对分析。

历史参照与差值：
Historical_Full: 原指标 {'dice': 0.9034965765204998, 'iou': 0.8293272953910553, 'hd95': 1.4233666940334322, 'asd': 0.4290264993301964}；R1−参照 {'dice': -0.0036520558396745395, 'iou': -0.005908235685441854, 'hd95': 1.518764776746648, 'asd': 0.3398991384925873}。Dice/IoU为fraction，乘100才为百分点差。
Historical_baseline: 原指标 {'dice': 0.8966063409597648, 'iou': 0.8182007425305878, 'hd95': 1.7618789072469767, 'asd': 0.5722597264440701}；R1−参照 {'dice': 0.0032381797210604413, 'iou': 0.005218317175025633, 'hd95': 1.1802525635331034, 'asd': 0.1966659113787136}。Dice/IoU为fraction，乘100才为百分点差。

旧Full90.35%只属旧机制；新结果允许相同或下降，不自动重训。参照同测试清单与统一评价，但部分历史import版本/预训练生成协议无法完全核验；baseline还缺TRC/旧TCSMix/CTW，差值不是单独模块因果。新版与old Full分数差值也不能单独归于动态尺度，R2尚未执行。

机制统计按预先固定的六个5000步区间及全程给出，见analysis/scale_distribution.csv和control_trajectory.png/SVG；raw trace全部保留，恢复重复step单独记录、最终逻辑轨迹取最后一次。old_q是同一新版教师输出的控制回放，无额外前向或随机数消耗；不是独立old训练。若晚期尺寸固定，如实保留。R2_FIXED_AREA_PROPOSAL只据完整真实面积在测试前生成建议，不执行R2。

离线质量诊断状态：ANALYZED，细节{'confidence_quality': {'status': 'DONE', 'rows': 136, 'labeled_patients': 7, 'validation_patients': 10, 'test_inputs': 0, 'GT_use': 'diagnostic only; never in unlabeled controller', 'middle_checkpoints': 'NA: only retained selected best and last teachers', 'state': 'independent process, reset cloned buffers per batch, no optimizer updates; post-step teachers not historical per-step logits'}, 'historical_comparison': {'status': 'DONE', 'matched_case_class_rows': 240, 'comparison': 'same reconstruction-grid evaluation/test list; older imported-source/pretrain-generating details incomplete', 'interpretation': 'R1-old Full delta alone does not isolate dynamic scheduling benefit; R2 not run'}}。原固定标注/验证样本GT只用于诊断；重复切片/checkpoint不算独立病例。v2_B概率间隔不是校准正确率；极小/错误前景类、缺类及fallback保留。保留全部改善/持平/变差配对行；定性例按最大差、median差、最小差选，slice取最大GT前景面积，并给匿名ID规则/provenance。预测和原医学数组留服务器，包中只交付既有匿名ID规范的规范化图及索引。

实际成本：{'recorded_logical_steps': 30000, 'raw_trace_rows': 30000, 'training_active_segment_seconds': 26440.013047698885, 'stable_timer_median_seconds': 0.6821030508726835, 'stable_recorded_steps': 28310, 'timer_scope': 'start after loader yields to control trace after optimizer; excludes validation/EMA/checkpoints following record and batch waiting', 'peak_allocated_bytes': 16563296256, 'median_controller_wall_ms': 2.804674208164215, 'host': 'n1', 'gpu': 'GPU0 A10040GB', 'precision': 'FP32', 'batch24': 24, 'original_teacher_forward_count': 4, 'log_overhead': 'append optimizer events/step trace and inherited revision telemetry; no duplicate old-teacher forward; no claim of paired full training overhead delta'}。训练用实际进程阶段时长，验证尾部包含在内；独立评价/分析/归档时长见state/stage_history.json。日志成本见reports/logging_cost_*.json，只计新trace/events/progress，不把控制器微基准当端到端差值。worker增强随机流和prefetch不能位级恢复；完整可恢复状态保存，不更改16worker协议。

失败与缺失：目前无阶段错误记录。缺失指标写NA；科学缺失与归档错误分别记录，FAILED/PARTIAL结果也归档已有证据。没有改batch/种子/控制器或自动从零重试。
