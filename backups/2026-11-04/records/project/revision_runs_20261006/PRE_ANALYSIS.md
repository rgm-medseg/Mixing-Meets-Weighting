# 启动前分析

时间2026-10-06T21:26:47.364189+08:00。所有可用历史材料已执行分析，不补写失落历史轨迹。

Full与baseline历史配置/源码/检查点来源已核验，统一评价在独立runs/Historical_*目录，原稿成绩不覆盖；如有数值差异，以新口径实际结果和来源解释。预训练文件确定，生成历史预算/seed仍NA，不推测。

固定尺度97×97，s=0.37890625，历史每200step的150条面积样本均相同；稀疏覆盖不代表全部训练step，记录s仍变化。E2/E3锁定实际面积误差0，不用mean(s)替代RMS/取整面积准则。原Full动态方案可能几何变化很小，该事实应影响解释。

E1均匀权重按控制模型自己的逐图w均值、参与像素为全mixed网格；stop-gradient与仅student1梯度已通过直接张量检查。E4宽18结构647参数对照TRC641，保留零门控与教师EMA。E5使用官方固定commit/nativeU-Net/loss/AdaMix，所有非原Full差异已在hyperparameter_provenance.md写明。

COVID本地baseline与Full所有5%/10%行历史标mm（n=4），但512→256缩放未更新spacing；原值精确重放并保留，另存正确spacing和回原网格的两种评价。最终XYZ重建轴序正确。其双向ASD与ACDC单向ASD不可混同。外部稿件Table4逐行原始映射本地未定位，仅该审计缺失项BLOCKED，未请求论文或等待写作助手。复制视图collate没有额外扰动；FFT实际交换相应mixed视图低频空间分量，复制输入下内容扰动近乎恒等，不预判为实现错误。

lambda扫描组件均ON，seed42/30000updates/valbest；.05是历史hook默认而非已证实优化出来的最优系数，不更改主值，不把扫描变化当seed方差。

- A1: PARTIAL — Historical sparse trace drawn; full-step traces added from new runs. Missing runs remain DEFERRED.
  依赖补齐：新E1–E5验证选中checkpoint/全步日志/统一test评价；A2/A3无E5 EMA要求。
- A2: PARTIAL — Existing teacher best/last checkpoints on 14 fixed already-labeled slices; new checkpoints DEFERRED. E5 has no EMA teacher by official definition.
  依赖补齐：新E1–E5验证选中checkpoint/全步日志/统一test评价；A2/A3无E5 EMA要求。
- A3: PARTIAL — True two-student/EMA outputs on four fixed already-labeled slices mixed offline; 2-pixel neighborhoods separated into GT-only/mix-only/overlap/interior. No anatomical semantics assigned to tokens; normalization denominator floor 1e-6.
  依赖补齐：新E1–E5验证选中checkpoint/全步日志/统一test评价；A2/A3无E5 EMA要求。
- A4: PARTIAL — ACDC frame/class raw records aggregated by patient for paired differences; COVID shown individually with n=4 per run. No slicing/checkpoint pseudo-replication.
  依赖补齐：新E1–E5验证选中checkpoint/全步日志/统一test评价；A2/A3无E5 EMA要求。
- A5: PARTIAL — Same A100/FP32/shapes; CUDA synchronized, warmup10 + timing30; repeats are timing only. Training wall/peak from new runs; historical training wall source audited separately.
  依赖补齐：新E1–E5验证选中checkpoint/全步日志/统一test评价；A2/A3无E5 EMA要求。
- A6: PARTIAL — Prespecified extreme/median patient gain selection, same frame/slice/display/colors; lowest gains retained even if adverse. Test used only for final descriptive display.
  依赖补齐：新E1–E5验证选中checkpoint/全步日志/统一test评价；A2/A3无E5 EMA要求。

每个脚本退出码、日志见reports/pre_analysis_execution.json。全部分析脚本现在可独立执行；队列结束后自动补齐，有失败或缺失保留PARTIAL/BLOCKED和原因。

## COVID缩放spacing追溯更正

最终evaluator沿axis=2正确重建XYZ，轴序判断已更正。512→256缩放后仍用原spacing是已证实的问题，旧mm标签不代表resize后的正确物理间距。已在独立目录保存固定预测的legacy重放、corrected derived center-grid、corrected native-grid三口径评价，无新增COVID训练。见reports/covid_geometry_audit.csv与covid_reevaluation_macro.csv。外部方法行仍缺失，不套用本地更正。
