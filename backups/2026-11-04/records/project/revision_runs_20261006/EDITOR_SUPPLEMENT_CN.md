
## 本轮补充核查与论文更新


固定尺度：FIXED_RECTANGLE_AUDIT.md及fixed_rectangle_historical_records.csv明确旧Full150条step200..30000稀疏记录；s范围0.377744894028–0.379582624435、均值0.379189255555，记录点97²比例100%，实际面积均值14.3569946%、固定误差0。历史batch切片ID与每时点teacher快照缺失，不能声称全30000步平均；E1逐步曲线独立标新run，不能冒充旧Full。E2/E3配置未变。

COVID：COVID_SPACING_AUDIT.md、covid_metrics_before_after.csv和macro CSV覆盖所有已定位本地预测/run。spacing-only保留预测/GT/重采样/后处理，仅改sx/sy；Z不缩放，Dice/IoU应完全不变。原网格另存结果改变了GT/预测重采样，须单独说明。原平面包括512²和630²，不全是512²。CSV含完整源/hash/axis/shape/spacing与真实重算的HD95/ASD；外部方法身份/输入缺失仍NA，不能根据比例估计。

协议：COMPARABILITY_AUDIT.md和protocol_comparison.csv逐项说明保持一致/预期改变/其他差异/无法核验。补清ICT实际最大系数5×.1=.5，不把参数consistency=.1当作最大有效权重。原预训练生成预算/seed、原训练当时清单与全部import依赖hash无法核验，不能写完全一致或位级精确续训。E4残差参数+.936%不是全网络增幅。E5实际执行train_entry→official_adapter，官方方法核心来自锁定commit的exp_acdc/train_adamix_ct.py；是有明示预算/验证/划分适配的官方方法实现，并未直接运行未经改动原main。

新增文件及原始统计实际进入证据包，保持五项单次结果包括下降。当前任务状态见reports/CURRENT_QUEUE_STATUS.csv与SUPPLEMENT_QUEUE_SNAPSHOT.json；未完成测试指标仍NA。原queue继续自动训练/评价/A1–A6/汇总。审计跟随进程位于同tmux的audits窗口，在queue最终归档后追加本节并再次生成一致快照包，避免最终summarizer覆盖本次写作交接。
