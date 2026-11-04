# 提交前五轮检查

1. **范围与实际计算：PASS。** 详见reports/five_round_review.json对应轮次的逐项证据。
2. **公式、实现与数值：PASS。** 详见reports/five_round_review.json对应轮次的逐项证据。
3. **数据使用与版本归属：PASS。** 详见reports/five_round_review.json对应轮次的逐项证据。
4. **证据完整与解释边界：PASS。** 详见reports/five_round_review.json对应轮次的逐项证据。
5. **正式训练前停止：PASS。** 详见reports/five_round_review.json对应轮次的逐项证据。

范围检查针对实际执行源码、loss/EMA/混合块、参数及原文件hash；公式检查逐行重算770条离线记录与240个真实步骤；数据检查实际病例归属和GT使用；交付检查日志/计时/恢复及图；边界检查真实运行四项拒绝命令、进程/tmux/原队列。不是仅比较开关或复述结论。

修正：报告公式转为明确文本公式避免转义控制字符；单切片支持计数从重复双组改为唯一像素，原表与执行脚本保留，概率/矩形/质量不变；旧Full空间CTW与E1均匀CTW来源明确分开。原兼容训练receipt计时低估已披露并改用实际阶段timestamp。没有因这些修正追加训练，实际总480次optimizer调用。原worker恢复非位级与正式效果未知继续保留。

正式R1/R2、重复short和用short面积锁R2均在写状态/启动训练前拒绝。用户明确要求本阶段停在正式训练前，状态WAITING_FOR_FULL_APPROVAL。
