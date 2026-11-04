作者确认更正（2026-10-09T23:03:41.867783+08:00）：论文主 Full 的 COVID HD95 为 **5%：22.79 mm；10%：11.88 mm**。此前将其替换为33.56／18.96 mm的建议：**“结果对应关系混淆，已按作者确认撤回”**，不再列为论文必须修改项，也不将另一组数值归入主 Full。原始实验产物及旧证据包保留。本次只更正记录，未训练、未重评价、未重选 checkpoint。论文 R1／主 Full Dice90.35%；“更换混合控制信号”Dice89.98%单独保留。

本包只处理A–D四项收尾证据。先读 REVIEW_CLOSURE_EVIDENCE.md，再查 CLOSURE_DECISIONS.csv 与 SOURCE_INDEX.csv。
原路径用于溯源，packed_path 可在解压后直接打开。INDEX_ONLY / CPU_METADATA_ONLY_WEIGHT_EXCLUDED 文件不随包导出；其中检查点仅登记SHA/元数据。
scripts/ 中都是只读CPU核查脚本；新输出仅写本目录。不要将当前CPU复算当作历史运行记录。

历史审计脚本和旧证据副本保留，不在本轮执行；旧脚本模板或原记录的其他组标签不得覆盖本次作者确认。当前更正索引见 covid_full_hd95_author_correction_20261009_230341_BJT/CHANGE_INDEX.csv。
