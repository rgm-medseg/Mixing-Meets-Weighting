# 写作交接

Full映射PARTIAL/NEEDS_MAPPING：保留作者说明及90.35%归属原运行，待另一个可定位入口/检查点关联才能将“与R1一致且实际矩形变化”作为该成绩的已验证历史机制；不能转移分数。ACDC测试集未用于原候选选择，按作者更正。

A2统一信号DONE：推荐引用A2_COMMON_SIGNAL_REPORT_CN.md统一205 GT-present/10患者表，baseline也有间隔正相关，因此描述信号性质与当前冻结质量，不归因TCSMix训练改善。Full/baseline q_B明确辅助，R1单切片代理、R2记录信号固定几何。210固定三前景宏值及5无前景FP保留。

部署等价/测速DONE：先引用全210×4 logits/mask和80体hash核验；主表采用本轮每次重新加载预热50的original/deployment分列，零/实输入与model/volume口径分列。节省诊断统计开销，TRC预测运算完整保留。641新增参数和已覆盖MAC口径可使用；完整FLOPs及历史Full/baseline同口径训练时间NA。旧测速仍为历史证据。

A3 DONE：引用confidence-transition定义、弱/强同内容配对、clean anatomical band与人工seam/cutout拆分；各方法best/last、2主带宽及1/3敏感性、pooled与patient equal都保留。uniform是同均值离线空间权重对照。权重不是校准的边界正确概率，离线区域贡献不是训练消融分数，无法证明纠错因果。

确需训练才能回答的问题仅列建议：CTW空间分配相对同均值uniform的最终分割收益；动态尺度相对固定面积收益的可重复性；去除或改进人工边界响应的训练效果。当前没有这些训练授权，未执行。

正文模块名TRC — Token-Guided Residual Conditioning；TCSMix — Teacher-Confidence-Scaled Mixing；CTW — Confidence-Transition Weighting。原MaskToken/AdaMix/CRW仅源码检索名。精确数值参见三个报告及CSV，图注同口径。

主表数字：Full best辅助q_B验证切片/患者ρ=0.807517/0.830303，baseline同口径0.803302/0.830303。Full零输入original/deployment CUDA mean=3.036089/2.093537 ms；native-volume=5.644241/4.738675 ms/切片。Full/R1/R2 best matched92患者等权clean boundary mean w=0.398393/0.402715/0.402385，seam-only=0.260337/0.261831/0.260888，remainder≈0.251。约97%整图像素位于w∈[0.25,0.27)，另图linear/log完整显示尾部。上述空间响应不是校准概率或训练收益。旧queue缓存3975，真实E1保存检查点/收据3976，仍暂停。
