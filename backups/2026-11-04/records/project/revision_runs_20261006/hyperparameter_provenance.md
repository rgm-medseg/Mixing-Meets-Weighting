# 系数与调度来源

图5扫描0/.01/.03/.05/.07/.10为TRC ON、TCSMix ON，seed42，30000updates，best validation Dice，保持现有原始扫描，无重新扫描。原始表与配置见reports/evidence_sources；数值90.1655/90.0328/90.0488/89.9908/90.0887/90.1823，不是多种子方差。

.05在V21.3候选hook中是共用默认值，并用于原Full；未发现证明该值通过预先数值优化确定的历史理由，保持NA，不因.10验证值更高改主系数。

TCSMix端点.18/.38不是训练过程观测范围。实际q批均高；固定尺度源自历史150个每200步记录，实际面积均97²/256²=14.3569946%。记录是稀疏时间覆盖，不能冒充全步轨迹。E2/E3固定s=.37890625，按round/clamp实际面积误差0；RMS边长仅辅助，未用test反推。A1新增逐step记录真实q/s/面积/LR/损失，旧轨迹不补写。

AdaMix-CT pinned commit85361fb91cf43a69a158d41dfa9e7ee9fce36db9，官方exp_acdc/train_adamix_ct.py。原生U-Net[16,32,64,128,256]，AdamW1e-4默认wd.01，无teacher/β-FFT/TRC/CTW；保留四加载器、DSCLossH、.95阈值、patch_size8、topk16、hard模式p.5、自步课程总步的四分之一。锁定30000updates以和Full比较，改变官方200epoch及val-loss选择为同协议每200step的student1 val-macroDice。每加载器B16、标注集重复至1312与原训练全集辅助池；7病例标签固定，官方图像PIL量化/增强照旧。初始化从头、无预训练；总预算不隐藏阶段。


## 已证实的COVID缩放spacing问题

代码与原NIfTI头共同证实：512→256的in-plane resize未更新spacing；最终evaluator沿axis=2正确重建XYZ，所以轴序本身正确。此前仅根据dataset ZXY堆叠作出的轴序判断已撤回，以完整重建链路为准。历史距离调用的mm标签没有反映resize后的正确像素间距；前述单位说明只描述标签/调用一致性。没有修改旧值，另存corrected-derived-center-grid及corrected-native-grid评价。具体患者shape/spacing/原GT文件SHA见reports/covid_geometry_audit.csv，全部三口径原始统计见covid_existing_prediction_reevaluation.csv。新native口径对预测nearest回原canonical网格，GT不缩放，spacing(Z,X,Y)。该问题不改变ACDC grid距离对照，也不新增COVID训练。
