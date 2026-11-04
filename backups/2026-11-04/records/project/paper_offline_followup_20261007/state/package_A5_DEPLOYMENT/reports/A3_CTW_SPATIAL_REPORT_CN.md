# CTW空间分析

DONE，限定冻结eval验证诊断。Full/R1/R2 best/last六个完整checkpoint，两学生和EMA严格加载。210不同验证切片/20帧/10患者，9批24模拟角色，末批6重复暴露，按来源患者分配拼接像素并患者等权，不把重复作为新增患者。108张混合输出/每策略状态，共1080张空间图；Full原策略NA，其他4状态原策略及6状态matched92均完成。没有历史训练重放、GT控制或模型更新。

实际调用同一混合内容weak student1与strong student2；unl组合uimg_a外/img_a内，l组合img_b外/uimg_b内，两个组合不相互比较。q=.5(max p1+max p2)，raw=两轴相邻绝对差之和，每图minmax/clamp1e-6，w=.25+normalized∈[.25,1.25]，e=mean_class|p1-stopgrad(p2)|，L=mean(w e)，λ=.05。原X03与R1/R2 ctwmaps在本输入上损失数值核验，合成logit梯度仅student1有梯度，权重detach。

原WeakStrongAugment的几何、nearest resize、cutout、ColorJitter全部保留，216次与literal增强输出和RNG后态逐元素一致。cutout人工标签0单独保存；旋转填充与cutout视为无效解剖/错误统计支持。source-GT四邻域类变化两侧边界经同transform，再裁取/paste；mixedGT边界含人工接缝另列。带宽2为主，1/3敏感性；四互斥区域用原B_anat，clean另排mix/cut邻近带、遮挡及旋转无效像素，w整图先算后筛，未区域重归一化。无边界distanceNA，其错误保留。

|method_id|region|pixels|support_patients|mean_w|mean_e|mean_we|delta_mean|
|---|---|---|---|---|---|---|---|
|Full_main|anatomy_clean|160752|10|0.398393|0.032760|0.016524|0.008148|
|Full_main|anatomy_seam_overlap|18634|10|0.403355|0.098145|0.042171|0.017057|
|Full_main|cutout_edge|161877|10|0.264718|0.024893|0.008972|0.002598|
|Full_main|remainder|6657303|10|0.250974|0.002494|0.000708|0.000069|
|Full_main|seam_only|217238|10|0.260337|0.010349|0.003638|0.000991|
|R1_v2_B|anatomy_clean|160752|10|0.402715|0.030945|0.016010|0.008091|
|R1_v2_B|anatomy_seam_overlap|18634|10|0.407312|0.094029|0.040766|0.016676|
|R1_v2_B|cutout_edge|161877|10|0.266047|0.024622|0.009039|0.002721|
|R1_v2_B|remainder|6657303|10|0.251002|0.002493|0.000720|0.000080|
|R1_v2_B|seam_only|217238|10|0.261831|0.010247|0.003615|0.000990|
|R2_fixed92|anatomy_clean|160752|10|0.402385|0.034273|0.017197|0.008431|
|R2_fixed92|anatomy_seam_overlap|18634|10|0.403269|0.102699|0.044062|0.017782|
|R2_fixed92|cutout_edge|161877|10|0.264674|0.024912|0.009026|0.002647|
|R2_fixed92|remainder|6657303|10|0.250925|0.002656|0.000758|0.000078|
|R2_fixed92|seam_only|217238|10|0.260888|0.010691|0.003800|0.001062|

区域CSV给每图/患者支持、mean/median w、e、w e、两个学生错误和分支不一致及完整图分母贡献；患者CSV median为精确支持像素中位数，另有image-region-median中位数以明确聚合。pooled及患者等权均保留，含前景/背景/RV/MYO/LV及无支持NA。像素loss贡献分母为完整65536，条件valid均值单列。uniform每图固定原w均值，预测/误差相同；delta=mean((w-mean(w))*e)，不是训练消融成绩。分位切点重复合并、不分割同权重，support fraction为实际值，不能称精确top20%。

需要判断clean anatomy、seam、cutout和错误响应的相对强度，应引用各区域完整表及width1/3；若seam或cutout高响应，同样保留。该定义支持“confidence-transition weighting对不同空间区域重新分配一致性损失”，不能将w称校准的边界正确概率或错误纠正率。正文不把CTW公式里的student confidence替换成TCSMix的teacher q_B。

English draft:
We analyzed CTW in frozen evaluation mode on validation images. The two students received spatially aligned weak and strong mixed views. CTW used the spatial variation in their mean maximum probability to assign weights between 0.25 and 1.25. We measured the responses at source anatomy boundaries, mixing seams, and cutout edges separately. The clean anatomy analysis excluded artificial-boundary neighborhoods and occluded pixels. A constant weight with the same image mean provided an offline comparison of spatial allocation. The numerical region results below describe where the frozen loss places emphasis.

完整input/map数组保存在state/input_arrays及state/map_arrays，仅服务器保留；源路径/哈希和重建脚本入包；2张小型w/e/置信度/错误/区域缓存不包含原医学图像/GT，用于算术复核。匿名示例包含固定2患者及clean富集min/nearestmedian/max，包含不利样本，非随机代表抽样。

精度口径：前向与空间图FP32；统计求和/均值使用float64，离线uniform标量为整图float64均值并独立构造常量数组核验均值完全相同，不当作E1训练loss的bitwise重放。R2 original与matched92在本配对中几何和输入相同，216对空间图逐元素相同，只作一致性记录，不新增独立观察。q_B列均为冻结离线批次信号；Full为辅助量，matched92几何不由该量控制；只有R1原策略用于该诊断矩形，不冒充历史batch轨迹，角色另表A3_controller_signal_roles.csv。

For Full_main, the patient-equal mean weight was 0.3984 in clean anatomy boundary bands, 0.2603 at seam-only bands, and 0.2510 in the remainder. The same-mean spatial comparison changed the clean-band conditional consistency term by 0.008148. These frozen-model observations describe spatial emphasis.

For R1_v2_B, the patient-equal mean weight was 0.4027 in clean anatomy boundary bands, 0.2618 at seam-only bands, and 0.2510 in the remainder. The same-mean spatial comparison changed the clean-band conditional consistency term by 0.008091. These frozen-model observations describe spatial emphasis.

For R2_fixed92, the patient-equal mean weight was 0.4024 in clean anatomy boundary bands, 0.2609 at seam-only bands, and 0.2509 in the remainder. The same-mean spatial comparison changed the clean-band conditional consistency term by 0.008431. These frozen-model observations describe spatial emphasis.

## 权重分布补充

Full_main：患者等权像素中97.100%权重落在[0.25,0.27)，不是指精确等于0.25。同均值uniform分布与空间分配均保留。新增linear/log双视图只改善尾部显示，不改变任何统计、阈值或方法。
R1_v2_B：患者等权像素中96.971%权重落在[0.25,0.27)，不是指精确等于0.25。同均值uniform分布与空间分配均保留。新增linear/log双视图只改善尾部显示，不改变任何统计、阈值或方法。
R2_fixed92：患者等权像素中97.096%权重落在[0.25,0.27)，不是指精确等于0.25。同均值uniform分布与空间分配均保留。新增linear/log双视图只改善尾部显示，不改变任何统计、阈值或方法。
