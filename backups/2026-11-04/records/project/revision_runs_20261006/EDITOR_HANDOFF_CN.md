# β-FFT 修订证据报告

更新时间：2026-10-06T22:05:07.582611+08:00；队列：RUNNING。训练完成、评价完成与归档分别记录。

|任务|状态|Dice (%)|IoU|HD95 (grid)|ASD (grid)|
|---|---|---:|---:|---:|---:|
|Historical_baseline|EVALUATED|89.66063409597648|0.8182007425305878|1.7618789072469767|0.5722597264440701|
|Historical_Full|EVALUATED|90.34965765204997|0.8293272953910553|1.4233666940334322|0.4290264993301964|
|E1|RUNNING|NA|NA|NA|NA|
|E2|PENDING|NA|NA|NA|NA|
|E3|PENDING|NA|NA|NA|NA|
|E4|PENDING|NA|NA|NA|NA|
|E5|PENDING|NA|NA|NA|NA|

- A1: PARTIAL — Historical sparse trace drawn; full-step traces added from new runs. Missing runs remain DEFERRED.
- A2: PARTIAL — Existing teacher best/last checkpoints on 14 fixed already-labeled slices; new checkpoints DEFERRED. E5 has no EMA teacher by official definition.
- A3: PARTIAL — True two-student/EMA outputs on four fixed already-labeled slices mixed offline; 2-pixel neighborhoods separated into GT-only/mix-only/overlap/interior. No anatomical semantics assigned to tokens; normalization denominator floor 1e-6.
- A4: PARTIAL — ACDC frame/class raw records aggregated by patient for paired differences; COVID shown individually with n=4 per run. No slicing/checkpoint pseudo-replication.
- A5: PARTIAL — Same A100/FP32/shapes; CUDA synchronized, warmup10 + timing30; repeats are timing only. Training wall/peak from new runs; historical training wall source audited separately.
- A6: PARTIAL — Prespecified extreme/median patient gain selection, same frame/slice/display/colors; lowest gains retained even if adverse. Test used only for final descriptive display.

固定矩形97×97，边长比例0.37890625；来源是历史 Full 每200步的150条记录，并非全程逐步平均。所记录面积恒为14.357%，这限制了E2动态几何优势的解释。

只报告单次运行差值，见 reports/single_run_differences.csv。负结果不删除；不做跨种子显著性或因果声明。

预训练沿用已核验的net/opt文件，文件产生时的实际命令、seed和预算无法从权重推出，保持NA。半监督预算每项30000updates；E5采用官方两学生AdamW1e-4、B16、四加载器、从头初始化，没有隐藏预训练。

checkpoint保留服务器，路径/哈希见CSV和reports/source_index.csv。16worker随机流与预取无法完整恢复，续训保存主进程随机状态及批次ID但不宣称位级精确。

所有范围、源码、历史口径差异见protocol_audit.md、implementation_audit.md、hyperparameter_provenance.md。未完成训练的数字保持NA，不能写作已完成的论文事实。


COVID评价缩放spacing问题：最终evaluator沿axis=2正确重建XYZ，轴序本身正确；512→256缩放后spacing未更新。历史记录的mm标签没有反映正确resize像素间距，原结果保留。已用固定模型的预测另存legacy重放、corrected derived center grid、corrected native grid三种固定口径。见reports/covid_geometry_audit.csv与covid_existing_prediction_reevaluation.csv，不选择更好数字替换口径。

|COVID run|评价口径|Dice|IoU|HD95|ASD|
|---|---|---:|---:|---:|---:|
|COVID10_Full|corrected_derived_center_grid|0.6794082017690449|0.5237523299362863|18.957128805392596|5.008397271886982|
|COVID10_Full|corrected_native_grid|0.6770425216753222|0.5210756333574784|20.103570449638184|5.018347891962928|
|COVID10_Full|legacy_replay|0.6794082017690449|0.5237523299362863|11.877109828896515|3.0315494897283286|
|COVID10_baseline|corrected_derived_center_grid|0.6550316795817169|0.4964359644862783|23.022861570142222|5.3852991579706755|
|COVID10_baseline|corrected_native_grid|0.6532490328933243|0.49448543197399153|24.901897145211898|5.440145779538981|
|COVID10_baseline|legacy_replay|0.6550316795817169|0.4964359644862783|13.891287168761059|3.1791635442421144|
|COVID5_Full|corrected_derived_center_grid|0.6731436651428727|0.514497054730846|33.56438112527303|6.350107949773349|
|COVID5_Full|corrected_native_grid|0.6704770507779239|0.5114286529628218|33.938036887065394|6.46748652935859|
|COVID5_Full|legacy_replay|0.6731436651428727|0.514497054730846|22.791388468075546|4.058622711161191|
|COVID5_baseline|corrected_derived_center_grid|0.6367743321888475|0.478921765140088|35.761468219124396|6.855459773400048|
|COVID5_baseline|corrected_native_grid|0.6350454028446889|0.47721132629645335|36.241783932257334|6.9947392805441835|
|COVID5_baseline|legacy_replay|0.6367743321888475|0.478921765140088|23.51210872051254|4.262073756032654|

# 给论文写作助手

E1比较同一控制模型的空间CTW与逐图均匀均值权重；均值相同不等于梯度范数相同。E2固定实际面积，E3复用同一尺度/均匀实现，交互定义(Full−E1)−(E2−E3)。E4普通1×1/LN/GELU/1×1残差，宽18，647参数，对照TRC641，误差+0.936%；三模型同时替换，零门控和EMA保留。E5是官方AdaMix-CT方法代码与原生U-Net的同划分适配，不是β-FFT中名为AdaMix的TCSMix。

CSV中Dice/IoU是[0,1]比例，dice_percent为百分数，差值pp为百分点。ACDC HD95/ASD为原H5体积网格单位；ASD为MedPy单向prediction→GT。case是patient+frame，patient列才是患者独立单位；class_id1/2/3对应RV/MYO/LV。A2前景/背景置信度按GT分区，checkpoint/slice不是独立患者。A3统计GT-only/mix-only/overlap/interior，混合边界允许与GT边界重叠。

每张图的文件名/标题与selection_and_provenance.csv、confidence_proxy.csv或spatial_response.csv共同给出run、checkpoint、case/slice及step。A1历史轨迹每200步采样，新轨迹每步记录，两者不拼接。A5计时重复不是重复训练；MAC计数不含归一化等，具体约定在cost.csv。A6预先固定最大/中位/最小患者增益，同病例切片显示，保留变差结果。

来源材料、配置、日志、CSV、SVG/PNG和脚本均进入revision_evidence_20261006.tar.gz，可离线审查；原始影像、GT数据数组、凭据和大权重不打包。压缩包SHA与内部SHA256SUMS用于完整性检查。

作者更正：ACDC测试成绩未参与候选方法选择。历史报告中的排序标签不能自行推导相反结论；如需证明实际决策时点仍应核对原作者记录。COVID外部文献行单位未知保持NA，不能自动统一写mm。

A6另保留COVID所有4例，5%/10%同切片/GT/颜色显示，来自既有baseline预测及原Full冻结检查点重新推理，无COVID训练。Corrected native grid在canonical原网格上对categorical预测nearest回采样、GT不缩放、spacing按(Z,X,Y)排序；derived-center grid则按SciPy grid_mode=False的像素中心跨度更新spacing。两口径明确区分，不跨表混用。外部论文行仍NA，未经同程序重评价不声称全表方法口径统一。

## 本轮补充核查与论文更新


固定尺度：FIXED_RECTANGLE_AUDIT.md及fixed_rectangle_historical_records.csv明确旧Full150条step200..30000稀疏记录；s范围0.377744894028–0.379582624435、均值0.379189255555，记录点97²比例100%，实际面积均值14.3569946%、固定误差0。历史batch切片ID与每时点teacher快照缺失，不能声称全30000步平均；E1逐步曲线独立标新run，不能冒充旧Full。E2/E3配置未变。

COVID：COVID_SPACING_AUDIT.md、covid_metrics_before_after.csv和macro CSV覆盖所有已定位本地预测/run。spacing-only保留预测/GT/重采样/后处理，仅改sx/sy；Z不缩放，Dice/IoU应完全不变。原网格另存结果改变了GT/预测重采样，须单独说明。原平面包括512²和630²，不全是512²。CSV含完整源/hash/axis/shape/spacing与真实重算的HD95/ASD；外部方法身份/输入缺失仍NA，不能根据比例估计。

协议：COMPARABILITY_AUDIT.md和protocol_comparison.csv逐项说明保持一致/预期改变/其他差异/无法核验。补清ICT实际最大系数5×.1=.5，不把参数consistency=.1当作最大有效权重。原预训练生成预算/seed、原训练当时清单与全部import依赖hash无法核验，不能写完全一致或位级精确续训。E4残差参数+.936%不是全网络增幅。E5实际执行train_entry→official_adapter，官方方法核心来自锁定commit的exp_acdc/train_adamix_ct.py；是有明示预算/验证/划分适配的官方方法实现，并未直接运行未经改动原main。

新增文件及原始统计实际进入证据包，保持五项单次结果包括下降。当前任务状态见reports/CURRENT_QUEUE_STATUS.csv与SUPPLEMENT_QUEUE_SNAPSHOT.json；未完成测试指标仍NA。原queue继续自动训练/评价/A1–A6/汇总。审计跟随进程位于同tmux的audits窗口，在queue最终归档后追加本节并再次生成一致快照包，避免最终summarizer覆盖本次写作交接。
