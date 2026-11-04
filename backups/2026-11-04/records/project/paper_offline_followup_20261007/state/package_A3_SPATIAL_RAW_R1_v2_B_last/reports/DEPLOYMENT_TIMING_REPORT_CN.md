# 部署等价与成本

DONE：840个方法×切片核验、80个方法×帧体哈希，logits exact=True，max_abs=0.0；mask、native-volume、逐类Dice/IoU/HD95/ASD一致。模型参数和BN状态前后哈希不变。指标距离单位原网格像素，双空全0，单空对角距离；不缩放病例轴，不做LCC。

GPU0 A100 FP32存储dtype，matmulTF32=False，cudnnTF32=True，deterministic=True，benchmark=False与旧环境一致。原路径全部诊断保留；部署仅独立bound adapter删除router entropy/utilization/norm/finite CPU同步和decoder类别占比argmax，不改任何预测运算。baseline两路径算子相同。adapter_diff.patch记录完整原forward到部署forward差异。

每次加载均预热50，再计时100；5轮方法/路径轮换，零输入与val.list首帧z0分开。资源线程创建/结束、块前后nvidia-smi不在host计时边界，计时期间并行监测线程可能产生轻微CPU调度扰动，所有方法同口径；外来GPU PID块保留invalid后等待重测。每次单模型驻GPU。CUDA-event吞吐=500/对应event总秒数；host墙钟另列。峰值allocated/reserved在加载+预热后reset，包括模型和输入；整卡占用独立列。

|method_id|path|input_kind|mean_ms|median_ms|p95_ms|throughput_slices_per_second|relative_to_baseline|
|---|---|---|---|---|---|---|---|
|baseline|original_runtime|zeros|1.759529|1.682432|1.907866|568.333924|1.000000|
|baseline|original_runtime|validation_first_slice|1.758159|1.683456|1.846784|568.776822|1.000000|
|baseline|deployment_no_diagnostics|zeros|1.767348|1.684992|2.091264|565.819450|1.000000|
|baseline|deployment_no_diagnostics|validation_first_slice|1.769603|1.683456|1.802086|565.098476|1.000000|
|Full_main|original_runtime|zeros|3.036089|2.952688|3.103013|329.371134|1.725512|
|Full_main|original_runtime|validation_first_slice|3.033750|2.952464|3.046314|329.625001|1.725527|
|Full_main|deployment_no_diagnostics|zeros|2.093537|2.014720|2.329498|477.660468|1.184564|
|Full_main|deployment_no_diagnostics|validation_first_slice|2.086373|2.017280|2.291354|479.300594|1.179006|
|R1_v2_B|original_runtime|zeros|3.032857|2.950320|3.033926|329.722118|1.723675|
|R1_v2_B|original_runtime|validation_first_slice|3.027179|2.950240|3.036424|330.340578|1.721789|
|R1_v2_B|deployment_no_diagnostics|zeros|2.085378|2.014208|2.077696|479.529359|1.179947|
|R1_v2_B|deployment_no_diagnostics|validation_first_slice|2.100916|2.020352|2.318541|475.982805|1.187225|
|R2_fixed92|original_runtime|zeros|3.032669|2.951664|3.123039|329.742507|1.723569|
|R2_fixed92|original_runtime|validation_first_slice|3.023365|2.943264|3.030451|330.757300|1.719620|
|R2_fixed92|deployment_no_diagnostics|zeros|2.101277|2.016256|2.146509|475.901156|1.188943|
|R2_fixed92|deployment_no_diagnostics|validation_first_slice|2.085825|2.012160|2.298470|479.426715|1.178696|

体路径3轮×20帧，预加载CPU图像到native mask，包含平面resize/H2D/forward/softmax argmax/D2H/restore，排除读写盘与GT评价，保持z和batch1：

|method_id|path|mean_ms_per_slice|throughput_slices_per_second|relative_to_baseline|
|---|---|---|---|---|
|baseline|original_runtime|4.369728|228.847201|1.000000|
|baseline|deployment_no_diagnostics|4.369525|228.857829|1.000000|
|Full_main|original_runtime|5.644241|177.171751|1.291669|
|Full_main|deployment_no_diagnostics|4.738675|211.029457|1.084483|
|R1_v2_B|original_runtime|5.661257|176.639219|1.295563|
|R1_v2_B|deployment_no_diagnostics|4.731563|211.346656|1.082855|
|R2_fixed92|original_runtime|5.633371|177.513607|1.289181|
|R2_fixed92|deployment_no_diagnostics|4.682373|213.566903|1.071598|

旧A5结果原样另存historical并保持旧包只读；旧口径含诊断且仅首次块预热，不能直接与新版时间解释训练创新。无诊断部署节省输出无关统计，TRC仍完整保留。单学生参数baseline1,826,828，其余1,827,469，增641；已覆盖MAC2.949644288/2.991587328 GMAC，完整非MAC FLOPs及FFT/反向训练FLOPs NA，历史Full/baseline完整同口径训练时间NA。
