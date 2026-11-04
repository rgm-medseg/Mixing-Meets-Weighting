# A5计算成本与效率

状态PARTIAL：参数、声明口径内MACs、真实forward测速、验证体推理和R1/R2训练成本已完成。完整非MAC推理FLOPs与全部训练FLOPs未完整覆盖，标NA；历史baseline/Full缺同口径训练段边界。缺口不影响部署参数与实测时延表，论文计算量列应明确为已覆盖MACs而非完整FLOPs。

部署按真实student1，strict加载model_1。四版本真实最佳checkpoint分别测速，单GPU常驻一个模型。TRC-only未发现对应ACDC真实权重，architecture-only共享实际Full实例相同结构计数，不填写分割质量或训练时间。Full/R1/R2参数与层结构相同，训练混合和CTW不进入部署模型。TRC净增641参数（相对baseline0.03509%）；三网络训练参数另列，总量不是部署成本。教师在原工厂ema=True detach_()，参数数目与可训练状态分开。

|method|parameters|covered_GMACs|mean_ms|p95_ms|slices_per_second|volume_ms_per_slice|
|---|---|---|---|---|---|---|
|baseline|1826828|2.9496|2.3928|2.4535|417.9266|5.0430|
|Full_main|1827469|2.9916|3.8287|4.0441|261.1878|6.5257|
|R1_v2_B|1827469|2.9916|3.8410|3.9990|260.3457|6.5248|
|R2_fixed92|1827469|2.9916|3.8627|3.9845|258.8891|6.5057|


表注：1×1×256×256、FP32 eval/inference_mode、GPU0 A100-SXM4-40GB；环境/UUID/TF32既有默认标志附原始JSON；未启用AMP/编译/量化。每个模型首次50预热，随后5个100-forward CUDA event块按预固定轮换顺序、结束同步，预热排除；重新加载时间排除。输入已在GPU，文件IO、H2D、输出保存以及外部softmax/argmax排除。原TRC forward自带概率路由统计、CPU同步和decoder argmax诊断，按真实原路径保留，实测开销包含这些已有实现。吞吐=500/500次event时间总和，mean/median/p95来自全部有效原始次数。整块host墙钟另附，不混用吞吐口径。

峰值allocated/reserved在模型与输入加载、首次预热后reset，包含一个常驻模型/输入；nvidia-smi整卡占用另列。测速期间每0.75秒检查外部GPU PID、块前/后核验，检测到干扰的块保留invalid且不入汇总。体推理输入是固定20个验证frame的预加载CPU数组，3轮，共每方法630切片；包括原order0缩放、H2D、forward、softmax/argmax、D2H与原尺寸恢复，不包括读盘、存盘和GT指标，保持batch1及z轴顺序。

MAC来自Conv/Linear模块hooks和两次einsum显式公式，torch.profiler原生bmm交叉验证。1 MAC=2 FLOPs，仅为已覆盖乘加口径。softmax/LayerNorm/残差融合补充公式另表，不与已计数项重复。未覆盖列表和原生零flops事件完整附A5_OPERATOR_COVERAGE.md与CSV。

R1成功训练26440.013048秒（7.344448小时）；R2成功训练26360.870724秒（7.322464小时），另有零更新启动失败10.039349秒。原R2 runtime_cost把失败加进活动段，本次按exit_code拆分，不修改历史文件。二者30000联合步、两学生各30000成功更新、物理重放0。活动段包含导入/加载/取数等待/EMA/验证/保存，Popen轮询结束检测误差≤5秒。局部step_seconds仅部分循环，不能替代活动段；控制器局部ms作为诊断单列。评价和分析额外时间见A5_training_cost.csv。

英文正文草稿：

The deployed model uses one student network. TRC adds 641 parameters to the baseline, giving 1,827,469 parameters in total. On the A100 GPU, the baseline and paper Full model took 2.393 and 3.829 ms per slice for model forward, respectively. The corresponding validation-volume paths took 5.043 and 6.526 ms per slice, including resizing and device transfers. TCSMix and CTW operate during training and share the same deployed network structure as TRC.


## 训练系统参数与已有训练成本分项

单网络参数位于parameters_total；三网络总参数位于system_parameters_total，训练总可训练数位于system_parameters_trainable。原始结构计数保留于raw_units/A5/parameters.json，汇总表分列避免部署与训练总量混用。

|method|student1|student2|EMA_teacher|system_total|system_trainable|teacher_trainable|
|---|---|---|---|---|---|---|
|baseline|1826828|1826828|1826828|5480484|3653656|0|
|Full_main|1827469|1827469|1827469|5482407|3654938|0|
|R1_v2_B|1827469|1827469|1827469|5482407|3654938|0|
|R2_fixed92|1827469|1827469|1827469|5482407|3654938|0|

|method_id|successful_training_seconds|successful_training_hours|failed_startup_seconds|joint_steps|optimizer1_success|optimizer2_success|evaluation_extra_seconds|analysis_extra_seconds|peak_allocated_GiB|
|---|---|---|---|---|---|---|---|---|---|
|baseline|NA|NA|NA|||||||
|Full_main|NA|NA|NA|||||||
|R1_v2_B|26440.013047698885|7.344448068805246|0|30000|30000|30000|25.07206778600812|25.252298582345247|15.425771713256836|
|R2_fixed92|26360.870723504573|7.322464089862382|10.03934931755066|30000|30000|30000|25.10974880307913|65.61116926372051|15.426748275756836|

计时活动段边界、局部单步计时遗漏与原始来源另见A5_training_cost.csv；失败启动独立列出。
