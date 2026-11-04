# A2/A5写作交接

新增内容使用TRC — Token-Guided Residual Conditioning；TCSMix — Teacher-Confidence-Scaled Mixing；CTW — Confidence-Transition Weighting。

A2建议放机制分析节：最佳R1教师预测背景平均最大概率0.9989，预测前景0.9620；错误前景仍有0.8211平均置信度。平均最大概率与GT-present伪标签前景macroDice的Spearman在切片层面为-0.247，患者层面0.091。分组及统计单位呈现不同关系，完整方法/检查点/类别比较见A2_paper_summary.csv、A2_group_summary.csv、A2_correlations.csv。负相关和弱相关保留。

We analyzed the frozen EMA teachers on 210 validation slices from 20 frames and 10 patients. For the best R1 teacher, predicted background pixels had a mean maximum probability of 0.9989, while predicted foreground pixels had a mean of 0.9620. Incorrect foreground predictions still had a mean confidence of 0.8211. The Spearman correlation between mean maximum probability and GT-present foreground macro Dice was -0.247 at the slice level and 0.091 at the patient level. These observations show that the relationship depends on the pixel group and the aggregation unit.

建议组合A2_confidence_quality与A2_reliability正文，A2_class_confidence和A2_correct_error_confidence放补充；全部图有SVG/PDF/PNG及图注。主要队列验证210切片、20帧、10患者；标注训练136切片单列；原34个小样本使用历史增强/中间层切片并原样保留。A2质量属于EMA教师离线切片诊断，原始argmax/原LCC分别报告。主版本90.35%、R1 89.98%、R2 90.20%是已有student1完整体测试结果，身份各自保留，A2/A5不产生动态混合独立性能增益结论。作者关于90.35%主版本动态机制的说明保留在预分析与A2报告，不转移到R1。

A5建议放效率节的表：

|method|parameters|covered_GMACs|mean_ms|p95_ms|slices_per_second|volume_ms_per_slice|
|---|---|---|---|---|---|---|
|baseline|1826828|2.9496|2.3928|2.4535|417.9266|5.0430|
|Full_main|1827469|2.9916|3.8287|4.0441|261.1878|6.5257|
|R1_v2_B|1827469|2.9916|3.8410|3.9990|260.3457|6.5248|
|R2_fixed92|1827469|2.9916|3.8627|3.9845|258.8891|6.5057|


The deployed model uses one student network. TRC adds 641 parameters to the baseline, giving 1,827,469 parameters in total. On the A100 GPU, the baseline and paper Full model took 2.393 and 3.829 ms per slice for model forward, respectively. The corresponding validation-volume paths took 5.043 and 6.526 ms per slice, including resizing and device transfers. TCSMix and CTW operate during training and share the same deployed network structure as TRC.

表注必须标已覆盖GMACs，完整FLOPs=NA/PARTIAL；单模型student1与三网络训练系统分列。模型forward时延与预加载CPU到原尺寸mask体时延分列；距离指标未在本次测速计入。R2成功训练7.3225小时，失败启动10.04秒另列。R1/R2计时范围相同，单次运行差异只描述。历史训练成本NA、中期教师NA、TRC-only仅architecture-only可结构计数，这些缺口应在表注准确说明。

证据映射：A2_SAMPLE_CLASS表→逐类质量与FP/FN；A2_confidence_bins→可靠性与ECE；A2_patient_correlation_inputs→相同切片/帧汇总；A5_parameters_compute和A5_OPERATOR_COVERAGE→结构/算子；A5_inference_raw/summary→500次有效原始测速和吞吐；A5_volume_runtime→3轮20frame；A5_training_cost+legacy/training_cost→历史训练边界。


## 实际控制代理值与质量的补充

|method_id|checkpoint_tag|level|spearman|valid_n|patients|
|---|---|---|---|---|---|
|Full_main|best|slice|-0.2174203137168519|205|10|
|Full_main|last|slice|-0.302851754191981|205|10|
|R1_v2_B|best|slice|0.7991739796629056|205|10|
|R1_v2_B|last|slice|0.7973631424989551|205|10|
|R2_fixed92|best|slice|0.7950285901337921|205|10|
|R2_fixed92|last|slice|0.7978339601615824|205|10|
|baseline|best|slice|NA|0|0|
|baseline|last|slice|NA|0|0|
|Full_main|best|patient|0.12727272727272726|10|10|
|Full_main|last|patient|0.09090909090909088|10|10|
|R1_v2_B|best|patient|0.7696969696969697|10|10|
|R1_v2_B|last|patient|0.8545454545454544|10|10|
|R2_fixed92|best|patient|0.8060606060606059|10|10|
|R2_fixed92|last|patient|0.8303030303030302|10|10|
|baseline|best|patient|NA|0|0|
|baseline|last|patient|NA|0|0|

R1的实际前景类别等权间隔代理与GT-present伪标签macroDice，在best切片/患者层面ρ=0.799174/0.769697，last为0.797363/0.854545。两个保留状态均呈正相关。真实可加载论文Full的全像素置信度代理在best切片/患者层面ρ=-0.217420/0.127273。方法与置信度定义分别列出；R2中q_B是记录信号，固定几何仍为92×92。控制代理与伪标签质量共享预测/LCC支持，本项描述当前冻结教师的条件关系，未分离训练机制的独立收益。

The foreground class-balanced margin proxy in R1 had a Spearman correlation of 0.799 with GT-present foreground macro Dice across 205 validation slices. The patient-level correlation was 0.770 across 10 patients. The final teacher retained positive associations of 0.797 and 0.855, respectively. For the separately loaded paper Full teacher, the all-pixel confidence proxy gave correlations of -0.217 at the slice level and 0.127 at the patient level. The foreground-conditioned signal tracked validation quality more closely in these frozen R1 snapshots.

证据：A2_actual_control_summary.csv、A2_patient_correlation_inputs.csv、A2_actual_control_quality及图注。

逐类precision/recall汇总：

|method_id|class_name|weighting|precision|recall|
|---|---|---|---|---|
|baseline|RV|pooled_pixel|0.9019954017485325|0.9164335806072399|
|baseline|RV|patient_equal|0.8716654585658276|0.933996254949444|
|baseline|MYO|pooled_pixel|0.8962623009129352|0.9090203073408056|
|baseline|MYO|patient_equal|0.8948723087158508|0.908652759389047|
|baseline|LV|pooled_pixel|0.9472458805439535|0.9555457770688375|
|baseline|LV|patient_equal|0.9201221120233148|0.9563341312339647|
|Full_main|RV|pooled_pixel|0.8871060339207348|0.9239024686853946|
|Full_main|RV|patient_equal|0.855410663273562|0.9372144481334063|
|Full_main|MYO|pooled_pixel|0.8974514802956639|0.9100825220584919|
|Full_main|MYO|patient_equal|0.8962622858052457|0.9096863953273205|
|Full_main|LV|pooled_pixel|0.9551132463967056|0.9567943675916504|
|Full_main|LV|patient_equal|0.9325199174543922|0.9567644755241164|
|R1_v2_B|RV|pooled_pixel|0.8899384043374219|0.9290303619927844|
|R1_v2_B|RV|patient_equal|0.8582717605726276|0.9397569483143604|
|R1_v2_B|MYO|pooled_pixel|0.8935924929047008|0.9165610297470226|
|R1_v2_B|MYO|patient_equal|0.8924933394690513|0.9162304944044084|
|R1_v2_B|LV|pooled_pixel|0.9567958536396689|0.9524435521575314|
|R1_v2_B|LV|patient_equal|0.9328615771178305|0.952536208109972|
|R2_fixed92|RV|pooled_pixel|0.8885194066285759|0.936904617130812|
|R2_fixed92|RV|patient_equal|0.8583870017714883|0.9446738128182203|
|R2_fixed92|MYO|pooled_pixel|0.8967945854948981|0.9155138463696808|
|R2_fixed92|MYO|patient_equal|0.8957840153239702|0.9151566062782065|
|R2_fixed92|LV|pooled_pixel|0.9586062555714895|0.9522895409917219|
|R2_fixed92|LV|patient_equal|0.9367809000299776|0.9524265693741258|


类别观察：最佳R1 EMA教师在验证集的MYO召回率最低，为0.916561；LV最高，为0.952444。各类别precision/recall的pooled-pixel和patient-equal结果均保留。

For the best R1 EMA teacher, MYO had a pooled-pixel recall of 0.9166, while LV had a recall of 0.9524. The class-specific results show where foreground confidence and recall differ.


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


背景支持观察：最佳R1 EMA教师的验证预测像素中，背景占95.7006%。其平均最大概率0.998857，预测前景为0.962001。全像素平均置信度因此主要反映背景像素的贡献；分组可靠性和前景类别等权代理提供互补信息。该陈述来自已有A2像素支持及概率求和，未新增前向。
