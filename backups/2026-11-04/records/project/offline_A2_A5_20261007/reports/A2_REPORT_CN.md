# A2教师置信度与质量

状态DONE，2026-10-07T19:46:11.001322+08:00。主要数据：验证210切片/20帧/10患者、真实标注训练136切片/14帧/7患者分开报告；四方法best/last共8个真实EMA教师状态，严格加载ema_model、eval/inference_mode。缺少6000/15000/24000完整教师状态，NA。训练更新0、EMA更新0、测试输入0。原136行来自34个样本×2检查点×2模式；保留并重放全部输入哈希。train_snapshot独立副本每批恢复参数和BN缓存，固定RNG，单列legacy数据组。

作者版本说明：作者提供：论文主版本TCSMix置信度变化有效改变实际矩形，Full Dice90.35%；本任务按真实状态分别分析，不重新裁定历史版本对应关系。。本次真实加载的Full_main路径/分支/步数/哈希附source_manifest；与R1/R2分别列身份。全文统一TRC — Token-Guided Residual Conditioning，TCSMix — Teacher-Confidence-Scaled Mixing，CTW — Confidence-Transition Weighting。外部AdaMix未纳入本次方法结果。

观察：最佳R1教师预测背景的平均最大概率0.998857，预测前景0.962001；前景正确率0.911651，错误前景像素平均置信度0.821101。置信度与质量在不同类别和统计单位呈现差异，完整表保留弱相关、负相关和病例反例。下面相关目标为原始伪标签处理后的GT-present前景macroDice；切片GT全背景时该宏值NA。固定三前景macroDice原规则双空0另列，其完整散点包括空切片。

|method|checkpoint|slice_rho|slice_n|patient_rho|patient_n|patient_control_rho|
|---|---|---|---|---|---|---|
|β-FFT baseline|best|-0.2282855755052287|205|0.12727272727272726|10|NA|
|β-FFT baseline|last|-0.24391416031472068|205|0.11515151515151514|10|NA|
|Full (paper main version)|best|-0.2174203137168519|205|0.12727272727272726|10|0.12727272727272726|
|Full (paper main version)|last|-0.302851754191981|205|0.09090909090909088|10|0.09090909090909088|
|R1 (TCSMix v2_B)|best|-0.2466412686975219|205|0.09090909090909088|10|0.7696969696969697|
|R1 (TCSMix v2_B)|last|-0.25942708297688005|205|0.13939393939393938|10|0.8545454545454544|
|R2 (fixed 92×92)|best|-0.33005296643569915|205|-0.17575757575757575|10|0.8060606060606059|
|R2 (fixed 92×92)|last|-0.2805399043116912|205|-0.07878787878787878|10|0.8303030303030302|


每个固定checkpoint/mode/cohort单独Spearman，无显著性检验；患者相关先在相同有效切片内等权、再帧等权，x/y共享同一有效名单，见A2_patient_correlation_inputs.csv。患者汇总并非新增训练重复。best与last的符号和幅度分别在表中，现有数据说明条件对应关系，未将它们合并为独立患者证据。

|method|group|pixels|mean_probability|accuracy|incorrect_mean_probability|pooled_ECE|patient_equal_ECE|
|---|---|---|---|---|---|---|---|
|β-FFT baseline|0|13176227|0.9989599618115141|0.9983157545783022|0.8217734935993588|0.0006727908984717024|0.0010443100793689653|
|β-FFT baseline|foreground|586333|0.9626285161512912|0.9141699341500479|0.8181788639488116|0.048458582001243325|0.05643398956632958|
|β-FFT baseline|1|200511|0.9627093227585104|0.9019954017485325|0.8343981217776272|0.060752525682251625|0.09291945672286651|
|β-FFT baseline|2|202424|0.9495281952195185|0.8962623009129352|0.8106371764086718|0.053280699673798775|0.055765009874817764|
|β-FFT baseline|3|183398|0.9769995384670908|0.9472458805439535|0.8016045240068312|0.029753657923137198|0.05411487171078836|
|Full (paper main version)|0|13172505|0.9988091344546156|0.9984071366835693|0.8129084240991387|0.0004891583390150128|0.0007778068155280483|
|Full (paper main version)|foreground|590055|0.9598834544898803|0.9116455245697435|0.81634212766694|0.04823792992013678|0.054448143958851856|
|Full (paper main version)|1|205538|0.9605363860580924|0.8871060339207348|0.8476693496105486|0.07345744185508174|0.10740231374994698|
|Full (paper main version)|2|202392|0.9427377070650406|0.8974514802956639|0.7904705389612584|0.04528622676937673|0.04754845772719577|
|Full (paper main version)|3|182125|0.9782003240253144|0.9551132463967056|0.7931064048054022|0.02308707762860892|0.04332642320347187|
|R1 (TCSMix v2_B)|0|13170848|0.9988572545072307|0.9985246963597181|0.8154299717578055|0.0004900446293509829|0.0007931486948952838|
|R1 (TCSMix v2_B)|foreground|591712|0.9620008508861004|0.9116512762965767|0.8211008796680128|0.050349574589523736|0.05681331560454034|
|R1 (TCSMix v2_B)|1|206021|0.9637313644872627|0.8899384043374219|0.8490099635894401|0.07379296014984082|0.10464510509811212|
|R1 (TCSMix v2_B)|2|204713|0.945731703884215|0.8935924929047008|0.7989908959377088|0.05213921097951433|0.05355517374863874|
|R1 (TCSMix v2_B)|3|180978|0.9784336976827934|0.9567958536396689|0.8017612323574679|0.021637844043124542|0.04390160178926564|
|R2 (fixed 92×92)|0|13170105|0.9988554481656731|0.998621347362075|0.8039846408273271|0.0003965857773813263|0.000687768068098852|
|R2 (fixed 92×92)|foreground|592455|0.9619675249734323|0.9127309247115815|0.81878672395825|0.049236600261850845|0.05404591510095352|
|R2 (fixed 92×92)|1|208099|0.9631524283833346|0.8885194066285759|0.8449120526003104|0.07477255376406823|0.10445437781267866|
|R2 (fixed 92×92)|2|203749|0.9462231585844355|0.8967945854948981|0.7992871934149414|0.04942857308953739|0.05091338600668686|
|R2 (fixed 92×92)|3|180607|0.9783640195317521|0.9586062555714895|0.7925634266197713|0.019757763960262645|0.039921138419183204|


分组0=预测背景，foreground=预测类1–3，类别1/2/3=RV/MYO/LV。GT条件分组以及背景→前景、前景→背景计数/分母见A2_confidence_correctness.csv。各类precision/recall及缺失类FP在A2_sample_class_quality.csv；GT缺该类时precision/recall=NA，Dice/IoU按原规则保留0。原始argmax和literal get_ACDC_masks(nms=1)逐类LCC结果分别保存质量。LCC定义来自对应运行源码，未增加处理。

可靠性分10个等宽[0,1]区间，最后闭右端；空区间NA。max概率用于校准，top1-top2间隔另作诊断。pooled-pixel按像素累计；patient-equal先各患者像素累计再等权平均有支持患者，分组支持人数随空区间变化。ECE=Σ_b n_b/N×|acc_b−conf_b|，条件像素范围在axis/group字段；patient-equal ECE=患者ECE均值。原训练控制依版本源码计算，不使用GT；sample_control_proxy为单切片离线响应，batch_control_value始终NA，本轮没有重建真实无标注训练批次。R2 q_B是记录信号、输出几何固定92，baseline没有TCSMix控制值。

英文正文草稿：

We analyzed the frozen EMA teachers on 210 validation slices from 20 frames and 10 patients. For the best R1 teacher, predicted background pixels had a mean maximum probability of 0.9989, while predicted foreground pixels had a mean of 0.9620. Incorrect foreground predictions still had a mean confidence of 0.8211. The Spearman correlation between mean maximum probability and GT-present foreground macro Dice was -0.247 at the slice level and 0.091 at the patient level. These observations show that the relationship depends on the pixel group and the aggregation unit.

图注均在figures/*_caption.md。两位预先固定患者patient002/019共42切片以缓存原始概率与掩码独立复算逐类质量、FP/FN、分组及每个区间支持；回执见A2_NUMERIC_REVIEW.json。未回写源文件。


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


背景支持观察：最佳R1 EMA教师的验证预测像素中，背景占95.7006%。其平均最大概率0.998857，预测前景为0.962001。全像素平均置信度因此主要反映背景像素的贡献；分组可靠性和前景类别等权代理提供互补信息。该陈述来自已有A2像素支持及概率求和，未新增前向。
