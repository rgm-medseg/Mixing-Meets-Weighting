# 短程检查报告

唯一SHORT_TCSMIX_V2运行已完成，240个联合SSL步、**480次真实optimizer.step调用**，本阶段500调用硬上限未触发。为满足500次optimizer.step硬上限，主体取240个联合步；每联合步调用两个optimizer.step，不能同时采用300–500联合步的建议。没有给A单独训练、没有其他smoke优化器更新、没有修复后训练重试。new方法正式指标NA。

fresh段1–200安全退出75，最后完整checkpoint200；resume段201–240正常退出0。两段同一锁定配置与run，从原Full相同pretrain初始化，非E1续训；controller B、原空间CTW及TRC。实际命令、解释器、时间及完整stdout/train.log均入包。停止后未接管旧队列或正式R1/R2。

|实际尺寸|步数|占比|
|---|---:|---:|
|85×85|6|2.5000%|
|86×86|19|7.9167%|
|87×87|51|21.2500%|
|88×88|93|38.7500%|
|89×89|63|26.2500%|
|90×90|8|3.3333%|

q_B范围0.755596975–0.860026379；实际面积11.02447510%–12.35961914%；最常见88²占38.75%，相邻step尺寸变化176次，fallback0。old_q回放这同一新教师输出时240步全97；不是另训old Full的对照，也不是旧Full全程轨迹。short_trace.csv覆盖连续1–240，逐step公式、每类margin/数量、面积、mask、lambda.05与LR.01核查PASS。

200/240完整checkpoint都实际严格载入三模型和两套optimizer，controller元数据hash相符、stateless=true，保留RNG/采样计划。resume日志与201–240真实轨迹证明续训；16worker增强流/预取不保存，不能声称位级无中断重现。短程checkpoint仅诊断，不可作为正式初始化。

|step|服务器checkpoint|SHA256|
|---|---|---|
|200|`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/runs/SHORT_TCSMIX_V2/checkpoints/recovery_boundary_200.ckpt`|`5451a99c378385e4fdf564f0c42635343fb986fd109c40d5b4e0446de3fe0502`|
|240|`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/runs/SHORT_TCSMIX_V2/checkpoints/last.ckpt`|`8137dfde0d6e60b0bd36cc5e2483fe71d9d9f786af43fe97f733f0c1532c3a2e`|

已通过：范围/损失块一致、原空间CTW与更新方向、无新增参数/教师前向、数据与预训练来源、图/目标共享mask、逐步控制公式、数值稳定、原子完整状态、同run恢复、自动预算退出、正式入口拒绝。未知/未证明：v2完整训练指标、全程尺度/分割收益、完整阶段稳定性、校准或因果解释；NaN/Inf测试拒绝策略属于代码检查。

耗时：校验与配置完成的入口timestamp到各段子进程结束，合计220.243s（约3.67min）。该口径排除timestamp之前早期imports/admission，resume之间阶段分别计数。原兼容receipt给194.075s，因其恢复偏移读取200步验证前的progress，遗漏上一段验证尾部，**不作为实际总耗时**；原receipt保留并在measurement_summary.json解释。正式估时采用179个稳定步median 0.683684s，加单个200步周期观测的验证/保存/安全退出尾部24.325s，非使用有偏receipt。

训练峰值CUDA allocated为16561985536bytes=15.4246GiB（FP32、B24/12标签、256²、双学生+EMA，同A100）。该值是本轮运行实测，不声称与old配对训练差值。相同Full29600 teacher预计算logits+LCC支持上的控制器/生成mask微基准：old median0.5660ms，B 2.4332ms，增量1.8672ms；临时allocated增量差41.159MiB。warmup10/同步计时30，无训练更新，排除teacher及LCC前向处理；不是端到端训练成本增幅。参数增量0，教师前向增量0。
