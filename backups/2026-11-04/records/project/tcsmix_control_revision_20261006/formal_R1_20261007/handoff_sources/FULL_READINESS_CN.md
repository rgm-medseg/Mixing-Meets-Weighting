# 正式训练准备状态

**WAITING_FOR_FULL_APPROVAL**。控制器实现、离线诊断、240步短程、同run保存恢复、范围与成本检查均完成，可以提交R1正式实验方案；本阶段不创建tmux、不启动正式Full、不恢复旧E1。R1/R2实际入口当前都在模型初始化前拒绝formal配置，这是用户明确阶段边界，不是运行故障。

R1：原Full仅换TCSMix v2_B，空间CTW/TRC/EMA及其余协议维持；configs/R1_planned.yaml；同一pretrain net+opt（SHA477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3）、seed42、相同7标注/136切片、B24/12、SGD.01、FP32、30000联合更新，单次，从原pretrain重新初始化。不得使用short200/240或E1/3976权重。

R2：configs/R2_planned.yaml同协议，只固定几何，B控制仍用于可比日志；fixed_side=NA待R1全程真实面积。lock_R2_from_R1.py要求TRAINED/30000及完整control_trace1..30000，按原方形比选择最接近mean实际像素面积的整数平方，另写R2_fixed_locked.yaml及绝对/相对/百分点误差，保留原template；不读测试指标。恢复产生重复记录原表保留，均值用最后一次对应最终逻辑step的记录并披露重复数。当前offline/short面积只能估计，不能锁作R1全程均值。R2运行准备依赖R1完成与尺度锁定，当前不能启动。

稳定训练计时median 0.683684s/联合步，纯计算约5.70h/30000；加现有每200步验证/存盘尾部估计约6.71h。规划**每项约6.7–7.5 GPU小时，两项约13.4–15.0 GPU小时**，另加评价/分析及资源等待；来自本机179个稳定短程步，不保证完整阶段速度，单个尾部还包含人为安全停止、误差需披露。新版R1完整指标全部NA。

仍未解决：B也可能在某一时刻所有固定batch同尺寸（Full30000 train_snapshot全部93）；类平衡放大极小/错误前景类，概率间隔不是校准准确率；仅有初始化/早期E1/末期Full静态来源，无Full中期teacher；短程不能证明完整训练收益；worker恢复非位级精确。选择B只据控制响应，不据测试或诊断Dice排名。

后续只有用户明确确认R1/R2正式训练后，才进入独立正式phase、构建新的tmux与普通服务器队列，R1训练→冻结val-best→确定固定面积→R2及统一评价/分析。当前短程入口上限250联合步且正式配置拒绝；不通过修改其max步数或拿short checkpoint冒充正式启动。正式阶段需单独批准的入口/队列并保留本证据快照。原E1–E5队列保持INTERRUPTED，原配置、checkpoint与历史结果均不变。
