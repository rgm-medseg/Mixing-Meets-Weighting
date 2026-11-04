# E1–E5 实验决策表

截至2026-10-06T22:57:05.937770+08:00，全队列INTERRUPTED；E1完整checkpoint第3976步，E2–E5正式run均PENDING/0。queue.json的latest_step=3975是控制器最后一次采样值，最终receipt/progress/checkpoint均为3976；本轮保留原状态文件。下表是建议，不修改队列执行顺序或配置。

|实验|实际干预与证据|目前能回答的问题|建议|依赖/准备状态|资源估计，GPU0 A100|
|---|---|---|---|---|---|
|E1|用每图原CTW mean广播空间；原L1/detach/.05/方向不变；原权重有空间SD，均匀后0，同输出损失改变|CTW空间分配相对均匀分配的单次控制；不证明跨seed显著性|保留；建议用户确认方案后按原配置续训；本轮不恢复|第3976步full-state可加载，剩余26024步；预训练、配置、source已锁定；worker随机流恢复不精确|剩余约6.13–6.34小时；已用0.9683h；历史/当前速度估算|
|E2|固定97；Full已记录150步97，相同位置RNG则mask同；Full缺29850步尺度日志|最多检验固定控制在完整新run上的行为；当前不能归因动态收益|暂缓；无须为维持原表格而训练近似空干预；尚不能断言全程严格等价|配置锁定、既有3步工程smoke PASS；正式未启动；缺真实Full全轨迹无法补造|若保留约7.07h，潜在冗余|
|E3|同E1均匀函数、同E2固定97；E1全部3976步97|与E1尾程几何是否分叉；目前尺度因子退化|暂缓；冗余风险高，等待方案决定是否取消原定义|同函数/固定值核验、旧smoke PASS；未来26024步q未知|若保留约7.07h，潜在冗余|
|E4|TRC641参数换1×1/LN/GELU/1×1残差647，gamma0，两学生/EMA均替换|共享token检索相对普通残差适配，控制其余实际算法；仅单次对照|保留价值，待统一方案后再启动|同预训练/split/source；既有3步fresh/restore smoke及EMA PASS，无新增依赖；必须继续记录实际尺度|约7.07h粗估；+6参数/student，残差MAC−10%，非总网变化；不能据此精确推算耗时|
|E5|官方核心双学生、四loader、原生AdaMix/DSCLossH，无EMA/TRC/CTW/FFT；wrapper适配协议|与最近外部方法同split/评价的受控比较，不是无改动官方CLI原样运行|保留价值，待方案后再启动|commit85361fb…已锁，依赖本地、四loader+3步保存/恢复smoke PASS；训练入口train_entry→official_adapter；预算/选择/PIL等差异有完整清单|30000 updates，四loader各B16；可靠总耗时NA，3步smoke含启动不足外推；旧smoke显存峰值约8.00GiB仅工程参考|

E1恢复核查实际加载了三个state_dict、两套SGD、主Python/NumPy/Torch/CUDA RNG，重放本epoch剩余6个batch ID且逐项一致；未调用训练入口、DataLoader训练或optimizer.step。epoch361、采样计划11批、已消耗5批；主LR两组均.01、FP32无AMP（amp_state=None），val-best metadata保留。150个参数/学生中90个已有momentum state；其余是未在当前loss图产生梯度的projection/prediction/contrastive heads，SGD按需创建state，完整载入原保存状态而未补造。checkpoint含全部模型参数与buffers，不是best-only权重。

恢复边界：16worker RNG、预取增强张量未保存；重启worker会改变其后数据增强随机实现，resume_exact=False；不宣称与不间断训练位级一致。diagnostic process不改变checkpoint或原进程RNG。本轮恢复可行性由实际加载与采样重放验证，尚未实际执行第3977步。

E2/E3差别的必要条件是动态分支在某执行step取到非97边长；连续s或日志名称不同不是干预。固定分支q本身仍来自softmax，动态分支额外argmax/class_hist只作统计，不新增随机抽样；同97时两者np.random.randint调用范围/次数相同。48个离线固定批次在同RNG下mask逐元素相同，证据限于这些批次，不能推广成历史30000步或未来训练等价证明。

E5完整命令与官方repo/source/adapter哈希在sources/E5_OFFICIAL_METHOD_AUDIT.json；所有E1–E5原配置入包。既有smoke是此前工程检查，不能当正式实验或把其checkpoint用于正式初始化。E4/E5不因TCSMix尺度退化被判为无价值。
