本轮仅执行E4/E5各一次seed42正式30,000联合迭代，E5无需预训练，额外预算0；每项隔离短程最多4步含失败。E3及第四格不加入队列。原E1保留3976步暂停，原E2–E5保持未启动，本轮使用独立目录。

E4直接参照可定位R1，唯一科学干预为decoder–head TRC替换成已有16→18→16普通残差适配层，Conv1×1带bias、逐位置LN、GELU及gamma0。v2_B源码、空间CTW、基础目标、预训练共享状态、SGD动量、常数学习率、EMA及选择均继承R1。使用E4自身当前教师形成控制量。实际模型参数、逐名结构和已覆盖MAC见MODEL_STRUCTURE_COMPARISON.csv与reports/MODEL_*。

E5锁定Senyh/AdaMix commit 85361fb91cf43a69a158d41dfa9e7ee9fce36db9，实际入口exp_acdc/train_adamix_ct.py。原始源码与方法文件保留，训练步逐字编译执行，保留官方区域选择、CE/Dice损失、双分支、课程、AdamW及增强，随机初始化，无额外教师。共同batch四路各6：主/辅助有标注各6，主/辅助无标注各6，总输入24、可见GT12。136监督切片重复到1176；伪标签池由官方all-training1312改为63无标注患者1176切片，伪标签流不读取GT。这是共同数据/预算适配，具体差异见E5_PROTOCOL_COMPARISON.csv。

E5主选择student1严格验证前景macro Dice每200步；另留官方student1监督val_loss≤准则每196个batch的epoch，官方PIL/bilinear验证输入保留。课程total_steps=30000//4=7500，cur_step0–29999。官方发表协议成绩与本次共同协议单次适配复现分别标识。

统一评价ACDC_v22_grid_directed_ASD：nearest平面输入缩放256²/预测还原，z轴不变，原H5重建网格spacing=(1,1,1)，无后处理。Dice/IoU fraction，HD95双向表面、ASD预测→GT，距离grid。双空全部0，单空距离体网格对角线。先冻结自身val-best与哈希，再评价全部40帧×3类并合并20患者ED/ES。E4主差R1−E4、E5主差R1−E5，反向差也保留；Dice/IoU乘100为pp，距离越低越好。

定性规则在测试前锁定：患者current−R1 Dice最大、最接近中位、最小，按患者ID处理并列；帧/类别选绝对差最大并按ID；切片从GT或任一预测存在目标类别的切片中选两方法联合错误像素最多，首z并列，零错误回退最大GT面积。所有坐标、方向、缩放、颜色一致：RV蓝、MYO橙、LV红；全部选择记录交付，保留改善与下降的真实方向。

只新增A1轨迹、A4/A6病例类别分析与模型/真实运行成本。既有A2/A3/A5引用paper_offline_followup_20261007已核验版本，不整套重跑。完整FLOPs、历史成本缺项保留NA；allocated/reserved及NVML分开。

R1/R2是空间CTW条件下混合对照，R1/C1是动态v2_B条件下CTW对照，分别保留结果；三格不构成完整交互估计。旧Full90.35%继续原运行归属且NEEDS_MAPPING，ACDC测试未参与原候选方法选择。历史预训练生成命令和原逐训练H5哈希NA，当前资产哈希不代替历史依据。

普通Python/Bash控制器在n1 GPU0独立tmux串行推进，客户端断开不影响。恢复主RNG、完整真实状态及剩余采样计划，worker随机流/预取不位级恢复。失败保留日志/退出码，另一READY任务继续，不从0自动重训。stop仅本轮安全边界保存，服务器重启后需resume。归档采用同一冻结快照、实际解包核验SHA，并纳入根协议CSV及C1补充CSV。
