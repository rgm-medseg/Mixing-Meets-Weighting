# TCSMix 机制核查

本轮结论：**B 在已覆盖迭代中成立，完整历史轨迹仍有 D 的证据缺口；未发现 A 型实现错误，也没有发现 C 型额外几何变化。** 公式中的连续 s 有变化，实际整数矩形在旧 Full 的 150 个已记录迭代及 E1 的全部 3976 个已执行迭代中均为 97×97。两者不能混称为“全部 Full 训练恒定”。本轮未恢复 E1、未训练、未更新参数、未访问测试数据，所有历史与锁定配置保持原字节。时间：2026-10-06T22:57:05.934435+08:00。

## 1. 教师输入到掩码的计算链

以下路径均为包内实际源文件；original_path 与 SHA256 见 source_manifest.csv。完整摘录见 sources/CODE_CHAIN_EXCERPTS.md。

|阶段|实际行为|代码与配置证据|
|---|---|---|
|教师输入|B24 中无标注12张，拆成6+6；共享几何后的 weak image；strong另一次前向产生目标，但不用于q_mix|historical_full_effective_source.py:333–357；dataset.py:436–461|
|原始输出|pre_a=ema_model(uimg_a)、pre_b=ema_model(uimg_b)，形状[6,4,256,256]；UNet最终out_conv输出logits，不预先softmax|historical_full_effective_source.py:345；unet.py:390；v14_mask_adamix_components.py:_install_router|
|概率|softmax(logits.detach(),dim=1)，类别维正确；max(dim=1)后为[6,256,256]|v14_mask_adamix_components.py:128–130；离线probability_sum误差≤2.3842e-7|
|batch平均|每组对样本和所有空间位置平均，然后两组平均；组等大小时等价全12张平均|v14_mask_adamix_components.py:128–130|
|s映射|q clamp[0,1]；s=.18+(.38−.18)q；min/max参数的保护范围不改变当前端点|v14_mask_adamix_components.py:125–131；Full Namespace及config .18/.38|
|取整/裁剪|int(round(H*s))、int(round(W*s))；Python round 为最近整数、.5向偶数；clamp[1,H−1]/[1,W−1]|v14_mask_adamix_components.py:132–133|
|最终掩码|矩形内0、外1；同一位置/大小扩展到6张，loss_mask一致；两组与双视图复用它|v14_mask_adamix_components.py:134–155；historical_full_effective_source.py:364–373|
|图/目标交换|weak/strong均以同一mask配对互补；目标与loss_mask对应；无其他s输入损失|historical_full_effective_source.py:364–397|

硬标签 get_ACDC_masks(...,nms=1)用于混合目标；生成矩形传入的是 **pre_a/pre_b 原始 logits**，没有传plab_a/plab_b。argmax仅另作patch class_histogram诊断，不参与尺度。未发现张量/维度误用。教师对象保持默认 train 模式；no_grad禁梯度，但不关闭dropout或BN行为。教师EMA在本步优化后更新整个state_dict，包括buffers。因此离线eval、train_snapshot和训练中某一步不是同一统计条件。额外synthetic_probability_mask_probe.csv用人工常数softmax分布验证实际动态函数：q约.25/.50/.90/.98/.995时取59/72/92/96/97，而相同输入下E2/E3一直97；表明动态s确实影响函数，不是被固定覆盖。该探针不是真实教师或历史证据。

日志中的side_fraction是计算得到的连续s；patch_area_ratio是由最终取整后patch_x*patch_y计算的实际交换面积。两者不可互换。原始日志没有逐步保存mask数组/坐标/宽高；在已确认256²、同一s控制两轴、无裁剪触发的实现下，可由面积与计算链唯一核验97×97。离线诊断还直接检查了mask零区域的宽、高及像素数。

E1实际入口 train_entry.py → build_beta_source → revision_effective_source.py，从原Full冻结循环引入revision_runtime；动态分支直接调用上述parent函数。E2/E3仅改runtime的mix_mode为fixed，s=97/256=.37890625；该值也存在于E1配置，但E1的dynamic分支**不读取它**。effective_entry_comparison.csv证明E1–E4生成循环与实际E1循环相同，差异在明确runtime配置。fixed分支仍计算q用于记录，但不据其改变边长。

旧Full每个run保存了有效循环和Namespace；未保存当时所有import模块的逐文件哈希。当前冻结组件与2026-09-07旧证据包副本及项目组件均为SHA256 75754fa75e6fd6fc01c97682262895998799c7ce3c0bc2dcbe058ef4c63dcaee；旧endpoint数值与其规则一致。它加强实现证据，但不能补造2026-08训练时未保存的模块哈希或完整shell argv。

## 2. 轨迹、来源和覆盖范围

|运行|独立尺度记录|step覆盖|实际尺寸种类/占比|实际面积范围|记录间变化次数|未覆盖|
|---|---:|---|---|---|---:|---|
|旧Full|150|200,400,…,30000|97×97：150/150=100%|14.3569946289%–14.3569946289%|0|其余29850步，包括1–199；完整轨迹变化次数NA|
|新增E1|3976|连续1–3976，无重复/缺行|97×97：3976/3976=100%|14.3569946289%–14.3569946289%|0|尚未执行3977–30000|

旧Full q范围0.988724470139–0.997913122177，s范围0.377744894028–0.379582624435，均值0.379189255555。E1 q范围0.985085248947–0.996754169464，s范围0.377017049789–0.379350833893，均值0.378668454599。完整原始逐步E1 trace和两份原始Full endpoint摘录均入包；scale_records.csv标明每条来源行号，scale_distribution.csv给出占比/面积/变化及证据性质。

额外找到controller stdout/stderr的同样150条endpoint，逐值与Full train.log一致，**没有增加独立step**。Full training_dynamics有30000行但没有q/s/mask字段；TFRecord只有total_loss/model1_loss/model2_loss/consistency_weight各30000条，没有尺度。正式run只找到best29600、last30000，未找到中间教师快照或保存掩码；另有X03工程smoke检查点，属于不同run，不能填补Full。history_source_inventory.csv及history_search.json记录搜索范围和文件SHA；不是宣称整个服务器不存在任何未索引备份。

97²匹配旧Full **已记录150个迭代**平均实际面积，取整后面积匹配误差0；不等于证明全程平均面积误差0。旧Full高置信度连续s发生变化而实际尺度未变化；E1直到3976步实际几何确已恒定。

## 3. 独立离线置信度诊断

固定输入在前向前锁定：24张无标注训练切片按列表索引等距选取（不读其GT作评价）；14张既有标注切片来自已锁定7例；全部20个验证frame各取固定中间z切片（10患者）。弱增强seed4261006；分12张批次为两等大小组，尾批明确较小。样本路径/hash和实际输入hash见diagnostic_batch_manifest.csv；本轮不访问测试。

4个现有教师checkpoint：Full best29600/last30000，E1 best3200/last3976。每个都做eval和train_snapshot两种模式，总48批；train_snapshot在每批前后恢复独立模型副本的参数和BN buffers、固定dropout RNG，模型state_digest前后完全相同。无优化器step。它近似当前训练前向的模式，**不是恢复历史batch/worker随机流/连续强弱前向序列**。checkpoint是post-step EMA状态，而本步训练q是pre-update teacher。所有48离线批次的动态mask与重置同一随机状态后的固定97mask完全一致，全部97²，不能据此补造历史轨迹。

|教师/step|无标注整体q（像素汇总）|预测背景比例|预测背景平均置信度|预测前景平均置信度|
|---|---:|---:|---:|---:|
|Historical_Full / 29600|0.996544898|95.6151%|0.998556137|0.952688217|
|Historical_Full / 30000|0.996657908|95.5862%|0.998596787|0.954666734|
|E1 / 3200|0.993817806|95.5349%|0.996770024|0.930663109|
|E1 / 3976|0.994165659|95.5157%|0.997037768|0.932988405|

|教师/step|有GT诊断域|真实前景比例|真实前景平均最大概率|真实背景平均最大概率|汇总前景macro Dice|
|---|---|---:|---:|---:|---:|
|Historical_Full/29600|labeled_train|5.8881%|0.95973110|0.99882382|0.95439703|
|Historical_Full/29600|validation|4.6680%|0.95747930|0.99907988|0.92937762|
|E1/3976|labeled_train|5.8881%|0.93957669|0.99804425|0.93590850|
|E1/3976|validation|4.6680%|0.94298512|0.99788052|0.91600892|

这里Dice是固定2D诊断样本的逐类汇总，并非模型正式测试/验证整卷得分；all3类double-empty记0，另给GT-present-class Dice防止遗漏类影响被隐藏。例：patient030_frame02:z5，Full29600的q=0.997173667而all3诊断Dice=0.45405、present-class Dice=0.68108；E1/3976 q=0.995913267，对应0.41840/0.62760。高全图q不能保证前景质量。E1/3976 train_snapshot描述性Spearman：14标注切片−0.10330、20验证中间切片0.45865；不同域/模式不稳定，不作独立样本检验/因果推断。

确认事实：背景在这个固定批次中占约95.5%，其高概率在全图平均中占主要权重；前景置信度也较高，但明显低于背景。不能据预测背景多断言前景全部塌缩：有GT样本也约94–95%为真实背景且存在较好前景分割。可能原因：预训练及学习后教师饱和、全像素聚合稀释前景不确定性、batch平均进一步压缩变化；本轮没有初始教师或历史每batch概率分布，不能把这些原因分别定量归因到Full全程。

## 4. 取整阈值与“动态”失效位置

H=W=256时d=round(46.08+51.2q)。每减一像素通常需q变化0.01953125。97区间为 **q>0.984765625** 到1；精确q=.984765625得到96.5，round取偶数96。96内部区间约0.965234375–0.984765625，两个精确半整数端点取96。float32 softmax/均值与Python浮点会影响恰好临界的数值，报告的是解析阈值。裁剪在当前范围不触发。

E1最小q只比97阈值高0.000319623947，但全部已执行step仍落在97区间；不能保证未来始终如此。四类softmax最大概率≥1/4，所以合法q实际范围[.25,1]，对应s约[.23,.38]和59–97像素；46是把形式端点q=0代入的值，不是4类概率可达到的下限。quantization_thresholds.csv列全部可达整数区间。

问题发生在**高且变化窄的全像素批均q，经线性映射后全部落入同一个整数取整格**，而非已查出的参数覆盖或硬标签错误。真实s变化与有效几何变化必须分开描述。随机位置仍每次变化；它不会成为“动态尺度”的证据。当前np.random.randint上界为H−d（exclusive），最后一个合法起点未被采到，是原有/固定控制共有的采样约定，不能将其当成尺度恒定的原因或在本轮静默修正。

## 5. 对实验与论文的影响

E1保留独立CTW空间分配对照价值。CTW_actual_difference.csv基于E1/3976两学生真实输出、12固定标注诊断样本、同一弱/强混合输入，原权重空间SD范围0.03474959–0.06213243，替代权重空间SD为0；空间损失减均匀损失0.000273042–0.001868169。均值一致不保证损失/梯度相同；本轮只前向，未生成参数梯度或更新。lambda .05、概率类别平均L1、student2 detach、student1方向仍由实际函数和既有独立梯度gate支持。

E2相对Full：已记录迭代几何一致，未记录Full迭代可能不同，只有独立控制实际未来全程轨迹才能排除。但在目前证据下，它不能直接支撑“置信度动态尺度优于固定尺度”。E3相对E1：1–3976已执行轨迹在同RNG条件下几何一致，尾程未知，因而存在很高冗余风险。原2×2的尺度因素在已观测范围没有实际干预；即使最终数值不同，也不能仅由开关名称解释为动态调度或协同收益。

E4仍可比较TRC与普通匹配残差适配层；E5仍可比较官方AdaMix-CT核心方法的同协议适配。TCSMix这项发现不否定它们或CTW。具体准备状态、资源和建议见EXPERIMENT_DECISION_TABLE.md与NEXT_STEP_PLAN_CN.md。建议属于方案，队列/config均未改。
