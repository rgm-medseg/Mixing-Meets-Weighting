# Full控制量来源对应

状态NEEDS_MAPPING。作者说明原文保留：“旧版TCSMix置信度变化有效改变实际矩形，机制与R1一致，论文主版本Full Dice为90.35%。”作者更正：ACDC测试集没有参与原论文候选方法选择。

本轮限定读取旧source_index/frozen_source_versions及已有四运行。90.35%可定位检查点仍为/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/best.ckpt，SHA c447a0a6d643405627cebf4718424b7d102fffd677b23f10ab5ef1e5227c3086，student1，step29600。其有效源码直接导入v14生成函数，已有冻结模块哈希与索引一致；该模块定义q为两组教师softmax(dim1)最大类概率均值，s=.18+.20q，Python round后裁剪1..255。未在该有效入口发现R1的control_runtime包装绑定。缺少原启动进程完整sys.path/运行时函数对象快照，不能把现存模块定义当作唯一完整历史运行证明。

该运行已有150条每200步endpoint，s范围0.3777448940–0.3795826244，均值0.3791892556；据已核查取整规则，已记录迭代矩形均97×97。这些记录不能证明全部30000迭代无变化；没有R1式逐迭代历史控制轨迹。其与作者“有效改变实际矩形且与R1一致”的唯一版本映射尚未解决。

R1实际v2_effective_source覆盖导入control_runtime生成器，显式复用EMA已有LCC伪标签，经controller.generate_mask和signals取前景类别等权间隔q_B，geometry取整。R2同信号，但geometry固定92×92。真实R1/R2逐迭代轨迹可定位，模型成绩各归各运行。候选来源表见Full_control_source_manifest.csv；R1不是90.35%检查点的替换来源。需要作者对应的另一条运行/入口/检查点身份才能解决层级映射，本轮不无限追查，也不新增训练。

A2原Full_main='old'是分析配置人工映射，不能单凭字符串证明论文版本。保留原控制代理列；所有方法重新计算同口径信号时，Full/baseline q_B标为offline auxiliary，R1是sample control proxy（非历史batch值），R2仅记录信号，固定输出几何。Full A3原策略NA；其CTW模型与公式可独立进行固定92 matched-mask诊断。
