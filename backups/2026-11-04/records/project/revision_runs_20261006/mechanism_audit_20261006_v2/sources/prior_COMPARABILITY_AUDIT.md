# 原Full与新增对照可比性核查

更新时间2026-10-06T22:55:27.056696+08:00。结论：可用配置与冻结循环中未发现E1–E4的额外科学变量改变；预期改变分别为E1均匀CTW、E2固定尺度、E3二者组合、E4普通残差层。不能宣称所有历史依赖、数据字节或恢复随机流都已证明完全一致。

protocol_comparison.csv逐项使用保持一致、预期改变、其他差异、无法核验四类，含真实run/checkpoint/hash、配置与完整队列命令。报告当前训练状态以state/queue.json为准。四个生成的循环源码hash相同，控制变量通过同一revision_runtime和独立cfg注入；实际E1源码与生成源码相同。原有72个预检锁定文件哈希仍全部匹配，未修改训练入口、控制器或五项配置。

标注七病例：patient021, patient038, patient049, patient050, patient058, patient099, patient100，首136切片列表SHA256 21e086cfa888ebc9f8c7d66b4889e43438030e7fc933e8e9afcdd36a72c6d9d4。当前train/val/test列表及逐slice标签身份实际装入reports/supplementary_sources和labeled_slice_identity.csv。旧Full日志记录同路径、labelnum7与对应136规则，但历史训练当时的清单hash未保存；不能把当前hash冒充历史hash。

初始化文件/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label7/unet_best_model.pth，SHA256 477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3，相同net/opt复用；预训练实际生成命令、seed、步数仍NA。历史pre_iterations=10000只是默认参数，不能补成已证实预算。E4新增普通层不从旧token权重初始化，其零gamma保证初始接口恒等。

逐loss定义在CSV并保留源码：双学生各自mixed CE/Dice、两向CPS、两向FFT pseudo-Dice；student1 weak/strong supervised与ICT；CTW仅student1。特别补清ICT有效权重=5×0.1×sigmoid_rampup(step//150,200)，最大0.5，而不是把args.consistency=.1写成有效最大权重。loss1+loss2+两视图supervised+两视图ICT+.05CTW表达式保持一致。CTW仍为类别均值概率L1、student2 detach、q/w detached、后向差分、逐图minmax下限1e-6，全部mixed像素参与。

E1与E3调用同一个逐图mean(w)广播实现；E2与E3调用同一个固定矩形实现和同一s；E4三模型同接口替换，647vs641，差6参数/部署student、18参数/三模型。所计残差MAC/pixel576vs640，差−10%，不含LN/激活/softmax；部署总参数差约0.000328%，不能把残差层百分比写成全网络百分比。

EMA=.99来自student1的整个state_dict，包括对应残差参数和buffers。B24/12标签、256²、SGD.01/momentum.9/wd1e-4、恢复预训练optimizer状态、恒定LR、30000SSLupdates、每200step验证student1前景macroDice严格提高选best保持原设置。历史名testloader实际仍读取val，本轮也保留重复val行为；ACDC测试不参与候选选择，测试只在best冻结后最终评价。

最终评价与原Full独立重评价采用同一evaluate.py，nearest输入/输出恢复H5原frame体积，无预测LCC；HD95为双向95百分位、ASD为prediction→GT单向，单位grid；both-empty全0、one-empty体积网格对角线。case/frame与class等权，A4另聚合为20独立患者。Full历史指标已精确重现，但这不证明缺失的历史训练依赖版本。

其他工程差异明确保留：原子checkpoint、val后刷新last元数据、sampler IDs/RNG、安全stop与detached逐step诊断。新鲜运行采样仍调用原parent iterator；诊断读取RNG而不抽取新随机数。resume的16worker增强随机流/预取不能完整恢复，不宣称位级精确。不同结构即使同seed也不保证相同dropout随机实现。原历史effective_source已冻结；历史所有import依赖的当时hash和运行环境版本未全部保存，无法核验，未额外重训Full。

E5单独核查：repo https://github.com/Senyh/AdaMix，commit 85361fb91cf43a69a158d41dfa9e7ee9fce36db9；方法配方来源/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/external/AdaMix/exp_acdc/train_adamix_ct.py，实际执行/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/train_entry.py → official_adapter.run_official。完整命令：

    /home/renguanming/.conda/envs/beta_fft/bin/python /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/train_entry.py --config /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/configs/E5_adamix_ct.yaml

当前PENDING。执行的是官方核心训练step/网络/损失/AdaMix的同协议适配，未直接运行原main；差异为official native U-Net/two students(no EMA),AdamW1e-4 wd.01,DSCLossH(confidence.95),four native AdaMix modules patch8/topk16/hard/p.5,age curriculum7500; no betaFFT/TRC/CTW; upstream200epochs adapted to30000updates, val-loss selection changed to common every200 val student1 macroDice; same7cases136slices; retain official PIL quantization/augmentation。详情reports/E5_OFFICIAL_METHOD_AUDIT.json和official_adapter.py，不能写成未经修改的官方命令原样复跑。
