# R2固定92正式启动报告

观测时刻2026-10-07T11:11:03.855012+08:00：**RUNNING / TRAIN，真实最新联合step 361/30000**，两学生成功更新各361/361。普通服务器Python/Bash流水线正在运行，已离开准备阶段；训练完成后自动冻结验证best、最终统一评价、三方法重点病例几何对照、四方法配对/阶段面积分析、汇总、五审和实际压缩归档。当前无未解决阻塞。

主机n1、GPU0 NVIDIA A100-SXM4-40GB；tmux **beta_fft_R2_fixed92_20261007**（pipeline/monitor/results）。控制器PID81963，训练PID82001。实际训练入口/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/scripts/train_R2_entry.py；当前pipeline通过显式 --resume 续接已修复的零更新入口准备，训练本身首次从原预训练joint_step0初始化，未重放/重启已更新模型。初始训练进程阶段启动见stage_history/logs，实际加载回执时间2026-10-07T11:06:04.189058+08:00。

范围与唯一变化：R2=原TRC+固定92×92区域混合+原空间CTW。已核验实际R2科学循环与R1有效源码逐字相同，controller.py及科学依赖沿用R1真实冻结版本，仅配置选择已有v2_fixed几何分支。teacher weak detached softmax和既有LCC支持复用；q_B类别等权间隔只日志，s_dynamic_B=.18+.20*q_B是参考；**s/s_actual=.359375、h/w_actual=92，实际8464像素/12.9150390625%**。实际已记录样本中q_B范围见STARTUP_ACTUAL_MASK_CHECK.json；所有已核查真实掩码固定92、图/目标与视图/批次共享断言通过，随机位置保留两次原randint和exclusive端点，无额外teacher forward/RNG。

面积定义冻结在R2_AREA_LOCK.json：R1全部1–30000步真实像素面积平均8492.147533333333，最近整数平方92²；下一候选93²误差156.85像素而92²误差28.15。源轨迹SHA **315139c2d54a89205988e0dd92eb28855ac3c42b84198b216d4c49fe2a35cb95**；有符号比例差-0.00042949727376301317，面积百分点差-0.0429497，相对差-0.331454%。没有按测试分数/最优步前轨迹/连续s平均重选；97不是本次定义。

初始化三分支公共骨干与原pretrain逐tensor相同、TRC gamma0，各1827469参数（router641）；两优化器原基础状态逐项匹配、router初始无动量状态。原文件/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label7/unet_best_model.pth，SHA477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3；没有使用R1/旧Full/短程/E1权重。与R1相同70/10/20病例、7标注患者和136切片、256²、batch24/label12、seed42、16worker、FP32、SGD lr.01常数/momentum.9/wd1e-4、EMA.99、空间CTW .05和全部原损失/增强/双视图/频率分支。30000是联合迭代，每学生正常各30000成功更新。

已读到并核查**step200完整检查点**，SHAbc7ee3ab928a110df5572005886d568e2e6c9abca583f8ceaae3aa28feb82f53，版本v2_fixed，信号v2_B、fixed92、cfg与面积锁定哈希正确，含学生2/教师/两优化器/RNG/采样器/LR/AMP状态；只CPU读取，无优化更新。每200验证并保存，严格student1验证foreground macroDice提高选best；先冻结选择与SHA后最终测试。主RNG/批次ID可恢复，16worker增强与prefetch不保证逐位恢复；服务器重启后运行resume.sh，不承诺tmux跨重启。

当前ETA **2026-10-07T18:09:57.097268+08:00**，采用当前完整活动训练进程墙钟/已完成联合步（含取数、已发生验证/保存等开销）；早期阶段仍会波动。已完成R1约7小时20分仅作参考，step_seconds未被冒作端到端耗时；各阶段检测延迟≤5秒。原零更新准备失败的10秒进程开销单独留在stage_history，成本汇总保留。

客户端独立性：实际attach后Ctrl-b d退出成功；同一PID前后step 40→124，不同heartbeat及实际trace时间证明继续推进，当前该会话无attached clients。重复start实测复用控制器，无第二训练。普通Python控制器在服务器detached tmux管理后续全部阶段，关闭电脑/VS Code/SSH/Codex不影响服务器进程。

R1已有预测误差CPU复核DONE：全部240个旧Full/R1类frame指标与原CSV逐项吻合，R1 Dice下降12/20病例；分类RV +0.1956pp、MYO -0.5413pp、LV -0.7499pp。patient065/075贡献HD95有符号净增量89.91%、单向ASD69.80%；不是全部性能原因，所有病例保留。重点三患者双frame的TP/FP/FN、逐slice覆盖/面积、6邻域三维域、双方向表面距离和ASD贡献、真实异常slice图与选择CSV均在analysis/error_R1及R1_ERROR_ANALYSIS_CN.md；远端假阳性、范围延伸、漏分割和局部误差同时如实记录。两主要病例中远端孤立域贡献了其ASD净差约85.18%（详见几何分解JSON），不能将HD95照此加性归因，也不证明某个模块造成错误。全部分析只读取原预测，无GPU推理、未改预测/GT/后处理，未反向调面积或算法。

启动问题与修复留痕：第一次入口在写有效源码前发现run目录未建，模型未初始化、优化器0次；失败退出码/traceback和压缩包保留在archives/pretrain_entry_failure_zero_updates，补齐目录后才第一次实际初始化。tmux3.0a不支持pane_start_path，死pane的cwd也为空；在首次优化更新前仅修复launcher为核验原绝对pipeline命令并重开死monitor，显式保存PRESTART_MANIFEST_V1和TMUX_COMPATIBILITY_REPAIR。科学源、配置和面积未变化，无自动从零重训；stage_errors保留历史问题，不伪称从未失败。当前manifest SHA452e4742bb903273f0d60c9c0a0a040d629e38f6c7d8784a9016d5e6aebbe05f、配置SHA002b456d2a45b06c4df558ef1e8b326e9a690db590527827ee8848667e5bcf1d、协议SHA3592154a081650d391d36027cb06b9260191904f549c995534d25c3e93a983eb；无Git仓库，用源SHA/工程patch标版本。

数据历史证据边界：列表和既有R1索引/样本哈希沿用；R1原40份预测/GT文件SHA已核验原索引。新增完整1372个H5当前哈希用于R2冻结，不能冒称R1历史运行期间已逐文件记录。原pretrain生成日志NA，不新增训练。

旧E1–E5保持暂停（E1完整checkpoint3976，E2–E5未启动），R1/旧结果和论文均未覆盖。R2性能允许改善/持平/下降，单次对照不能证明跨种子稳定性、拆分信息/波动/课程效应或CTW协同。最终分数现在未产生，NA不拿旧90.35填充；旧90.35属于旧机制。

## 实际操作命令

```bash
tmux attach-session -t beta_fft_R2_fixed92_20261007
# Ctrl-b d 仅detach
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/scripts/status.sh
tail -f /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/runs/R2_fixed92/train.log
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/scripts/start_tmux.sh
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/scripts/stop.sh
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/scripts/resume.sh
```

start重复调用复用，stop只请求R2安全保存并退出、不强杀其他任务，resume从同R2完整状态继续；已完成训练只续后续阶段，不重训。

## 可下载证据与后续交付

启动包：TCSMix_R2_fixed92_start_evidence.tar.gz；外层SHA和完成回执在打包后另写startup_archive_receipt.json，包内是此前一致快照，trace/optimizer ledger按共同完成step截断，医学原数组/全预测/大权重不入包。包内真实报告/CSV/异常slicePNG+SVG、面积锁、协议/配置、源/差异、初始化/CP回执、执行命令和关键日志无需登录服务器即可审查。
后台最终生成R2_results/class/case/patient、case_class_metrics、完整R2_control_trace、optimizer_calls/validation_trace、四方法配对/阶段面积图、R1_R2_ERROR_ANALYSIS_CN/FINAL_REPORT_R2_CN/EDITOR_HANDOFF_R2_CN、五审、SHA256SUMS与TCSMix_R2_fixed92_final_evidence.tar.gz及外层校验；原权重/全预测保留服务器并给实际索引/哈希。最终归档后另写完成回执，明确区分归档前快照与最终DONE/FAILED状态。
