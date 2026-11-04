# R1正式启动报告

**RUNNING / TRAIN**，报告时刻2026-10-07T00:48:19.545688+08:00，服务器n1、GPU0 A10040GB。已授权R1正式训练实际启动，不再等待确认；只运行本次R1，旧E1仍停3976，E2–E5未启动，R2未启动。

启动目录：`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007`；run目录：`/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/runs/R1_full_v2B`；tmux：`beta_fft_R1_v2B_20261007`（pipeline/monitor/results三窗口）。pipeline PID70652，实际训练PID70675；/proc确认训练PPID属于该服务器pipeline，pipeline由tmux pane bash管理，与Codex/客户端无关。NVIDIA查询另报告GPU计算PID4901，本地/proc映射未确立；操作使用已核验本地父子进程，不向GPU查询PID发送信号。

实际启动命令：
```bash
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/scripts/start_tmux.sh
```
实际训练命令：`/bin/bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/scripts/train_R1.sh`；该脚本exec服务器绝对Python解释器与train_R1_entry.py。完整解析argv/初始化来源见reports/command_*.json，环境见environment_freeze.txt。工作/数据/解释器/权重均为服务器持久路径，不需要本地代理、挂载、在线登录或后续聊天驱动。

方法已锁定TRC+TCSMix v2_B+原空间CTW（λ.05、原L1/stop-gradient/方向）；controller直接复制已验收文件，SHA`f923e7f9bf0ef9bdae94bbd400d54f52fbbc6064efac9bcece365465657f795c`。控制来源为原teacher weak logits→softmax(dim1)/detach→已有每类LCC前景支持→每类概率间隔均值→有支持类等权；空前景0；原s=.18+.20q、round/clamp、随机位置/批次共享/双视图互补不变。scope/loss/EMA/mix块逐字检查PASS，无新增参数或teacher前向；冻结配置SHA`e9d6af11c56e8b1b8439fbbd6bb3d3d51681c79eaea43b4064bc78907dc35b66`，协议SHA`97594183409307f5670adc6211d1cfc9e53d4d3cc3bb93842d940fc372320213`。预启动工程修订V1 manifest保留，均发生在首次正式训练前；已冻结当前有效版本，训练后未修改源码/参数。

实际起点为原`/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_ACDC/Pre_model/label7/unet_best_model.pth`，SHA`477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3`，半监督step0；两学生与教师共有权重逐张量等于该pretrain，三分支TRC gamma0、router641、每模型1827469参数；原optimizer net/opt加载规则保持，学习率.01。未使用short240、E1/3976、末期Full或best初始化，没有新预训练，没有新smoke更新。ACDC70/10/20病例，原7标注136切片，B24/12、256²、seed42、16worker、FP32、原增强/EMA.99/损失与验证每200保持。

预算**30000联合迭代**，正常两学生各30000次SGD调用、共60000；与旧short480调用计数完全分开，不将正式总预算减半。当前heartbeat联合step=532，读取实际trace覆盖至539，两optimizer各539/539成功调用；日志与控制记录从1连续推进，早期已记录尺寸[85, 86, 87, 88, 89, 90, 91]。这些是未完成正式实验的局部行为，不能当全程结论。正式Dice/IoU/HD95/ASD均NA，旧90.35%仍属于old。

真实连接/分离验证：2026-10-07T00:41:40.049674+08:00 step75 / heartbeat2026-10-07T00:41:36.600487+08:00；在工具PTY实际attach服务器tmux并按Ctrl-b,d退出后，2026-10-07T00:42:59.988541+08:00 step180 / heartbeat2026-10-07T00:42:57.259890+08:00，无附着客户端且同pipeline/train PID继续。两个tmux capture、进程status、GPU与状态JSON在STARTUP_OBSERVATION_1/2、TMUX_ATTACH_DETACH.json。重复start实际复用PID，未产生第二个控制器。原tmux三窗口保留；原82项输入hash不变。

本次只读核查的完整恢复checkpoint为step400，SHA`a2e93fbdd0438bede8e22f02f4dbf4fe6f6288021193c45e9f2485d86931d7f4`；模型/双optimizer/RNG/sampler/LR/AMP=None/controller元数据齐全，见FIRST_FULL_CHECKPOINT_AUDIT.json。首次验证按student1原规则运行，validation_trace.csv可核查；日志名含_test的旧字段来自split=val，不是真测试。当前best只由验证选择，最终测试须TRAINED/30000与postcheck冻结selected identity后进入。主状态/采样位置可恢复；16worker增强随机流/prefetch仍非位级恢复，未为恢复改变协议。

当前正式阶段ETA训练结束约2026-10-07T07:56:10.155971+08:00，剩余7.13 GPU小时；依据current active formal segment elapsed / progressed joint iterations, includes validation/loading，初期观测、包含初始化/验证，不保证完整阶段速度。稳定step速度会持续重估；评价、诊断、分析、打包另计。GPU查询和heartbeat各自时间记录，不能把早前空闲快照当训练中显存。真实相关损失/学习率、mask面积/位置、支持数、数据索引和物理调用均逐步保存。

后续由普通服务器pipeline独立执行：R1到30000→完整训练核查→冻结val-best（先用完整面积生成R2建议）→统一ACDC最终测试并保存预测→六个预定5000步分布/全程控制图、标注/验证final teacher诊断、全部历史病例/类差值和改善/典型/最差匿名例→实际报告/写作交接→五轮审查→实际静态压缩包+SHA256。结束停止。R2只有面积建议，不接队列、不启动；历史缺失单独PARTIAL。失败保存阶段退出码、traceback、轨迹与checkpoint并归档已有证据，自动训练重试0次；不换seed/batch/机制。归档失败与科学缺失分别记录。

操作命令（分别按需要执行）：
```bash
tmux ls
tmux attach -t beta_fft_R1_v2B_20261007
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/scripts/status.sh
tail -n 30 /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/runs/R1_full_v2B/train.log
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/scripts/stop.sh
bash /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/scripts/resume.sh
cat /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/results/R1_results.csv
ls -lh /home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007/TCSMix_v2B_R1_final_evidence.tar.gz*
```
在tmux里Ctrl-b后d离开。stop只请求本R1在下一完整迭代边界存盘安全退出，不杀无关进程；resume核验同一代码/config/init与本run完整last，训练完成则跳过训练继续缺失阶段，DONE不重训。电脑/VSCode/SSH关闭不影响服务器任务；服务器重启或系统杀进程需本runcheckpoint恢复，不承诺tmux跨重启自启。

最终输出：results/R1_results.csv、R1_case_metrics.csv、R1_class_metrics.csv、R1_patient_metrics.csv、完整R1_control_trace.csv与prediction_index.csv；analysis/图和原始统计；FINAL_REPORT_R1_CN.md、EDITOR_HANDOFF_R1_CN.md、R2_FIXED_AREA_PROPOSAL.md；`TCSMix_v2B_R1_final_evidence.tar.gz`及外层SHA。大型权重、原始影像/GT与测试npz预测留服务器；最终包包含匿名定性渲染、实际报告表图脚本patch/hash/log，写作助手无需登录查看。
