# β-FFT 补实验启动报告

报告时间：2026-10-06T21:33:02.742180+08:00（Asia/Shanghai）。本报告是正式后台启动验收，五项完整训练尚未结束，新实验测试指标仍为NA。

## 实际运行位置

服务器/执行namespace：n1；项目：/home/renguanming/jupyterlab/rgm/beta-FFT-main；持久工作目录：/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006，位于NFS /home（约2.9P可用）。当前训练在服务器的Nomad/OCI作业namespace内执行，数据、源码和检查点存于持久盘。电脑关机或SSH断连不会终止tmux及队列；服务器重启、容器/服务作业终止属于需恢复的中断。

绝对解释器：/home/renguanming/.conda/envs/beta_fft/bin/python，Python3.10，PyTorch2.4.1+cu121，torchvision0.19.1+cu121。GPU0：NVIDIA A100-SXM4-40GB，驱动550.54.14。启动前0MiB/40960MiB占用、无GPU计算进程；正式E1已产生新step，记录峰值分配15.42GiB，FP32。没有停止其他任务或修改系统服务。

普通队列：/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/run_queue.py。tmux会话：beta_fft_revision_20261006，窗口queue、monitor。控制器PID：28224；训练PID：28255；当前正式E1第200步，预算30000。锁、心跳与状态：/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/state/controller.lock、/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/state/queue.json；主manifest：/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/run_manifest.json。调度不依赖agent、浏览器、客户端或持续SSH。

|任务|当前状态|PID|已记录step|锁定预算updates|输出目录|
|---|---|---:|---:|---:|---|
|E1|RUNNING|28255|200|30000|/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/runs/E1|
|E2|PENDING|NA|0|30000|/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/runs/E2|
|E3|PENDING|NA|0|30000|/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/runs/E3|
|E4|PENDING|NA|0|30000|/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/runs/E4|
|E5|PENDING|NA|0|30000|/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/runs/E5|

## 验收证据

预检PASS，完成时间2026-10-06T21:29:39.322730+08:00：22个Python脚本、9个Shell入口、32个冻结源码文件、9个配置/诊断文件的哈希已锁定。五个入口各完成独立smoke：新建2步前向/反向/保存，再恢复1步，最终预测通过统一评价（仅既有val frame）。smoke目录独立，不用于正式初始化或论文数值。CTW逐图均值、detached、student2 stop-gradient/student1更新方向、97²面积、E4零门控及EMA均通过直接检查。另有各向异性物理距离/轴变换检查和队列真实非零退出捕获→独立任务继续的检查。

真实TTY执行tmux attach并按Ctrl-b后d detach，E1从第8步继续到第37步，控制器与训练PID保持存活，验收时心跳年龄2.54秒。证据：reports/LAUNCH_VERIFICATION.json、launch_heartbeat_snapshot.json、launch_progress_before_attach.json、tmux_sessions.txt、tmux_panes.txt、tmux_capture_queue.txt、tmux_capture_monitor.txt和launch_gpu.txt；全部随证据包保存。不是仅创建空会话。

## 已完成的核查与预分析

ACDC真实70/10/20患者划分，10%标签为patient021、038、049、050、058、099、100的136切片。E1–E4沿用历史Full的seed42、B24含12标注、30000updates、恒定SGD .01、每200步student1验证macro Dice选择、同一预训练net/opt文件。统一重评价精确得到baseline Dice89.6606341% / Full90.3496577%；ACDC表面距离为H5重建grid单位，ASD为prediction→GT单向。作者更正已保留：ACDC测试不参与候选方法选择。

E2/E3固定s=0.37890625，97×97交换矩形，实际面积97²/256²=14.3569946%，与历史Full每200步的150条记录匹配，误差0。来源configs/fixed_scale.json、analysis/A1/historical_full_scale.csv。记录边长比例仍变化，但所有采样点取整后面积相同；这是稀疏覆盖，不能冒充全步平均或据此宣称动态几何明显改变。E1/E3采用控制模型自身w的detached逐图均值，均值相同不意味着梯度范数相同。

E4只采用一个宽18的1×1/LN/GELU/1×1残差配置，647参数vs TRC641（+0.936%）；所计MAC/pixel为576vs640（−10%，仅投影和token einsum，不含norm/activation），同接口、零gamma与对应教师EMA，未搜索多个宽度。

E5固定官方AdaMix commit85361fb91cf43a69a158d41dfa9e7ee9fce36db9，入口exp_acdc/train_adamix_ct.py，原生U-Net/两学生/四加载器/CE-Dice及AdaMix课程，AdamW1e-4、每加载器B16、scratch初始化，无隐藏预训练。官方200epoch预算/val-loss选择改为锁定30000updates与共用每200step的student1验证Dice选择；所有方法/增强/优化器差异单列hyperparameter_provenance.md，未把β-FFT中的TCSMix称作外部官方方法。源码与依赖均已在服务器准备，后台无需网络下载或在线跟踪登录。

COVID最终沿axis=2重建XYZ，轴序正确；已证实512→256缩放后未更新spacing。16项历史病例评价精确重放，差值全为0，旧记录不覆盖；已另存fixed-prediction的corrected-derived-center-grid、corrected-native-grid两口径，共48项病例统计。原网格修正也会轻微改变Dice，因为使用原canonical GT，不能从三口径中挑更好数字替换。未新增COVID训练。来源reports/covid_geometry_audit.csv、covid_existing_prediction_reevaluation.csv、covid_reevaluation_macro.csv。外部Table4行的身份/程序/单位映射仍缺失，仅该审计项BLOCKED；预训练文件生成命令/seed/预算未定位，保持NA，同文件初始化对照可执行。系数扫描保持原组件开关/seed42/30000/验证指标；0.05沿用历史hook默认，未据更高扫描值调整。

- A1：PARTIAL；已完成历史材料部分，新运行依赖由队列结束时补齐。
- A2：PARTIAL；已完成历史材料部分，新运行依赖由队列结束时补齐。
- A3：PARTIAL；已完成历史材料部分，新运行依赖由队列结束时补齐。
- A4：PARTIAL；已完成历史材料部分，新运行依赖由队列结束时补齐。
- A5：PARTIAL；已完成历史材料部分，新运行依赖由队列结束时补齐。
- A6：PARTIAL；已完成历史材料部分，新运行依赖由队列结束时补齐。

现有材料的评价/分析入口均退出0，见reports/pre_analysis_execution.json与PRE_ANALYSIS.md。病例以患者为独立单位；COVID全部4例均展示；定性图保留变差病例，描述性相关与单次差值不作为因果或跨种子显著性结论。

## 自动完成与恢复边界

按E1→E2→E3→E4→E5串行推进，每项30000updates；GPU忙则WAITING_RESOURCE每30秒检查。训练完成后冻结验证best身份、统一最终评价，再推进下一任务。五项结束后自动运行A1–A6、summarize_results.py、package_results.py。FAILED保留traceback/退出码、不无限重试、不静默改batch或预算，独立任务继续；训练、评价、分析与归档分别记录状态。

last.ckpt原子保存双学生、EMA（E5无EMA）、优化器、step、LR、主进程Python/NumPy/PyTorch/CUDA RNG与采样batch IDs（E5还存AdaMix age）。默认每200步保存，safe stop在当前step结束后再保存。16worker随机流与预取无法完整恢复，resume_exact=false；恢复同一run但不宣称位级精确。stop.sh仅对本轮发出停止请求；等状态变INTERRUPTED后再resume，DONE跳过、FAILED保留。跨服务器/容器重启须运行resume.sh，tmux不会自行跨重启继续。

## 实际预算与产物

新增总预算150000updates，无新增预训练、seed搜索、baseline/Full/八组合/其他预算/系数扫描/其他数据集训练。历史Full日志实际25434.563秒=7.065小时；E1–E4合计仅可粗略参考28.26历史训练小时，当前实际耗时可能不同；E5无可比历史耗时，未知。最终评价与分析另有成本，总完成时间未知。

结果目录：/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006。experiment_results.csv保留五项配置及历史参照，未完成值NA；case_metrics.csv、class_metrics.csv、analysis/A1–A6含真实原始统计与PNG/SVG；protocol_audit.md、implementation_audit.md、hyperparameter_provenance.md、FINAL_REPORT_CN.md、EDITOR_HANDOFF_CN.md提供写作交接。检查点和预测留runs/，大权重不归档。

当前证据包：/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/revision_evidence_20261006.tar.gz，是启动时的历史分析与脚本证据快照，完整新实验结果待后台完成后自动更新同一路径。包含配置、补丁、冻结源码/官方方法、日志、CSV、图与本报告，排除原始医学数组、凭据、依赖二进制和大型权重。内部SHA256SUMS针对归档的冻结字节逐项验证；实时日志/心跳随后继续变化。外层SHA见revision_evidence_20261006.tar.gz.sha256，归档验收reports/archive_receipt.json。

## 可直接复制的操作

查看会话：
```bash
tmux list-sessions
```
进入会话：
```bash
tmux attach -t beta_fft_revision_20261006
```
按Ctrl-b再按1切换monitor；按Ctrl-b再按d离开会话，训练继续。queue窗口控制器输出到logs/queue.log，训练输出在各E任务日志。

查看当前状态：
```bash
/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/status.sh
```
查看当前E1日志（后续活动任务及真实日志路径由status.sh给出）：
```bash
tail -f /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/logs/E1.log
```
安全停止：
```bash
/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/stop.sh
```
等待status.sh显示INTERRUPTED后，同run恢复：
```bash
/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/resume.sh
```
查看结果目录与归档校验：
```bash
ls -lh /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006
cd /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006
sha256sum -c revision_evidence_20261006.tar.gz.sha256
```

## 首个正式保存周期已通过

第200步已完成正式验证并写出best和last完整检查点，训练随后继续（追加验收第285步）。last SHA256：565d509ebf82c5e4b9919fb51d0486a4981983930f9c4b0c1403907135c3cc31。双学生、EMA、两优化器、主RNG、采样状态、LR和amp_state字段实测存在。证据reports/START_CHECKPOINT_VERIFICATION.json；验证分数不作最终测试结果。
