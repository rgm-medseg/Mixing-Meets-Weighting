# β-FFT ACDC10% 修订实验

工作目录/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006。普通Python队列scripts/run_queue.py负责E1→E2→E3→E4→E5、最终评价、A1–A6、汇总和打包；脱离agent/SSH继续。正式训练由通过的state/preflight.json及源码/配置SHA gate控制；不重复完成项、不读取smoke作初始化。

配置configs/E*.yaml采用JSON（YAML1.2合法子集），protocol.json是固定原协议，fixed_scale.json是锁定面积来源。scripts/train_E*.sh显式cwd和绝对环境。run_manifest.json记录命令/配置SHA/预算/依赖/状态，state/queue.json为原子心跳，logs留traceback和退出码。锁state/controller.lock防双控制器。state/stop_requested为安全停止请求；下个训练step完成后保存last并退出。

`/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/start_tmux.sh`启动；`/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/status.sh`查看；`/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/stop.sh`安全停止；`/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/resume.sh`同run恢复。失败配置保持FAILED且无无限自动重试，其他独立配置继续。已完成DONE不重训。中断续训主状态和batch IDs保留，workerRNG边界见implementation_audit.md。

会话beta_fft_revision_20261006，`tmux attach -t beta_fft_revision_20261006`进入，Ctrl-b随后d detach。tmux能跨SSH断连，不能跨服务器/容器重启；重启后运行resume.sh，不修改系统启动服务。当前执行namespace中n1有GPU与持久NFS盘；容器/作业被系统终止同样属于需手动恢复的中断，不保证tmux存活。

最后自动生成experiment_results.csv、case_metrics.csv、class_metrics.csv、analysis真实统计/PNG/SVG、FINAL_REPORT_CN.md、EDITOR_HANDOFF_CN.md、SHA256SUMS.txt和revision_evidence_20261006.tar.gz。内部CSV分数为比例，ACDC表面距离grid；COVID历史mm标签存在缩放spacing问题，修正距离另列mm。归档排除原始影像/GT数组/权重/凭据，仅checkpoint路径+hash。负结果保留。
