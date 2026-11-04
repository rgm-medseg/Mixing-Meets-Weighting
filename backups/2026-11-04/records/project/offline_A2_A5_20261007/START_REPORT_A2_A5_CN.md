# A2/A5启动报告

2026-10-07T19:34:29.199067+08:00。独立tmux beta_fft_A2_A5_20261007，控制器PID84847，子进程84850，阶段A2。真实进度{'stage': 'A2', 'time': '2026-10-07T19:34:28.731316+08:00', 'method': 'baseline', 'checkpoint': 'last', 'cohort': 'labeled_train', 'patient': 'patient021', 'completed_units': 33}。原始输入/教师/部署分支清单已锁定，A2先行；预计后续分析10–25分钟，实际耗时取决于CPU统计和GPU等待。

已实际attach后detach，重复start复用原控制器；首次因容器PID和NVML PID不同等待自有CUDA上下文，已优雅停止分析、保留0患者单位的启动证据，并在独立common适配GPU PID映射后resume。未结束他人进程，未改原模型/训练源码。见GPU_PID_STARTUP_NOTE和修复patch。当前身份9c039632d8918000110d4d8d541296761dd96f663068b817b651dd2940f1b740，先前身份材料保存在versions。

操作：./start.sh启动或复用；./status.sh查看控制器/进度；./stop.sh在单元边界停；./resume.sh补缺失单元。路径/home/renguanming/jupyterlab/rgm/beta-FFT-main/offline_A2_A5_20261007。服务器重启后需resume.sh。旧beta_fft_revision_20261006保持INTERRUPTED；R1/R2不重训。
