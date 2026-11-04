C1当前 DONE / FINISHED；时间 2026-10-08T08:15:20.656629+08:00。

服务器n1，GPU0 GPU-0a89fa49-2afe-ac0e-2fe5-2ba2541aeb59，NVIDIA A100-SXM4-40GB。tmux：beta_fft_C1_CTW_uniform_20261008。控制器/子进程身份及完整命令见state/status.json。

正式逻辑step=30000/30000，两优化器成功物理调用=30000/30000。预计余下训练0.00小时，依据R1 actual successful Popen segment 26440.013s；评价分析归档另计。

首个200步完整检查点：{"status": "PASS", "time": "2026-10-08T00:55:08.974186+08:00", "path": "/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_C1_CTW_uniform_20261008/state/first200_snapshot.ckpt", "sha256": "70a560805eb82cab1615c738883332ae753aebf0032bc4579d74d1a342c62876", "step": 200, "keys": ["amp_state", "args", "best_metrics", "controller_metadata", "ema_model", "epoch", "global_step", "lr_state", "lr_state_optimizer2", "model_1", "model_2", "optimizer1", "optimizer2", "random_state", "resume_exact", "sampler_state", "scheduler_state"], "optimizer_physical_calls": {"optimizer1": 200, "optimizer2": 200}, "independent_model_optimizer_reload": false, "worker_resume_exact": false}。初始化、短程、绑定和梯度收据见reports；未出现的阶段不宣称已通过。

唯一科学干预：逐图自身整图原权重均值均匀CTW；TRC和TCSMix v2_B保留。E1保持3976暂停，E2–E5未启动。完成C1后停止。

查看：scripts/status.sh；安全停止：scripts/stop.sh；原状态恢复：scripts/resume.sh。进入：tmux attach -t beta_fft_C1_CTW_uniform_20261008；detach：Ctrl-b d。服务器重启后需执行resume.sh；tmux不跨重启保活。
