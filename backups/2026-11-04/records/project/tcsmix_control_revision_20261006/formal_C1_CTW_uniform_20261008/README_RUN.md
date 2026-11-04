唯一正式实验C1_CTW_mean_uniform_v2B：TRC + Teacher-Confidence-Scaled Mixing v2_B + Confidence-Transition Weighting每图同均值均匀空间分配。一次seed42/30000联合迭代，外加最多2步隔离短程检查。

启动scripts/start_tmux.sh；状态scripts/status.sh；安全停止scripts/stop.sh；同运行恢复scripts/resume.sh。detached服务器tmux beta_fft_C1_CTW_uniform_20261008，独立普通Python控制器自动预检→短程→训练→核查→评价→分析→五审→实际解包核验。每5秒状态更新，GPU每30秒快照；繁忙WAITING_FOR_RESOURCE每5秒等待。

原E1保持3976暂停、E2–E5未启动；R1/R2只读。模型/GT/测试结果不用于筛选新配置。16 worker RNG/prefetch不能位级恢复；回滚物理更新全保留。训练失败需显式恢复，不从零自动重训。停止请求仅本实验标志文件，训练完成联合迭代完整保存后退出。控制器/训练独占锁及/proc PID起始tick+完整cmd身份，stale PID不发送信号。

FP32/no AMP，训练原TRC诊断保留。SGD lr .01常量、momentum .9/weight_decay 1e−4，原预训练优化器状态移植，EMA=.99，CTW=.05。合成检查和1+1短程隔离临时状态不进入正式训练。源码旧名仅兼容；论文正式名见协议。

当前哈希清单、完整patch及NA历史来源见manifests/reports。旧Full历史机制仍NEEDS_MAPPING，90.35%不转移。服务器重启需resume.sh，tmux不跨重启。
