# 补充审计跟随进程

更新时间2026-10-06T22:09:17.161056+08:00。tmux beta_fft_revision_20261006的audits窗口运行普通CPU脚本，PID 44622，phase FOLLOWING_EXISTING_QUEUE。原queue控制器PID28224和E1 PID28255保持不变。本进程不提交训练、不改配置、不访问GPU。

脚本/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/supplementary_audits.py --watch 已完成三项报告/CSV和初始证据包；每10分钟刷新E1真实调度快照、可比性与状态，原queue结束并退出后再次补齐EDITOR_HANDOFF及完整归档。心跳state/supplementary_audits.json；日志logs/supplementary_audits.log、logs/supplement_delivery.log。

原resume.sh仍有效；服务器/容器重启导致两个进程均退出时，使用组合入口同时恢复既有queue和审计（锁和进程匹配避免重复）：

    /home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/resume_with_audits.sh

安全停训练仍使用/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/stop.sh。审计会等待同run恢复；SSH断连不终止两个tmux进程。未修改原有预检锁定入口或控制器。初始审计执行版本已保存reports/supplementary_sources/supplementary_audits_initial_version.py及其真实SHA，病例缓存保留计算版本/坐标系；后续补充只完善来源文件与坐标说明，不改数字。

本文启动审计结果保持NA的新测试项将由原queue完结后自动补齐；当前状态快照见reports/CURRENT_QUEUE_STATUS.csv，真实缺失项为预训练生成历史、原历史列表/import版本哈希/原shell argv和外部COVID行的身份/输入/单位映射。没有训练任务因此BLOCKED，97²来源与面积匹配通过。
