# 独立正式R1（仅一项）

工作目录：/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R1_20261007。用户2026-10-07明确授权R1正式训练，上一阶段停止边界仅对R1解除；R2与旧E1–E5保持停止。
固定方法TRC+已验收TCSMix v2_B+原空间CTW，ACDC10%、seed42、30,000联合步；正常两学生各30,000次SGD调用。正式起点是原pretrain net/opt，不使用short/E1/末期Full。controller.py直接复制且SHA锁定，无新候选或搜索。

服务流水线：preflight→资源串行等待→R1正式训练→训练后核查/冻结val-best（先生成面积建议）→一次最终测试→全程分析/固定标注验证teacher诊断/全部历史差值→报告→五轮检查→静态归档。完成停止；不启动R2，不调用旧resume。中断/失败保持原配置及全部证据，自动归档已有结果；自动训练重试0次，恢复由本run resume.sh完成，同定义不重新从0。

输入、代码与完整配置以run_manifest.json、protocol.json及configs/R1_full_v2B.yaml为准。原loss/EMA/mix逐字核验，旧模块Mask Token/AdaMix/CRW分别对应TRC/本项目TCSMix/CTW，外部AdaMix另标。评价统一grid单位，不能把HD95/ASD标mm，详细规则见protocol.json与最终报告。

操作（服务器绝对路径）：
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
在tmux里按Ctrl-b再按d退出连接，服务器流水线继续。电脑关机/SSH断开不会取消任务；服务器重启需核查本run完整checkpoint再resume，不修改系统自启动。

GPU0启动条件：无GPU0 compute进程、used<2048MiB、free>=30000MiB、util<=10%，每30秒查询、5秒heartbeat；占用时WAITING_RESOURCE，不换设备/batch或终止他人。pipeline/training/launcher各自文件锁，重复start复用本次活进程。

checkpoint保存两学生/EMA、两optimizer、constantLR/AMP=None、联合step、无状态controller版本/configSHA、Python/NumPy/Torch/CUDA RNG与sampled IDs/位置。16worker增强流/预取不能位级恢复。optimizer_calls.csv保留物理更新包括恢复重放，完整trace保留全部原始行，逻辑统计取最后一次；断电前未flush极短区间可能缺证，不能补造。

状态是atomic JSON：WAITING_RESOURCE/RUNNING/TRAINED/EVALUATED/ANALYZED/DONE/FAILED/INTERRUPTED；阶段日志/退出码/error独立保存，训练完成不等于结果DONE。最终包排除大型weights、原始医学数组、npz预测和凭据；prediction_index.csv/checkpoint索引提供服务器路径与hash，定性图沿用匿名ACDC ID和image-only规范化展示。

单次运行允许持平或下降。旧90.35%仍属于old，新版成绩在完成前为NA；R1分数差异不能单独解释动态调度收益。R2_FIXED_AREA_PROPOSAL只据完整真实轨迹面积，测试前生成，不执行R2。
