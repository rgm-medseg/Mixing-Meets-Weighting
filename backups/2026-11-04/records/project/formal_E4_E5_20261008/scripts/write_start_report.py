"""Current verified startup receipt and directly usable server commands."""
from datetime import datetime,timedelta
from common import *
def main():
    from run_queue import live,last_trace
    s=read_json(MAIN_WORK/'state/status.json');assert s['status']=='RUNNING' and s['stage']=='TRAIN' and s['task']=='E4'
    assert live(s['pipeline_identity']) and live(s['child_identity'])
    init=read_json(MAIN_WORK/'tasks/E4/reports/INITIALIZATION_ACTUAL.json');assert init['fresh'] and init['joint_step']==0
    attach=read_json(MAIN_WORK/'reports/TMUX_ATTACH_DETACH.json');assert attach['status']=='PASS' and attach['heartbeat_after_detach_verified']
    first=read_json(MAIN_WORK/'tasks/E4/reports/FIRST_200_CHECKPOINT.json');assert first['status']=='PASS' and first['step']==200
    for task in ['E4','E5']:
        r=read_json(MAIN_WORK/'tasks'/task/'reports/SHORT_CHECK.json');assert r['status']=='PASS' and r['physical_joint_updates']==2
    trace=read_table(MAIN_WORK/'runs/E5_isolated_short/results/E5_control_trace.csv')
    steps=[float(r['step_seconds']) for r in trace];low=min(steps)*30000*1.2/3600;high=sum(steps)/len(steps)*30000*1.5/3600
    nowdt=datetime.now().astimezone();e4h=s['ETA_current_training_remaining_hours'];earliest=nowdt+timedelta(hours=e4h+low+.25);latest=nowdt+timedelta(hours=e4h+high+1)
    actual=last_trace('E4');step=int(actual['step'])
    text=f'''E4已正式启动，E5预检通过并在独立队列READY等待。当前时刻{now()}，E4逻辑step={step}/30000，两学生物理调用={actual['optimizer1_calls']}/{actual['optimizer2_calls']}。最新状态以state/status.json与status.sh为准。

服务器n1，GPU0 NVIDIA A100-SXM4-40GB，UUID GPU-0a89fa49-2afe-ac0e-2fe5-2ba2541aeb59。解释器{PYTHON}，torch2.4.1+cu121，FP32，seed42。控制器容器PID{s['pipeline_identity']['pid']}，E4训练容器PID{s['child_identity']['pid']}；NVML主机PID见tasks/E4/reports/GPU_PID_BINDING_*.json，两种PID分别保存。磁盘约2.9PiB可用；GPU资源每30秒记录allocated/reserved之外的NVML证据。完整进程身份、启动ticks、绝对命令、当前日志与心跳在state/status.json。

正式预算E4 30000＋E5 30000；E5必要预训练0。各smoke2个物理联合更新、无失败训练尝试，分别1＋1真实恢复，独立重载与零更新恢复探测通过。静态准备第一次因官方val_slices.list缺失终止，0个训练更新；已从相同20验证帧210原生切片完成数据接口适配，失败日志保留。E4正式初始化来自锁定原预训练，共享权重/动量逐状态一致，新普通残差组初始空，三分支gamma0；模型/优化器/RNG未继承smoke。

E4唯一科学干预TRC→已有普通残差16→18→16（bias、LayerNorm、GELU、gamma0），647对641参数，单学生1827475。其余v2_B控制器、空间CTW.05、基础目标、SGD常数.01、EMA.99、数据/验证均继承R1；实际函数与源码哈希核验。E5为锁定官方commit85361fb91cf43a69a158d41dfa9e7ee9fce36db9的ACDC CT共同协议适配，官方native U-Net182?参数精确见MODEL_STRUCTURE_COMPARISON.csv（实测1813764）、AdamW1e-4、CE/Dice、置信阈值.95、原课程/增强，随机初始化，无额外教师/本论文模块。四路各6合计24输入/12GT，伪标签流严格63无标注患者且GT不读取；主选student1 strict共同valDice每200，另存官方val-loss每196。全部适配在E5_PROTOCOL_COMPARISON.csv明列。

已实际attach到tmux并通过Ctrl-b d分离，分离后的step和心跳继续增长，证据reports/TMUX_ATTACH_DETACH.json。首个200步完整checkpoint独立CPU加载PASS，SHA256 {first['sha256']}，18项字段含两学生、EMA、两优化器、双组LR、RNG和采样状态。独立模型/优化器重建加载在两项smoke完成；此首200检查不扰动正在训练的状态。

E4当前余下训练约{e4h:.2f}小时，依据本进程实际进度/墙钟，并以R1约7.4小时为历史参照。E5只有2步短程吞吐，冷启动与稳态混合，估计训练{low:.1f}–{high:.1f}小时；下界为观测最短步×30000×1.2，上界为两步均值×30000×1.5，包含验证/IO余量，非性能选择。正式200步后控制器改用真实进度更新。评价、CPU分析、五审和压缩另留约0.25–1小时，粗略队列结束{earliest.strftime('%m-%d %H:%M')}–{latest.strftime('%m-%d %H:%M')}（Asia/Shanghai），实际心跳与阶段优先。

流水线为E4训练→完整更新数核查→自身val-best冻结→统一测试评价→A1/A4/A6及成本→E5相同步骤→汇总→五审→一致快照归档。普通服务器Python/Bash控制器自主推进，关闭电脑/VS Code/SSH/Codex不影响。独立任务失败保留traceback、状态和成本，继续另一READY任务。停止只创建本轮标志，安全保存完整last后暂停；恢复从自身完整last继续。worker随机流/预取不位级恢复；服务器重启后执行resume.sh。已完成结果核验后跳过，锁避免多开。

E3及第四格未执行。旧E1仍3976步暂停，旧E2–E5未启动；R1/R2/C1受保护且哈希未变。R1/R2、R1/C1分别是两项条件对照，旧Full90.35%保留原版本且NEEDS_MAPPING。ACDCtest未参与原候选选择。当前不报告E4/E5最终测试指标；完成后所有持平/下降病例与全部40×3行均保留。

直接复制的命令：

    tmux list-sessions
    tmux attach -t beta_fft_E4_E5_20261008

分离：先Ctrl-b，再d。

    {MAIN_WORK}/scripts/status.sh
    {MAIN_WORK}/scripts/stop.sh
    {MAIN_WORK}/scripts/resume.sh

E4当前完整训练日志：

    tail -f {s['current_log']}

E4继承原程序训练日志：

    tail -f {MAIN_WORK}/tasks/E4/runs/E4_plain_residual_v2B/train.log

E5训练阶段日志将由控制器写到：

    {MAIN_WORK}/tasks/E5/logs/TRAIN_*.log

结果：{MAIN_WORK}/results/和tasks/E4/results/、tasks/E5/results/。
最终报告：{MAIN_WORK}/FINAL_REPORT_E4_E5_CN.md
最终包：{MAIN_WORK}/E4_E5_final_evidence_20261008.tar.gz及.sha256。
启动包：{MAIN_WORK}/E4_E5_start_evidence_20261008.tar.gz及.sha256；实际解包核验记录START_ARCHIVE_RECEIPT.json。
'''
    text=text.replace('native U-Net182?参数精确见MODEL_STRUCTURE_COMPARISON.csv（实测1813764）','native U-Net1813764参数')
    (MAIN_WORK/'START_REPORT_E4_E5_CN.md').write_text(text)
    atomic_json(MAIN_WORK/'reports/START_VERIFIED.json',dict(status='PASS',time=now(),current_step=step,state=s,initialization_receipt_sha256=sha(MAIN_WORK/'tasks/E4/reports/INITIALIZATION_ACTUAL.json'),tmux_detach_verified=True,first200_verified=True,E5_short_steps=2,E3=False,old_queue='INTERRUPTED',ETA_E5_hours=[low,high]))
    print('Start verified',step,'E5 estimate',low,high)
if __name__=='__main__':main()
