# 固定矩形来源审计

更新时间2026-10-27T15:19:39.604515+08:00。结论：97×97匹配旧Full已记录150个时点的平均实际交换面积，绝对误差0、相对误差0、面积百分点误差0。配置保持锁定，未发现需要暂停E2/E3的来源或计算错误。不是旧Full全部30000步的均值估计。

|来源|n|step范围|s最小|s最大|s均值|平均实际面积|97×97比例|
|---|---:|---|---:|---:|---:|---:|---:|
|旧Full每200步真实记录|150|200–30000|0.377744894028|0.379582624435|0.379189255555|14.356994629%|100.000000%|
|新增E1逐step记录|3976|1–3976|0.377017049789|0.379350833893|0.378668454599|14.356994629%|100.000000%|


真实来源：/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/train.log；SHA256 50a8922dfc74495006fb5f77f249b4d94347ab84900c570b952726dd6b098d9f；run /home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT。采样点step200、400、…、30000（150条），不是通过教师检查点离线重推。全部原行及行号在reports/fixed_rectangle_historical_excerpt.log，数值在reports/fixed_rectangle_historical_records.csv。

历史q范围0.988724470139–0.997913122177、均值0.995946277777。s的RMS=0.379189384276。s=.18+.20q，Python round(256s)并clamp[1,255]。映射端点.18/.38对应可取整边长46/97；97确为这一定义的最大边长，但q高不证明前景可靠，q还包含大量背景。97取整所需q约≥0.984765625。

固定比例97/256=.37890625，固定面积9409/65536=.1435699462890625。选择按实际取整矩形面积的最近可表示值，不按mean(s)直接定边长。忽略取整时RMS边长只是辅助；未读取测试成绩或重训Full。

原batch24含12标注、12无标注；两个6例无标注weak视图的teacher最大类别概率分别平均后取均值，单个矩形共享batch/两视图，原随机位置np.random.randint(0,max(1,256-patch_size))。位置、真实切片ID和增强随机状态没有逐batch日志，因此采样批次ID为NA；seed42不能补回失落的worker随机流。

checkpoint仅作历史run身份核验：best step29600和last step30000，路径/hash见fixed_rectangle_audit.json。它们未用于估计固定尺度；中间teacher快照未逐时点保留。q是在该step EMA更新之前计算，不能把post-step checkpoint冒充同一前向teacher。

E1的真实逐步日志来自/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/runs/E1/audit/revision_trace.csv，覆盖范围见表；它是新的均匀CTW运行，teacher轨迹也可能不同，不能冒充旧Full历史，也不据E1面积重新选E2/E3尺度。重复恢复造成的重复step将按真实执行记录保留，并报告distinct_steps。刷新时忽略的未完整尾行=0。图analysis/A1/supplement_schedule.png及SVG按两列分开展示，旧Full仅画散点，不插值补出失落历史。
