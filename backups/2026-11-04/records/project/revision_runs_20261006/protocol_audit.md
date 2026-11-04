# 协议核查

服务器n1，持久NFS目录/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006，实际A100-SXM4-40GB，Python /home/renguanming/.conda/envs/beta_fft/bin/python（torch2.4.1+cu121）。无历史原稿/权重覆盖，无数据大规模复制。本轮只新增ACDC10%E1–E5各单次，未自动重跑其他配置。

Full来源/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/effective_source.py与train.log，baseline来源/home/renguanming/jupyterlab/rgm/beta-FFT-main/runs/beta_fft_acdc_official_full_seed42_fresh_ckpt/source_snapshot.py与train.log。每项事实和代码行见reports/fact_provenance.csv，每个原始文件/检查点SHA见reports/source_index.csv。当前拷贝源码和产生论文数值的历史运行分列，不用当前代码推测失落历史日志。

病例划分70/10/20患者，对应1312训练切片、20验证frame体积、40测试frame体积。标注病例：patient021, patient038, patient049, patient050, patient058, patient099, patient100，历史规则首136切片。seed42，2D256²，B24含12标注，30000updates，SGD .01/momentum.9/wd1e-4，预训练optimizer状态恢复；循环无LR调度。验证每200update，student1前景macroDice严格提高才选择best。历史名为test的重复验证loader实际split=val；ACDC测试不参与方法选择，遵循作者更正。

原始预训练文件net/opt及SHA已核验。生成该文件的训练日志、实际命令/seed/预算未定位，保持NA，不能用当前官方pre_train默认10000来补写历史事实。E1–E4仍能定义为完全复用同一已知文件初始化的对照。

最终统一评价逐切片nearest resize、回原H5网格重建frame体积，无预测LCC，RV/MYO/LV与frame等权平均。HD95为MedPy双向95百分位，ASD为prediction→GT单向距离，无spacing参数，单位grid。空预测/非空GT给体积网格对角线惩罚；都空时Dice/IoU/HD95/ASD=0。这与验证辅助函数空预测HD=0的口径不同；锁定原验证Dice选择，最终评价使用统一完整程序。A4统计配对差值时先把两个frame聚合到patient，独立单位为20患者。

COVID本地baseline和Full的5%/10%行历史均标mm，patient-volume macro、n=4；baseline确认receipt和Full调用都指向covid19_volume_evaluator.py，但512→256缩放后仍保留原spacing，距离未体现正确像素间距。最终沿axis=2重建XYZ的轴序正确。历史口径逐病例精确复现，另存两种修正spacing/原网格口径，不覆盖旧值。该程序ASD是合并双向表面距离均值，与ACDC单向ASD不同。外部文献Table4的逐行身份/程序/spacing映射未在本地定位，仅该单位审计项BLOCKED，继续五项训练和其余分析，不索取论文/审查报告。见reports/covid_metric_unit_audit.csv。

历史Full实际日志耗时7.065小时，当前机器耗时不能据此保证。E1–E4合计可参考约28.26历史小时，E5未知；全部预算150000新增SSL/co-trainingupdates，没有新增预训练。计时smoke不作为正式初始化或科学结果。


## 已证实的COVID缩放spacing问题

代码与原NIfTI头共同证实：512→256的in-plane resize未更新spacing；最终evaluator沿axis=2正确重建XYZ，所以轴序本身正确。此前仅根据dataset ZXY堆叠作出的轴序判断已撤回，以完整重建链路为准。历史距离调用的mm标签没有反映resize后的正确像素间距；前述单位说明只描述标签/调用一致性。没有修改旧值，另存corrected-derived-center-grid及corrected-native-grid评价。具体患者shape/spacing/原GT文件SHA见reports/covid_geometry_audit.csv，全部三口径原始统计见covid_existing_prediction_reevaluation.csv。新native口径对预测nearest回原canonical网格，GT不缩放，spacing(Z,X,Y)。该问题不改变ACDC grid距离对照，也不新增COVID训练。
