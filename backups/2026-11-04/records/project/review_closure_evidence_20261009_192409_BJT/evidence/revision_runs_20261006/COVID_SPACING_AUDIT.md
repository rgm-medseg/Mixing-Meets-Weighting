# COVID-19 spacing与重评价审计

更新时间2026-10-06T22:05:07.412809+08:00。独立CPU评价，没有新COVID训练、模型推理或GPU占用。共发现14个已有本地run，保留旧run与canonical1337 run各自身份，不按分数挑run、不把旧seed混入canonical主对照。外部稿件逐行身份/预测/空间信息仍缺失，NA/BLOCKED写入CSV，未猜测方法或按比例修正指标。

|病例|canonical XYZ尺寸|旧评价XYZ尺寸|原spacing XYZ|旧spacing XYZ|spacing-only修正XYZ|
|---|---|---|---|---|---|
|coronacases_002|[512, 512, 200]|[256, 256, 200]|[0.68359375, 0.68359375, 1.5]|[0.68359375, 0.68359375, 1.5]|[1.3698682598039216, 1.3698682598039216, 1.5]|
|coronacases_009|[512, 512, 256]|[256, 256, 256]|[0.6835939884185791, 0.6835939884185791, 1.0]|[0.6835939884185791, 0.6835939884185791, 1.0]|[1.3698687375760545, 1.3698687375760545, 1.0]|
|radiopaedia_27_86410_0|[630, 630, 66]|[256, 256, 66]|[0.6835939884185791, 0.6835939884185791, 6.0]|[0.6835939884185791, 0.6835939884185791, 6.0]|[1.6861985047658286, 1.6861985047658286, 6.0]|
|radiopaedia_40_86625_0|[630, 630, 93]|[256, 256, 93]|[0.6835939884185791, 0.6835939884185791, 4.0]|[0.6835939884185791, 0.6835939884185791, 4.0]|[1.6861985047658286, 1.6861985047658286, 4.0]|


并非所有原平面都是512²：两例coronacases为512²，两例radiopaedia为630²，均变为256²；Z切片数分别200、256、66、93，原序号0..N−1逐一保留。原始NIfTI header/orientation/canonical尺寸与GT/image affine一致性在reports/covid_spacing_geometry.csv。615个实际derived NPZ的路径与SHA/GT键在covid_derived_slice_sources.csv；逐病例GT严格等于原canonical GT各slice nearest到256²，不做Z插值。

数组顺序：原canonical NIfTI XYZ；dataset/prediction文件存ZXY；旧evaluator reconstruct_patient_volume沿axis=2把2D切片堆回XYZ，spacing(sx,sy,sz)与XYZ对应，轴序正确。旧问题是resize后仍存原sx/sy，并非应缩放Z spacing。新spacing-only评价使用同一binary_volume_metrics、相同prediction/GT XYZ数组，只把sx改为sx×(Nx−1)/255、sy改为sy×(Ny−1)/255，sz严格保持原值。

比例根据实际SciPy zoom(grid_mode=False)的首末像素中心跨度，而不是统一×2或所有轴同时缩放。代码来源/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_COVID19/covid19_official20_v2.py:80、/home/renguanming/jupyterlab/rgm/beta-FFT-main/train_COVID19/covid19_official20_v2.py:172；重建/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/source/train_COVID19/covid19_volume_evaluator.py:83；评价/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/source/train_COVID19/covid19_volume_evaluator.py:45。补充脚本/home/renguanming/jupyterlab/rgm/beta-FFT-main/revision_runs_20261006/scripts/supplementary_audits.py直接调用同一个冻结evaluator，见SHA清单。无需比例推测HD95/ASD，两项从surface EDT实算。

covid_metrics_before_after.csv逐方法/预算/run/患者/口径列明prediction与GT源、hash、shape、spacing、axis对应、代码位置、before/after/delta四指标和变化范围。Dice/IoU为比例；修正后距离mm，旧距离只保留原mm标签及geometry-error注释。原预测/GT/receipt不覆盖，legacy最大逐指标重放差值0.0。既有baseline/Full核验过的同预测hash结果复用，其余已有预测直接重评价，并逐病例缓存以便中断恢复。

spacing_only_derived_center_grid：prediction、GT、重采样、后处理全部不变，Dice/IoU严格逐病例相等，只有平面内spacing改正。新计算以XYZ调用；已核验缓存的derived修正原来以ZXY和相应(sz,sx,sy)调用，联合轴排列数学等价且预检已独立验证，CSV的before_call/after_call字段保留真实数组次序、spacing与shape，normalized XYZ字段用于几何对照，不混淆两者。native_grid_separate_reevaluation：仅对先前已完成该口径的baseline/Full保留另一个独立结果；实际调用为ZXY、spacing(sz,sx,sy)，预测nearest回原canonical网格，GT换为未缩放原GT，Dice/IoU可能变化，这是网格/GT重采样重评价，不能统称spacing修正，也不将这部分与spacing-only平均混用。其他本地方法本轮只需、并已做spacing-only；未新增native-grid评价要求。

后处理：读取既有categorical预测并binary>0，未加新LCC、TTA、ensemble或阈值搜索。empty-both时Dice/IoU1、HD95/ASD0；empty-one时Dice/IoU0、表面距离inf，保持既有程序而非换成有限惩罚。HD95为合并两向表面距离95百分位；ASD为合并两向样本均值，并非两方向均值再等权。指标先在完整3D患者上计算，再4患者等权macro，不把615切片作独立样本。未发生输入缺失的本地病例条数：72；实际缺失原因见CSV和receipt。

|method/run|预算|口径|Dice前→后|IoU前→后|HD95前→后|ASD前→后|
|---|---|---|---|---|---|---|
|baseline/COVID19_10P_BASELINE_TEST_ONCE|10%|spacing_only_derived_center_grid|0.655085906→0.655085906|0.498952334→0.498952334|15.723608416→26.745913845|3.440977247→5.847297658|
|TRC+TCSMix/COVID19_10P_MASK_TOKEN_ADAMIX_TEST_ONCE|10%|spacing_only_derived_center_grid|0.648234299→0.648234299|0.493756409→0.493756409|14.955020531→24.531323880|3.330197656→5.564033457|
|baseline/COVID19_5P_BASELINE_TEST_ONCE|5%|spacing_only_derived_center_grid|0.659408365→0.659408365|0.501012166→0.501012166|27.289580746→40.079835917|4.748241487→7.659926464|
|TRC+TCSMix/COVID19_5P_MASK_TOKEN_ADAMIX_TEST_ONCE|5%|spacing_only_derived_center_grid|0.678263749→0.678263749|0.520039247→0.520039247|24.090574390→32.235529604|4.292742123→6.587375673|
|baseline/COVID19_10P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE|10%|spacing_only_derived_center_grid|0.655031680→0.655031680|0.496435964→0.496435964|13.891287169→23.022861570|3.179163544→5.385299158|
|baseline/COVID19_10P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE|10%|native_grid_separate_reevaluation|0.655031680→0.653249033|0.496435964→0.494485432|13.891287169→24.901897145|3.179163544→5.440145780|
|TRC+TCSMix/COVID19_10P_MASK_TOKEN_ADAMIX_CANON1337_R1_CONFIRMATORY_TEST_ONCE|10%|spacing_only_derived_center_grid|0.691513624→0.691513624|0.534522265→0.534522265|20.414115307→30.387889240|3.804238079→6.222291209|
|baseline/COVID19_5P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE|5%|spacing_only_derived_center_grid|0.636774332→0.636774332|0.478921765→0.478921765|23.512108721→35.761468219|4.262073756→6.855459773|
|baseline/COVID19_5P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE|5%|native_grid_separate_reevaluation|0.636774332→0.635045403|0.478921765→0.477211326|23.512108721→36.241783932|4.262073756→6.994739281|
|TRC+TCSMix/COVID19_5P_MASK_TOKEN_ADAMIX_CANON1337_R1_CONFIRMATORY_TEST_ONCE|5%|spacing_only_derived_center_grid|0.674060641→0.674060641|0.512772869→0.512772869|25.407801307→37.313545143|4.608154159→7.369873699|
|TCSMix/V19_8_COVID19_10P_ADAMIX_CANON1337_R1_CONFIRMATORY_TEST_ONCE|10%|spacing_only_derived_center_grid|0.665593175→0.665593175|0.506488384→0.506488384|16.571001932→26.111573905|3.441124353→5.671726966|
|TRC/V19_8_COVID19_10P_MASK_TOKEN_CANON1337_R1_CONFIRMATORY_TEST_ONCE|10%|spacing_only_derived_center_grid|0.689546752→0.689546752|0.532528836→0.532528836|13.620162991→22.846381997|3.082898102→5.180099334|
|TCSMix/V19_8_COVID19_5P_ADAMIX_CANON1337_R1_CONFIRMATORY_TEST_ONCE|5%|spacing_only_derived_center_grid|0.657109520→0.657109520|0.499986268→0.499986268|23.895198408→34.297831004|4.470431251→7.065880668|
|TRC/V19_8_COVID19_5P_MASK_TOKEN_CANON1337_R1_CONFIRMATORY_TEST_ONCE|5%|spacing_only_derived_center_grid|0.680752920→0.680752920|0.520801191→0.520801191|23.209683748→34.375152070|4.624150294→7.346407015|
|Full/COVID5_FINAL3|5%|spacing_only_derived_center_grid|0.673143665→0.673143665|0.514497055→0.514497055|22.791388468→33.564381125|4.058622711→6.350107950|
|Full/COVID5_FINAL3|5%|native_grid_separate_reevaluation|0.673143665→0.670477051|0.514497055→0.511428653|22.791388468→33.938036887|4.058622711→6.467486529|
|Full/COVID10_FINAL3|10%|spacing_only_derived_center_grid|0.679408202→0.679408202|0.523752330→0.523752330|11.877109829→18.957128805|3.031549490→5.008397272|
|Full/COVID10_FINAL3|10%|native_grid_separate_reevaluation|0.679408202→0.677042522|0.523752330→0.521075633|11.877109829→20.103570450|3.031549490→5.018347892|


论文应更新：区分原历史距离记录与正确spacing-only物理距离，原记录保持溯源；把512/630各自到256的平面变化及Z不变写清。原网格结果单独注明GT/预测重采样改变。尚不能证明外部文献方法使用同一口径，因此不能给外部整表统一加mm或比较混合单位。两种修正口径均保留，不据指标优劣选择一种。
