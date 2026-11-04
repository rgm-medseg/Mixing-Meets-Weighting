# R1主要误差病例核查（R2启动前，已有预测CPU分析）

状态DONE，时刻2026-10-07T10:58:28.625305+08:00。读取真实冻结检查点预测与GT，重新计算全部40个frame×3类指标并逐项匹配原CSV（误差<1e-9）；未训练、未修改预测/GT/后处理，GPU推理0次。R1选择step29000；旧Full选择step29600。来源、数组z,y,x、原尺寸、预测/GT/检查点哈希见analysis/error_R1/prediction_provenance.csv。

20病例中R1 Dice下降12例；分类差值（fraction，×100为pp）：[{'model': 'R1_v2_B', 'class_name': 'RV', 'dice_delta': 0.0019555476692891733, 'iou_delta': 0.00242179089133627, 'hd95_delta': -0.29369360507710757, 'asd_delta': -0.06294825287380643}, {'model': 'R1_v2_B', 'class_name': 'MYO', 'dice_delta': -0.005413050456933857, 'iou_delta': -0.00794931290868324, 'hd95_delta': 1.9405820935320839, 'asd_delta': 0.44897077905237326}, {'model': 'R1_v2_B', 'class_name': 'LV', 'dice_delta': -0.0074986647313787765, 'iou_delta': -0.012197185038979192, 'hd95_delta': 2.9094058417849658, 'asd_delta': 0.633674889299195}]。
patient065与075对HD95、ASD有符号净增量的贡献复核为：{'hd95': {'all_net_sum': 182.2517732095977, 'two_patients_net_sum': 163.86239265350633, 'net_share': 0.8990990307954768, 'meaning': 'signed delta sum share, includes cancellation; not exclusive cause attribution or exclusion-adjusted main score'}, 'asd': {'all_net_sum': 40.78789661911049, 'two_patients_net_sum': 28.471255880026128, 'net_share': 0.6980319712462543, 'meaning': 'signed delta sum share, includes cancellation; not exclusive cause attribution or exclusion-adjusted main score'}}。这是全体差值中可含正负抵消的净占比，不能将两个病例当成全部原因；全体20病例、120类frame行均保留，不生成剔除病例主结果。

三维连通域与表面均用6邻域。HD95为预测→GT和GT→预测两组表面距离拼接后的95分位，ASD只用预测→GT。表面统计均在原H5 grid、spacing=(1,1,1)，不是mm。完整距离频数、逐slice距离贡献、最大20个表面点及其最近对应点坐标均另存CSV；复算均与原评价一致。诊断中≥10/20 grid仅描述远端程度，不是模型阈值或新增后处理。

确认的几何现象（各优先frame的全部类别）：
- patient083_frame01 RV：Dice差-1.556pp，HD95差+0.000，单向ASD差+0.053；FN增加214像素；远端孤立域对单向ASD的加性贡献差+0.000 grid。
- patient083_frame01 MYO：Dice差-5.261pp，HD95差+1.828，单向ASD差+0.694；FN增加100像素；FP增加194像素；远端孤立域对单向ASD的加性贡献差-0.027 grid。
- patient083_frame01 LV：Dice差-2.749pp，HD95差+1.147，单向ASD差+0.368；FN增加36像素；FP增加76像素；远端孤立域对单向ASD的加性贡献差+0.000 grid。
- patient083_frame02 RV：Dice差-2.815pp，HD95差+0.000，单向ASD差+0.083；FN增加348像素；远端孤立域对单向ASD的加性贡献差+0.000 grid。
- patient083_frame02 MYO：Dice差-3.215pp，HD95差+1.212，单向ASD差+0.707；FN增加3像素；FP增加200像素；远端孤立域对单向ASD的加性贡献差+0.000 grid。
- patient083_frame02 LV：Dice差-7.861pp，HD95差+11.000，单向ASD差+2.279；1个与GT零交叠且最小距离≥10 grid的孤立域；FN增加6像素；FP增加164像素；远端孤立域对单向ASD的加性贡献差+0.017 grid。
- patient065_frame01 RV：Dice差-1.247pp，HD95差+0.000，单向ASD差+0.932；1个与GT零交叠且最小距离≥10 grid的孤立域；FN增加90像素；FP增加29像素；远端孤立域对单向ASD的加性贡献差+0.925 grid。
- patient065_frame01 MYO：Dice差-3.551pp，HD95差+30.210，单向ASD差+3.735；1个与GT零交叠且最小距离≥10 grid的孤立域；FP增加290像素；远端孤立域对单向ASD的加性贡献差+3.675 grid。
- patient065_frame01 LV：Dice差-3.036pp，HD95差+28.554，单向ASD差+4.204；1个与GT零交叠且最小距离≥10 grid的孤立域；FP增加279像素；远端孤立域对单向ASD的加性贡献差+3.848 grid。
- patient065_frame02 RV：Dice差+0.324pp，HD95差+0.000，单向ASD差+0.082；2个与GT零交叠且最小距离≥10 grid的孤立域；GT无该类切片上424个预测像素（旧447）；FN增加21像素；远端孤立域对单向ASD的加性贡献差+0.104 grid。
- patient065_frame02 MYO：Dice差-3.170pp，HD95差+0.000，单向ASD差+2.168；2个与GT零交叠且最小距离≥10 grid的孤立域；GT无该类切片上188个预测像素（旧0）；FP增加217像素；远端孤立域对单向ASD的加性贡献差+2.076 grid。
- patient065_frame02 LV：Dice差-7.238pp，HD95差+35.623，单向ASD差+4.493；1个与GT零交叠且最小距离≥10 grid的孤立域；GT无该类切片上262个预测像素（旧0）；FN增加2像素；FP增加242像素；远端孤立域对单向ASD的加性贡献差+4.356 grid。
- patient075_frame01 RV：Dice差-0.137pp，HD95差+0.000，单向ASD差-0.001；FN增加39像素；远端孤立域对单向ASD的加性贡献差+0.000 grid。
- patient075_frame01 MYO：Dice差-6.050pp，HD95差+20.536，单向ASD差+3.356；2个与GT零交叠且最小距离≥10 grid的孤立域；FP增加833像素；远端孤立域对单向ASD的加性贡献差+1.720 grid。
- patient075_frame01 LV：Dice差-1.394pp，HD95差+0.083，单向ASD差+1.822；3个与GT零交叠且最小距离≥10 grid的孤立域；FN增加11像素；FP增加190像素；远端孤立域对单向ASD的加性贡献差+1.838 grid。
- patient075_frame02 RV：Dice差+0.206pp，HD95差+0.000，单向ASD差-0.029；GT无该类切片上727个预测像素（旧751）；FN增加52像素；远端孤立域对单向ASD的加性贡献差+0.000 grid。
- patient075_frame02 MYO：Dice差-6.013pp，HD95差+22.690，单向ASD差+3.911；7个与GT零交叠且最小距离≥10 grid的孤立域；FP增加1034像素；远端孤立域对单向ASD的加性贡献差+1.853 grid。
- patient075_frame02 LV：Dice差-3.712pp，HD95差+26.166，单向ASD差+3.799；3个与GT零交叠且最小距离≥10 grid的孤立域；FP增加283像素；远端孤立域对单向ASD的加性贡献差+3.858 grid。

表面点方向可区分远端假阳性与漏分割的位置；范围延伸用GT空切片预测像素描述，漏分割用FN/GT→预测尾部描述。孤立域、范围延伸和局部偏移可能共存，这些是两个冻结模型输出的现象，不证明某个训练模块造成错误。没有把所有距离变化统称为孤立假阳性。

定性选择：在每个优先frame内选HD95差最大的类别；分别选R1预测→GT距离总和最高的slice（其对单向ASD贡献最大），以及双向≥该体积HD95尾部距离总和最高的slice；逐规则首次tie。不是最大GT面积slice。统一原生方向、颜色、图像强度范围及尺度，图中所有类均显示，具体slice/class/volume指标见qualitative_selection.csv。匿名ACDC ID，不携带DICOM元数据；原始医学数组/全预测不打包。

证据缺口：上述预测、GT和评价均可用；这里只覆盖现有冻结输出，不是全程机制因果或跨种子证据。R1 q_B与质量使用此前固定标注/验证样本的独立诊断结果，另存analysis/R1_confidence_quality*.csv，与此测试误差描述分开。固定92已按全程面积锁定，未按这些误差调整。

进一步确认：patient065与075的全部12个类别/frame行中，远端零GT交叠域（6邻域、最小距GT≥10 grid）对单向ASD的加性贡献项差之和为24.252260，全ASD差之和为28.471256，净占比85.18%；剩余贡献项差之和4.218996。因此这两病例的ASD恶化主要伴随远端假阳性的距离贡献增大，仍保留局部轮廓/漏分割等其他误差。它是两个冻结预测的几何分解，不证明TCSMix导致了这些错误；HD95是分位数，不能照此作加性归因。原始列和公式见analysis/R1_GEOMETRY_ATTRIBUTION_SUMMARY.json及mask_volume_metrics.csv。
