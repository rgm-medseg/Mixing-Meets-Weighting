"""R2 results and manuscript handoff; actual negative or missing results are retained."""
from common import *

def main():
    state=read_json(WORK/'state/status.json');path=WORK/'results/EVALUATION_RECEIPT.json';evaluation=read_json(path) if path.exists() else None
    selected=read_json(WORK/'reports/SELECTED_CHECKPOINT.json') if (WORK/'reports/SELECTED_CHECKPOINT.json').exists() else {}
    analysis=read_json(WORK/'analysis/ANALYSIS_RECEIPT.json') if (WORK/'analysis/ANALYSIS_RECEIPT.json').exists() else dict(status='PARTIAL',reason='analysis not completed')
    if not evaluation:
        write_csv(WORK/'results/R2_results.csv',[dict(run='R2_fixed92',status='NA',dice='NA',iou='NA',hd95='NA',asd='NA',reason='frozen validation-best final evaluation not completed')])
    for name,fields in [('R2_class_metrics.csv',['class_name','dice','iou','hd95','asd','reason']),('R2_case_metrics.csv',['case','patient','dice','iou','hd95','asd','reason']),('R2_patient_metrics.csv',['patient','dice','iou','hd95','asd','reason'])]:
        if not (WORK/'results'/name).exists():write_csv(WORK/'results'/name,[],fields)
    comparison=read_table(WORK/'analysis/method_comparison.csv') if (WORK/'analysis/method_comparison.csv').exists() else []
    table='\n'.join('|'+str(r['method'])+'|'+'|'.join(str(r.get(k,'NA')) for k in ['dice','iou','hd95','asd'])+'|' for r in comparison)
    values=evaluation['metrics'] if evaluation else {k:'NA' for k in ['dice','iou','hd95','asd']}
    lock=read_json(WORK/'R2_AREA_LOCK.json');errors=read_json(WORK/'state/stage_errors.json') if (WORK/'state/stage_errors.json').exists() else []
    timing=read_json(WORK/'analysis/runtime_cost.json') if (WORK/'analysis/runtime_cost.json').exists() else {}
    (WORK/'FINAL_REPORT_R2_CN.md').write_text(f'''# R2固定面积正式对照结果

报告时刻{now()}，阶段快照{state['stage']}/{state['status']}。完成训练、完成评价、完成归档是不同事件；最终流水线状态见state/status.json，归档后另写final_archive_receipt.json及外层SHA256，二者不会伪装成包内此前快照。
仅一项ACDC10% seed42 R2：TRC + 固定92×92区域混合 + 原空间CTW。旧E1–E5保持暂停，R1和旧主结果未重训或覆盖。真实执行源来自R1冻结版本，controller.py字节相同，原训练科学循环字节相同；配置选择既有v2_fixed分支。完整工程差异见reports/R1_to_R2_engineering.patch、有效入口及run_manifest.json。

固定面积锁定：全R1 step1–30000 actual_area_pixels平均{lock['mean_actual_pixels']}，原方形整数平方距离最小92²=8464，平局选小边；源SHA{lock['source_sha256']}。固定比例.359375、实际面积.129150390625；有符号面积差{lock['signed_area_error']}，面积百分点差{lock['error_percentage_points']}，相对差{lock['relative_error']}。只匹配全程平均，不能声称每5000步阶段或每步匹配。固定定义未按测试分数、最优步或连续s均值重选。

q_B始终是原weak教师概率/既有LCC前景支持的类别等权间隔，仅用于日志；s_dynamic_B=.18+.20*q_B是参考值；s及s_actual=.359375、h/w_actual=92。图像与目标共用原互补掩码，两次位置randint及端点规则不变；合法位置范围因尺寸不同，两个独立训练轨迹并非逐步相同。原空间CTW(.05、概率L1、stop-gradient、原更新方向)、TRC4token/dim16、EMA.99、其他损失/增强/双视图与频率分支沿用R1。

原预训练net/opt重新初始化、joint_step0、TRC gamma0；没有用R1/旧Full/短程/E1权重起跑。初始化三分支公共状态及两优化器基础状态逐tensor核验，见INITIALIZATION_ACTUAL.json。原预训练生成日志未知如实NA，未新增预训练。70/10/20病例，7标注患者/136切片，256²、batch24/label12、16worker、FP32、SGD lr.01常数、momentum.9、wd1e-4；实际数据索引/哈希及名单附包。预算30000联合步，每学生正常各30000成功更新；恢复物理重放另存ledger，主RNG/采样器ID可恢复，worker增强和prefetch不保证位级恢复。

验证每200，student1前景macroDice严格提高更新best，平局沿用原规则。选择step{selected.get('selected_step','NA')}，checkpoint SHA{selected.get('sha256','NA')}；先冻结选择再最终测试，ACDC测试未用于此前候选方法选择，本次也未用于固定面积、控制器或检查点选择。

|方法|Dice fraction|IoU fraction|HD95 grid|ASD grid|
|---|---|---|---|---|
{table}

R2实际评价：{values}。40个ED/ES frame volumes×RV/MYO/LV等权平均，20patient两frame均值另存。输入/输出nearest order0、输出回原H5平面、z不重采样、spacing=(1,1,1)重建grid不是mm；ASD为MedPy prediction→GT单向，HD95双向表面距离合并95分位；无预测后处理。双空四项0，单空Dice/IoU0、距离为grid体积对角。同统一协议历史CSV复用，缺参照记NA，不从论文表猜病例值。

配对包含R2−R1、R2−旧Full、R2−β-FFT、R1−旧Full，全部改善/持平/下降行保留，见analysis/paired_*，没有剔除坏病例主结果。R1误差描述见R1_ERROR_ANALYSIS_CN.md及训练后R1_R2_ERROR_ANALYSIS_CN.md；patient065/075/083双frame的TP/FP/FN、逐slice面积/覆盖、6邻域三维连通域、双向表面分布及单向ASD贡献、实际大误差slice对照图均附CSV。对全部40frame复算并匹配原结果。这是测试后几何描述，不反向调整训练/固定面积/后处理。

R1 q_B与质量沿用已冻结标注/验证诊断与其来源索引，区别于测试误差描述；R2训练期间不增加GPU推理。本次只有单次训练，病例配对不是训练重复；R1/R2比较动态策略与近平均面积固定策略，不能拆分置信度信息、尺度波动和课程效应，不能据此证明CTW协同或跨种子稳定性。R2更好/持平/变差均保留，不自动搜索或再训。旧90.35%仍属于旧TCSMix Full，新机制不自动替换论文主结果和其他数据集。

实际成本：{timing}。端到端训练用train_segments中的活动墙钟（含取数/验证/保存），各阶段含轮询检测≤5s延迟；step_seconds不替代完整墙钟。新增trace/events/progress日志成本见logging_cost；未隔离旧日志开销，不声称严格端到端额外成本差。
分析状态：{analysis}。阶段错误：{errors or '无已记录阶段错误'}。大型权重/原医学影像/全预测留服务器，包内实际索引和哈希；匿名ID统一方向规范化渲染图可独立审查。无客户端在线依赖，服务器重启需要resume.sh；tmux不承诺跨重启存活。缺失或失败仍归档已有证据，科学核查与归档校验分开。
''')
    (WORK/'EDITOR_HANDOFF_R2_CN.md').write_text(f'''# 给写作助手：R2固定92面积对照

R2={values}，选中step{selected.get('selected_step','NA')}，SHA{selected.get('sha256','NA')}。实际来源results/R2_results.csv，分数fraction×100展示百分比，HD95/ASD为grid不是mm。旧Full90.35%归旧机制，v2_B R1与固定R2独立列明，不能把旧分数移到新版；本轮未修改论文。
唯一方法变化是v2_B动态尺寸变为固定92；公共科学循环、模型/损失/空间CTW/EMA/初始化/增强与评价保持R1。q_B继续记录但不控制掩码，s_dynamic_B参考，s_actual=.359375。R2面积比完整R1平均低约.04295面积pp（相对-.33145%），平均匹配不等于每阶段匹配。

主要对照原始CSV、配对病例和阶段面积图在analysis；全部不利行保留。R1/R2一次对照只比较整体动态和近平均固定策略，不能分离置信度信息、尺度波动/课程贡献，也不能证明与CTW协同、跨种子稳定性。训练结果下降允许报告，不引出自动搜索。
R1_ERROR_ANALYSIS_CN.md和R1_R2_ERROR_ANALYSIS_CN.md只描述冻结测试预测几何；不能用误差slice倒推固定尺度/控制器/最佳checkpoint。HD95需要双向表面距离，ASD单向prediction→GT；孤立域和GT空slice范围延伸有实际mask/连通域/方向距离贡献证据，局部偏移和漏分割可共存，不能统称一个原因。案例选择为真实距离贡献最大的slice，规则/匿名ID/方向/颜色/指标附CSV，不只展示最大GT面积。
R1固定标注/验证q_B质量诊断是另一组证据，未使用测试选择方法；切片或病例不等于独立训练重复。历史生成预训练日志与部分旧import版本缺失如实标NA，不强称全部历史完全可核验。

完整协议、有效代码/工程patch、运行命令、初始化/检查点回执、数据/预测/权重哈希和科学核查附包。snapshot状态是归档前时刻，最终状态与外层校验由完成回执另记。当前分析状态{analysis.get('status')}，实际错误{errors or '无'}。
''')
    print('R2 report/handoff written; all actual or NA results retained.',flush=True)

if __name__=='__main__':main()
