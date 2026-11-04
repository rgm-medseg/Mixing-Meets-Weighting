"""Write current/final scientific handoff without hiding absent or unfavorable results."""
from common import *

def main():
    status=read_json(WORK/'state/status.json');evalpath=WORK/'results/EVALUATION_RECEIPT.json'
    evaluation=read_json(evalpath) if evalpath.exists() else None
    analyses=read_json(WORK/'analysis/ANALYSIS_RECEIPT.json') if (WORK/'analysis/ANALYSIS_RECEIPT.json').exists() else dict(status='PARTIAL',reason='analysis unavailable')
    result=read_table(WORK/'results/R1_results.csv') if (WORK/'results/R1_results.csv').exists() else []
    if not result:
        write_csv(WORK/'results/R1_results.csv',[dict(run='R1_v2_B',status='NA',dice='NA',iou='NA',hd95='NA',asd='NA',reason='final selected-checkpoint evaluation not completed')])
    for name,fields in [('R1_case_metrics.csv',['case','patient','dice','iou','hd95','asd','reason']),('R1_class_metrics.csv',['class_name','dice','iou','hd95','asd','reason'])]:
        if not (WORK/'results'/name).exists():write_csv(WORK/'results'/name,[],fields)
    values=evaluation['metrics'] if evaluation else {k:'NA' for k in ['dice','iou','hd95','asd']}
    metrics='\n'.join(f'|{k}|{v}|' for k,v in values.items())
    deltas=[]
    for ref in ['Historical_Full','Historical_baseline']:
        p=WORK/f'reports/{ref}_EVALUATION_RECEIPT.json'
        if not p.exists():deltas.append(f'{ref}: NA，历史同协议评价缺失');continue
        old=read_json(p)
        dd={k:values[k]-old['metrics'][k] if evaluation else 'NA' for k in values}
        deltas.append(f'{ref}: 原指标 {old["metrics"]}；R1−参照 {dd}。Dice/IoU为fraction，乘100才为百分点差。')
    timing=read_json(WORK/'analysis/runtime_cost.json') if (WORK/'analysis/runtime_cost.json').exists() else {}
    selection=read_json(WORK/'reports/SELECTED_CHECKPOINT.json') if (WORK/'reports/SELECTED_CHECKPOINT.json').exists() else dict(selected_step='NA',sha256='NA')
    errors=read_json(WORK/'state/stage_errors.json') if (WORK/'state/stage_errors.json').exists() else []
    (WORK/'FINAL_REPORT_R1_CN.md').write_text(f'''# R1新版Full实际结果

报告时刻{now()}。仅R1：TRC+锁定TCSMix v2_B+原空间CTW；seed42、ACDC10%、同原pretrain net/opt、30000联合迭代。R2和旧E1–E5未启动/恢复。当前阶段{status.get('stage')}，状态{status.get('status')}；训练完成不自动等同评价与归档完成。

|指标|实际值（fraction或grid距离）|
|---|---|
{metrics}

评价完成：{bool(evaluation)}。选择step={selection.get('selected_step')}，checkpoint SHA={selection.get('sha256')}；按原严格student1验证前景macroDice，测试只用于冻结后的最终评价。两学生实际调用数/重复更新见reports/TRAIN_POSTCHECK.json、results/optimizer_calls.csv、完整R1_control_trace.csv。初始化实际三分支/零门控/预训练哈希见INITIALIZATION_ACTUAL.json；原始预训练生成日志未知，未补造。

评价协议为原统一ACDC_v22_grid_directed_ASD：输入切片nearest resize256²，输出nearest回原H5切片平面，z轴不重采样；原重建grid单位spacing(1,1,1)，HD95/ASD不能标mm；无预测后处理，ASD为预测→GT方向。双空四指标0，单空Dice/IoU0、距离为体积grid对角。总体等权平均40个ED/ES frame体积×3前景类；病例/患者/类表单独保留，20个患者的两frame平均用于配对分析。

历史参照与差值：
{chr(10).join(deltas)}

旧Full90.35%只属旧机制；新结果允许相同或下降，不自动重训。参照同测试清单与统一评价，但部分历史import版本/预训练生成协议无法完全核验；baseline还缺TRC/旧TCSMix/CTW，差值不是单独模块因果。新版与old Full分数差值也不能单独归于动态尺度，R2尚未执行。

机制统计按预先固定的六个5000步区间及全程给出，见analysis/scale_distribution.csv和control_trajectory.png/SVG；raw trace全部保留，恢复重复step单独记录、最终逻辑轨迹取最后一次。old_q是同一新版教师输出的控制回放，无额外前向或随机数消耗；不是独立old训练。若晚期尺寸固定，如实保留。R2_FIXED_AREA_PROPOSAL只据完整真实面积在测试前生成建议，不执行R2。

离线质量诊断状态：{analyses.get('status')}，细节{analyses.get('details',analyses)}。原固定标注/验证样本GT只用于诊断；重复切片/checkpoint不算独立病例。v2_B概率间隔不是校准正确率；极小/错误前景类、缺类及fallback保留。保留全部改善/持平/变差配对行；定性例按最大差、median差、最小差选，slice取最大GT前景面积，并给匿名ID规则/provenance。预测和原医学数组留服务器，包中只交付既有匿名ID规范的规范化图及索引。

实际成本：{timing}。训练用实际进程阶段时长，验证尾部包含在内；独立评价/分析/归档时长见state/stage_history.json。日志成本见reports/logging_cost_*.json，只计新trace/events/progress，不把控制器微基准当端到端差值。worker增强随机流和prefetch不能位级恢复；完整可恢复状态保存，不更改16worker协议。

失败与缺失：{errors or '目前无阶段错误记录'}。缺失指标写NA；科学缺失与归档错误分别记录，FAILED/PARTIAL结果也归档已有证据。没有改batch/种子/控制器或自动从零重试。
''')
    (WORK/'EDITOR_HANDOFF_R1_CN.md').write_text(f'''# 给写作助手：R1 TCSMix v2_B

新版公式只替换教师控制信号：原weak参考视图softmax(dim1)/detach，复用原argmax+每类LCC伪标签支持；先按有支持前景类计算mean(P_c−max其他P)，再类等权；空前景q0。仍s=.18+.20q，Python round和原clamp，原随机位置、共享一个矩形、图/目标同步互补混合。TRC和原空间CTW(.05、L1/stop-gradient/更新方向)及其他Full设置保留。不要把本项目旧名AdaMix与外部AdaMix论文混同。

本次正式指标：{values}；检查点step{selection.get('selected_step')}，SHA{selection.get('sha256')}。NA表示尚未得到，不能拿旧90.35填入新版列；旧90.35%/统一复评90.349657652%继续列在old Full。
实际协议、命令、初始化和代码哈希见protocol.json、run_manifest.json及configs；测量与平均单位见FINAL_REPORT_R1_CN.md。HD95/ASD是原H5重建grid距离，不能改成mm；ASD为定向值。

新旧配对差值见analysis/paired_case_class_differences.csv和paired_aggregated_differences.csv，全部病例含不利行保留。不能仅根据R1相对old分数差值归因于动态控制；后续匹配平均实际面积的R2才是另一个对照，本轮只给面积建议。单次运行不支持多种子显著性、泛化或新版CTW交互结论。
图注：control_trajectory为真实全程v2_B轨迹，old曲线是同教师输出回放；阶段分布预先固定5000步，重复step按最终轨迹处理且raw全部保留。confidence_quality为独立final teacher回放（eval/train_snapshot），原标注/验证GT只诊断，不是全程teacher质量或测试选参。patient_differences包含正/零/负差值；qualitative选择规则与匿名ID表一起交付，不能只保留改善例。
细小错误前景、部分阶段取整固定、代理信号未校准及恢复边界均须如实写出。状态/缺失：{analyses}。本轮不修改论文题目或模块名，正式评价允许下降，不自动再训。
''')
    print('Scientific report and editor handoff written, missing formal values remain NA.')

if __name__=='__main__':main()
