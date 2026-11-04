"""Retain every experiment outcome and generate reviewer/editor handoff."""
import json
from common import *

def main():
    statepath=WORK/'state/queue.json'
    state=read_json(statepath) if statepath.exists() else read_json(WORK/'run_manifest.json')
    rows=[]; class_rows=[]
    for task in ['Historical_baseline','Historical_Full']+[f'E{i}' for i in range(1,6)]:
        if task.startswith('Historical'): cfg=read_json(WORK/f'configs/{task}.json')
        else: _,cfg=config_for(task)
        receipt=WORK/f'runs/{task}/evaluation/EVALUATION_RECEIPT.json'
        r=read_json(receipt) if receipt.exists() else {}
        cp=r.get('checkpoint',cfg.get('checkpoint',str(Path(cfg['run_root'])/'checkpoints/best.ckpt') if 'run_root' in cfg else 'NA'))
        m=r.get('metrics',{})
        status='EVALUATED' if r else state['tasks'].get(task,{}).get('status','NA')
        rows.append(dict(task=task,status=status,dataset='ACDC',ratio='10%',protocol='ACDC_v22_grid_directed_ASD',seed=42,
                         training_updates=30000,pretrain_budget='scratch/no pretrain' if task=='E5' else 'reused historical checkpoint; generation budget NA',
                         run=str(WORK/'runs'/task),checkpoint=cp,checkpoint_sha256=r.get('checkpoint_sha256','NA'),checkpoint_step=r.get('checkpoint_step','NA'),
                         dice=m.get('dice','NA'),dice_percent=100*m['dice'] if 'dice' in m else 'NA',iou=m.get('iou','NA'),
                         hd95=m.get('hd95','NA'),asd=m.get('asd','NA'),boundary_unit='grid',source=str(receipt),
                         reason=state['tasks'].get(task,{}).get('reason',''),result_type='unified reevaluation of frozen historical checkpoint' if task.startswith('Historical') else 'single new training'))
        cls=receipt.parent/'class_metrics.csv'
        if cls.exists(): class_rows+=read_table(cls)
    write_csv(WORK/'experiment_results.csv',rows)
    write_csv(WORK/'class_metrics.csv',class_rows,['run','class_name','unit','dice','iou','hd95','asd'])
    mapping={r['task']:r for r in rows}; comparisons=[]
    for r in rows:
        full=mapping['Historical_Full']
        if isinstance(r['dice'],float) and isinstance(full['dice'],float):
            comparisons.append(dict(task=r['task'],dice_difference_pp=100*(r['dice']-full['dice']),reference='Historical_Full'))
    if all(isinstance(mapping[t]['dice'],float) for t in ('Historical_Full','E1','E2','E3')):
        # effect of spatial weighting at dynamic scale minus at fixed scale.
        interaction=(mapping['Historical_Full']['dice']-mapping['E1']['dice'])-(mapping['E2']['dice']-mapping['E3']['dice'])
        comparisons.append(dict(task='2x2_interaction',dice_difference_pp=100*interaction,reference='(Full-E1)-(E2-E3)'))
    write_csv(WORK/'reports/single_run_differences.csv',comparisons,['task','dice_difference_pp','reference'])
    statuses={}
    for i in range(1,7):
        path=WORK/f'analysis/A{i}/status.json'
        statuses[f'A{i}']=read_json(path) if path.exists() else dict(status='DEFERRED',reason='Not run yet')
    table='|任务|状态|Dice (%)|IoU|HD95 (grid)|ASD (grid)|\n|---|---|---:|---:|---:|---:|\n'
    table+='\n'.join(f"|{r['task']}|{r['status']}|{r['dice_percent']}|{r['iou']}|{r['hd95']}|{r['asd']}|" for r in rows)
    analysis='\n'.join(f"- {k}: {v['status']} — {v.get('reason','')}" for k,v in statuses.items())
    scale=read_json(WORK/'configs/fixed_scale.json')
    report=f"# β-FFT 修订证据报告\n\n更新时间：{now()}；队列：{state.get('status','未启动')}。训练完成、评价完成与归档分别记录。\n\n{table}\n\n{analysis}\n\n固定矩形97×97，边长比例{scale['fixed_side']}；来源是历史 Full 每200步的150条记录，并非全程逐步平均。所记录面积恒为14.357%，这限制了E2动态几何优势的解释。\n\n只报告单次运行差值，见 reports/single_run_differences.csv。负结果不删除；不做跨种子显著性或因果声明。\n\n预训练沿用已核验的net/opt文件，文件产生时的实际命令、seed和预算无法从权重推出，保持NA。半监督预算每项30000updates；E5采用官方两学生AdamW1e-4、B16、四加载器、从头初始化，没有隐藏预训练。\n\ncheckpoint保留服务器，路径/哈希见CSV和reports/source_index.csv。16worker随机流与预取无法完整恢复，续训保存主进程随机状态及批次ID但不宣称位级精确。\n\n所有范围、源码、历史口径差异见protocol_audit.md、implementation_audit.md、hyperparameter_provenance.md。未完成训练的数字保持NA，不能写作已完成的论文事实。\n"
    (WORK/'FINAL_REPORT_CN.md').write_text(report)
    covid=WORK/'reports/covid_reevaluation_macro.csv'
    covid_addendum='\n\nCOVID评价缩放spacing问题：最终evaluator沿axis=2正确重建XYZ，轴序本身正确；512→256缩放后spacing未更新。历史记录的mm标签没有反映正确resize像素间距，原结果保留。已用固定模型的预测另存legacy重放、corrected derived center grid、corrected native grid三种固定口径。见reports/covid_geometry_audit.csv与covid_existing_prediction_reevaluation.csv，不选择更好数字替换口径。\n'
    if covid.exists():
        data=read_table(covid)
        covid_addendum+='\n|COVID run|评价口径|Dice|IoU|HD95|ASD|\n|---|---|---:|---:|---:|---:|\n'
        covid_addendum+='\n'.join(f"|{r['run']}|{r['policy']}|{r['dice']}|{r['iou']}|{r['hd95_mm']}|{r['asd_mm']}|" for r in data)+'\n'
    report+=covid_addendum
    (WORK/'FINAL_REPORT_CN.md').write_text(report)
    handoff=report+"\n# 给论文写作助手\n\nE1比较同一控制模型的空间CTW与逐图均匀均值权重；均值相同不等于梯度范数相同。E2固定实际面积，E3复用同一尺度/均匀实现，交互定义(Full−E1)−(E2−E3)。E4普通1×1/LN/GELU/1×1残差，宽18，647参数，对照TRC641，误差+0.936%；三模型同时替换，零门控和EMA保留。E5是官方AdaMix-CT方法代码与原生U-Net的同划分适配，不是β-FFT中名为AdaMix的TCSMix。\n\nCSV中Dice/IoU是[0,1]比例，dice_percent为百分数，差值pp为百分点。ACDC HD95/ASD为原H5体积网格单位；ASD为MedPy单向prediction→GT。case是patient+frame，patient列才是患者独立单位；class_id1/2/3对应RV/MYO/LV。A2前景/背景置信度按GT分区，checkpoint/slice不是独立患者。A3统计GT-only/mix-only/overlap/interior，混合边界允许与GT边界重叠。\n\n每张图的文件名/标题与selection_and_provenance.csv、confidence_proxy.csv或spatial_response.csv共同给出run、checkpoint、case/slice及step。A1历史轨迹每200步采样，新轨迹每步记录，两者不拼接。A5计时重复不是重复训练；MAC计数不含归一化等，具体约定在cost.csv。A6预先固定最大/中位/最小患者增益，同病例切片显示，保留变差结果。\n\n来源材料、配置、日志、CSV、SVG/PNG和脚本均进入revision_evidence_20261006.tar.gz，可离线审查；原始影像、GT数据数组、凭据和大权重不打包。压缩包SHA与内部SHA256SUMS用于完整性检查。\n\n作者更正：ACDC测试成绩未参与候选方法选择。历史报告中的排序标签不能自行推导相反结论；如需证明实际决策时点仍应核对原作者记录。COVID外部文献行单位未知保持NA，不能自动统一写mm。\n"
    handoff+='\nA6另保留COVID所有4例，5%/10%同切片/GT/颜色显示，来自既有baseline预测及原Full冻结检查点重新推理，无COVID训练。Corrected native grid在canonical原网格上对categorical预测nearest回采样、GT不缩放、spacing按(Z,X,Y)排序；derived-center grid则按SciPy grid_mode=False的像素中心跨度更新spacing。两口径明确区分，不跨表混用。外部论文行仍NA，未经同程序重评价不声称全表方法口径统一。\n'
    (WORK/'EDITOR_HANDOFF_CN.md').write_text(handoff)
    atomic_json(WORK/'reports/result_status.json',dict(experiments={r['task']:r['status'] for r in rows},analyses=statuses,time=now()))
    print(table)

if __name__=='__main__': main()
