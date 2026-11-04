"""Additional A2 controller plot/text and A5 source audit, entirely from completed CSVs."""
import ast,shutil
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
from finish import figsave,mdtable
def main():
 co=rows(W/'results/A2_correlations.csv');sample=rows(W/'results/A2_sample_class_quality.csv');patient=rows(W/'results/A2_patient_correlation_inputs.csv');correct=rows(W/'results/A2_confidence_correctness.csv')
 rr=[r for r in co if r['cohort']=='validation' and r['mode']=='eval' and r['prediction_kind']=='original_pseudolabel' and r['class_id']=='GT_present_foreground_macro' and r['signal']=='sample_control_proxy' and r['quality']=='dice']
 csvout(W/'results/A2_actual_control_summary.csv',rr)
 methods=[m for m in config()['methods'] if m['id']!='baseline'];fig,axs=plt.subplots(1,3,figsize=(12.8,4.2),layout='constrained')
 for ax,m in zip(axs,methods):
  for tag,color,marker in [('best','#0072B2','o'),('last','#D55E00','x')]:
   ss=[r for r in sample if r['method_id']==m['id'] and r['checkpoint_tag']==tag and r['mode']=='eval' and r['cohort']=='validation' and r['prediction_kind']=='original_pseudolabel' and r['class_id']=='GT_present_foreground_macro' and r['dice']!='NA']
   pp=[r for r in patient if r['method_id']==m['id'] and r['checkpoint_tag']==tag and r['mode']=='eval' and r['cohort']=='validation' and r['prediction_kind']=='original_pseudolabel' and r['class_id']=='GT_present_foreground_macro' and r['signal']=='sample_control_proxy' and r['quality']=='dice']
   ax.scatter([float(r['sample_control_proxy']) for r in ss],[float(r['dice']) for r in ss],s=12,color=color,marker=marker,alpha=.35,label=tag+' slices')
   ax.scatter([float(r['confidence']) for r in pp],[float(r['quality_value']) for r in pp],s=40,marker='D',facecolors='none',edgecolors=color,label=tag+' patients')
  ax.set(xlabel='Sample mean max probability' if m['id']=='Full_main' else 'Sample foreground margin proxy',ylabel='GT-present macro Dice (fraction)',ylim=(0,1.02),title=m['name']+('\nrecorded signal; fixed geometry' if m['id']=='R2_fixed92' else ''));ax.legend(fontsize=7)
 figsave(fig,'A2_actual_control_quality','Actual version-specific single-sample controller proxies on205 GT-foreground-present validation slices, separately best/last EMA teachers. Loadable paper Full uses all-pixel maximum probability; R1 uses equal foreground-class probability margins on literal LCC support. R2 records the same margin signal while keeping92×92 geometry. Open diamonds are10 patient summaries: identical valid slices are averaged equally within frame, then frames equally. Baseline has no TCSMix control proxy. These are offline proxies; actual historical unlabeled batch_control_value is NA. No cross-checkpoint significance test or dynamic-mixing performance attribution.')
 def row(mid,tag,level):return next(r for r in rr if r['method_id']==mid and r['checkpoint_tag']==tag and r['level']==level)
 bestslice=row('R1_v2_B','best','slice');bestpat=row('R1_v2_B','best','patient');lastslice=row('R1_v2_B','last','slice');lastpat=row('R1_v2_B','last','patient');fulls=row('Full_main','best','slice');fullp=row('Full_main','best','patient')
 en=f"The foreground class-balanced margin proxy in R1 had a Spearman correlation of {float(bestslice['spearman']):.3f} with GT-present foreground macro Dice across {bestslice['valid_n']} validation slices. The patient-level correlation was {float(bestpat['spearman']):.3f} across {bestpat['valid_n']} patients. The final teacher retained positive associations of {float(lastslice['spearman']):.3f} and {float(lastpat['spearman']):.3f}, respectively. For the separately loaded paper Full teacher, the all-pixel confidence proxy gave correlations of {float(fulls['spearman']):.3f} at the slice level and {float(fullp['spearman']):.3f} at the patient level. The foreground-conditioned signal tracked validation quality more closely in these frozen R1 snapshots."
 tab=[{k:r[k] for k in ['method_id','checkpoint_tag','level','spearman','valid_n','patients']} for r in rr]
 text='\n\n## 实际控制代理值与质量的补充\n\n'+mdtable(tab,['method_id','checkpoint_tag','level','spearman','valid_n','patients'])+f'\nR1的实际前景类别等权间隔代理与GT-present伪标签macroDice，在best切片/患者层面ρ={float(bestslice["spearman"]):.6f}/{float(bestpat["spearman"]):.6f}，last为{float(lastslice["spearman"]):.6f}/{float(lastpat["spearman"]):.6f}。两个保留状态均呈正相关。真实可加载论文Full的全像素置信度代理在best切片/患者层面ρ={float(fulls["spearman"]):.6f}/{float(fullp["spearman"]):.6f}。方法与置信度定义分别列出；R2中q_B是记录信号，固定几何仍为92×92。控制代理与伪标签质量共享预测/LCC支持，本项描述当前冻结教师的条件关系，未分离训练机制的独立收益。\n\n'+en+'\n\n证据：A2_actual_control_summary.csv、A2_patient_correlation_inputs.csv、A2_actual_control_quality及图注。\n'
 precision=[]
 for m in config()['methods']:
  for c,name in [('1','RV'),('2','MYO'),('3','LV')]:
   for weight in ['pooled_pixel','patient_equal']:
    pred=next(r for r in correct if r['method_id']==m['id'] and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['axis']=='predicted' and r['group']==c and r['weighting']==weight)
    gt=next(r for r in correct if r['method_id']==m['id'] and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['axis']=='GT' and r['group']==c and r['weighting']==weight)
    precision.append(dict(method_id=m['id'],checkpoint_tag='best',branch='ema_model',mode='eval',cohort='validation',class_name=name,weighting=weight,precision=pred['accuracy'],recall=gt['accuracy'],predicted_support_pixels=pred['pixel_count'],GT_support_pixels=gt['pixel_count'],precision_support_patients=pred['support_patients'],recall_support_patients=gt['support_patients'],definition='pooled TP/predicted-class and TP/GT-class; patient_equal averages patient conditional rates with support'))
 csvout(W/'results/A2_class_precision_recall_summary.csv',precision)
 text+='\n逐类precision/recall汇总：\n\n'+mdtable(precision,['method_id','class_name','weighting','precision','recall'])
 for path in [W/'reports/A2_REPORT_CN.md',W/'WRITING_HANDOFF_A2_A5_CN.md',W/'FINAL_REPORT_A2_A5_CN.md']:
  old=path.read_text()
  if '## 实际控制代理值与质量的补充' not in old:atomic(path,old+text)
 paths=[]
 for m in config()['methods']:
  root=Path(m['run']);source=root/'v2_effective_source.py' if (root/'v2_effective_source.py').exists() else Path(m['source_path']);calls=[]
  sizes={'uimg_a':6,'uimg_b':6,'uimg_a_s':6,'uimg_b_s':6,'net_input_unl_1':6,'net_input_l_1':6,'net_input_unl_2':6,'net_input_l_2':6,'input_volume_batch':18,'input_volume_batch_s':18,'image_patch_last_FFT.unsqueeze(1)':24}
  source_text=source.read_text();tree=ast.parse(source_text)
  for n in ast.walk(tree):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['model_1','model_2','ema_model']:
    arg=ast.get_source_segment(source_text,n.args[0]);assert arg in sizes
    calls.append(dict(method_id=m['id'],network=n.func.id,line=n.lineno,input_expression=arg,input_batch=sizes[arg],calls_each_joint_iteration=1,source=str(source),source_sha256=sha(source),full_training_FLOPs='NA'))
  assert sum(r['network']=='model_1' for r in calls)==5 and sum(r['network']=='model_2' for r in calls)==3 and sum(r['network']=='ema_model' for r in calls)==4
  paths.extend(calls)
  ns='\n\n'.join(ast.get_source_segment(source_text,n) for n in tree.body if isinstance(n,ast.FunctionDef) and n.name in ['get_ACDC_masks','get_ACDC_2DLargestCC'])
  atomic(W/'reports'/f"LITERAL_PSEUDOLABEL_{m['id']}.py",ns+'\n')
 csvout(W/'results/A5_training_forward_paths.csv',paths)
 js(W/'reports/POST_REVIEW_RECEIPT.json',dict(time=now(),new_training_updates=0,new_inference=0,source='completed A2/A5 CSV and literal source',controller_plot=True,class_precision_recall=True,training_forward_counts='student1:5 calls batches6,6,18,18,24; student2:3 calls batches6,6,24; EMA:4 calls batch6',all_original_results_retained=True))
if __name__=='__main__':main()
