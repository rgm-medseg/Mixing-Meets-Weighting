import re,shutil
from collections import defaultdict
import numpy as np
from scipy.stats import spearmanr
from common import *
O=ROOT/'offline_A2_A5_20261007'
def main():
 cfg=config();sources=[];full=cfg['methods'][1];lines=[]
 for m in cfg['methods']:
  for cp in m['checkpoints']:
   sources.append(dict(method_id=m['id'],checkpoint_tag=cp['tag'],global_step=cp['step'],checkpoint_path=cp['path'],checkpoint_sha256=cp['sha256'],A2_branch='ema_model',A5_branch='model_1',A3_branches='model_1;model_2;ema_model',actual_source=str(Path(m['run'])/('v2_effective_source.py' if m['id'].startswith('R') else Path(m['source_path']).name)),control_evidence='bound control_runtime -> controller.generate_mask q_B' if m['id'].startswith('R') else 'effective import v14 generator; sparse endpoint mean-max' if m['id']=='Full_main' else 'no TCSMix',paper_version_mapping='NEEDS_MAPPING' if m['id']=='Full_main' else 'verified run identity',score_owner={'Full_main':'90.34965765205% historical Full','R1_v2_B':'89.98445206808% R1','R2_fixed92':'90.20444131485% R2','baseline':'89.660634% baseline'}[m['id']]))
 # Limited source inquiry: indexed historical Full; no searching/selection by new test scores.
 log=Path(full['run'])/'train.log';endpoints=[]
 for line in log.read_text().splitlines():
  if 'adamix_endpoint=' in line:
   st=json.loads(line.split('adamix_endpoint=',1)[1]);step=int(re.search(r'iteration (\d+)',line)[1]);s=st['side_fraction'];endpoints.append(dict(step=step,**st,inferred_rectangle_h=round(256*s),inferred_rectangle_w=round(256*s),evidence='sampled endpoint plus audited formula; not all-iteration masks'))
 csvout(W/'results/Full_recorded_control_endpoints.csv',endpoints)
 for p in [Path(full['source_path']),ROOT/'train_ACDC/v14_mask_adamix_components.py',R1/'scripts/control_runtime.py',R1/'scripts/controller.py',R1/'scripts/revision_runtime.py']:
  text=p.read_text().splitlines();lines.append('\nSOURCE '+str(p)+' SHA '+sha(p)+'\n')
  for i,line in enumerate(text):
   if any(k in line for k in ['generate_adamix','conf_a =','teacher_conf =','side_fraction =','patch_x =','patch_y =','from control_runtime','ctw_maps','boundary_weight','loss = ((p1','contour =','from revision_runtime']):lines.append(f'{i+1}: {line}\n')
 atomic(W/'reports/CONTROL_CODE_SNIPPETS.txt',''.join(lines));csvout(W/'results/Full_control_source_manifest.csv',sources)
 atomic(W/'reports/FULL_CONTROL_MAPPING_CN.md',f'''# Full控制量来源对应\n\n状态NEEDS_MAPPING。作者说明原文保留：“旧版TCSMix置信度变化有效改变实际矩形，机制与R1一致，论文主版本Full Dice为90.35%。”作者更正：ACDC测试集没有参与原论文候选方法选择。\n\n本轮限定读取旧source_index/frozen_source_versions及已有四运行。90.35%可定位检查点仍为{full['checkpoints'][0]['path']}，SHA {full['checkpoints'][0]['sha256']}，student1，step29600。其有效源码直接导入v14生成函数，已有冻结模块哈希与索引一致；该模块定义q为两组教师softmax(dim1)最大类概率均值，s=.18+.20q，Python round后裁剪1..255。未在该有效入口发现R1的control_runtime包装绑定。缺少原启动进程完整sys.path/运行时函数对象快照，不能把现存模块定义当作唯一完整历史运行证明。\n\n该运行已有{len(endpoints)}条每200步endpoint，s范围{min(r['side_fraction'] for r in endpoints):.10f}–{max(r['side_fraction'] for r in endpoints):.10f}，均值{np.mean([r['side_fraction'] for r in endpoints]):.10f}；据已核查取整规则，已记录迭代矩形均97×97。这些记录不能证明全部30000迭代无变化；没有R1式逐迭代历史控制轨迹。其与作者“有效改变实际矩形且与R1一致”的唯一版本映射尚未解决。\n\nR1实际v2_effective_source覆盖导入control_runtime生成器，显式复用EMA已有LCC伪标签，经controller.generate_mask和signals取前景类别等权间隔q_B，geometry取整。R2同信号，但geometry固定92×92。真实R1/R2逐迭代轨迹可定位，模型成绩各归各运行。候选来源表见Full_control_source_manifest.csv；R1不是90.35%检查点的替换来源。需要作者对应的另一条运行/入口/检查点身份才能解决层级映射，本轮不无限追查，也不新增训练。\n\nA2原Full_main='old'是分析配置人工映射，不能单凭字符串证明论文版本。保留原控制代理列；所有方法重新计算同口径信号时，Full/baseline q_B标为offline auxiliary，R1是sample control proxy（非历史batch值），R2仅记录信号，固定输出几何。Full A3原策略NA；其CTW模型与公式可独立进行固定92 matched-mask诊断。\n''')
 rr=rows(O/'results/A2_sample_class_quality.csv');shutil.copy2(O/'results/A2_sample_class_quality.csv',W/'raw_units/A2_sample_class_quality_source.csv')
 rr=[r for r in rr if r['cohort'] in ['validation','labeled_train'] and r['mode']=='eval'];corr=[];pi=[];zero=[]
 for m in cfg['methods']:
  for cp in ['best','last']:
   for cohort in ['validation','labeled_train']:
    for kind in ['raw_argmax','original_pseudolabel']:
     base=[r for r in rr if r['method_id']==m['id'] and r['checkpoint_tag']==cp and r['cohort']==cohort and r['prediction_kind']==kind]
     for macro in ['GT_present_foreground_macro','fixed_three_foreground_macro']:
      selected=[r for r in base if r['class_name']==macro];valid=[r for r in selected if r['dice'] not in ['NA','']]
      if cohort=='validation':assert len(selected)==210 and len(valid)==(205 if macro.startswith('GT_present') else 210)
      for signal in ['mean_max_probability','old_q_auxiliary','q_B_auxiliary']:
       xx=np.array([float(r[signal]) for r in valid]);yy=np.array([float(r['dice']) for r in valid]);prefix=dict(method_id=m['id'],checkpoint_tag=cp,cohort=cohort,mode='eval',prediction_kind=kind,quality=macro,signal=signal,signal_role='offline auxiliary' if signal=='q_B_auxiliary' and m['id'] in ['baseline','Full_main'] else 'recorded q_B; fixed geometry' if signal=='q_B_auxiliary' and m['id']=='R2_fixed92' else 'sample control proxy; batch historical NA' if signal=='q_B_auxiliary' else 'common offline signal')
       corr.append(dict(**prefix,level='slice',n=len(valid),spearman=float(spearmanr(xx,yy).statistic),excluded=len(selected)-len(valid),p_value='NA: descriptive only'))
       frame=defaultdict(list)
       for r in valid:frame[(r['patient'],r['frame'])].append(r)
       pats=defaultdict(list)
       for (patient,fr),ss in frame.items():pats[patient].append((np.mean([float(r[signal]) for r in ss]),np.mean([float(r['dice']) for r in ss]),ss))
       px=[];py=[]
       for patient,ff in sorted(pats.items()):
        x=float(np.mean([f[0] for f in ff]));y=float(np.mean([f[1] for f in ff]));px.append(x);py.append(y)
        pi.append(dict(**prefix,patient=patient,signal_value=x,quality_value=y,valid_slices=sum(len(f[2]) for f in ff),valid_frames=len(ff),paired_samples=';'.join(r['sample'] for f in ff for r in f[2]),aggregation='equal valid slices within frame then equal valid frames; identical x/y support'))
       corr.append(dict(**prefix,level='patient',n=len(pats),spearman=float(spearmanr(px,py).statistic),excluded=len(selected)-len(valid),p_value='NA: descriptive only'))
     absent={r['sample'] for r in base if r['class_name']=='GT_present_foreground_macro' and r['dice']=='NA'}
     for r in base:
      if r['sample'] in absent and r['class_name'] in ['RV','MYO','LV']:zero.append(r)
 csvout(W/'results/A2_common_signal_correlations.csv',corr);csvout(W/'results/A2_common_signal_patient_inputs.csv',pi);csvout(W/'results/A2_zero_foreground_false_positives.csv',zero)
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams['svg.fonttype']='none'
 for level in ['slice','patient']:
  fig,axs=plt.subplots(2,4,figsize=(12,6),sharex='row',sharey=True)
  for col,m in enumerate(cfg['methods']):
   for row,signal in enumerate(['mean_max_probability','q_B_auxiliary']):
    ax=axs[row,col]
    for cp,color in [('best','#147c93'),('last','#bc5837')]:
     if level=='slice':ss=[r for r in rr if r['cohort']=='validation' and r['method_id']==m['id'] and r['checkpoint_tag']==cp and r['prediction_kind']=='original_pseudolabel' and r['class_name']=='GT_present_foreground_macro' and r['dice']!='NA'];xx=[float(r[signal]) for r in ss];yy=[float(r['dice']) for r in ss]
     else:ss=[r for r in pi if r['cohort']=='validation' and r['method_id']==m['id'] and r['checkpoint_tag']==cp and r['prediction_kind']=='original_pseudolabel' and r['quality']=='GT_present_foreground_macro' and r['signal']==signal];xx=[r['signal_value'] for r in ss];yy=[r['quality_value'] for r in ss]
     rho=next(r['spearman'] for r in corr if r['method_id']==m['id'] and r['checkpoint_tag']==cp and r['cohort']=='validation' and r['prediction_kind']=='original_pseudolabel' and r['quality']=='GT_present_foreground_macro' and r['signal']==signal and r['level']==level)
     ax.scatter(xx,yy,s=12 if level=='slice' else 25,alpha=.55,color=color,label=f'{cp}: rho={rho:.3f}')
    ax.set(xlim=(.9,1) if row==0 else (0,1),ylim=(0,1),title=m['id'],xlabel='Mean maximum probability' if row==0 else 'Foreground class-balanced margin');ax.legend(fontsize=7)
  axs[0,0].set_ylabel('GT-present macro Dice');axs[1,0].set_ylabel('GT-present macro Dice');fig.tight_layout()
  for ext in ['svg','pdf','png']:fig.savefig(W/f'figures/A2_common_signal_{level}.{ext}',dpi=160)
  plt.close(fig)
  atomic(W/f'figures/A2_common_signal_{level}_caption.md','Validation frozen EMA teachers, eval, best/last separate. 205 GT-present slices or 10 patients with paired valid slices and equal-frame averaging. Original classwise LCC pseudo-label Dice. Uniform signal definitions; Full/baseline foreground margin is an offline auxiliary statistic; R1 is a single-slice control proxy, R2 records the signal but fixes geometry. No historical batch signal is reconstructed. Main max-probability range [0.9,1]; all values verified within range. All fallback/support and 5 no-foreground slice false positives retained in CSV. Correlations describe signal properties, including baseline, and do not estimate training gains.\n')
 js(unitdir('A2')/'complete.json',dict(status='DONE',identity=identity(),source_sha256=sha(O/'results/A2_sample_class_quality.csv'),correlation_rows=len(corr),mapping='NEEDS_MAPPING',GPU_forwards=0))
 print('A2 DONE; Full NEEDS_MAPPING',flush=True)
if __name__=='__main__':main()
