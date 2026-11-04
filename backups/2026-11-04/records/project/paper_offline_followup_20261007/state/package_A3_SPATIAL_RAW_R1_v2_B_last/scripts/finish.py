import shutil
from collections import defaultdict
import numpy as np
from scipy.stats import rankdata
from scipy.ndimage import distance_transform_edt
from common import *
O=ROOT/'offline_A2_A5_20261007'
def table(rr,fields):
 return '|'+ '|'.join(fields)+'|\n|'+'|'.join(['---']*len(fields))+'|\n'+''.join('|'+ '|'.join(f'{r.get(f,"NA"):.6f}' if isinstance(r.get(f),float) else str(r.get(f,'NA')) for f in fields)+'|\n' for r in rr)
def savefig(fig,name,caption):
 import matplotlib.pyplot as plt
 # Matplotlib locators also create tick objects outside the visible limits; explicitly filter them.
 for ax in fig.axes:
  lo,hi=sorted(ax.get_xlim());ax.set_xticks([x for x in ax.get_xticks() if lo-1e-12<=x<=hi+1e-12])
  lo,hi=sorted(ax.get_ylim());ax.set_yticks([y for y in ax.get_yticks() if lo-1e-12<=y<=hi+1e-12])
 fig.tight_layout();fig.canvas.draw();width,height=fig.canvas.get_width_height();bad=[]
 # All text extents must intersect the canvas; clipping assessment saved transparently.
 for text in fig.findobj(__import__('matplotlib').text.Text):
  if text.get_visible() and text.get_text():
   bb=text.get_window_extent();
   if bb.x1<0 or bb.y1<0 or bb.x0>width or bb.y0>height:bad.append(text.get_text())
 assert not bad,('figure text outside canvas',bad)
 for ext in ['svg','pdf','png']:fig.savefig(W/f'figures/{name}.{ext}',dpi=170)
 atomic(W/f'figures/{name}_caption.md',caption+'\n');plt.close(fig)
def figures():
 import matplotlib;matplotlib.use('Agg');import matplotlib.pyplot as plt
 plt.rcParams['svg.fonttype']='none';plt.rcParams['font.size']=9
 summ=rows(W/'results/A5_timing_summary.csv');vol=rows(W/'results/A5_volume_summary.csv');methods=[m['id'] for m in config()['methods']];fig,axs=plt.subplots(1,3,figsize=(12,3.8))
 for k,inputkind in enumerate(config()['inputs']):
  for path,color,offset in [('original_runtime','#147c93',-.17),('deployment_no_diagnostics','#bd673d',.17)]:
   yy=[float(next(r['mean_ms'] for r in summ if r['method_id']==m and r['path']==path and r['input_kind']==inputkind)) for m in methods];axs[k].bar(np.arange(4)+offset,yy,.34,color=color,label=path.replace('_',' '));axs[k].set(xticks=np.arange(4),xticklabels=['baseline','Full','R1','R2'],ylabel='CUDA model call (ms)',title=inputkind.replace('_',' '))
 for path,color,offset in [('original_runtime','#147c93',-.17),('deployment_no_diagnostics','#bd673d',.17)]:axs[2].bar(np.arange(4)+offset,[float(next(r['mean_ms_per_slice'] for r in vol if r['method_id']==m and r['path']==path)) for m in methods],.34,color=color,label=path.replace('_',' '))
 axs[2].set(xticks=np.arange(4),xticklabels=['baseline','Full','R1','R2'],ylabel='Native volume path (ms/slice)',title='Preloaded CPU volume to mask');axs[0].legend(fontsize=6)
 savefig(fig,'A5_deployment_timing','Four real best student1 checkpoints on A100 GPU0, FP32 storage, same backend. Every model reload receives 50 warmups; 5 rounds x100 measured calls per path/input. CUDA events measure resident-input model calls, host boundary excludes monitor setup/exit. Volume 3x20 frames includes in-plane nearest resize, transfers, softmax/argmax and native restoration. Diagnostic suppression retains all TRC prediction operations. All 840 logits and masks and 80 native volume hashes were checked before timing. Means include every valid block; invalids retained separately.')
 ag=rows(W/'results/A3_aggregate_statistics.csv');methods3=methods[1:];regions=['anatomy_clean','anatomy_only','seam_only','anatomy_seam_overlap','remainder','cutout_edge'];fig,axs=plt.subplots(1,2,figsize=(12,4.5))
 for i,m in enumerate(methods3):
  rr=[next(r for r in ag if r['method_id']==m and r['checkpoint_tag']=='best' and r['strategy']=='matched92' and r['width']=='2' and r['region']==reg and r['weighting']=='patient_equal') for reg in regions]
  axs[0].plot(np.arange(len(regions)),[float(r['mean_w']) if not r['mean_w'].startswith('NA') else np.nan for r in rr],marker='o',label=m);axs[1].plot(np.arange(len(regions)),[float(r['delta_mean']) if not r['delta_mean'].startswith('NA') else np.nan for r in rr],marker='o',label=m)
 for ax in axs:ax.set(xticks=np.arange(len(regions)),xticklabels=[s.replace('_','\n') for s in regions]);ax.tick_params(axis='x',labelsize=7);ax.legend(fontsize=7)
 axs[0].set(ylabel='Patient-equal mean CTW weight',ylim=(.25,1.25));axs[1].set(ylabel='Patient-equal mean (w - mean(w)) x e');axs[1].axhline(0,color='gray',lw=.6)
 savefig(fig,'A3_region_and_uniform','Frozen eval best students; common matched 92x92 mask and identical predeclared weak/strong source pairings. Ten source patients have equal weight. Boundary band width2 grid pixels. Anatomy_clean excludes mix/cutout neighborhoods, occlusion and rotation padding; the four mutually exclusive regions use the uncleaned anatomy band. CTW is confidence-transition weighting. Same-mean uniform is a constant per full image with unchanged predictions and errors; the plotted delta is a conditional regional mean, not a retrained ablation score. Full original strategy remains NA pending mapping; all best/last and width1/3 results retained.')
 maps=rows(W/'results/A3_map_manifest.csv')
 # Weight distribution includes the constant-per-image same-mean reference, with source patients equal.
 bins=np.linspace(.25,1.25,51);histrows=[];fig,axs=plt.subplots(1,3,figsize=(12,3.5))
 for ax,method in zip(axs,methods3):
  hist=defaultdict(lambda:np.zeros(50));uniformhist=defaultdict(lambda:np.zeros(50))
  for r in maps:
   if r['method_id']!=method or r['checkpoint_tag']!='best' or r['strategy']!='matched92':continue
   with np.load(r['map_array_path']) as z:
    w=z['w'];pid=z['patient_ids'];mu=float(np.mean(w,dtype=np.float64))
    for p in np.unique(pid):
     mask=pid==p;hist[int(p)]+=np.histogram(w[mask],bins)[0];uniformhist[int(p)]+=np.histogram([mu],bins)[0]*int(mask.sum())
  h=np.mean([a/a.sum() for a in hist.values()],axis=0);u=np.mean([a/a.sum() for a in uniformhist.values()],axis=0)
  ax.step((bins[1:]+bins[:-1])/2,h,where='mid',label='Spatial CTW');ax.step((bins[1:]+bins[:-1])/2,u,where='mid',label='Same-mean uniform',linestyle='--');ax.set(xlim=(.25,1.25),xlabel='Weight',ylabel='Patient-equal pixel fraction',title=method);ax.legend(fontsize=7)
  for i in range(50):histrows.append(dict(method_id=method,checkpoint_tag='best',strategy='matched92',weighting='patient_equal',bin_low=float(bins[i]),bin_high=float(bins[i+1]),spatial_fraction=float(h[i]),same_mean_uniform_fraction=float(u[i]),support_patients=len(hist)))
 csvout(W/'results/A3_weight_distribution.csv',histrows)
 savefig(fig,'A3_weight_distribution','Best eval states under predeclared matched92 masks. Fixed 50 equal-width bins over [0.25,1.25], last bin includes1.25. Patient-equal normalized pixel histograms; per-image uniform uses the same full-image mean w and is assigned to the same source pixels. Histograms include every image pixel, including occlusion/padding, as in the original weight normalization; clean error/anatomy analyses use separate valid masks. This displays spatial and global mean effects independently of training accuracy.')
 sel=[r for r in maps if r['method_id']=='R1_v2_B' and r['checkpoint_tag']=='best' and r['strategy']=='matched92'];rs=rows(W/'results/A3_region_statistics.csv');scores={};lookup=defaultdict(list)
 for s in rs:
  if s['width']=='2' and s['region'] in ['anatomy_clean','remainder']:lookup[(s['map_array_path'],s['region'])].append(s)
 for r in maps:
  clean=lookup[(r['map_array_path'],'anatomy_clean')];rest=lookup[(r['map_array_path'],'remainder')];nc=sum(int(x['pixels']) for x in clean);nr=sum(int(x['pixels']) for x in rest)
  if nc and nr:scores[r['map_array_path']]=sum(float(x['w_sum']) for x in clean)/nc-sum(float(x['w_sum']) for x in rest)/nr
 selected=[]
 for patient in ['patient002','patient019']:
  r=next((r for r in sel if patient in r['outside_sample']),None)
  if r is None:r=next(r for r in sel if patient in r['inside_sample'])
  selected.append((patient+'_preselected',r))
 ordered=sorted(scores,key=lambda k:(scores[k],k));median=float(np.median(list(scores.values())));nearest=min(scores,key=lambda k:(abs(scores[k]-median),k))
 for name,path in [('minimum_enrichment',ordered[0]),('median_enrichment',nearest),('maximum_enrichment',ordered[-1])]:selected.append((name,next(r for r in maps if r['map_array_path']==path)))
 chosen=[]
 for name,r in selected:
  with np.load(r['map_array_path']) as zz:a={k:zz[k] for k in zz.files}
  plan=read(unitdir('INPUTS')/f"batch_{r['batch']}.json")
  outslot=next(s for s in plan['slots'] if s['sample']==r['outside_sample']);inslot=next(s for s in plan['slots'] if s['sample']==r['inside_sample'])
  with np.load(outslot['input_array_path']) as z:outside_source=z['weak'][0]
  with np.load(inslot['input_array_path']) as z:inside_source=z['weak'][0]
  markers=np.zeros((*a['anat'].shape,3));markers[a['anat']]=[1,0,0];markers[a['seam']]=[1,.65,0];markers[a['occlusion']]=[.75,.1,.5]
  fig,axs=plt.subplots(3,4,figsize=(12,9));images=[outside_source,inside_source,a['image1'],a['image2'],a['gt'],a['strong_artificial_gt'],a['pred1'],a['pred2'],a['w'],a['err1'].astype(float),markers,a['q']]
  titles=['Outside source before mix','Inside source before mix','Mixed weak image','Aligned strong image','True anatomy GT','Strong artificial label','Student1 raw prediction','Student2 raw prediction','CTW weight','Student1 GT error','Anatomy / seam / cutout','Student mean max probability']
  for i,(ax,img,title) in enumerate(zip(axs.flat,images,titles)):
   ax.set_title(title);ax.axis('off')
   if i in [4,5,6,7]:im=ax.imshow(img,cmap='tab10',vmin=0,vmax=9,interpolation='nearest')
   elif i==8:im=ax.imshow(img,cmap='viridis',vmin=.25,vmax=1.25);cb=fig.colorbar(im,ax=ax,fraction=.046,pad=.04);cb.set_ticks([.25,.5,.75,1,1.25])
   elif i==10:ax.imshow(img)
   else:ax.imshow(img,cmap='gray' if i<4 else 'Reds' if i==9 else 'viridis',vmin=None if i<4 else 0,vmax=None if i<4 else 1)
   if i in [2,3,8,9]:
    ax.contour(a['anat'],levels=[.5],colors=['#e33232'],linewidths=.45);ax.contour(a['seam'],levels=[.5],colors=['#f1a340'],linewidths=.8)
    if a['occlusion'].any():ax.contour(a['occlusion'],levels=[.5],colors=['#c51b7d'],linewidths=.6)
  fig.suptitle(f"{r['method_id']} {r['checkpoint_tag']} {r['strategy']}: {name}; enrichment={scores.get(r['map_array_path'],np.nan):.5f}",fontsize=10)
  cap='Frozen eval, offline source pair, 256-grid coordinates. Original source images are shown after the shared recorded geometric transform/nearest resize and before mixing, to keep their coordinates aligned with the mixed views. Red=source anatomy boundary; orange=rectangle seam; magenta=strong cutout. GT composition shares source masks across views. Strong artificial labels are separately shown; occlusion is excluded from primary anatomy/error statistics. Student predictions are raw argmax; teacher LCC is only used for controller support. The final probability panel is the students\' mean maximum probability q used by CTW, separate from teacher q_B. Fixed weight color scale0.25–1.25, probability scale0–1. Selection was predeclared: two fixed verification patients (R1-best matched-mask base examples) and minimum/closest-to-median/maximum clean-anatomy enrichment across all1080 diagnostic maps with clean/remainder support. All methods, checkpoints and strategies participate; no-support maps remain in statistics but have NA enrichment. These are structured examples, not random representative samples. '+json.dumps({k:r[k] for k in ['method_id','checkpoint_tag','strategy','batch','pair_index','outside_sample','inside_sample','rectangle_h','rectangle_w','y','x']})
  savefig(fig,'A3_example_'+name,cap);chosen.append(dict(selection=name,**r,clean_anatomy_enrichment=scores.get(r['map_array_path'],'NA')))
 csvout(W/'results/A3_example_selection.csv',chosen)
 # Small non-image caches enable independent regional arithmetic; full arrays stay on server.
 for name,r in selected[:2]:
  with np.load(r['map_array_path']) as z:a={k:z[k] for k in ['w','e','confidence1','confidence2','raw','normalized','anat','seam','cut_boundary','valid','valid1','valid2','err1','err2','occlusion','disagree','patient_ids']}
  dest=W/f'raw_units/review_cache/{name}.npz';dest.parent.mkdir(parents=True,exist_ok=True);np.savez_compressed(dest,**a)
def review():
 before=read(W/'manifests/protected.json');changes=[]
 for p,h in before.items():
  actualsha=sha(p)
  if actualsha!=h:changes.append(dict(path=p,before=h,after=actualsha))
 assert not changes,changes
 js(W/'reports/PROTECTED_AFTER.json',dict(status='PASS',files=len(before),changes=changes,training_updates=0,optimizer_updates=0,EMA_updates=0))
 cfg=config();old=read(ROOT/'revision_runs_20261006/state/queue.json');saved=read(ROOT/'revision_runs_20261006/runs/E1/TRAIN_RECEIPT.json')
 assert old['status']=='INTERRUPTED' and saved['status']=='INTERRUPTED' and saved['global_step']==3976
 assert read(W/'reports/E1_PAUSED_STATE_RECEIPT.json')['saved_checkpoint_step']==3976
 assert all(old['tasks'][s]['latest_step']==0 for s in ['E2','E3','E4','E5'])
 # Independent paired aggregation and rank correlation, without calling A2 implementation.
 source=rows(W/'raw_units/A2_sample_class_quality_source.csv');target=rows(W/'results/A2_common_signal_correlations.csv');checks=[]
 for r in target:
  if r['cohort']!='validation':continue
  ss=[s for s in source if all(s[k]==r[k] for k in ['method_id','checkpoint_tag','cohort','mode','prediction_kind']) and s['class_name']==r['quality'] and s['dice'] not in ['NA','']]
  if r['level']=='slice':xx=[float(s[r['signal']]) for s in ss];yy=[float(s['dice']) for s in ss]
  else:
   frames=defaultdict(list)
   for s in ss:frames[(s['patient'],s['frame'])].append(s)
   patients=defaultdict(list)
   for (pat,frame),f in frames.items():patients[pat].append((sum(float(s[r['signal']]) for s in f)/len(f),sum(float(s['dice']) for s in f)/len(f)))
   xx=[sum(a for a,b in f)/len(f) for _,f in sorted(patients.items())];yy=[sum(b for a,b in f)/len(f) for _,f in sorted(patients.items())]
  rho=float(np.corrcoef(rankdata(xx),rankdata(yy))[0,1]);err=abs(rho-float(r['spearman']));assert err<1e-12 and len(xx)==int(r['n']);checks.append(dict(method_id=r['method_id'],checkpoint_tag=r['checkpoint_tag'],signal=r['signal'],quality=r['quality'],prediction_kind=r['prediction_kind'],level=r['level'],n=len(xx),rho_independent=rho,absolute_difference=err))
 csvout(W/'reports/A2_INDEPENDENT_RECOMPUTATION.csv',checks)
 # Two fixed source patients, all R1-best matched-mask exposures, four-way partition and clean boundary.
 maps=rows(W/'results/A3_map_manifest.csv');rr=rows(W/'results/A3_region_statistics.csv');numeric=[]
 for pat in ['patient002','patient019']:
  pid=cfg['a3_patients'].index(pat);count=defaultdict(int);ws=defaultdict(float);wes=defaultdict(float);delta=defaultdict(float)
  for r in maps:
   if r['method_id']!='R1_v2_B' or r['checkpoint_tag']!='best' or r['strategy']!='matched92' or (pat not in r['outside_sample'] and pat not in r['inside_sample']):continue
   with np.load(r['map_array_path']) as a:
    pm=a['patient_ids']==pid;an=distance_transform_edt(~a['anat'])<=2 if a['anat'].any() else np.zeros(pm.shape,bool);se=distance_transform_edt(~a['seam'])<=2;cu=distance_transform_edt(~a['cut_boundary'])<=2 if a['cut_boundary'].any() else np.zeros(pm.shape,bool)
    regions=dict(all=np.ones(pm.shape,bool),anatomy_only=an&~se,seam_only=se&~an,anatomy_seam_overlap=an&se,remainder=~an&~se,anatomy_clean=an&~se&~cu&a['valid']);assert sum(int((v&pm).sum()) for k,v in regions.items() if k in ['anatomy_only','seam_only','anatomy_seam_overlap','remainder'])==int(pm.sum())
    mean=float(np.mean(a['w'],dtype=np.float64))
    for reg,mask in regions.items():
     mask=mask&pm;count[reg]+=int(mask.sum());ws[reg]+=float(np.sum(a['w'][mask],dtype=np.float64));wes[reg]+=float(np.sum(a['w'][mask]*a['e'][mask],dtype=np.float64));delta[reg]+=float(np.sum((a['w'][mask]-mean)*a['e'][mask],dtype=np.float64))
  for reg,n in count.items():
   ss=[r for r in rr if r['method_id']=='R1_v2_B' and r['checkpoint_tag']=='best' and r['strategy']=='matched92' and r['width']=='2' and r['patient']==pat and r['region']==reg];n0=sum(int(r['pixels']) for r in ss);w0=sum(float(r['w_sum']) for r in ss);we0=sum(float(r['we_sum']) for r in ss);de0=sum(float(r['delta_we_sum']) for r in ss)
   assert n==n0 and abs(ws[reg]-w0)<1e-7 and abs(wes[reg]-we0)<1e-7 and abs(delta[reg]-de0)<1e-7
   numeric.append(dict(patient=pat,region=reg,pixels=n,w_sum=ws[reg],we_sum=wes[reg],delta_sum=delta[reg],count_difference=n-n0,loss_sum_difference=wes[reg]-we0))
 csvout(W/'reports/A3_FIXED_PATIENT_NUMERIC_REVIEW.csv',numeric)
 timing=[read(p) for p in unitdir('TIMING').glob('timing_*.json')];volume=[read(p) for p in unitdir('TIMING').glob('volume_*.json')];assert len(timing)==80 and len(volume)==480
 assert all(u['warmup']==50 and len(u['cuda_event_ms'])==100 and u['valid'] for u in timing)
 assert all(u['warmup_on_load']==50 and u['valid'] for u in volume)
 uniform=rows(W/'results/A3_uniform_mean_matched.csv');assert all(float(r['mean_difference'])==0 for r in uniform)
 uniform_checks=[]
 for r in maps:
  with np.load(r['map_array_path']) as a:
   weights=a['w'];mu=float(weights.mean(dtype=np.float64));constant=np.full(weights.shape,mu,dtype=np.float64);difference=float(constant.mean(dtype=np.float64)-weights.mean(dtype=np.float64))
   assert difference==0.
   recorded=[q for q in uniform if q['map_array_path']==r['map_array_path']]
   assert recorded and all(float(q['image_original_mean_w'])==mu and float(q['image_uniform_mean_w'])==mu for q in recorded)
   uniform_checks.append(dict(map_array_path=r['map_array_path'],spatial_mean_float64=mu,constructed_constant_mean=float(constant.mean(dtype=np.float64)),difference=difference))
 csvout(W/'reports/A3_UNIFORM_INDEPENDENT_MEAN_CHECK.csv',uniform_checks)
 r2original={(r['checkpoint_tag'],r['batch'],r['pair_index']):r for r in maps if r['method_id']=='R2_fixed92' and r['strategy']=='original_strategy'};same=[]
 for r in maps:
  if r['method_id']!='R2_fixed92' or r['strategy']!='matched92':continue
  other=r2original[(r['checkpoint_tag'],r['batch'],r['pair_index'])]
  assert r['weak_input_sha256']==other['weak_input_sha256'] and r['strong_input_sha256']==other['strong_input_sha256']
  with np.load(r['map_array_path']) as a,np.load(other['map_array_path']) as b:
   assert all(np.array_equal(a[k],b[k]) for k in ['w','e','pred1','pred2','gt','patient_ids'])
  same.append(dict(checkpoint_tag=r['checkpoint_tag'],batch=r['batch'],pair_index=r['pair_index'],input_and_CTW_maps_exact=True,independence='same source exposure; strategy labels are not new independent observations'))
 csvout(W/'reports/A3_R2_MATCHED_ORIGINAL_EQUIVALENCE.csv',same)
 eq=read(W/'results/DEPLOYMENT_EQUIVALENCE_RECEIPT.json');align=read(W/'results/A3_alignment_receipt.json');assert eq['status']==align['status']=='PASS'
 # Unit keys guarantee no duplicate rows; numerical and plotted summaries regenerated from raw units.
 assert len({(u['method_id'],u['path'],u['input_kind'],u['round']) for u in timing})==80
 assert len({(u['method_id'],u['path'],u['round'],u['frame']) for u in volume})==480
 review=dict(source='PARTIAL: immutable files PASS; Full author/code/trace unique mapping NEEDS_MAPPING',prediction_alignment='PASS',statistics_timing='PASS',figure_text='PASS: rendered bounds/CSV identities checked; human visual receipt separately if present',reproduction_archive='PENDING actual extraction; recovery receipt checked at delivery',A2_independent_checks=len(checks),A3_two_patient_checks=len(numeric),timing_blocks=80,volume_units=480,uniform_rows=len(uniform),training_updates=0)
 js(W/'reports/FIVE_PASS_REVIEW.json',review)
def reports():
 maps=rows(W/'results/A3_map_manifest.csv');roles=[]
 for r in maps:
  role='offline auxiliary margin; Full original controller mapping NA' if r['method_id']=='Full_main' else 'offline batch control used for diagnostic rectangle; not historical trajectory' if r['method_id']=='R1_v2_B' and r['strategy']=='original_strategy' else 'offline recorded batch margin; fixed/matched92 geometry independent of q_B'
  roles.append({**{k:r[k] for k in ['method_id','checkpoint_tag','strategy','batch','pair_index','q_B','rectangle_h','rectangle_w']},'q_B_role':role,'historical_batch_control':'NA: frozen validation diagnostic'})
 csvout(W/'results/A3_controller_signal_roles.csv',roles)
 corr=rows(W/'results/A2_common_signal_correlations.csv');target=[dict(r,spearman=float(r['spearman'])) for r in corr if r['cohort']=='validation' and r['quality']=='GT_present_foreground_macro' and r['prediction_kind']=='original_pseudolabel' and r['signal'] in ['mean_max_probability','q_B_auxiliary']]
 summ=[{**r,**{k:float(r[k]) for k in ['mean_ms','median_ms','p95_ms','throughput_slices_per_second','relative_to_baseline']}} for r in rows(W/'results/A5_timing_summary.csv')];vol=[{**r,**{k:float(r[k]) for k in ['mean_ms_per_slice','throughput_slices_per_second','relative_to_baseline']}} for r in rows(W/'results/A5_volume_summary.csv')]
 for p in ['A5_parameters_compute.csv','A5_training_cost.csv','A5_inference_summary.csv','A5_volume_summary.csv']:
  dst=W/'results/historical'/p;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(O/'results'/p,dst)
 atomic(W/'reports/A2_COMMON_SIGNAL_REPORT_CN.md','''# 统一A2统计\n\nDONE。直接复用已验证原概率/质量CSV，没有额外GPU前向。验证210张切片，GT-present有效205；全部210固定三前景macro和5张无前景切片的逐类FP同时保留。两种预测处理、训练标注子集分别保存，best/last不合并。患者同一有效切片配对，先切片等权再帧等权，共10患者。\n\n'''+table(target,['method_id','checkpoint_tag','signal','level','n','spearman'])+'\n统一前景等权间隔在baseline也呈强正相关，描述信号与冻结教师质量的条件关系。Full/baseline该量为离线辅助；R1是单切片控制代理，R2记录该信号但固定几何。Full真实论文版本映射NEEDS_MAPPING，不能把q_B相关结果改写成Full历史使用该控制量。无显著性检验、无因果训练收益结论。\n')
 eq=read(W/'results/DEPLOYMENT_EQUIVALENCE_RECEIPT.json');mem=rows(W/'results/A5_memory.csv')
 atomic(W/'reports/DEPLOYMENT_TIMING_REPORT_CN.md',f'''# 部署等价与成本\n\nDONE：{eq['method_slice_checks']}个方法×切片核验、{eq['method_volumes']}个方法×帧体哈希，logits exact={eq['all_logits_exact']}，max_abs={eq['max_abs']}；mask、native-volume、逐类Dice/IoU/HD95/ASD一致。模型参数和BN状态前后哈希不变。指标距离单位原网格像素，双空全0，单空对角距离；不缩放病例轴，不做LCC。\n\nGPU0 A100 FP32存储dtype，matmulTF32=False，cudnnTF32=True，deterministic=True，benchmark=False与旧环境一致。原路径全部诊断保留；部署仅独立bound adapter删除router entropy/utilization/norm/finite CPU同步和decoder类别占比argmax，不改任何预测运算。baseline两路径算子相同。adapter_diff.patch记录完整原forward到部署forward差异。\n\n每次加载均预热50，再计时100；5轮方法/路径轮换，零输入与val.list首帧z0分开。资源线程创建/结束、块前后nvidia-smi不在host计时边界，计时期间并行监测线程可能产生轻微CPU调度扰动，所有方法同口径；外来GPU PID块保留invalid后等待重测。每次单模型驻GPU。CUDA-event吞吐=500/对应event总秒数；host墙钟另列。峰值allocated/reserved在加载+预热后reset，包括模型和输入；整卡占用独立列。\n\n'''+table(summ,['method_id','path','input_kind','mean_ms','median_ms','p95_ms','throughput_slices_per_second','relative_to_baseline'])+'\n体路径3轮×20帧，预加载CPU图像到native mask，包含平面resize/H2D/forward/softmax argmax/D2H/restore，排除读写盘与GT评价，保持z和batch1：\n\n'+table(vol,['method_id','path','mean_ms_per_slice','throughput_slices_per_second','relative_to_baseline'])+'\n旧A5结果原样另存historical并保持旧包只读；旧口径含诊断且仅首次块预热，不能直接与新版时间解释训练创新。无诊断部署节省输出无关统计，TRC仍完整保留。单学生参数baseline1,826,828，其余1,827,469，增641；已覆盖MAC2.949644288/2.991587328 GMAC，完整非MAC FLOPs及FFT/反向训练FLOPs NA，历史Full/baseline完整同口径训练时间NA。\n')
 ag=rows(W/'results/A3_aggregate_statistics.csv');selected=[{**r,**{k:float(r[k]) if not r[k].startswith('NA') else 'NA' for k in ['mean_w','mean_e','mean_we','delta_mean']}} for r in ag if r['checkpoint_tag']=='best' and r['strategy']=='matched92' and r['width']=='2' and r['weighting']=='patient_equal' and r['region'] in ['anatomy_clean','seam_only','anatomy_seam_overlap','remainder','cutout_edge']]
 atomic(W/'reports/A3_CTW_SPATIAL_REPORT_CN.md','''# CTW空间分析\n\nDONE，限定冻结eval验证诊断。Full/R1/R2 best/last六个完整checkpoint，两学生和EMA严格加载。210不同验证切片/20帧/10患者，9批24模拟角色，末批6重复暴露，按来源患者分配拼接像素并患者等权，不把重复作为新增患者。108张混合输出/每策略状态，共1080张空间图；Full原策略NA，其他4状态原策略及6状态matched92均完成。没有历史训练重放、GT控制或模型更新。\n\n实际调用同一混合内容weak student1与strong student2；unl组合uimg_a外/img_a内，l组合img_b外/uimg_b内，两个组合不相互比较。q=.5(max p1+max p2)，raw=两轴相邻绝对差之和，每图minmax/clamp1e-6，w=.25+normalized∈[.25,1.25]，e=mean_class|p1-stopgrad(p2)|，L=mean(w e)，λ=.05。原X03与R1/R2 ctwmaps在本输入上损失数值核验，合成logit梯度仅student1有梯度，权重detach。\n\n原WeakStrongAugment的几何、nearest resize、cutout、ColorJitter全部保留，216次与literal增强输出和RNG后态逐元素一致。cutout人工标签0单独保存；旋转填充与cutout视为无效解剖/错误统计支持。source-GT四邻域类变化两侧边界经同transform，再裁取/paste；mixedGT边界含人工接缝另列。带宽2为主，1/3敏感性；四互斥区域用原B_anat，clean另排mix/cut邻近带、遮挡及旋转无效像素，w整图先算后筛，未区域重归一化。无边界distanceNA，其错误保留。\n\n'''+table(selected,['method_id','region','pixels','support_patients','mean_w','mean_e','mean_we','delta_mean'])+'\n区域CSV给每图/患者支持、mean/median w、e、w e、两个学生错误和分支不一致及完整图分母贡献；患者CSV median为精确支持像素中位数，另有image-region-median中位数以明确聚合。pooled及患者等权均保留，含前景/背景/RV/MYO/LV及无支持NA。像素loss贡献分母为完整65536，条件valid均值单列。uniform每图固定原w均值，预测/误差相同；delta=mean((w-mean(w))*e)，不是训练消融成绩。分位切点重复合并、不分割同权重，support fraction为实际值，不能称精确top20%。\n\n需要判断clean anatomy、seam、cutout和错误响应的相对强度，应引用各区域完整表及width1/3；若seam或cutout高响应，同样保留。该定义支持“confidence-transition weighting对不同空间区域重新分配一致性损失”，不能将w称校准的边界正确概率或错误纠正率。正文不把CTW公式里的student confidence替换成TCSMix的teacher q_B。\n\nEnglish draft:\nWe analyzed CTW in frozen evaluation mode on validation images. The two students received spatially aligned weak and strong mixed views. CTW used the spatial variation in their mean maximum probability to assign weights between 0.25 and 1.25. We measured the responses at source anatomy boundaries, mixing seams, and cutout edges separately. The clean anatomy analysis excluded artificial-boundary neighborhoods and occluded pixels. A constant weight with the same image mean provided an offline comparison of spatial allocation. The numerical region results below describe where the frozen loss places emphasis.\n\n完整input/map数组保存在state/input_arrays及state/map_arrays，仅服务器保留；源路径/哈希和重建脚本入包；2张小型w/e/置信度/错误/区域缓存不包含原医学图像/GT，用于算术复核。匿名示例包含固定2患者及clean富集min/nearestmedian/max，包含不利样本，非随机代表抽样。\n')
 # Add concise evidence-based numeric English observations, generated from the table.
 p=W/'reports/A3_CTW_SPATIAL_REPORT_CN.md';atomic(p,p.read_text()+'\n精度口径：前向与空间图FP32；统计求和/均值使用float64，离线uniform标量为整图float64均值并独立构造常量数组核验均值完全相同，不当作E1训练loss的bitwise重放。R2 original与matched92在本配对中几何和输入相同，216对空间图逐元素相同，只作一致性记录，不新增独立观察。q_B列均为冻结离线批次信号；Full为辅助量，matched92几何不由该量控制；只有R1原策略用于该诊断矩形，不冒充历史batch轨迹，角色另表A3_controller_signal_roles.csv。\n')
 for m in ['Full_main','R1_v2_B','R2_fixed92']:
  rr={r['region']:r for r in selected if r['method_id']==m};sentence=f"\nFor {m}, the patient-equal mean weight was {rr['anatomy_clean']['mean_w']:.4f} in clean anatomy boundary bands, {rr['seam_only']['mean_w']:.4f} at seam-only bands, and {rr['remainder']['mean_w']:.4f} in the remainder. The same-mean spatial comparison changed the clean-band conditional consistency term by {rr['anatomy_clean']['delta_mean']:.6f}. These frozen-model observations describe spatial emphasis.\n";p=W/'reports/A3_CTW_SPATIAL_REPORT_CN.md';atomic(p,p.read_text()+sentence)
 atomic(W/'WRITING_HANDOFF_CN.md','''# 写作交接\n\nFull映射PARTIAL/NEEDS_MAPPING：保留作者说明及90.35%归属原运行，待另一个可定位入口/检查点关联才能将“与R1一致且实际矩形变化”作为该成绩的已验证历史机制；不能转移分数。ACDC测试集未用于原候选选择，按作者更正。\n\nA2统一信号DONE：推荐引用A2_COMMON_SIGNAL_REPORT_CN.md统一205 GT-present/10患者表，baseline也有间隔正相关，因此描述信号性质与当前冻结质量，不归因TCSMix训练改善。Full/baseline q_B明确辅助，R1单切片代理、R2记录信号固定几何。210固定三前景宏值及5无前景FP保留。\n\n部署等价/测速DONE：先引用全210×4 logits/mask和80体hash核验；主表采用本轮每次重新加载预热50的original/deployment分列，零/实输入与model/volume口径分列。节省诊断统计开销，TRC预测运算完整保留。641新增参数和已覆盖MAC口径可使用；完整FLOPs及历史Full/baseline同口径训练时间NA。旧测速仍为历史证据。\n\nA3 DONE：引用confidence-transition定义、弱/强同内容配对、clean anatomical band与人工seam/cutout拆分；各方法best/last、2主带宽及1/3敏感性、pooled与patient equal都保留。uniform是同均值离线空间权重对照。权重不是校准的边界正确概率，离线区域贡献不是训练消融分数，无法证明纠错因果。\n\n确需训练才能回答的问题仅列建议：CTW空间分配相对同均值uniform的最终分割收益；动态尺度相对固定面积收益的可重复性；去除或改进人工边界响应的训练效果。当前没有这些训练授权，未执行。\n\n正文模块名TRC — Token-Guided Residual Conditioning；TCSMix — Teacher-Confidence-Scaled Mixing；CTW — Confidence-Transition Weighting。原MaskToken/AdaMix/CRW仅源码检索名。精确数值参见三个报告及CSV，图注同口径。\n''')
 atomic(W/'FINAL_REPORT_CN.md','''# 本轮离线补充交付\n\nFull来源映射PARTIAL（NEEDS_MAPPING）；A2统一汇总DONE；部署等价DONE、规范测速DONE；A3空间诊断DONE。训练/优化器/EMA更新均0，旧E1第3976步暂停，E2–E5未启动，R1/R2不重跑。\n\n三项报告及数据：reports/FULL_CONTROL_MAPPING_CN.md、A2_COMMON_SIGNAL_REPORT_CN.md、DEPLOYMENT_TIMING_REPORT_CN.md、A3_CTW_SPATIAL_REPORT_CN.md；CSV均在results，PDF/SVG/PNG与图注在figures，来源/代码/配置/环境与原始单元实际入包。Full mapping缺口准确列出，不影响其student1部署检查和固定92×92 CTW诊断。\n\n部署840个切片检查、80个原尺寸体结果及全部指标一致；每路径/输入/方法500次有效forward，3×20帧体路径。A3六个状态10策略共1080幅混合图，按10来源患者统计，完整arrays留服务器。旧成绩和旧包只读。主要数值、支持与不支持结论参见WRITING_HANDOFF_CN.md及报告表，保留弱/负相关、不利例子和所有计时异常。\n\n普通tmux队列支持start/status/stop/resume，安全停止在完成单元边界，重启需resume。完整快照、按需每包<30MiB拆包、内部SHA256及外部SHA256由delivery阶段生成并实际解包核验。最终archive receipt保存在包外避免自引用。\n''')
def main():
 from measurement import summarize as timing_summary
 from a3 import summarize as spatial_summary
 timing_summary();spatial_summary();figures();review();reports()
 from presentation_addendum import main as addendum
 addendum()
 report_receipt()
 print('REPORTS AND FIVE-ROUND CORE CHECKS COMPLETE',flush=True)
def report_receipt():
 js(W/'results/REPORTS_RECEIPT.json',dict(status='DONE',identity=identity(),producer_sha256=sha(__file__),presentation_sha256=sha(W/'scripts/presentation_addendum.py'),numerical_review=read(W/'reports/FIVE_PASS_REVIEW.json'),scope='completed raw regeneration/independent numerical reviews; presentation-only additions require no GPU/statistics rerun'))
if __name__=='__main__':main()
