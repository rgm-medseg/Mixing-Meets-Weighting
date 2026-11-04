import shutil,platform,importlib.metadata as meta,subprocess
import h5py,numpy as np,torch
from scipy.ndimage import zoom
from common import *
def main():
 for d in ['configs','manifests','logs','state','results','figures','reports','scripts/frozen/train_ACDC/networks','legacy']:(W/d).mkdir(parents=True,exist_ok=True)
 frozen=['v14_mask_adamix_components.py','networks/unet.py','networks/net_factory.py','dataloaders/dataset.py']
 pairs=[]
 for n in frozen:
  src=R1/'source/train_ACDC'/n;dst=W/'scripts/frozen/train_ACDC'/n;dst.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(src,dst);pairs.append(dict(source=str(src),source_sha256=sha(src),copy=str(dst),copy_sha256=sha(dst)))
 src=R1/'scripts/controller.py';dst=W/'scripts/frozen/controller.py';shutil.copy2(src,dst);pairs.append(dict(source=str(src),source_sha256=sha(src),copy=str(dst),copy_sha256=sha(dst)))
 csvout(W/'manifests/source_copies.csv',pairs)
 definitions=[('baseline','β-FFT baseline','original_baseline',ROOT/'runs/beta_fft_acdc_official_full_seed42_fresh_ckpt',False,'none'),('Full_main','Full (paper main version)','historical_loadable_Full_90.35',ROOT/'runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',True,'old'),('R1_v2_B','R1 (TCSMix v2_B)','R1_v2_B',R1/'runs/R1_full_v2B',True,'v2_B'),('R2_fixed92','R2 (fixed 92×92)','R2_fixed92',R2/'runs/R2_fixed92',True,'fixed')]
 methods=[];manifest=[];protect={p['source']:p['source_sha256'] for p in pairs}
 for mid,name,version,run,trc,ctrl in definitions:
  source=run/('effective_source.py' if (run/'effective_source.py').exists() else 'source_snapshot.py')
  cps=[];cfgpath=W/'configs'/f'{mid}_checkpoint_args.json'
  for tag in ['best','last']:
   cp=run/f'checkpoints/{tag}.ckpt';ck=torch.load(cp,map_location='cpu',weights_only=False);ch=sha(cp);protect[str(cp)]=ch
   if mid=='R1_v2_B' and tag=='best':assert ch=='5df2de9b8df16e4d89df9265f104b4b8c7c321245d9eaa0479971672b4d7a1a9'
   if mid=='R2_fixed92' and tag=='best':assert ch=='b56292a270ccf6679b5597e9d556eda686df91778e496e715174aaa367b710de'
   if tag=='best':js(cfgpath,ck['args'])
   cps.append(dict(tag=tag,path=str(cp),sha256=ch,step=int(ck['global_step']),teacher='ema_model' in ck))
   for branch in ['ema_model']+(['model_1'] if tag=='best' else []):
    manifest.append(dict(method_id=mid,author_display_name=name,version=version,configuration_path=str(cfgpath),configuration_sha256=sha(cfgpath),source_path=str(source),source_sha256=sha(source),adapter_path=str(W/'scripts/adapter.py'),adapter_sha256=sha(W/'scripts/adapter.py'),checkpoint_tag=tag,checkpoint_path=str(cp),checkpoint_sha256=ch,global_step=int(ck['global_step']),checkpoint_selection='existing strict validation student1 best' if tag=='best' else 'existing final training state',actual_load_branch=branch,branch_present=branch in ck,data_list=str(DATA/'val.list')+';'+str(DATA/'train_slices.list'),data_list_sha256=sha(DATA/'val.list')+';'+sha(DATA/'train_slices.list'),precision='FP32',available_analysis='A2 teacher eval' if branch=='ema_model' and branch in ck else 'A5 deployed student' if branch=='model_1' else 'NA missing teacher'))
   del ck
  protect[str(source)]=sha(source);protect[str(cfgpath)]=sha(cfgpath)
  methods.append(dict(id=mid,name=name,version=version,run=str(run),trc=trc,controller=ctrl,source_path=str(source),checkpoints=cps))
 for n in ['confidence_quality.csv','confidence_quality_correlations.csv','diagnostic_samples.csv']:
  src=R1/'analysis'/n;shutil.copy2(src,W/'legacy'/n);protect[str(src)]=sha(src)
 for f in ['analysis/runtime_cost.json','state/stage_history.json','state/status.json','reports/TRAIN_POSTCHECK.json']:
  for rr,mid in [(R1,'R1'),(R2,'R2')]:
   p=rr/f
   if p.exists():shutil.copy2(p,W/'legacy'/f'{mid}_{p.name}');protect[str(p)]=sha(p)
 inputs=[];samples=[];val=(DATA/'val.list').read_text().splitlines();lab=(DATA/'train_slices.list').read_text().splitlines()[:136]
 for cohort,names in [('validation',val),('labeled_train',lab)]:
  for name in names:
   p=DATA/('data' if cohort=='validation' else 'data/slices')/f'{name}.h5'
   with h5py.File(p) as f:image=f['image'][:];gt=f['label'][:]
   assert image.shape==gt.shape and set(np.unique(gt)).issubset({0,1,2,3})
   ch=sha(p);protect[str(p)]=ch;inputs.append(dict(cohort=cohort,id=name,path=str(p),sha256=ch,image_shape=str(image.shape),GT_source='label dataset, real annotation',GT_shape=str(gt.shape)))
   for z in (range(image.shape[0]) if image.ndim==3 else [None]):
    a=image[z] if z is not None else image;g=gt[z] if z is not None else gt;h,w=a.shape
    x=zoom(a,(256/h,256/w),order=0).astype(np.float32);y=zoom(g,(256/h,256/w),order=0).astype(np.uint8)
    zi=int(z) if z is not None else int(name.rsplit('_',1)[1]);frame=name.split('_')[0]+'_'+name.split('_')[1]
    samples.append(dict(sample=f'{name}:z{z}' if z is not None else name,patient=name.split('_')[0],frame=frame,slice_index=zi,cohort=cohort,source=str(p),source_sha256=ch,input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),gt_processed_sha256=hashlib.sha256(y.tobytes()).hexdigest(),preprocessing='original eval nearest order0 to256; no new normalization/augmentation',GT_real=True))
 for p in [DATA/'val.list',DATA/'train_slices.list',ROOT/'revision_runs_20261006/state/queue.json']:
  protect[str(p)]=sha(p);shutil.copy2(p,W/'manifests'/p.name)
 assert len(samples)==346
 csvout(W/'manifests/input_volumes.csv',inputs);csvout(W/'manifests/fixed_samples.csv',samples)
 verification=sorted({r['patient'] for r in samples if r['cohort']=='validation'})[:2]
 cfg=dict(methods=methods,classes={'0':'Background','1':'RV','2':'MYO','3':'LV'},sample_manifest=str(W/'manifests/fixed_samples.csv'),val_cases=val,verification_patients=verification,modes=['eval'],legacy_train_snapshot='R1 only; each split batch restores all frozen buffers and fixed RNG',bins=list(np.linspace(0,1,11)),warmup=50,blocks=5,repeats=100,volume_rounds=3,precision='FP32',input=[1,1,256,256],author_version_note='作者提供：论文主版本TCSMix置信度变化有效改变实际矩形，Full Dice90.35%；本任务按真实状态分别分析，不重新裁定历史版本对应关系。',scope='A2+A5 only; no training/EMA/optimizer updates')
 js(W/'configs/analysis.json',cfg)
 manifest.append(dict(method_id='TRC_architecture_only',author_display_name='TRC architecture only',version='same frozen UNet+TRC as Full_main',source_path=str(W/'scripts/frozen/train_ACDC/v14_mask_adamix_components.py'),source_sha256=sha(W/'scripts/frozen/train_ACDC/v14_mask_adamix_components.py'),adapter_path=str(W/'scripts/adapter.py'),adapter_sha256=sha(W/'scripts/adapter.py'),checkpoint_path='NA: no corresponding trained ACDC TRC-only state',checkpoint_sha256='NA',global_step='NA',actual_load_branch='architecture-only shared instantiated Full structure',precision='FP32',available_analysis='A5 structure count only'))
 csvout(W/'manifests/source_manifest.csv',manifest)
 csvout(W/'manifests/missing_states.csv',[dict(method_id=m['id'],requested_step=step,status='NA',reason='only best and last retained in this run; no complete teacher state at requested step') for m in methods for step in [6000,15000,24000]])
 js(W/'manifests/protected.json',protect)
 env=dict(time=now(),host=platform.node(),python=sys.version,torch=torch.__version__,cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),GPU=gpu(),packages={n:meta.version(n) for n in ['numpy','scipy','h5py','matplotlib','medpy']},count_tool='torch.profiler with_flops + exact module hooks and TRC einsum formulas; no package installation',AGENTS='No AGENTS.md in project or ancestor directories',old_queue=read(ROOT/'revision_runs_20261006/state/queue.json')['status'])
 js(W/'reports/environment.json',env)
 atomic(W/'PRE_ANALYSIS_A2_A5_CN.md',f'''# A2/A5预分析\n\n{now()}。GPU UUID {env['GPU']['uuid']}，当前空闲。旧队列{env['old_queue']}；R1/R2状态DONE、last30000、best29000，已读取真实checkpoint键与步数，已交叉核对已知SHA。\n\n四方法best/last全部含ema_model；A2拟strict加载冻结真实教师，主要覆盖验证10患者20帧210切片，补充7标注训练患者136切片。统一名单在查看新预测前固定；验证患者{verification}固定用于独立复算。旧136条记录实际是34个样本×2检查点×2模式，将保留原CSV及哈希，重放原增强输入并单列旧train_snapshot。仅使用真实GT，测试输入0。\n\nA5部署student1，baseline与三个Full版本分别真实测速；TRC-only未找到对应ACDC单模块状态，仅列architecture-only结构计数，与可加载Full共享结构来源。1×1×256²、FP32；50预热、5×100轮换块、完整验证体推理3轮。原TRC forward含同步CPU诊断统计，照原可执行路径保留并说明成本。现有计数库缺少fvcore/thop，使用torch.profiler加公式，完整非MAC运算FLOPs若不能可靠覆盖则PARTIAL。无新增依赖。\n\n6000/15000/24000教师状态缺失，标NA，不训练补齐。历史预训练及完整训练成本缺边界项标NA。作者提供的90.35%机制版本说明保留，实际加载历史Full与R1/R2分别列身份，不转移分数或机制证据。\n\n预计GPU/CPU分析与核验约10–25分钟，GPU忙则等待；脚本准备额外计时。独立tmux顺序A2→A5→核验→归档，断点以方法/检查点/患者和测速块身份复用。原源码/配置/检查点只读，保护清单执行前后复核。\n''')
 print('Prepared frozen sample manifest and source inventory',len(samples),flush=True)
if __name__=='__main__':main()
