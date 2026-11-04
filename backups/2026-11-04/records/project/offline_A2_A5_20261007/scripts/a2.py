import gc,random,ast
from collections import defaultdict
import h5py,numpy as np,torch
from scipy.ndimage import zoom
from scipy.stats import spearmanr
from common import *
from adapter import model,pseudo,control,digest
GROUPS=[(axis,c) for axis in ['predicted','GT'] for c in [0,1,2,3,'foreground','all']]
def load_sample(s):
 with h5py.File(s['source']) as f:
  a=f['image'][:];g=f['label'][:]
 if a.ndim==3:a=a[int(s['slice_index'])];g=g[int(s['slice_index'])]
 h,w=a.shape;x=zoom(a,(256/h,256/w),order=0).astype(np.float32);y=zoom(g,(256/h,256/w),order=0).astype(np.uint8)
 assert hashlib.sha256(x.tobytes()).hexdigest()==s['input_sha256']
 assert hashlib.sha256(y.tobytes()).hexdigest()==s['gt_processed_sha256']
 return x,y
def pixel_stats(p,g,conf,margin):
 p=p.ravel();g=g.ravel();conf=conf.ravel().astype(np.float64);margin=margin.ravel().astype(np.float64);ok=p==g
 bi=np.minimum((conf*10).astype(int),9);hi=np.minimum((conf*50).astype(int),49);mi=np.minimum((margin*50).astype(int),49)
 rr=[]
 for axis,c in GROUPS:
  base=p if axis=='predicted' else g;mask=base==c if isinstance(c,int) else base>0 if c=='foreground' else np.ones(p.shape,bool)
  co=mask&ok;er=mask&~ok;n=int(mask.sum())
  counts=np.bincount(bi[mask],minlength=10);sums=np.bincount(bi[mask],weights=conf[mask],minlength=10);correct=np.bincount(bi[co],minlength=10)
  rr.append(dict(axis=axis,group=str(c),n=n,correct=int(co.sum()),conf_sum=float(conf[mask].sum()),margin_sum=float(margin[mask].sum()),correct_conf_sum=float(conf[co].sum()),error_conf_sum=float(conf[er].sum()),bins_n=counts.tolist(),bins_conf_sum=sums.tolist(),bins_correct=correct.tolist(),confidence_hist_correct=np.bincount(hi[co],minlength=50).tolist(),confidence_hist_error=np.bincount(hi[er],minlength=50).tolist(),margin_hist_correct=np.bincount(mi[co],minlength=50).tolist(),margin_hist_error=np.bincount(mi[er],minlength=50).tolist()))
 return rr
def summarize_slice(method,cp,mode,s,logits,gt):
 with torch.inference_mode():
  probs=logits.softmax(1);top=probs.topk(2,dim=1).values
  conf=top[0,0].cpu().numpy();margin=(top[0,0]-top[0,1]).cpu().numpy();raw=probs.argmax(1)[0].cpu().numpy().astype(np.uint8)
  yp=pseudo(logits,method);post=yp[0].cpu().numpy().astype(np.uint8);ctl=control(logits,yp,method)
 base=dict(method_id=method['id'],checkpoint_tag=cp['tag'],checkpoint_step=cp['step'],checkpoint_sha256=cp['sha256'],branch='ema_model',mode=mode,cohort=s['cohort'],sample=s['sample'],patient=s['patient'],frame=s['frame'],slice_index=s['slice_index'],input_sha256=s['input_sha256'],mean_max_probability=float(np.mean(conf,dtype=np.float64)),mean_top1_top2_margin=float(np.mean(margin,dtype=np.float64)),GT_foreground_fraction=float((gt>0).mean()),raw_predicted_foreground_fraction=float((raw>0).mean()),pseudo_predicted_foreground_fraction=float((post>0).mean()),GT_present_classes=';'.join(map(str,np.unique(gt))),raw_predicted_classes=';'.join(map(str,np.unique(raw))),pseudo_predicted_classes=';'.join(map(str,np.unique(post))),pseudo_changed_pixels=int((raw!=post).sum()),**{k:v for k,v in ctl.items() if k!='support_counts'})
 qualities=[]
 for kind,pred in [('raw_argmax',raw),('original_pseudolabel',post)]:
  cc=np.bincount((gt.astype(int)*4+pred).ravel(),minlength=16).reshape(4,4)
  ds=[];ious=[];presentd=[];presenti=[]
  for c in [1,2,3]:
   tp=int(cc[c,c]);gn=int(cc[c].sum());pn=int(cc[:,c].sum());fp=pn-tp;fn=gn-tp
   d=2*tp/(gn+pn) if gn+pn else 0.;iou=tp/(gn+pn-tp) if gn+pn-tp else 0.;ds.append(d);ious.append(iou)
   if gn:presentd.append(d);presenti.append(iou)
   pm=raw==c;gm=gt==c
   qualities.append(dict(**base,prediction_kind=kind,class_id=c,class_name={1:'RV',2:'MYO',3:'LV'}[c],dice=d,iou=iou,precision=tp/pn if gn and pn else 'NA',recall=tp/gn if gn else 'NA',TP=tp,FP=fp,FN=fn,GT_pixels=gn,predicted_pixels=pn,predicted_class_mean_confidence=float(conf[pm].mean(dtype=np.float64)) if pm.any() else 'NA',GT_class_mean_confidence=float(conf[gm].mean(dtype=np.float64)) if gm.any() else 'NA',support_pixels=ctl['support_counts'][c],aggregation='class slice grid; both empty Dice/IoU0'))
  for label,d,io in [('fixed_three_foreground_macro',float(np.mean(ds)),float(np.mean(ious))),('GT_present_foreground_macro',float(np.mean(presentd)) if presentd else 'NA',float(np.mean(presenti)) if presenti else 'NA')]:
   qualities.append(dict(**base,prediction_kind=kind,class_id=label,class_name=label,dice=d,iou=io,precision='NA',recall='NA',TP='NA',FP='NA',FN='NA',GT_pixels=int((gt>0).sum()),predicted_pixels=int((pred>0).sum()),predicted_class_mean_confidence='NA',GT_class_mean_confidence='NA',support_pixels=sum(ctl['support_counts'].values()),aggregation='equal class mean; all GT-absent gives NA only for GT-present macro'))
 bgfp=int(((gt==0)&(raw>0)).sum());fgfn=int(((gt>0)&(raw==0)).sum())
 counters=dict(background_to_foreground=bgfp,background_to_foreground_denominator=int((gt==0).sum()),foreground_to_background=fgfn,foreground_to_background_denominator=int((gt>0).sum()),pixels=int(gt.size))
 arrays=dict(prediction=raw,pseudolabel=post,GT=gt,confidence=conf,margin=margin)
 return dict(quality=qualities,pixel_stats=pixel_stats(raw,gt,conf,margin),counters=counters),arrays
def combine_pixel(stats):
 out={}
 for r in stats:
  key=(r['axis'],r['group'])
  if key not in out:out[key]={k:(np.zeros(len(v),dtype=float) if isinstance(v,list) else 0.) for k,v in r.items() if k not in ['axis','group']}
  for k,v in r.items():
   if k not in ['axis','group']:out[key][k]+=np.asarray(v) if isinstance(v,list) else v
 return out
def meanfinite(v):
 a=[float(x) for x in v if x!='NA' and x is not None and np.isfinite(float(x))]
 return float(np.mean(a)) if a else 'NA'
def ece(r):
 n=np.asarray(r['bins_n']);sel=n>0
 return float(np.sum(np.abs(np.asarray(r['bins_correct'])[sel]-np.asarray(r['bins_conf_sum'])[sel]))/r['n']) if r['n'] else 'NA'
def aggregate():
 allunits=[read(p) for p in sorted(unitdir('A2').glob('*.json'))];quality=[r for u in allunits for d in u['slices'] for r in d['quality']]
 csvout(W/'results/A2_sample_class_quality.csv',quality)
 patients=[];corr=[];bins=[];correctness=[];hist=[]
 grouped=defaultdict(list)
 for r in quality:grouped[(r['method_id'],r['checkpoint_tag'],r['checkpoint_step'],r['mode'],r['cohort'],r['prediction_kind'],str(r['class_id']),r['patient'])].append(r)
 for key,rr in grouped.items():
  if key[6]=='GT_present_foreground_macro':rr=[r for r in rr if r['dice']!='NA']
  if not rr:continue
  frames=defaultdict(list)
  for r in rr:frames[r['frame']].append(r)
  numeric=['dice','iou','mean_max_probability','mean_top1_top2_margin','sample_control_proxy','GT_foreground_fraction','raw_predicted_foreground_fraction','pseudo_predicted_foreground_fraction']
  d=dict(zip(['method_id','checkpoint_tag','checkpoint_step','mode','cohort','prediction_kind','class_id','patient'],key));d.update(n_slices=len(rr),n_frames=len(frames),aggregation='equal slices within frame then equal frames within patient, same rule for confidence and quality')
  for n in numeric:d[n]=meanfinite([meanfinite([r[n] for r in f]) for f in frames.values()])
  patients.append(d)
 csvout(W/'results/A2_patient_summary.csv',patients)
 paired_patient_inputs=[]
 for level,rs in [('slice',quality),('patient',patients)]:
  gg=defaultdict(list)
  for r in rs:gg[(r['method_id'],r['checkpoint_tag'],r['checkpoint_step'],r['mode'],r['cohort'],r['prediction_kind'],str(r['class_id']))].append(r)
  for key,rr in gg.items():
   for x in ['mean_max_probability','mean_top1_top2_margin','sample_control_proxy']+(['predicted_class_mean_confidence','GT_class_mean_confidence'] if level=='slice' else []):
    for y in ['dice','iou']:
     valid=[r for r in rr if r[x]!='NA' and r[y]!='NA'];a=[float(r[x]) for r in valid];b=[float(r[y]) for r in valid]
     if level=='patient':
      pairgroups=defaultdict(list)
      for s in quality:
       kk=(s['method_id'],s['checkpoint_tag'],s['checkpoint_step'],s['mode'],s['cohort'],s['prediction_kind'],str(s['class_id']))
       if kk==key and s[x]!='NA' and s[y]!='NA':pairgroups[s['patient']].append(s)
      valid=[];a=[];b=[]
      for pat,qq in pairgroups.items():
       ff=defaultdict(list)
       for s in qq:ff[s['frame']].append(s)
       xv=float(np.mean([np.mean([float(s[x]) for s in f]) for f in ff.values()]));yv=float(np.mean([np.mean([float(s[y]) for s in f]) for f in ff.values()]))
       a.append(xv);b.append(yv);valid.append(dict(patient=pat,n_frames=len(ff)))
       paired_patient_inputs.append(dict(zip(['method_id','checkpoint_tag','checkpoint_step','mode','cohort','prediction_kind','class_id'],key),patient=pat,signal=x,quality=y,confidence=xv,quality_value=yv,valid_slices=len(qq),valid_frames=len(ff),weighting='same valid slices for x/y; equal slices within each frame then equal frames'))
     reason='constant input or fewer than3 valid observations' if len(valid)<3 or len(set(a))<2 or len(set(b))<2 else ''
     rho='NA' if reason else float(spearmanr(a,b).statistic)
     corr.append(dict(zip(['method_id','checkpoint_tag','checkpoint_step','mode','cohort','prediction_kind','class_id'],key),level=level,signal=x,quality=y,spearman=rho,valid_n=len(valid),patients=len({r['patient'] for r in valid}),frames=len({r['frame'] for r in valid}) if level=='slice' else sum(int(r['n_frames']) for r in valid),checkpoint_count=1,NA_reason=reason,no_significance_test=True))
 csvout(W/'results/A2_correlations.csv',corr)
 csvout(W/'results/A2_patient_correlation_inputs.csv',paired_patient_inputs)
 gu=defaultdict(list)
 for u in allunits:gu[(u['method_id'],u['checkpoint_tag'],u['checkpoint_step'],u['mode'],u['cohort'])].append(u)
 for key,uu in gu.items():
  per=[]
  for u in uu:
   dd=combine_pixel([s for d in u['slices'] for s in d['pixel_stats']]);per.append((u['patient'],dd))
  pooled=combine_pixel([dict(axis=a,group=g,**{k:v.tolist() if isinstance(v,np.ndarray) else v for k,v in r.items()}) for _,dd in per for (a,g),r in dd.items()])
  base=dict(zip(['method_id','checkpoint_tag','checkpoint_step','mode','cohort'],key))
  for (axis,g),p in pooled.items():
   pp=[d[(axis,g)] for _,d in per];supported=[r for r in pp if r['n']]
   for weighting in ['pooled_pixel','patient_equal']:
    for bi in range(10):
     active=[r for r in pp if r['bins_n'][bi]>0];n=int(p['bins_n'][bi]);cn=int(p['bins_correct'][bi])
     c=float(p['bins_conf_sum'][bi]/n) if n else 'NA';acc=cn/n if n else 'NA'
     if weighting=='patient_equal':c=meanfinite([r['bins_conf_sum'][bi]/r['bins_n'][bi] for r in active]);acc=meanfinite([r['bins_correct'][bi]/r['bins_n'][bi] for r in active])
     bins.append(dict(**base,axis=axis,group=g,weighting=weighting,bin_index=bi,bin_left=bi/10,bin_right=(bi+1)/10,last_bin_closed=bi==9,pixel_count=n,correct_pixels=cn,support_patients=len(active),mean_confidence=c,accuracy=acc,ECE=ece(p) if weighting=='pooled_pixel' else meanfinite([ece(r) for r in supported]),ECE_scope='conditional group, max probability, sum bin n/N abs(acc-conf); patient_equal=mean of patient ECE'))
    n=int(p['n']);ok=int(p['correct']);err=n-ok
    cnf=p['conf_sum']/n if n else 'NA';mar=p['margin_sum']/n if n else 'NA';acc=ok/n if n else 'NA';cc=p['correct_conf_sum']/ok if ok else 'NA';ec=p['error_conf_sum']/err if err else 'NA'
    if weighting=='patient_equal':
     cnf=meanfinite([r['conf_sum']/r['n'] for r in supported]);mar=meanfinite([r['margin_sum']/r['n'] for r in supported]);acc=meanfinite([r['correct']/r['n'] for r in supported]);cc=meanfinite([r['correct_conf_sum']/r['correct'] for r in pp if r['correct']]);ec=meanfinite([r['error_conf_sum']/(r['n']-r['correct']) for r in pp if r['n']>r['correct']])
    correctness.append(dict(**base,axis=axis,group=g,weighting=weighting,pixel_count=n,correct_pixels=ok,error_pixels=err,support_patients=len(supported),mean_confidence=cnf,mean_margin=mar,accuracy=acc,correct_mean_confidence=cc,error_mean_confidence=ec,ECE=ece(p) if weighting=='pooled_pixel' else meanfinite([ece(r) for r in supported]),denominator='all pixels in specified group; class accuracy is multiclass correctness'))
   for value in ['confidence','margin']:
    for outcome in ['correct','error']:
     hk=f'{value}_hist_{outcome}';total=sum(p[hk])
     for bi,n in enumerate(p[hk]):hist.append(dict(**base,axis=axis,group=g,value=value,outcome=outcome,bin_left=bi/50,bin_right=(bi+1)/50,pixel_count=int(n),conditional_fraction=float(n/total) if total else 'NA',distribution_bins=50))
  counters=[d['counters'] for u in uu for d in u['slices']]
  for name,den in [('background_to_foreground','background_to_foreground_denominator'),('foreground_to_background','foreground_to_background_denominator')]:
   n=sum(d[name] for d in counters);de=sum(d[den] for d in counters)
   for weighting in ['pooled_pixel','patient_equal']:
    pr=[]
    for u in uu:
     cc=[d['counters'] for d in u['slices']];pd=sum(d[den] for d in cc)
     if pd:pr.append(sum(d[name] for d in cc)/pd)
    correctness.append(dict(**base,axis='binary_error',group=name,weighting=weighting,pixel_count=de,correct_pixels='NA',error_pixels=n,support_patients=len(pr),mean_confidence='NA',mean_margin='NA',accuracy='NA',correct_mean_confidence='NA',error_mean_confidence='NA',ECE='NA',error_rate=n/de if weighting=='pooled_pixel' and de else meanfinite(pr) if weighting=='patient_equal' else 'NA',denominator='GT background pixels' if name=='background_to_foreground' else 'GT foreground pixels'))
 csvout(W/'results/A2_confidence_bins.csv',bins);csvout(W/'results/A2_confidence_correctness.csv',correctness);csvout(W/'results/A2_confidence_distributions.csv',hist)
 js(W/'results/A2_RECEIPT.json',dict(status='DONE',time=now(),teacher_checkpoints=8,primary_samples_per_checkpoint=346,primary_validation_slices=210,primary_validation_frames=20,primary_validation_patients=10,labeled_train_slices=136,labeled_train_patients=7,legacy_unique_samples=34,quality_rows=len(quality),optimizer_updates=0,EMA_updates=0,test_inputs=0,early_middle_states='NA',mode='eval main; legacy train_snapshot separate',unit_identity=identity()))
def legacy_inputs():
 from dataloaders.dataset import WeakStrongAugment
 path=R1/'scripts/accepted_diagnostic_inputs.py'
 ns=dict(Path=Path,DATA=DATA,PROJECT=ROOT,WORK=W,read_json=read,random=random,np=np,torch=torch,h5py=h5py,zoom=zoom,WeakStrongAugment=WeakStrongAugment,sha=sha,hashlib=hashlib,write_csv=lambda p,r:csvout(W/'legacy/replayed_diagnostic_samples.csv',r))
 for node in ast.parse(path.read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name=='samples':exec(compile(ast.Module(body=[node],type_ignores=[]),str(path),'exec'),ns)
 ii=ns['samples']();old=rows(W/'legacy/diagnostic_samples.csv');new=rows(W/'legacy/replayed_diagnostic_samples.csv')
 assert {r['sample']:r['input_sha256'] for r in old}=={r['sample']:r['input_sha256'] for r in new}
 js(W/'reports/LEGACY_INPUT_REPLAY.json',dict(status='PASS',all_source_input_hashes_match=True,all_manifest_samples=len(old),real_GT_analyzed_samples=sum(s['gt'] is not None for s in ii),original_augmented_input_relation='14 labeled slices share source IDs with enlarged cohort but use historical seeded weak augmentation;20 validation middle slices are members of enlarged all-slice cohort'))
 return [dict(sample=s['id'],patient=s['patient'],frame=s['id'].split(':')[0] if ':z' in s['id'] else '_'.join(s['id'].split('_')[:2]),slice_index=int(s['id'].split(':z')[1]) if ':z' in s['id'] else int(s['id'].rsplit('_',1)[1]),cohort=s['cohort']+'_legacy',x=s['x'],gt=s['gt'],input_sha256=hashlib.sha256(s['x'].numpy().tobytes()).hexdigest()) for s in ii if s['gt'] is not None]
def main():
 torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 cfg=config();ss=rows(W/'manifests/fixed_samples.csv');unitdir('A2').mkdir(parents=True,exist_ok=True)
 for method in cfg['methods']:
  for cp in method['checkpoints']:
   wait_gpu();m,receipt=model(method,cp,'ema_model');before=digest(m)
   js(W/'reports'/f"LOAD_{method['id']}_{cp['tag']}_teacher.json",receipt)
   groups=defaultdict(list)
   for s in ss:groups[(s['cohort'],s['patient'])].append(s)
   for (cohort,patient),samples in groups.items():
    stopped();dst=unitdir('A2')/f"{method['id']}_{cp['tag']}_eval_{cohort}_{patient}.json"
    if dst.exists():continue
    wait_gpu(own=True);out=[]
    for s in samples:
     x,gt=load_sample(s)
     with torch.inference_mode():logit=m(torch.from_numpy(x)[None,None].cuda())
     d,arr=summarize_slice(method,cp,'eval',s,logit,gt);out.append(d)
     if method['id']=='R1_v2_B' and cp['tag']=='best' and cohort=='validation' and patient in cfg['verification_patients']:
      dp=W/'state/verification_pixels'/f"{s['sample'].replace(':','_')}.npz";dp.parent.mkdir(parents=True,exist_ok=True)
      with dp.with_suffix('.tmp').open('wb') as f:np.savez_compressed(f,**arr)
      os.replace(dp.with_suffix('.tmp'),dp)
    js(dst,dict(identity=identity(),method_id=method['id'],checkpoint_tag=cp['tag'],checkpoint_step=cp['step'],mode='eval',cohort=cohort,patient=patient,slices=out));progress('A2',method=method['id'],checkpoint=cp['tag'],cohort=cohort,patient=patient,completed_units=len(list(unitdir('A2').glob('*.json'))));print('A2',method['id'],cp['tag'],cohort,patient,len(out),flush=True)
   assert digest(m)==before
   del m;gc.collect();torch.cuda.empty_cache()
 # Independently replay the historical small sample batches and BN snapshots.
 ii=legacy_inputs();method=next(m for m in cfg['methods'] if m['id']=='R1_v2_B')
 for cp in method['checkpoints']:
  for mode in ['eval','train_snapshot']:
   marker=unitdir('A2')/f"legacy_complete_{cp['tag']}_{mode}.marker"
   if marker.exists():continue
   wait_gpu();m,_=model(method,cp,'ema_model');frozen={k:v.detach().clone() for k,v in m.state_dict().items()};before=digest(m);m.train(mode=='train_snapshot');patientout=defaultdict(list)
   for cohort in ['labeled_train_legacy','validation_legacy']:
    sub=[s for s in ii if s['cohort']==cohort]
    for start in range(0,len(sub),12):
     stopped();batch=sub[start:start+12];n=len(batch)//2;assert 2*n==len(batch)
     m.load_state_dict(frozen,strict=True);torch.manual_seed(4261100+start//12);torch.cuda.manual_seed_all(4261100+start//12)
     with torch.no_grad():
      x=torch.stack([s['x'] for s in batch]).cuda();la=m(x[:n]);lb=m(x[n:]);ll=torch.cat([la,lb])
      for i,s in enumerate(batch):d,_=summarize_slice(method,cp,mode,s,ll[i:i+1],s['gt']);patientout[(cohort,s['patient'])].append(d)
     m.load_state_dict(frozen,strict=True)
   assert digest(m)==before
   for (cohort,patient),ds in patientout.items():js(unitdir('A2')/f"{method['id']}_{cp['tag']}_{mode}_{cohort}_{patient}.json",dict(identity=identity(),method_id=method['id'],checkpoint_tag=cp['tag'],checkpoint_step=cp['step'],mode=mode,cohort=cohort,patient=patient,slices=ds))
   atomic(marker,'completed immutable batches\n');del m,frozen;gc.collect();torch.cuda.empty_cache()
 aggregate()
if __name__=='__main__':main()
