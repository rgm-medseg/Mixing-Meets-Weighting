import gc
from collections import defaultdict
import numpy as np,torch
from scipy.ndimage import distance_transform_edt
from scipy.stats import spearmanr
from common import *
from adapter import model,pseudo,signals,geometry,functions,digest
from measurement import imagehash,backend,unload
from a3_inputs import boundary
actual=functions(R1/'scripts/revision_runtime.py',['ctw_maps'])['ctw_maps']
def bands(b,width):return distance_transform_edt(~b)<=width if b.any() else np.zeros(b.shape,bool)
def masks(a,width):
 anatomy=bands(a['anat'],width);seam=bands(a['seam'],width);cut=bands(a['cut_boundary'],width);valid=a['valid'];clean=anatomy&~seam&~cut&valid
 return dict(all=np.ones(anatomy.shape,bool),anatomy_only=anatomy&~seam,seam_only=seam&~anatomy,anatomy_seam_overlap=anatomy&seam,remainder=~anatomy&~seam,anatomy=anatomy,anatomy_clean=clean,seam=seam,cutout_edge=cut,occlusion=a['occlusion'],valid=valid,mixed_GT_boundary=bands(boundary(a['gt']),width),foreground=(a['gt']>0)&valid,background=(a['gt']==0)&valid,RV=(a['gt']==1)&valid,MYO=(a['gt']==2)&valid,LV=(a['gt']==3)&valid,foreground_interior=(a['gt']>0)&~anatomy&valid,background_interior=(a['gt']==0)&~anatomy&valid,student1_correct=(~a['err1'])&valid,student1_error=a['err1']&valid,student2_correct=(~a['err2'])&valid,student2_error=a['err2']&valid,branches_agree=(~a['disagree'])&valid,branches_disagree=a['disagree']&valid)
def summarize_map(meta,a):
 rr=[];dist=[];quant=[];uniform=[];w=a['w'];e=a['e'];we=w*e;wu=float(w.mean(dtype=np.float64));total=float(we.sum(dtype=np.float64));N=w.size
 for width in config()['a3_widths']:
  regions=masks(a,width)
  assert np.all(sum(regions[k].astype(int) for k in ['anatomy_only','seam_only','anatomy_seam_overlap','remainder'])==1)
  for patient_id in np.unique(a['patient_ids']):
   patient=config()['a3_patients'][int(patient_id)];pm=a['patient_ids']==patient_id
   for region,r in regions.items():
    r=r&pm;n=int(r.sum());safe=n or 1;valid=r&a['valid'];nv=int(valid.sum());v1=r&a['valid1'];n1=int(v1.sum());v2=r&a['valid2'];n2=int(v2.sum());delta=(w-wu)*e
    rr.append(dict(**meta,width=width,patient=patient,region=region,pixels=n,total_image_pixels=N,valid_error_pixels=nv,student1_error_denominator=n1,student2_error_denominator=n2,w_sum=float(w[r].sum(dtype=np.float64)),e_sum=float(e[r].sum(dtype=np.float64)),we_sum=float(we[r].sum(dtype=np.float64)),uniform_we_sum=float((wu*e[r]).sum(dtype=np.float64)),delta_we_sum=float(delta[r].sum(dtype=np.float64)),mean_w=float(w[r].mean(dtype=np.float64)) if n else 'NA',median_w=float(np.median(w[r])) if n else 'NA',mean_e=float(e[r].mean(dtype=np.float64)) if n else 'NA',mean_we=float(we[r].mean(dtype=np.float64)) if n else 'NA',student1_error_count=int(a['err1'][v1].sum()),student2_error_count=int(a['err2'][v2].sum()),disagreement_count=int(a['disagree'][valid].sum()),student1_error_rate=float(a['err1'][v1].mean()) if n1 else 'NA',student2_error_rate=float(a['err2'][v2].mean()) if n2 else 'NA',disagreement_rate=float(a['disagree'][valid].mean()) if nv else 'NA',loss_contribution_full_image_mean=float(we[r].sum(dtype=np.float64)/N),loss_fraction=float(we[r].sum(dtype=np.float64)/total) if total else 'NA',uniform_contribution_full_image_mean=float(wu*e[r].sum(dtype=np.float64)/N),conditional_valid_mean_we=float(we[valid].mean(dtype=np.float64)) if nv else 'NA'))
 for patient_id in np.unique(a['patient_ids']):
  patient=config()['a3_patients'][int(patient_id)];pm=a['patient_ids']==patient_id;domain=pm&a['valid'];n=int(domain.sum())
  uniform.append(dict(**meta,patient=patient,image_original_mean_w=wu,image_uniform_mean_w=wu,mean_difference=0.,full_image_L=float(we.mean(dtype=np.float64)),full_image_L_uniform=float((wu*e).mean(dtype=np.float64)),full_image_delta_L=float(((w-wu)*e).mean(dtype=np.float64)),patient_support=n,patient_conditional_delta_L=float(((w-wu)*e)[domain].mean(dtype=np.float64)) if n else 'NA',scope='same predictions/input; uniform constant is whole-image mean, not region/patient renormalization'))
  for name,b in [('anatomy',a['anat']),('mix_seam',a['seam'])]:
   if not b.any():dist.append(dict(**meta,patient=patient,distance_to=name,distance_band='NA',pixels=n,mean_w='NA',spearman='NA',reason='no boundary pixels'));continue
   d=distance_transform_edt(~b);clean_domain=domain&~bands(a['seam'],2)&~bands(a['cut_boundary'],2) if name=='anatomy' else domain
   rho=float(spearmanr(d[clean_domain],w[clean_domain]).statistic) if clean_domain.sum()>1 and np.ptp(d[clean_domain]) and np.ptp(w[clean_domain]) else 'NA'
   for lo,hi in [(0,2),(2,5),(5,10),(10,20),(20,np.inf)]:
    r=clean_domain&(d<=hi)&(d>=lo if lo==0 else d>lo);n0=int(r.sum());dist.append(dict(**meta,patient=patient,distance_to=name,distance_band=f'{lo}-{hi}',pixels=n0,mean_w=float(w[r].mean(dtype=np.float64)) if n0 else 'NA',spearman=rho,domain='clean valid excludes seam/cutout width2' if name=='anatomy' else 'valid',unit='256-grid pixels'))
  vals=w[domain]
  if n:
   cuts=np.unique(np.quantile(vals,config()['quantiles']));edges=cuts[1:-1];ids=np.searchsorted(edges,vals,side='left');err=a['err1'][domain];dis=a['disagree'][domain]
   for k in np.unique(ids):
    r=ids==k;nq=int(r.sum());quant.append(dict(**meta,patient=patient,quantile_group=int(k),cutpoints=';'.join(map(str,cuts)),pixels=nq,actual_support_fraction=nq/n,mean_w=float(vals[r].mean()),min_w=float(vals[r].min()),max_w=float(vals[r].max()),GT_error_fraction=float(err[r].mean()),branch_disagreement_fraction=float(dis[r].mean()),ties='unique cutpoints; equal weights never split',domain='valid per source patient'))
 return rr,dist,quant,uniform
def run():
 backend();cfg=config();ud=unitdir('A3');patients=sorted({r['patient'] for r in rows(W/'results/A3_input_pair_manifest.csv')});assert cfg['a3_patients']==patients
 # Direction check uses synthetic logits only; model parameters never receive backward.
 x=torch.randn(2,4,8,8,requires_grad=True);y=torch.randn(2,4,8,8,requires_grad=True);raw,norm,orig,w,e=actual(x,y,False);loss=(w*e).mean();g1,g2=torch.autograd.grad(loss,(x,y),allow_unused=True);assert g1 is not None and g1.abs().sum()>0 and g2 is None and not w.requires_grad
 js(W/'reports/A3_SYNTHETIC_GRADIENT_CHECK.json',dict(status='PASS',student1_nonzero=True,student2_gradient=None,weight_detached=True,scope='independent synthetic logits only; no model parameter backward',lambda_coefficient=.05))
 for method in cfg['methods'][1:]:
  for cp in method['checkpoints']:
   strategies=['matched92']+(['original_strategy'] if method['id']!='Full_main' else [])
   pending=[(strategy,b) for strategy in strategies for b in range(9) if not(ud/f"{method['id']}_{cp['tag']}_{strategy}_{b}.json").exists()]
   if not pending:continue
   wait_gpu(own=torch.cuda.is_initialized());m1,r1=model(method,cp,'model_1');m2,r2=model(method,cp,'model_2');mt,rt=model(method,cp,'ema_model');before=[digest(m) for m in [m1,m2,mt]]
   for strategy,b in pending:
    stopped();wait_gpu(own=True);plan=read(unitdir('INPUTS')/f'batch_{b}.json');slots=plan['slots'];arrays=[dict(np.load(r['input_array_path'])) for r in slots];weak=torch.from_numpy(np.stack([a['weak'] for a in arrays])).cuda();strong=torch.from_numpy(np.stack([a['strong'] for a in arrays])).cuda()
    with torch.inference_mode():
     la=mt(weak[12:18]);lb=mt(weak[18:24]);pla=pseudo(la,method);plb=pseudo(lb,method);st=signals(la,lb,pla,plb,[1,2,3],0);q=st['q_margin'];s,ph,pw=geometry(q,256,256)
     if strategy=='matched92' or method['id']=='R2_fixed92':ph=pw=92;s=92/256
     y0=int(plan['position_uniform_y']*(256-ph));x0=int(plan['position_uniform_x']*(256-pw));outside=np.ones((256,256),bool);outside[y0:y0+ph,x0:x0+pw]=False;mask=torch.from_numpy(outside.astype(np.float32))[None,None].cuda()
     inputs1=[weak[12:18]*mask+weak[:6]*(1-mask),weak[6:12]*mask+weak[18:24]*(1-mask)];inputs2=[strong[12:18]*mask+strong[:6]*(1-mask),strong[6:12]*mask+strong[18:24]*(1-mask)]
     outputs1=[m1(v) for v in inputs1];outputs2=[m2(v) for v in inputs2]
    rr=[];dist=[];quant=[];uniform=[];maps=[]
    for group in range(2):
     o1,o2=outputs1[group],outputs2[group];raw,norm,original,w,e=actual(o1,o2,False);p1=o1.softmax(1);p2=o2.softmax(1)
     # Cross-check literal historical X03 loss on the exact same aligned logits.
     from v21_3_acdc10_third_innovation_components import compute_v21_3_innovation_x
     check,_=compute_v21_3_innovation_x('V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',o1,o2,la,lb,o1,o2,0,30000);assert torch.allclose(check,(w*e).mean(),atol=1e-8,rtol=1e-6)
     confidence1=p1.max(1).values.cpu().numpy();confidence2=p2.max(1).values.cpu().numpy();pred1=p1.argmax(1).cpu().numpy();pred2=p2.argmax(1).cpu().numpy()
     for j in range(6):
      outside_slot=12+j if group==0 else 6+j;inside_slot=j if group==0 else 18+j;ao=arrays[outside_slot];ai=arrays[inside_slot]
      def compose(key):return np.where(outside,ao[key],ai[key])
      gt=compose('gt');occ=compose('occlusion');gv=compose('geometry_valid');valid=gv&~occ
      patientids=np.where(outside,patients.index(slots[outside_slot]['patient']),patients.index(slots[inside_slot]['patient'])).astype(np.uint8)
      a=dict(w=w[j].cpu().numpy(),e=e[j].cpu().numpy(),q=.5*(confidence1[j]+confidence2[j]),confidence1=confidence1[j],confidence2=confidence2[j],raw=raw[j].cpu().numpy(),normalized=norm[j].cpu().numpy(),we=(w[j]*e[j]).cpu().numpy(),pred1=pred1[j],pred2=pred2[j],gt=gt,strong_artificial_gt=compose('strong_artificial_gt'),anat=compose('anatomy'),seam=boundary(outside.astype(np.uint8)),cut_boundary=boundary(occ.astype(np.uint8)),occlusion=occ,valid=valid,valid1=gv,valid2=valid,err1=pred1[j]!=gt,err2=pred2[j]!=gt,disagree=pred1[j]!=pred2[j],patient_ids=patientids,image1=inputs1[group][j,0].cpu().numpy(),image2=inputs2[group][j,0].cpu().numpy(),outside=outside)
      assert np.allclose(a['we'],a['w']*a['e'],atol=1e-8);assert a['w'].min()>=.25 and a['w'].max()<=1.25
      meta=dict(method_id=method['id'],checkpoint_tag=cp['tag'],checkpoint_sha256=cp['sha256'],mode='eval',strategy=strategy,batch=b,pair_index=group*6+j,direction='student1_to_stopgrad_student2',outside_sample=slots[outside_slot]['sample'],inside_sample=slots[inside_slot]['sample'],rectangle_h=ph,rectangle_w=pw,y=y0,x=x0,q_B=q,active_foreground_classes=st['active_classes'],fallback=st['fallback'])
      dst=W/'state/map_arrays'/identity()/f"{method['id']}_{cp['tag']}_{strategy}_{b}_{group*6+j}.npz";dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.tmp.npz');np.savez_compressed(tmp,**a);os.replace(tmp,dst);meta.update(map_array_path=str(dst),map_array_sha256=sha(dst),weak_input_sha256=imagehash(a['image1']),strong_input_sha256=imagehash(a['image2']))
      r,d,q0,u=summarize_map(meta,a);rr.extend(r);dist.extend(d);quant.extend(q0);uniform.extend(u);maps.append(meta)
    after=[digest(m) for m in [m1,m2,mt]];assert after==before
    js(ud/f"{method['id']}_{cp['tag']}_{strategy}_{b}.json",dict(identity=identity(),status='PASS',method_id=method['id'],checkpoint_tag=cp['tag'],checkpoint_sha256=cp['sha256'],strategy=strategy,batch=b,load_receipts=[r1,r2,rt],model_states_before=before,model_states_after=after,regions=rr,distances=dist,quantiles=quant,uniform=uniform,maps=maps))
    del weak,strong,outputs1,outputs2;progress('A3',method=method['id'],checkpoint=cp['tag'],strategy=strategy,batch=b);print('A3',method['id'],cp['tag'],strategy,b,flush=True)
   del m1,m2,mt;unload()
 summarize()
def summarize():
 vv=[read(p) for p in sorted(unitdir('A3').glob('*.json'))];assert len(vv)==90
 rr=[r for u in vv for r in u['regions']];csvout(W/'results/A3_region_statistics.csv',rr);csvout(W/'results/A3_distance_statistics.csv',[r for u in vv for r in u['distances']]);csvout(W/'results/A3_uniform_mean_matched.csv',[r for u in vv for r in u['uniform']]);csvout(W/'results/A3_weight_quantiles.csv',[r for u in vv for r in u['quantiles']]);maps=[r for u in vv for r in u['maps']];csvout(W/'results/A3_map_manifest.csv',maps)
 groups=defaultdict(list)
 for r in rr:groups[(r['method_id'],r['checkpoint_tag'],r['strategy'],r['width'],r['patient'],r['region'])].append(r)
 sums=[]
 for key,ss in groups.items():
  m,cp,strategy,width,patient,region=key;n=sum(r['pixels'] for r in ss);ne=sum(r['valid_error_pixels'] for r in ss);n1=sum(r['student1_error_denominator'] for r in ss);n2=sum(r['student2_error_denominator'] for r in ss);ws=sum(r['w_sum'] for r in ss);es=sum(r['e_sum'] for r in ss);wes=sum(r['we_sum'] for r in ss)
  sums.append(dict(method_id=m,checkpoint_tag=cp,strategy=strategy,mode='eval',width=width,patient=patient,region=region,pixels=n,image_source_exposures=len(ss),mean_w=ws/n if n else 'NA',mean_e=es/n if n else 'NA',mean_we=wes/n if n else 'NA',median_of_image_region_medians=float(np.median([r['median_w'] for r in ss if r['pixels']])) if n else 'NA',median_w='NA: exact pooled patient median computed separately for primary regions',w_sum=ws,e_sum=es,we_sum=wes,uniform_we_sum=sum(r['uniform_we_sum'] for r in ss),delta_we_sum=sum(r['delta_we_sum'] for r in ss),student1_error_rate=sum(r['student1_error_count'] for r in ss)/n1 if n1 else 'NA',student2_error_rate=sum(r['student2_error_count'] for r in ss)/n2 if n2 else 'NA',disagreement_rate=sum(r['disagreement_count'] for r in ss)/ne if ne else 'NA',valid_error_pixels=ne,student1_error_denominator=n1,student2_error_denominator=n2))
 # Exact pooled patient median for every region: process each method/checkpoint/strategy independently.
 primary={}
 for group in sorted({(r['method_id'],r['checkpoint_tag'],r['strategy']) for r in maps}):
  buckets=defaultdict(list)
  for r in maps:
   if (r['method_id'],r['checkpoint_tag'],r['strategy'])!=group:continue
   with np.load(r['map_array_path']) as z:a={k:z[k] for k in ['w','anat','seam','cut_boundary','valid','occlusion','gt','err1','err2','disagree','patient_ids']}
   for width in config()['a3_widths']:
    regs=masks(a,width)
    for pid in np.unique(a['patient_ids']):
     pm=a['patient_ids']==pid;pat=config()['a3_patients'][int(pid)]
     for reg,mask0 in regs.items():
      val=a['w'][mask0&pm]
      if len(val):buckets[(width,pat,reg)].append(val)
  for key,v in buckets.items():primary[(*group,*key)]=float(np.median(np.concatenate(v)))
  del buckets;gc.collect();print('A3 medians',group,flush=True)
 for r in sums:r['median_w']=primary.get((r['method_id'],r['checkpoint_tag'],r['strategy'],r['width'],r['patient'],r['region']),'NA: no support')
 csvout(W/'results/A3_patient_summary.csv',sums)
 ag=[]
 for key in sorted({(r['method_id'],r['checkpoint_tag'],r['strategy'],r['width'],r['region']) for r in sums}):
  ss=[r for r in sums if (r['method_id'],r['checkpoint_tag'],r['strategy'],r['width'],r['region'])==key];m,cp,strat,width,reg=key;n=sum(r['pixels'] for r in ss);valid=[r for r in ss if r['pixels']]
  for weighting in ['pooled_pixel','patient_equal']:
   ag.append(dict(method_id=m,checkpoint_tag=cp,strategy=strat,width=width,region=reg,weighting=weighting,pixels=n,support_patients=len(valid),mean_w=sum(r['w_sum'] for r in ss)/n if n and weighting=='pooled_pixel' else float(np.mean([r['mean_w'] for r in valid])) if valid else 'NA',mean_e=sum(r['e_sum'] for r in ss)/n if n and weighting=='pooled_pixel' else float(np.mean([r['mean_e'] for r in valid])) if valid else 'NA',mean_we=sum(r['we_sum'] for r in ss)/n if n and weighting=='pooled_pixel' else float(np.mean([r['mean_we'] for r in valid])) if valid else 'NA',delta_mean=sum(r['delta_we_sum'] for r in ss)/n if n and weighting=='pooled_pixel' else float(np.mean([r['delta_we_sum']/r['pixels'] for r in valid])) if valid else 'NA'))
 csvout(W/'results/A3_aggregate_statistics.csv',ag)
 js(W/'results/A3_RECEIPT.json',dict(status='DONE',units=90,maps=len(maps),checkpoints=6,mode='eval',patient_count=10,matched_mask=True,Full_original_strategy='NA: NEEDS_MAPPING',R1_R2_original_strategy=True,training_updates=0,model_BN_states_unchanged=True,identity=identity()))
if __name__=='__main__':run()
