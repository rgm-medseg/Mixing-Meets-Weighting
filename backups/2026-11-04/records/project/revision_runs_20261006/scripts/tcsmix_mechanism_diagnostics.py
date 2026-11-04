"""Read-only mechanism diagnostics. Never invokes training or queue recovery.

Writes only to a new, explicitly supplied output directory. Raw patient arrays,
checkpoint weights, optimizer states and random states are not exported.
"""
import argparse
import collections
import hashlib
import json
import math
import os
import random
import re
import shutil
import struct
import subprocess
import time
from pathlib import Path

os.environ.setdefault('CUDA_VISIBLE_DEVICES', '0')
os.environ.setdefault('OMP_NUM_THREADS', '1')
os.environ.setdefault('MKL_NUM_THREADS', '1')
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
import h5py
import numpy as np
import torch
from scipy.ndimage import zoom
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import WORK, PROJECT, FULL, DATA, now, sha, read_json, read_table, write_csv, atomic_json, bootstrap
bootstrap()
import revision_runtime as runtime
import v14_mask_adamix_components as parent
from dataloaders.dataset import WeakStrongAugment
from evaluate import load_model

plt.rcParams.update({'font.size': 9, 'svg.fonttype': 'none', 'axes.spines.top': False,
                     'axes.spines.right': False})

def dumpfig(fig, out, name):
    fig.savefig(out / f'{name}.png', dpi=170, bbox_inches='tight')
    fig.savefig(out / f'{name}.svg', bbox_inches='tight')
    plt.close(fig)

def state_digest(model):
    h = hashlib.sha256()
    for name, t in model.state_dict().items():
        h.update(name.encode()); h.update(t.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def scalestats(rows, run, evidence, out):
    counts = collections.Counter((r['rectangle_h'], r['rectangle_w']) for r in rows)
    steps = [r['step'] for r in rows]
    changes = sum((a['rectangle_h'], a['rectangle_w']) != (b['rectangle_h'], b['rectangle_w'])
                  for a, b in zip(rows, rows[1:]))
    for (h, w), count in sorted(counts.items()):
        out.append(dict(run=run, evidence=evidence, n_records=len(rows), n_distinct_steps=len(set(steps)),
                        step_min=min(steps), step_max=max(steps), rectangle_h=h, rectangle_w=w,
                        count=count, fraction=count/len(rows), n_sizes=len(counts),
                        area_fraction=h*w/65536, area_min=min(r['actual_area'] for r in rows),
                        area_max=max(r['actual_area'] for r in rows), changes_between_recorded_rows=changes,
                        full_iteration_changes=changes if evidence=='all executed updates' else 'NA: sparse records',
                        q_min=min(r['q'] for r in rows), q_max=max(r['q'] for r in rows),
                        q_mean=np.mean([r['q'] for r in rows]), s_min=min(r['s'] for r in rows),
                        s_max=max(r['s'] for r in rows), s_mean=np.mean([r['s'] for r in rows]),
                        coverage='1..3976 every executed update; future 3977..30000 unknown' if run=='E1'
                        else '200,400,...,30000; other 29850 updates unrecorded'))

def parse_scale(path):
    rows=[]; lines=[]
    for lineno, line in enumerate(path.read_text(errors='replace').splitlines(), 1):
        match = re.search(r'iteration (\d+) : adamix_endpoint=(\{.*\})', line)
        if not match: continue
        st=json.loads(match[2]); s=st['side_fraction']; d=round(256*s)
        assert d*d/65536==st['patch_area_ratio']
        rows.append(dict(step=int(match[1]),q=st['teacher_conf'],s=s,actual_area=st['patch_area_ratio'],
                         rectangle_h=d,rectangle_w=d,source_line=lineno))
        lines.append(f'{lineno}: {line}')
    return rows,lines

def history(out):
    rows,lines=parse_scale(FULL/'train.log')
    controller=PROJECT/'runtime/acdc10_third_innovation_codex_prep_top6_v21_3/controller/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT_stdout_stderr.log'
    mirror,mlines=parse_scale(controller)
    assert [{k:v for k,v in r.items() if k!='source_line'} for r in rows]==[
        {k:v for k,v in r.items() if k!='source_line'} for r in mirror]
    (out/'sources/historical_scale_log_excerpt.txt').write_text('\n'.join(lines)+'\n')
    (out/'sources/controller_scale_log_excerpt.txt').write_text('\n'.join(mlines)+'\n')
    trace=read_table(WORK/'runs/E1/audit/revision_trace.csv')
    e1=[]
    for r in trace:
        step=int(r['step']); s=float(r['s']); d=round(256*s); area=float(r['actual_area'])
        assert d*d/65536==area
        e1.append(dict(step=step,q=float(r['q']),s=s,actual_area=area,rectangle_h=d,rectangle_w=d,source_line=step+1))
    assert [r['step'] for r in e1]==list(range(1,3977))
    assert [r['step'] for r in rows]==list(range(200,30001,200))
    distribution=[]
    scalestats(rows,'Historical_Full','every 200 updates',distribution)
    scalestats(e1,'E1','all executed updates',distribution)
    write_csv(out/'scale_distribution.csv',distribution)
    write_csv(out/'scale_records.csv',[dict(run=run,**r) for run,rr in [('Historical_Full',rows),('E1',e1)] for r in rr])
    # Read TFRecord summaries without adding a dependency or interpreting losses as scales.
    from tensorboardX.proto.event_pb2 import Event
    eventinfo=[]
    for f in sorted((FULL/'logs').glob('events.*')):
        tags=collections.defaultdict(list)
        with f.open('rb') as h:
            while header:=h.read(12):
                assert len(header)==12
                n=struct.unpack('<Q',header[:8])[0]; data=h.read(n); assert len(data)==n
                assert len(h.read(4))==4
                e=Event(); e.ParseFromString(data)
                for value in e.summary.value: tags[value.tag].append(e.step)
        eventinfo.append(dict(path=str(f),sha256=sha(f),tags={k:dict(n=len(v),step_min=min(v),step_max=max(v)) for k,v in tags.items()}))
    files=subprocess.run(['rg','--files','-uuu'],cwd=PROJECT,capture_output=True,text=True,check=True).stdout.splitlines()
    matches=[f for f in files if 'V21_3_X03_CONTOUR_RELIABILITY_WEIGHT' in f and
             Path(f).suffix in ('.log','.csv','.json','.ckpt','.pth','.npz','.npy')]
    inventory=[]
    for name in matches:
        p=PROJECT/name
        kind=('separate smoke, not Full trajectory' if '/smoke/' in name else
              'copied evidence, not independent trajectory' if name.startswith('revision_runs_') else
              'original Full' if name.startswith(str(FULL.relative_to(PROJECT))) else 'configuration/audit/controller')
        inventory.append(dict(path=str(p),bytes=p.stat().st_size,sha256=sha(p),role=kind))
    write_csv(out/'history_source_inventory.csv',inventory)
    dynamics=read_table(FULL/'audit/training_dynamics.csv')
    atomic_json(out/'history_search.json',dict(time=now(),scope='project filenames via rg --files -uuu; exact Full run, mirrored controller, TFRecord summaries and dynamics schema',
        matched_files=len(matches),controller_records=len(mirror),controller_adds_distinct_scale_steps=0,
        events=eventinfo,dynamics_records=len(dynamics),dynamics_columns=list(dynamics[0]),
        dynamics_has_q_or_scale=False,formal_checkpoints=['best.ckpt step29600','last.ckpt step30000'],
        no_formal_intermediate_checkpoint_found=True,no_saved_mask_array_found=True,
        excluded_smoke='runtime/.../smoke/X03 checkpoints belong to engineering smoke; never substitute for Full history'))
    return distribution,rows,e1

def stats(a):
    a=np.asarray(a)
    if not a.size: return {k:'NA: empty region' for k in ('mean','std','min','p01','p05','p50','p95','p99','max')}
    q=np.quantile(a,[.01,.05,.5,.95,.99])
    return dict(mean=float(a.mean()),std=float(a.std()),min=float(a.min()),p01=q[0],p05=q[1],p50=q[2],p95=q[3],p99=q[4],max=float(a.max()))

def dice(pred,gt):
    vals=[]; present=[]
    for c in (1,2,3):
        den=(pred==c).sum()+(gt==c).sum()
        value=2*((pred==c)&(gt==c)).sum()/den if den else 0.
        vals.append(float(value))
        if (gt==c).any(): present.append(value)
    return dict(dice_RV=vals[0],dice_MYO=vals[1],dice_LV=vals[2],fg_macro_dice=float(np.mean(vals)),
                gt_present_macro_dice=float(np.mean(present)) if present else 'NA: no GT foreground class')

def samples(out):
    ids=(DATA/'train_slices.list').read_text().splitlines()
    labeled=read_json(WORK/'configs/diagnostic_cases.json')['slices']
    # Index-spaced selection, declared before loading any model or inspecting output.
    unlab=[ids[136+int(i)] for i in np.linspace(0,len(ids)-137,24).round().astype(int)]
    assert len(set(unlab))==24 and not set(unlab)&set(ids[:136])
    random.seed(4261006); np.random.seed(4261006); torch.manual_seed(4261006)
    aug=WeakStrongAugment([256,256]); data=collections.defaultdict(list); manifest=[]
    for cohort,names in [('unlabeled_train',unlab),('labeled_train',labeled)]:
        for name in names:
            p=DATA/'data/slices'/f'{name}.h5'
            with h5py.File(p,'r') as f: image=f['image'][:]; gt=f['label'][:]
            s=aug(dict(image=image,label=gt))
            record=dict(id=name,patient=name.split('_')[0],input=s['image'],strong=s['image_strong'],
                        gt=s['label'].numpy() if cohort=='labeled_train' else None)
            data[cohort].append(record)
            manifest.append(dict(cohort=cohort,sample_id=name,patient=record['patient'],source=str(p),sha256=sha(p),
                                 input_sha256=hashlib.sha256(s['image'].numpy().tobytes()).hexdigest(),
                                 label_used=cohort=='labeled_train',view='frozen WeakStrongAugment weak, seed4261006'))
    for name in (DATA/'val.list').read_text().splitlines():
        p=DATA/'data'/f'{name}.h5'
        with h5py.File(p,'r') as f: image=f['image'][:]; gt=f['label'][:]
        z=image.shape[0]//2; h,w=image[z].shape
        x=zoom(image[z],(256/h,256/w),order=0).astype(np.float32)
        y=zoom(gt[z],(256/h,256/w),order=0)
        data['validation'].append(dict(id=f'{name}:z{z}',patient=name.split('_')[0],input=torch.from_numpy(x)[None],gt=y))
        manifest.append(dict(cohort='validation',sample_id=f'{name}:z{z}',patient=name.split('_')[0],source=str(p),sha256=sha(p),
                             input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),label_used=True,
                             view='fixed middle z=Nz//2, nearest256, no augmentation'))
    write_csv(out/'diagnostic_batch_manifest.csv',manifest)
    atomic_json(out/'diagnostic_protocol.json',dict(time=now(),selection='unlabeled24 index-spaced; previously locked14 labeled slices; middle slice of all20 validation frames',
        seed=4261006,grouping='contiguous chunks of12; two equal subgroups, six+six except final short chunks',
        samples={k:len(v) for k,v in data.items()},image_size=[256,256],label_access='only previously labeled training slices and validation; unlabeled GT not retained or scored',
        test_access=False,parameter_updates=0,optimizer_steps=0,
        modes=['eval','train_snapshot'],train_snapshot='training-mode dropout/BN; checkpoint buffers restored before and after each diagnostic batch; parameters never updated',
        continuation_state='independent process RNG/model copies, checkpoint files and queue unchanged',
        dice_definition='per-slice or pooled diagnostic all3 foreground classes; double-empty class0; present-GT-class score also provided; not final test-volume Dice'))
    return data

def diagnostics(out,data):
    checkpoints=[('Historical_Full',FULL/'checkpoints/best.ckpt'),('Historical_Full',FULL/'checkpoints/last.ckpt'),
                 ('E1',WORK/'runs/E1/checkpoints/best.ckpt'),('E1',WORK/'runs/E1/checkpoints/last.ckpt')]
    rows=[]; hist=[]; pooled=[]; comparisons=[]; metadata=[]; probability_stats=[]; bins=np.linspace(0,1,101)
    for run,cp in checkpoints:
        cfg=read_json(WORK/'configs/Historical_Full.json') if run=='Historical_Full' else read_json(WORK/'configs/E1_uniform.yaml')
        model,ck=load_model(cfg,cp,'ema_model'); ch=sha(cp); step=ck['global_step']
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()}; initial=state_digest(model)
        metadata.append(dict(run=run,checkpoint=str(cp),sha256=ch,step=step,keys=sorted(ck),teacher_state_digest=initial))
        for mode in ('eval','train_snapshot'):
            for cohort,samples_ in data.items():
                acc=collections.defaultdict(list)
                for batch_index,start in enumerate(range(0,len(samples_),12)):
                    batch=samples_[start:start+12]; n=len(batch)//2; assert len(batch)%2==0
                    model.load_state_dict(frozen); model.train(mode=='train_snapshot')
                    x=torch.stack([s['input'] for s in batch]).cuda()
                    torch.manual_seed(4261100+batch_index); torch.cuda.manual_seed_all(4261100+batch_index)
                    with torch.inference_mode():
                        la=model(x[:n]); lb=model(x[n:]); logits=torch.cat([la,lb]); prob=logits.softmax(1)
                        q=.5*(la.softmax(1).max(1).values.mean()+lb.softmax(1).max(1).values.mean())
                        qvalue=q.item(); s=.18+.2*qvalue; d=round(256*s)
                        np.random.seed(4261200+batch_index)
                        m,l,st=parent.generate_adamix_bcp_lite_v1_mask(x[:n],la,lb)
                        zero=m[0]==0; points=zero.nonzero(); hh=int(points[:,0].max()-points[:,0].min()+1); ww=int(points[:,1].max()-points[:,1].min()+1)
                        assert hh==ww==d and int(zero.sum())==d*d
                        assert abs(qvalue-st['teacher_conf'])<1e-7 and torch.equal(m,l)
                        original_rng=np.random.get_state()
                        comp={}
                        for task in ('E1','E2','E3'):
                            runtime.CFG=read_json(next((WORK/'configs').glob(f'{task}_*.yaml')))
                            np.random.seed(4261200+batch_index)
                            mm,ll,ss=runtime.generate_adamix_bcp_lite_v1_mask(x[:n],la,lb)
                            comp[task]=mm.detach().clone()
                        np.random.set_state(original_rng)
                        comparisons.append(dict(run=run,step=step,checkpoint_sha256=ch,mode=mode,cohort=cohort,batch=batch_index,
                            q=qvalue,s=s,dynamic_h=hh,dynamic_w=ww,fixed_h=97,fixed_w=97,
                            E1_E2_masks_identical=bool(torch.equal(comp['E1'],comp['E2'])),E1_E3_masks_identical=bool(torch.equal(comp['E1'],comp['E3'])),
                            y=int(points[:,0].min()),x=int(points[:,1].min()),scope='offline same inputs/checkpoint/RNG; not historical positions'))
                        p=prob.cpu().numpy(); log=logits.cpu().numpy(); conf=p.max(1); pred=p.argmax(1)
                    model.load_state_dict(frozen)
                    base=dict(run=run,checkpoint=str(cp),checkpoint_sha256=ch,checkpoint_step=step,mode=mode,cohort=cohort,
                              batch=batch_index,batch_size=len(batch),offline=True)
                    rows.append(dict(**base,scope='batch',sample_id=';'.join(v['id'] for v in batch),region='all',
                        pixels=int(conf.size),pixel_fraction=1.,q_batch=qvalue,s=s,rectangle_h=d,rectangle_w=d,
                        probability_sum_max_error=float(np.abs(p.sum(1)-1).max()),logit_min=float(log.min()),logit_max=float(log.max()),
                        **{f'confidence_{k}':v for k,v in stats(conf).items()}))
                    for i,item in enumerate(batch):
                        gt=item['gt']; fg=pred[i]>0
                        masks={'all':np.ones(fg.shape,bool),'pred_fg':fg,'pred_bg':~fg}
                        if gt is not None: masks.update(gt_fg=gt>0,gt_bg=gt==0)
                        ds=dice(pred[i],gt) if gt is not None else {}
                        acc['confidence'].append(conf[i]); acc['pred'].append(pred[i]); acc['pbg'].append(p[i,0])
                        for cls in range(4): acc[f'probability_class{cls}'].append(p[i,cls])
                        if gt is not None: acc['gt'].append(gt)
                        for region,mask in masks.items():
                            cpix=conf[i][mask]
                            row=dict(**base,scope='sample',sample_id=item['id'],patient=item['patient'],region=region,
                                pixels=int(mask.sum()),pixel_fraction=float(mask.mean()),q_batch=qvalue,s=s,rectangle_h=d,rectangle_w=d,
                                confidence_contribution=float(cpix.sum()/conf[i].size),**{f'confidence_{k}':v for k,v in stats(cpix).items()},**ds)
                            for cls in range(4): row[f'p_class{cls}_mean']=float(p[i,cls][mask].mean()) if mask.any() else 'NA: empty region'
                            if gt is not None and mask.any():
                                row['pixel_accuracy']=float((pred[i][mask]==gt[mask]).mean())
                                truep=np.take_along_axis(p[i],gt[None].astype(np.int64),axis=0)[0]
                                row['true_class_probability_mean']=float(truep[mask].mean())
                            rows.append(row)
                conf=np.stack(acc['confidence']); pred=np.stack(acc['pred']); gt=np.stack(acc['gt']) if acc['gt'] else None
                masks={'all':np.ones(conf.shape,bool),'pred_fg':pred>0,'pred_bg':pred==0}
                if gt is not None: masks.update(gt_fg=gt>0,gt_bg=gt==0)
                for region,mask in masks.items():
                    vals=conf[mask]; count,_=np.histogram(vals,bins)
                    pr=dict(run=run,checkpoint_sha256=ch,checkpoint_step=step,mode=mode,cohort=cohort,region=region,
                            samples=len(samples_),patients=len({v['patient'] for v in samples_}),pixels=int(mask.sum()),
                            pixel_fraction=float(mask.mean()),q=float(conf.mean()),confidence_contribution=float(vals.sum(dtype=np.float64)/conf.size),
                            **{f'confidence_{k}':v for k,v in stats(vals).items()})
                    if gt is not None:
                        pr.update(dice(pred,gt))
                        pr['pixel_accuracy']=float((pred[mask]==gt[mask]).mean()) if mask.any() else 'NA'
                    pooled.append(pr)
                    for j,c in enumerate(count): hist.append(dict(run=run,checkpoint_step=step,checkpoint_sha256=ch,mode=mode,cohort=cohort,
                        region=region,bin_left=bins[j],bin_right=bins[j+1],count=int(c),denominator=int(vals.size),statistic='maximum class probability'))
                # Raw class-probability distribution distinguishes actual softmax from hard labels.
                for cls in range(4):
                    vals=np.stack(acc[f'probability_class{cls}']); count,_=np.histogram(vals,bins)
                    probability_stats.append(dict(run=run,checkpoint_step=step,checkpoint_sha256=ch,mode=mode,cohort=cohort,
                        class_id=cls,pixels=int(vals.size),**stats(vals)))
                    for j,c in enumerate(count): hist.append(dict(run=run,checkpoint_step=step,checkpoint_sha256=ch,mode=mode,cohort=cohort,
                        region='all',bin_left=bins[j],bin_right=bins[j+1],count=int(c),denominator=int(vals.size),statistic=f'softmax class{cls} probability'))
        assert state_digest(model)==initial
        metadata[-1]['teacher_state_unchanged_after_diagnostics']=True
        print(f'diagnosed {run} step{step}',flush=True)
        del model,ck,frozen; torch.cuda.empty_cache()
    fields=list(dict.fromkeys(k for r in rows for k in r))
    write_csv(out/'confidence_diagnostics.csv',rows,fields)
    write_csv(out/'confidence_pooled_statistics.csv',pooled,list(dict.fromkeys(k for r in pooled for k in r)))
    write_csv(out/'confidence_probability_histograms.csv',hist,list(dict.fromkeys(k for r in hist for k in r)))
    write_csv(out/'softmax_probability_statistics.csv',probability_stats)
    qstats=[]
    for m in metadata:
        for mode in ('eval','train_snapshot'):
            for cohort in data:
                qq=[r['q_batch'] for r in rows if r['scope']=='batch' and r['checkpoint_sha256']==m['sha256'] and r['mode']==mode and r['cohort']==cohort]
                qstats.append(dict(run=m['run'],step=m['step'],checkpoint_sha256=m['sha256'],mode=mode,cohort=cohort,n_batches=len(qq),
                    **stats(qq),scope='offline batch-average q distribution, not temporal trajectory'))
    write_csv(out/'batch_confidence_distribution.csv',qstats)
    write_csv(out/'offline_mask_comparison.csv',comparisons)
    atomic_json(out/'teacher_checkpoint_inventory.json',metadata)
    cor=[]
    for m in metadata:
        for mode in ('eval','train_snapshot'):
            for cohort in ('labeled_train','validation'):
                rr=[r for r in rows if r.get('scope')=='sample' and r.get('region')=='all' and r['checkpoint_sha256']==m['sha256'] and r['mode']==mode and r['cohort']==cohort]
                sp=spearmanr([r['confidence_mean'] for r in rr],[r['fg_macro_dice'] for r in rr])
                cor.append(dict(run=m['run'],step=m['step'],checkpoint_sha256=m['sha256'],mode=mode,cohort=cohort,samples=len(rr),patients=len({r['patient'] for r in rr}),
                    spearman=float(sp.statistic) if math.isfinite(sp.statistic) else 'NA: constant',interpretation='descriptive only; repeated slices/checkpoints are not independent patients; no test selection'))
    write_csv(out/'confidence_quality_correlations.csv',cor)
    return rows,pooled,hist,comparisons

def restore_check(out):
    cp=WORK/'runs/E1/checkpoints/last.ckpt'; ck=torch.load(cp,map_location='cpu',weights_only=False)
    cfg=read_json(WORK/'configs/E1_uniform.yaml'); runtime.CFG=cfg
    models=[runtime.BCP_net(ema=i==2) for i in range(3)]
    for model,key in zip(models,['model_1','model_2','ema_model']): model.load_state_dict(ck[key],strict=True)
    optimizer_info=[]
    for model,key in zip(models[:2],['optimizer1','optimizer2']):
        opt=parent.make_optimizer(model,.01,True); opt.load_state_dict(ck[key]); shapes=True
        for p,st in opt.state.items():
            if 'momentum_buffer' in st: shapes &= st['momentum_buffer'].shape==p.shape
        assert shapes
        absent=[name for name,p in model.named_parameters() if not opt.state.get(p)]
        optimizer_info.append(dict(key=key,state_entries=len(opt.state),parameters=sum(len(g['params']) for g in opt.param_groups),
                                   parameters_without_momentum=absent,
                                   absent_state_interpretation='SGD allocates momentum only on parameters that receive gradients; restored optimizer equals saved state, not fabricated missing buffers',
                                   momentum_shapes_match=True,learning_rates=[g['lr'] for g in opt.param_groups]))
    rng=runtime.rng_state(); runtime.restore_rng(ck['random_state']); restored=runtime.rng_state()
    assert restored['python_random']==ck['random_state']['python_random']
    assert np.array_equal(restored['numpy_random'][1],ck['random_state']['numpy_random'][1])
    assert torch.equal(restored['torch_random'],ck['random_state']['torch_random'])
    assert all(torch.equal(a,b) for a,b in zip(restored['torch_cuda_random'],ck['random_state']['torch_cuda_random']))
    runtime.restore_rng(rng)
    sampler=runtime.StatefulSampler(list(range(136)),list(range(136,1312)),24,12)
    runtime.set_sampler_resume(ck['sampler_state']); replay=list(iter(sampler)); state=ck['sampler_state']
    assert replay==[tuple(v) for v in state['batches'][state['consumed']:]]
    assert all(len(b)==24 and all(0<=v<136 for v in b[:12]) and all(136<=v<1312 for v in b[12:]) for b in replay)
    report=dict(time=now(),checkpoint=str(cp),sha256=sha(cp),global_step=ck['global_step'],epoch=ck['epoch'],
        checkpoint_keys=sorted(ck),strict_students_and_teacher_load=True,optimizers=optimizer_info,
        main_rng_restoration=True,sampler_plan_length=len(state['batches']),sampler_consumed=state['consumed'],
        replay_batches=len(replay),saved_ID_plan_matches=True,lr_state=ck['lr_state'],amp_state=ck['amp_state'],
        best_metrics=ck['best_metrics'],resume_exact=ck['resume_exact'],
        unverified_or_missing=['16 worker augmentation RNG and prefetched transformed tensors are not saved',
                              'bitwise identity after resume is not promised',
                              'no continuation training or optimizer step executed in this audit'],
        planned_resume_step=3977,remaining_budget_updates=30000-3976)
    atomic_json(out/'E1_checkpoint_restoration_check.json',report)
    del models,ck; torch.cuda.empty_cache()
    return report

def ctw_diagnostic(out,data):
    cfg=read_json(WORK/'configs/E1_uniform.yaml'); cp=WORK/'runs/E1/checkpoints/last.ckpt'
    a,ck=load_model(cfg,cp,'model_1'); b,_=load_model(cfg,cp,'model_2'); t,_=load_model(cfg,cp,'ema_model')
    rr=data['labeled_train'][:12]; x=torch.stack([v['input'] for v in rr]).cuda(); xs=torch.stack([v['strong'] for v in rr]).cuda()
    with torch.inference_mode():
        la=t(x[:6]); lb=t(x[6:]); np.random.seed(4261300)
        m,_,_=parent.generate_adamix_bcp_lite_v1_mask(x[:6],la,lb); m=torch.cat([m,m])[:,None]
        donor=list(range(6,12))+list(range(6))
        mx=x*m+x[donor]*(1-m); ms=xs*m+xs[donor]*(1-m)
        o1=a(mx); o2=b(ms)
        raw,norm,w,u,e=runtime.ctw_maps(o1,o2,True)
    rows=[]
    for i,v in enumerate(rr):
        spatial=float((w[i]*e[i]).mean()); uniform=float((u[i]*e[i]).mean())
        rows.append(dict(run='E1',step=3976,checkpoint_sha256=sha(cp),sample_id=v['id'],donor_id=rr[donor[i]]['id'],mode='offline eval weak/strong mixed labeled inputs',
            raw_weight_min=float(w[i].min()),raw_weight_max=float(w[i].max()),raw_weight_mean=float(w[i].mean()),raw_weight_spatial_std=float(w[i].std(unbiased=False)),
            uniform_weight_min=float(u[i].min()),uniform_weight_max=float(u[i].max()),uniform_weight_spatial_std=float(u[i].std(unbiased=False)),
            mean_absolute_weight_change=float((w[i]-u[i]).abs().mean()),mean_weight_match_error=float((w[i].mean()-u[i].mean()).abs()),
            original_ctw_loss=spatial,uniform_ctw_loss=uniform,spatial_minus_uniform=spatial-uniform,
            identity='difference equals spatial covariance(weight, L1); equal mean weights do not imply equal loss or gradients'))
    write_csv(out/'CTW_actual_difference.csv',rows)
    fig,axs=plt.subplots(1,2,figsize=(9,3),layout='constrained')
    axs[0].bar(range(12),[r['raw_weight_spatial_std'] for r in rows],color='#0072B2',label='Original spatial CTW')
    axs[0].set(xlabel='Fixed labeled diagnostic sample',ylabel='Within-image weight SD'); axs[0].legend(fontsize=8)
    axs[1].scatter([r['original_ctw_loss'] for r in rows],[r['uniform_ctw_loss'] for r in rows],color='#D55E00')
    lim=max(r[k] for r in rows for k in ('original_ctw_loss','uniform_ctw_loss')); axs[1].plot([0,lim],[0,lim],c='gray',ls='--',lw=.8)
    axs[1].set(xlabel='Spatial CTW loss',ylabel='Uniform CTW loss')
    fig.suptitle('E1 step3976 | read-only same-logit CTW comparison')
    dumpfig(fig,out,'CTW_weight_and_loss_comparison')
    del a,b,t,ck; torch.cuda.empty_cache()
    return rows

def figures(out,rows,pooled,hist,distribution):
    fig,axs=plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for run,step,col in [('Historical_Full',29600,'#0072B2'),('Historical_Full',30000,'#56B4E9'),('E1',3200,'#E69F00'),('E1',3976,'#D55E00')]:
        hh=[r for r in hist if r['run']==run and r['checkpoint_step']==step and r['mode']=='train_snapshot' and r['cohort']=='unlabeled_train' and r['region']=='all' and r['statistic']=='maximum class probability']
        axs[0,0].plot([(r['bin_left']+r['bin_right'])/2 for r in hh],[r['count']/r['denominator'] for r in hh],color=col,label=f'{run} {step}')
        dd=[r for r in rows if r.get('scope')=='sample' and r['region']=='all' and r['run']==run and r['checkpoint_step']==step and r['mode']=='train_snapshot' and r['cohort']=='validation']
        axs[1,0].scatter([r['confidence_mean'] for r in dd],[r['fg_macro_dice'] for r in dd],s=15,color=col,label=f'{run} {step}',alpha=.7)
    axs[0,0].set(xlim=(.25,1),xlabel='Maximum softmax probability',ylabel='Pixel fraction per 0.01 bin',title='Fixed unlabeled training batch distribution')
    axs[0,0].set_yscale('log'); axs[0,0].set_ylim(1e-5,1.1); axs[0,0].legend(fontsize=7)
    pp=[r for r in pooled if r['cohort']=='unlabeled_train' and r['mode']=='train_snapshot' and r['region']=='all']
    labels=[]
    for i,r in enumerate(pp):
        match=[v for v in pooled if v['checkpoint_sha256']==r['checkpoint_sha256'] and v['mode']==r['mode'] and v['cohort']==r['cohort']]
        bg=next(v for v in match if v['region']=='pred_bg'); fg=next(v for v in match if v['region']=='pred_fg')
        axs[0,1].bar(i,bg['confidence_contribution'],color='#0072B2',label='Predicted background' if i==0 else None)
        axs[0,1].bar(i,fg['confidence_contribution'],bottom=bg['confidence_contribution'],color='#E69F00',label='Predicted foreground' if i==0 else None)
        labels.append(f"{r['run']}\n{r['checkpoint_step']}")
    axs[0,1].set(xticks=range(len(labels)),xticklabels=labels,ylabel='Contribution to mean max probability',title='Confidence decomposition by predicted region')
    axs[0,1].legend(fontsize=7)
    axs[1,0].set(xlabel='Per-slice mean max probability',ylabel='All-3 foreground macro Dice',title='Fixed validation middle slices; offline teacher')
    axs[1,0].legend(fontsize=7)
    q=np.linspace(.95,1,1001); sides=np.array([round(256*(.18+.2*v)) for v in q])
    axs[1,1].step(q,sides,where='mid',color='#009E73'); axs[1,1].axvline(.984765625,c='#CC79A7',ls='--',label='97 starts above q=0.984765625')
    for r in distribution:
        y=97.15 if r['run']=='Historical_Full' else 97.3
        axs[1,1].plot([r['q_min'],r['q_max']],[y,y],lw=3,label=f"{r['run']} recorded q range")
    axs[1,1].set(xlabel='Batch mean max probability q',ylabel='Rounded rectangle side (pixels)',title='Current quantization near high confidence')
    axs[1,1].legend(fontsize=6)
    fig.suptitle('TCSMix mechanism diagnostic | fixed inputs, no training, no test access')
    dumpfig(fig,out,'confidence_and_quantization')

def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',required=True); arg=parser.parse_args()
    out=Path(arg.output).resolve(); out.mkdir(parents=True,exist_ok=False); (out/'sources').mkdir()
    assert (WORK/'state/stop_requested').exists() and read_json(WORK/'state/queue.json')['status']=='INTERRUPTED'
    gate=read_json(WORK/'state/preflight.json')
    locked={**gate['script_sha256'],**gate.get('shell_sha256',{}),**gate.get('source_sha256',{}),**gate['config_sha256']}
    tracked={str(WORK/k):v for k,v in locked.items()}
    for p in [FULL/'train.log',FULL/'effective_source.py',FULL/'checkpoints/best.ckpt',FULL/'checkpoints/last.ckpt',
              WORK/'runs/E1/checkpoints/last.ckpt',WORK/'runs/E1/checkpoints/best.ckpt',WORK/'runs/E1/audit/revision_trace.csv',
              WORK/'runs/E1/TRAIN_RECEIPT.json',WORK/'state/queue.json',WORK/'state/stop_requested']:
        tracked[str(p)]=sha(p)
    assert all(sha(p)==h for p,h in tracked.items())
    atomic_json(out/'input_hashes_before.json',tracked)
    distribution,_,_=history(out)
    data=samples(out)
    start=time.monotonic(); rows,pooled,hist,comparisons=diagnostics(out,data)
    restore=restore_check(out); ctw=ctw_diagnostic(out,data); figures(out,rows,pooled,hist,distribution)
    after={p:sha(p) for p in tracked}; assert after==tracked
    atomic_json(out/'input_hashes_after.json',after)
    atomic_json(out/'DIAGNOSTIC_RECEIPT.json',dict(time=now(),status='DONE',device=torch.cuda.get_device_name(0),
        diagnostic_seconds=time.monotonic()-start,parameter_updates=0,optimizer_steps=0,test_access=False,
        queue_status=read_json(WORK/'state/queue.json')['status'],stop_requested=True,
        E1_resume_executed=False,locked_files_unchanged=len(locked),all_tracked_inputs_unchanged=True,
        E1_step=restore['global_step'],confidence_rows=len(rows),offline_batches=len(comparisons),
        offline_dynamic_sides=sorted({r['dynamic_h'] for r in comparisons}),
        offline_dynamic_fixed_masks_equal=sum(r['E1_E2_masks_identical'] for r in comparisons),
        E2_E3_status='PENDING, not launched',script=str(Path(__file__).resolve()),script_sha256=sha(__file__)))
    print(json.dumps(dict(output=str(out),status='DONE',seconds=time.monotonic()-start),ensure_ascii=False),flush=True)

if __name__=='__main__':
    torch.set_num_threads(1)
    main()
