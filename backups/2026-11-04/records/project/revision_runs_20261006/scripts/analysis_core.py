"""Data-derived A1-A6 evidence, with raw tables, provenance and honest status."""
import argparse
import json
import math
import re
from pathlib import Path
import h5py
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from scipy.ndimage import binary_dilation, binary_erosion, zoom
from scipy.stats import spearmanr
from common import *

PALETTE=['#000000','#E69F00','#009E73','#56B4E9']
plt.rcParams.update({'font.size':9,'axes.spines.top':False,'axes.spines.right':False,'svg.fonttype':'none','figure.dpi':120})

def savefig(fig,base):
    base=Path(base); base.parent.mkdir(parents=True,exist_ok=True)
    fig.savefig(base.with_suffix('.png'),dpi=180,bbox_inches='tight')
    fig.savefig(base.with_suffix('.svg'),bbox_inches='tight')
    atomic_json(base.with_suffix('.provenance.json'),dict(figure=str(base),analysis=base.parent.name,
                identity_tables=['confidence_proxy.csv','spatial_response.csv','selection_and_provenance.csv','../A4/paired_patient_differences.csv','../../case_metrics.csv'],
                run_case_checkpoint_scope='See the applicable table in this analysis directory; multiple checkpoints are never treated as independent patients',time=now()))
    plt.close(fig)

def available_configs():
    configs=[read_json(WORK/f'configs/{task}.json') for task in ('Historical_baseline','Historical_Full')]
    for i in range(1,6):
        _,cfg=config_for(f'E{i}')
        cfg['checkpoint']=str(Path(cfg['run_root'])/'checkpoints/best.ckpt')
        if Path(cfg['checkpoint']).exists(): configs.append(cfg)
    return configs

def diag_ids():
    lock=WORK/'configs/diagnostic_cases.json'
    if not lock.exists():
        protocol=read_json(WORK/'protocol.json')
        all_ids=(DATA/'train_slices.list').read_text().splitlines()[:136]
        samples=[]
        for patient in protocol['labeled_patients']:
            ids=sorted(s for s in all_ids if s.startswith(patient+'_'))
            # Fixed index quartiles; no prediction quality is used.
            samples += [ids[len(ids)//4],ids[3*len(ids)//4]]
        atomic_json(lock,dict(rule='two fixed index quartiles of each of the seven already labeled patients',
                    slices=samples,test_access=False,qualitative_rule='largest/median/smallest patient Dice difference; frame01 if available; middle foreground-bearing slice'))
    return read_json(lock)['slices']

def completed_training(tasks):
    for task in tasks:
        path=WORK/f'runs/{task}/TRAIN_RECEIPT.json'
        if not path.exists(): return False
        receipt=read_json(path)
        if receipt['status']!='TRAINED' or receipt['global_step']!=30000: return False
    return True

def A1():
    out=WORK/'analysis/A1'; out.mkdir(parents=True,exist_ok=True)
    series={'Historical_Full (200-step sparse log)':read_table(out/'historical_full_scale.csv')}
    for cfg in available_configs():
        trace=WORK/'runs'/cfg['task']/'audit/revision_trace.csv'
        if trace.exists(): series[cfg['task']]=read_table(trace)
    for name,rows in series.items():
        if not rows: continue
        keys=['q','s','actual_area' if 'actual_area' in rows[0] else 'area','lr','total_loss','ctw_loss']
        fig,axes=plt.subplots(3,2,figsize=(8,7),layout='constrained')
        for ax,key in zip(axes.flat,keys):
            if key in rows[0]:
                ax.plot([int(r['step']) for r in rows],[float(r[key]) for r in rows],lw=.8)
                ax.set(xlabel='Update',ylabel=key)
            else:
                ax.text(.5,.5,'NA: not recorded in this run',ha='center',transform=ax.transAxes); ax.axis('off')
        fig.suptitle(name)
        savefig(fig,out/re.sub(r'[^a-zA-Z0-9_]','_',name))
    full_trace=series['Historical_Full (200-step sparse log)']
    stats=dict(samples=len(full_trace),s_min=min(float(x['s']) for x in full_trace),s_max=max(float(x['s']) for x in full_trace),
               distinct_actual_areas=len({x['area'] for x in full_trace}),coverage='every 200 updates, not all steps')
    atomic_json(out/'historical_statistics.json',stats)
    done=completed_training([f'E{i}' for i in range(1,6)])
    analysis_status('A1','DONE' if done else 'PARTIAL','Historical sparse trace drawn; full-step traces added from new runs. Missing runs remain DEFERRED.',runs=list(series),statistics=stats)

def A2():
    import torch
    from evaluate import load_model
    from dataloaders.dataset import WeakStrongAugment
    out=WORK/'analysis/A2'; out.mkdir(parents=True,exist_ok=True)
    ids=diag_ids(); rows=[]
    for cfg in available_configs():
        if cfg['model_family']=='official_adamix_ct': continue
        base=Path(cfg['checkpoint']).parent
        for cp in [base/'best.ckpt',base/'last.ckpt']:
            if not cp.exists(): continue
            model,ck=load_model(cfg,cp,'ema_model')
            checkpoint_hash=sha(cp)
            torch.manual_seed(42002); np.random.seed(42002)
            import random
            random.seed(42002)
            augment=WeakStrongAugment([256,256])
            for case in ids:
                with h5py.File(DATA/'data/slices'/f'{case}.h5','r') as f:
                    sample=augment(dict(image=f['image'][:],label=f['label'][:]))
                # Teacher reference weak input. Offline eval mode is explicit.
                x=sample['image'][None].cuda(); gt=sample['label'].numpy()
                with torch.no_grad(): probs=model(x).softmax(1)[0]; conf=probs.max(0).values.cpu().numpy(); pred=probs.argmax(0).cpu().numpy()
                scores=[]
                for cls in range(1,4):
                    denom=(pred==cls).sum()+(gt==cls).sum()
                    scores.append(2*((pred==cls)&(gt==cls)).sum()/denom if denom else 0.)
                fg=gt>0; bg=~fg
                rows.append(dict(run=cfg['task'],checkpoint=str(cp),checkpoint_sha256=checkpoint_hash,step=ck['global_step'],
                    slice=case,patient=case.split('_')[0],pixels=gt.size,mode='teacher eval, fixed weak augmentation seed42002',
                    q=float(conf.mean()),foreground_confidence=float(conf[fg].mean()) if fg.any() else '',
                    background_confidence=float(conf[bg].mean()) if bg.any() else '',foreground_fraction=float(fg.mean()),foreground_macro_dice=float(np.mean(scores))))
                rows[-1].update(foreground_present_classes=sum((gt==c).any() for c in range(1,4)),all_foreground_empty=not bool(fg.any()),
                                dice_definition='all3 foreground classes, double-empty class Dice0; fixed labeled slice diagnostic, not whole-volume score')
            del model; torch.cuda.empty_cache()
    write_csv(out/'confidence_proxy.csv',rows)
    correlations=[]
    for run in sorted({x['run'] for x in rows}):
        data=[x for x in rows if x['run']==run]
        stat=spearmanr([x['q'] for x in data],[x['foreground_macro_dice'] for x in data])
        correlations.append(dict(run=run,observations=len(data),patients=len({x['patient'] for x in data}),
                                 spearman=float(stat.statistic) if np.isfinite(stat.statistic) else '',
                                 interpretation='descriptive repeated checkpoints/slices, no independent-sample test or causal conclusion'))
    write_csv(out/'descriptive_correlations.csv',correlations)
    fig,ax=plt.subplots(figsize=(5,4),layout='constrained')
    for run in sorted({x['run'] for x in rows}):
        d=[x for x in rows if x['run']==run]
        ax.scatter([x['q'] for x in d],[x['foreground_macro_dice'] for x in d],s=15,label=run,alpha=.6)
    ax.set(xlabel='Teacher mean max probability',ylabel='Foreground macro Dice'); ax.legend(fontsize=6)
    savefig(fig,out/'confidence_vs_dice')
    analysis_status('A2','DONE' if completed_training(['E1','E2','E3','E4']) else 'PARTIAL','Existing teacher best/last checkpoints on 14 fixed already-labeled slices; new checkpoints DEFERRED. E5 has no EMA teacher by official definition.',samples=14,correlations=correlations)

def boundary(x):
    result=np.zeros(x.shape,bool)
    result[1:,:]|=x[1:,:]!=x[:-1,:]; result[:,1:]|=x[:,1:]!=x[:,:-1]
    return result

def A3():
    import torch,random
    from evaluate import load_model
    from dataloaders.dataset import WeakStrongAugment
    import revision_runtime as r
    out=WORK/'analysis/A3'; out.mkdir(parents=True,exist_ok=True)
    samples=[]; ids=diag_ids()[:4]
    augment=WeakStrongAugment([256,256]); rows=[]
    random.seed(42003); np.random.seed(42003); torch.manual_seed(42003)
    for case in ids:
        with h5py.File(DATA/'data/slices'/f'{case}.h5','r') as f:
            samples.append(augment(dict(image=f['image'][:],label=f['label'][:])))
    weak=torch.stack([s['image'] for s in samples]).cuda(); strong=torch.stack([s['image_strong'] for s in samples]).cuda()
    truth=np.stack([s['label'].numpy() for s in samples])
    for cfg in available_configs():
        if cfg['trc']=='off': continue
        model1,ck=load_model(cfg,branch='model_1'); model2,_=load_model(cfg,branch='model_2'); teacher,_=load_model(cfg,branch='ema_model')
        cfg=dict(cfg,weight_mode=cfg.get('weight_mode','spatial'),mix_mode=cfg.get('mix_mode','dynamic'))
        r.configure(cfg); np.random.seed(42003)
        with torch.no_grad():
            logits=teacher(weak)
            mask,_,stats=r.generate_adamix_bcp_lite_v1_mask(weak[:2],logits[:2],logits[2:])
            order=[2,3,0,1]
            m=torch.cat([mask,mask]); mix=weak*m[:,None]+weak[order]*(1-m[:,None]); mix_s=strong*m[:,None]+strong[order]*(1-m[:,None])
            out1=model1(mix); out2=model2(mix_s)
            raw,norm,w,used,error=r.ctw_maps(out1,out2,cfg['weight_mode']=='uniform')
        mixed_gt=truth*m.cpu().numpy()+truth[order]*(1-m.cpu().numpy())
        pred=out1.argmax(1).cpu().numpy(); raw=raw.cpu().numpy(); norm=norm.cpu().numpy(); w=w.cpu().numpy(); used=used.cpu().numpy()
        for index in range(4):
            gt_b=boundary(mixed_gt[index]); mix_b=boundary(m[index].cpu().numpy())
            near_gt=binary_dilation(gt_b,iterations=2); near_mix=binary_dilation(mix_b,iterations=2)
            regions={'gt_only':near_gt&~near_mix,'mix_only':near_mix&~near_gt,'overlap':near_gt&near_mix,'interior':~(near_gt|near_mix)}
            for region,pixels in regions.items():
                rows.append(dict(run=cfg['task'],checkpoint=str(cfg['checkpoint']),checkpoint_step=ck['global_step'],checkpoint_sha256=sha(cfg['checkpoint']),
                                 source_slice=ids[index],donor_slice=ids[order[index]],region=region,pixels=int(pixels.sum()),
                                 raw_weight_mean=float(w[index][pixels].mean()) if pixels.any() else '',
                                 used_weight_mean=float(used[index][pixels].mean()) if pixels.any() else '',
                                 q=stats['teacher_conf'],s=stats['side_fraction'],area=stats['patch_area_ratio']))
            fig,axes=plt.subplots(2,3,figsize=(8,5),layout='constrained')
            panels=[(raw[index],'Raw confidence difference','magma'),(norm[index],'Normalized response','viridis'),
                    (w[index],'Original CTW weight','viridis'),(m[index].cpu().numpy(),'Shared mix mask','gray'),
                    (gt_b,'GT boundary (mixed labeled sources)','gray'),(pred[index]!=mixed_gt[index],'Prediction error vs weak GT','gray')]
            for ax,(data,title,cmap) in zip(axes.flat,panels):
                im=ax.imshow(data,cmap=cmap); ax.set_title(title,fontsize=8); ax.axis('off')
                if title in ('Raw confidence difference','Normalized response','Original CTW weight'): fig.colorbar(im,ax=ax,fraction=.04)
            fig.suptitle(f"{cfg['task']} | ckpt {sha(cfg['checkpoint'])[:12]} | step {ck['global_step']}\n{ids[index]} + {ids[order[index]]}",fontsize=8)
            savefig(fig,out/f"{cfg['task']}_{ids[index]}")
        del model1,model2,teacher; torch.cuda.empty_cache()
    write_csv(out/'spatial_response.csv',rows)
    analysis_status('A3','DONE' if completed_training(['E1','E2','E3','E4']) else 'PARTIAL',
        'True two-student/EMA outputs on four fixed already-labeled slices mixed offline; 2-pixel neighborhoods separated into GT-only/mix-only/overlap/interior. No anatomical semantics assigned to tokens; normalization denominator floor 1e-6.',
        diagnostic_only=True,test_access=False)

def A4():
    out=WORK/'analysis/A4'; out.mkdir(parents=True,exist_ok=True)
    raw=[]
    for cfg in available_configs():
        path=WORK/'runs'/cfg['task']/'evaluation/case_class_metrics.csv'
        if path.exists(): raw+=read_table(path)
    write_csv(WORK/'case_metrics.csv',raw,['run','checkpoint','case','patient','class_id','class_name','unit','dice','iou','hd95','asd'])
    grouped={}
    for row in raw:
        grouped.setdefault((row['run'],row['patient']),[]).append(row)
    patients=[]
    for (run,patient),data in grouped.items():
        patients.append(dict(run=run,patient=patient,frame_volumes=len({x['case'] for x in data}),
                             **{k:float(np.mean([float(x[k]) for x in data])) for k in ('dice','iou','hd95','asd')}))
    write_csv(out/'patient_metrics.csv',patients)
    mapping={(r['run'],r['patient']):r for r in patients}; diffs=[]
    for r in patients:
        for reference in ('Historical_baseline','Historical_Full'):
            b=mapping.get((reference,r['patient']))
            if b and r['run']!=reference:
                diffs.append(dict(run=r['run'],reference=reference,patient=r['patient'],
                                  **{k+'_difference':r[k]-b[k] for k in ('dice','iou','hd95','asd')}))
    write_csv(out/'paired_patient_differences.csv',diffs)
    # COVID-19 four-patient Full evidence, retain each patient rather than pool slices.
    covid=[]
    for task in ('COVID5_FINAL3','COVID10_FINAL3'):
        source=PROJECT/f'final_report/three_innovation_postfull_test_v22_1/raw_test/{task}/patient_metrics.json'
        if source.exists():
            payload=read_json(source)
            for patient,data in payload.get('per_patient',{}).items(): covid.append(dict(run=task,patient=patient,source=str(source),**data))
    for percent in (5,10):
        source=PROJECT/f'formal_test/covid19_seed1337/COVID19_{percent}P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json'
        if source.exists():
            payload=read_json(source)
            for patient,data in payload['metric_result']['per_patient'].items():
                covid.append(dict(run=f'COVID{percent}_baseline',patient=patient,source=str(source),**data))
    write_csv(out/'covid_four_patient_full.csv',covid)
    covidmap={(x['run'],x['patient']):x for x in covid}
    paired_covid=[]
    for percent in (5,10):
        full=[x for x in covid if x['run']==f'COVID{percent}_FINAL3']
        for row in full:
            ref=covidmap.get((f'COVID{percent}_baseline',row['patient']))
            if ref:
                paired_covid.append(dict(ratio=f'{percent}%',patient=row['patient'],baseline_dice=ref['dice'],full_dice=row['dice'],
                                         dice_difference_pp=100*(row['dice']-ref['dice']),baseline_source=ref['source'],full_source=row['source']))
        if full:
            fig,ax=plt.subplots(figsize=(7,3.5),layout='constrained')
            patients=[x['patient'] for x in full]; positions=np.arange(len(patients))
            ax.bar(positions-.18,[100*covidmap[(f'COVID{percent}_baseline',x)]['dice'] for x in patients],.36,label='Baseline',color='#0072B2')
            ax.bar(positions+.18,[100*covidmap[(f'COVID{percent}_FINAL3',x)]['dice'] for x in patients],.36,label='Full',color='#D55E00')
            ax.set_xticks(positions,patients,rotation=15,ha='right'); ax.set_ylabel('Patient-volume Dice (%)'); ax.set_title(f'COVID-19 {percent}%, four held-out patients'); ax.legend()
            savefig(fig,out/f'covid{percent}_four_patients')
    write_csv(out/'covid_patient_paired_differences.csv',paired_covid)
    corrected=WORK/'reports/covid_existing_prediction_reevaluation.csv'
    if corrected.exists():
        all_policies=read_table(corrected)
        write_csv(out/'covid_all_geometry_policies.csv',all_policies)
        native=[r for r in all_policies if r['policy']=='corrected_native_grid']
        index={(r['run'],r['patient']):r for r in native}
        native_pairs=[]
        for percent in (5,10):
            for patient in sorted({r['patient'] for r in native}):
                b=index[(f'COVID{percent}_baseline',patient)]; f=index[(f'COVID{percent}_Full',patient)]
                native_pairs.append(dict(ratio=f'{percent}%',patient=patient,policy='corrected_native_grid',unit='mm',
                                    baseline_dice=b['dice'],full_dice=f['dice'],dice_difference_pp=100*(float(f['dice'])-float(b['dice'])),
                                    baseline_hd95_mm=b['hd95_mm'],full_hd95_mm=f['hd95_mm'],baseline_asd_mm=b['asd_mm'],full_asd_mm=f['asd_mm'],
                                    baseline_checkpoint_sha256=b['checkpoint_sha256'],full_checkpoint_sha256=f['checkpoint_sha256']))
        write_csv(out/'covid_native_grid_patient_differences.csv',native_pairs)
    if diffs:
        fig,ax=plt.subplots(figsize=(8,4),layout='constrained')
        d=[x for x in diffs if x['reference']=='Historical_Full']
        for run in sorted({x['run'] for x in d}):
            r=[x for x in d if x['run']==run]
            ax.plot([x['patient'] for x in r],[100*x['dice_difference'] for x in r],'.',label=run)
        ax.axhline(0,c='gray',lw=.6); ax.tick_params(axis='x',rotation=90); ax.set_ylabel('Patient Dice difference (pp)'); ax.legend(fontsize=7)
        savefig(fig,out/'patient_differences')
    analysis_status('A4','DONE' if len({r['run'] for r in raw})==7 else 'PARTIAL',
        'ACDC frame/class raw records aggregated by patient for paired differences; COVID shown individually with n=4 per run. No slicing/checkpoint pseudo-replication.',patients=len({r['patient'] for r in raw}),covid_records=len(covid))

def A5():
    import torch,time
    from evaluate import load_model
    out=WORK/'analysis/A5'; out.mkdir(parents=True,exist_ok=True)
    rows=[]
    for cfg in available_configs():
        model,ck=load_model(cfg)
        count=sum(p.numel() for p in model.parameters())
        x=torch.zeros((1,1,256,256),device='cuda'); mac=[0]
        def hook(module,args,output):
            mac[0]+=output.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
        hooks=[m.register_forward_hook(hook) for m in model.modules() if isinstance(m,torch.nn.Conv2d)]
        with torch.no_grad(): model(x)
        for h in hooks: h.remove()
        router=getattr(getattr(model,'decoder',None),'mask_token_router',None)
        if router is not None and hasattr(router,'tokens'): mac[0]+=2*4*16*256*256
        for batch in (1,24):
            x=torch.zeros((batch,1,256,256),device='cuda')
            with torch.inference_mode():
                for _ in range(10): model(x)
                torch.cuda.synchronize(); torch.cuda.reset_peak_memory_stats(); timings=[]
                for _ in range(30):
                    start=time.perf_counter(); model(x); torch.cuda.synchronize(); timings.append((time.perf_counter()-start)*1000)
            receipt=WORK/'runs'/cfg['task']/'TRAIN_RECEIPT.json'
            progress=WORK/'runs'/cfg['task']/'progress.json'
            train=read_json(receipt) if receipt.exists() else {}
            rows.append(dict(run=cfg['task'],checkpoint=str(cfg['checkpoint']),checkpoint_sha256=sha(cfg['checkpoint']),device=torch.cuda.get_device_name(),
                precision='FP32',batch=batch,input='Bx1x256x256',deployed_student1_parameters=count,
                training_model_parameters=count*(2 if cfg['task']=='E5' else 3),student1_MACs=mac[0]*batch,
                MAC_convention='Conv2d and two token einsums; excludes norm/activation/softmax/pooling/resize; FLOPs=2MAC only for counted operations',
                warmup=10,repeats=30,inference_ms_mean=float(np.mean(timings)),inference_ms_sd=float(np.std(timings)),peak_inference_cuda_bytes=torch.cuda.max_memory_allocated(),
                actual_training_seconds=train.get('training_seconds','NA'),
                actual_peak_training_cuda_bytes=read_json(progress).get('peak_cuda_bytes','NA') if progress.exists() else 'NA',
                training_scope='native official two students' if cfg['task']=='E5' else 'two students + EMA teacher; mixing/CTW training only'))
            write_csv(out/f"{cfg['task']}_batch{batch}_raw_timing.csv",[dict(repeat=i,ms=t) for i,t in enumerate(timings)])
        del model; torch.cuda.empty_cache()
    write_csv(out/'cost.csv',rows)
    analysis_status('A5','DONE' if completed_training([f'E{i}' for i in range(1,6)]) else 'PARTIAL','Same A100/FP32/shapes; CUDA synchronized, warmup10 + timing30; repeats are timing only. Training wall/peak from new runs; historical training wall source audited separately.')

def A6():
    out=WORK/'analysis/A6'; out.mkdir(parents=True,exist_ok=True)
    diag_ids()  # Freeze rule independent of predictions before figure selection.
    paired=WORK/'analysis/A4/paired_patient_differences.csv'
    if not paired.exists():
        analysis_status('A6','BLOCKED','A4 paired patient records missing'); return
    rows=read_table(paired); selected=[]
    for run in sorted({r['run'] for r in rows}):
        d=sorted([r for r in rows if r['run']==run and r['reference']=='Historical_baseline'],key=lambda r:(float(r['dice_difference']),r['patient']))
        if not d: continue
        for label,index in [('smallest_gain_or_worsening',0),('typical_median',len(d)//2),('largest_gain',len(d)-1)]:
            patient=d[index]['patient']
            frames=sorted(c for c in (DATA/'test.list').read_text().splitlines() if c.startswith(patient+'_'))
            case=frames[0]
            with h5py.File(DATA/'data'/f'{case}.h5','r') as f: image=f['image'][:]; gt=f['label'][:]
            slices=np.flatnonzero((gt>0).sum((1,2))>0)
            z=int(slices[len(slices)//2]) if len(slices) else len(gt)//2
            panels=[(image[z],'Input','gray'),(gt[z],'GT',ListedColormap(PALETTE))]
            identity=[]
            for name in dict.fromkeys(['Historical_baseline','Historical_Full',run]):
                pred_path=WORK/f'runs/{name}/evaluation/predictions/{case}.npz'
                if not pred_path.exists(): continue
                prediction=np.load(pred_path)['prediction']
                receipt=read_json(pred_path.parent.parent/'EVALUATION_RECEIPT.json')
                identity.append(dict(run=name,checkpoint=receipt['checkpoint'],sha256=receipt['checkpoint_sha256']))
                panels.append((prediction[z],name+'\nckpt '+receipt['checkpoint_sha256'][:12],ListedColormap(PALETTE)))
            fig,axes=plt.subplots(1,len(panels),figsize=(2.1*len(panels),2.5),layout='constrained')
            for ax,(data,title,cmap) in zip(np.atleast_1d(axes),panels):
                ax.imshow(data,cmap=cmap,vmin=0 if title!='Input' else None,vmax=3 if title!='Input' else None); ax.set_title(title,fontsize=8); ax.axis('off')
            fig.suptitle(f'{label} | {case} | slice {z} | Dice diff {100*float(d[index]["dice_difference"]):+.3f} pp',fontsize=8)
            slug=f'{run}_{label}_{case}_slice{z}'; savefig(fig,out/slug)
            selected.append(dict(run=run,selection=label,case=case,slice=z,dice_difference=d[index]['dice_difference'],figure=slug,identities=json.dumps(identity)))
    write_csv(out/'selection_and_provenance.csv',selected)
    # All four COVID held-out patients; deterministic GT-based slice selection.
    covid_manifest=PROJECT/'data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv'
    covid_records=[]
    covid_rows=read_table(covid_manifest)
    for patient in sorted({r['patient_id'] for r in covid_rows}):
        path10=WORK/f'runs/COVID10_Full/evaluation/predictions/{patient}.npz'
        if not path10.exists(): continue
        rows_patient=sorted([r for r in covid_rows if r['patient_id']==patient],key=lambda r:int(r['slice_idx']))
        foreground=[r for r in rows_patient if int(r['foreground_pixels'])>0]
        selected_row=(foreground or rows_patient)[len(foreground or rows_patient)//2]
        z=int(selected_row['slice_idx']); p=Path(selected_row['derived_path']); p=p if p.is_absolute() else PROJECT/p
        payload=np.load(p)
        panels=[(payload['image'],'Input','gray'),(payload['mask'],'GT',ListedColormap([PALETTE[0],PALETTE[1]]))]
        identity=[]
        for percent in (5,10):
            base_dir=PROJECT/f'formal_test/covid19_seed1337/COVID19_{percent}P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE'
            receipt=read_json(base_dir/'FORMAL_TEST_RECEIPT.json')
            base=base_dir/f'predictions/{patient}_prediction.npz'
            full=WORK/f'runs/COVID{percent}_Full/evaluation/predictions/{patient}.npz'
            if not full.exists(): continue
            f=np.load(full); b=np.load(base)
            full_sha=str(f['checkpoint_sha256']); base_sha=receipt['checkpoint_sha256']
            panels += [(b['prediction'][z],f'Baseline {percent}%\nckpt {base_sha[:10]}',ListedColormap([PALETTE[0],PALETTE[1]])),
                       (f['prediction'][z],f'Full {percent}%\nckpt {full_sha[:10]}',ListedColormap([PALETTE[0],PALETTE[1]]))]
            identity.append(dict(ratio=f'{percent}%',baseline_checkpoint=receipt['checkpoint_path'],baseline_sha256=base_sha,
                                 full_checkpoint=str(f['checkpoint']),full_sha256=full_sha))
        fig,axes=plt.subplots(1,len(panels),figsize=(2*len(panels),2.8),layout='constrained')
        for ax,(data,title,cmap) in zip(axes,panels):
            ax.imshow(data,cmap=cmap,vmin=0 if title!='Input' else None,vmax=1 if title!='Input' else None); ax.set_title(title,fontsize=8); ax.axis('off')
        fig.suptitle(f'COVID-19 | {patient} | slice {z} | all four patients retained',fontsize=8)
        slug=f'COVID_all_four_{patient}_slice{z}'; savefig(fig,out/slug)
        covid_records.append(dict(patient=patient,slice=z,figure=slug,selection='all four test patients; middle foreground-bearing GT slice',identities=json.dumps(identity)))
    write_csv(out/'covid_all_four_selection.csv',covid_records,['patient','slice','figure','selection','identities'])
    analysis_status('A6','DONE' if len({r['run'] for r in selected})>=6 else 'PARTIAL',
        'Prespecified extreme/median patient gain selection, same frame/slice/display/colors; lowest gains retained even if adverse. Test used only for final descriptive display.',rule=read_json(WORK/'configs/diagnostic_cases.json')['qualitative_rule'])

def main(task):
    p=argparse.ArgumentParser(); p.add_argument('--pre',action='store_true'); p.parse_args()
    bootstrap(); globals()[task]()

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('task',choices=['A1','A2','A3','A4','A5','A6']); a=p.parse_args(); main(a.task)
