"""Same teacher prediction replay for old control and foreground-balanced A."""
import ast
import argparse
import collections
import hashlib
import json
import random
import time
from pathlib import Path
import h5py
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from scipy.stats import spearmanr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
bootstrap()
import revision_runtime as runtime
from dataloaders.dataset import WeakStrongAugment
from controller import signals,geometry,generate_mask

# Literal existing pseudo-label/LCC functions, not a new postprocessing rule.
body=(WORK/'source/historical_full.py').read_text()
ns=dict(torch=torch,np=np,F=F)
for node in ast.parse(body).body:
    if isinstance(node,ast.FunctionDef) and node.name in ['get_ACDC_2DLargestCC','get_ACDC_masks']:
        exec(ast.get_source_segment(body,node),ns)
pseudolabel=ns['get_ACDC_masks']

def model_digest(model):
    h=hashlib.sha256()
    for k,v in model.state_dict().items(): h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()

def samples():
    ids=(DATA/'train_slices.list').read_text().splitlines()
    prev=PROJECT/'revision_runs_20261006/configs/diagnostic_cases.json'
    names=read_json(prev)['slices']; random.seed(4261006);np.random.seed(4261006);torch.manual_seed(4261006)
    aug=WeakStrongAugment([256,256]); result=[]; manifest=[]
    unlab=[ids[136+int(i)] for i in np.linspace(0,len(ids)-137,24).round().astype(int)]
    # Add fixed extreme-index slices, without consulting unlabeled GT.
    patients=sorted({s.split('_')[0] for s in ids[136:]})[:6]
    edge=[]
    for patient in patients:
        pp=sorted([v for v in ids[136:] if v.startswith(patient+'_')],key=lambda s:(s.split('_')[1],int(s.rsplit('_',1)[1])))
        edge.extend([pp[0],pp[-1]])
    for cohort,ll in [('unlabeled_train',unlab),('unlabeled_edge',edge),('labeled_train',names)]:
        for name in ll:
            p=DATA/'data/slices'/f'{name}.h5'
            with h5py.File(p,'r') as f:
                image=f['image'][:]
                # Unlabeled true labels never loaded; dummy augmentation label is discarded.
                gt=f['label'][:] if cohort=='labeled_train' else np.zeros(image.shape,dtype=np.uint8)
            s=aug(dict(image=image,label=gt)); x=s['image']; label=s['label'].numpy() if cohort=='labeled_train' else None
            item=dict(id=name,patient=name.split('_')[0],x=x,gt=label,cohort=cohort)
            result.append(item);manifest.append(dict(sample=name,cohort=cohort,patient=item['patient'],source=str(p),sha256=sha(p),
                input_sha256=hashlib.sha256(x.numpy().tobytes()).hexdigest(),GT_used=label is not None,selection='fixed index-spaced / extreme slice indices / prior locked labeled quartiles'))
    for name in (DATA/'val.list').read_text().splitlines():
        p=DATA/'data'/f'{name}.h5'
        with h5py.File(p,'r') as f: image=f['image'][:];gt=f['label'][:]
        z=image.shape[0]//2;h,w=image[z].shape;x=zoom(image[z],(256/h,256/w),order=0).astype(np.float32);y=zoom(gt[z],(256/h,256/w),order=0)
        item=dict(id=f'{name}:z{z}',patient=name.split('_')[0],x=torch.from_numpy(x)[None],gt=y,cohort='validation');result.append(item)
        manifest.append(dict(sample=item['id'],cohort='validation',patient=item['patient'],source=str(p),sha256=sha(p),input_sha256=hashlib.sha256(x.tobytes()).hexdigest(),GT_used=True,selection='middle z=Nz//2 of each val frame'))
    write_csv(WORK/'analysis/diagnostic_samples.csv',manifest)
    return result

def scores(pred,gt):
    ds=[];present=[]
    for c in [1,2,3]:
        den=(pred==c).sum()+(gt==c).sum(); d=float(2*((pred==c)&(gt==c)).sum()/den) if den else 0.
        ds.append(d)
        if (gt==c).any():present.append(d)
    return dict(dice_RV=ds[0],dice_MYO=ds[1],dice_LV=ds[2],macro_dice=np.mean(ds),gt_present_macro_dice=np.mean(present) if present else 'NA:no GT foreground')

def row(st,run,cp,ch,step,mode,cohort,scope,samples_,pred=None,gt=None):
    rr=dict(run=run,checkpoint=str(cp),checkpoint_sha256=ch,checkpoint_step=step,mode=mode,cohort=cohort,scope=scope,
        samples=';'.join(s['id'] for s in samples_),n_samples=len(samples_),patients=';'.join(sorted({s['patient'] for s in samples_})),
        n_patients=len({s['patient'] for s in samples_}),old_q=st['old_q'],q_A=st['q_fg'],q_B=st['q_margin'],fallback=st['fallback'],
        active_classes=st['active_classes'],foreground_pixels=st['foreground_pixels'],pixels=st['pixels'],
        predicted_background_fraction=st['predicted_background_fraction'],background_confidence=st['background_confidence'])
    for c in [1,2,3]: rr.update({f'n_{c}':st['foreground_counts'][str(c)],f'q_class{c}':st['q_by_class'][str(c)],f'margin_class{c}':st['margin_by_class'][str(c)]})
    for key,q in [('old',st['old_q']),('A',st['q_fg']),('B',st['q_margin'])]:
        s,h,w=geometry(q,256,256); rr.update({f'{key}_s':s,f'{key}_h':h,f'{key}_w':w,f'{key}_area':h*w/65536})
    if gt is not None:rr.update(scores(pred,gt))
    return rr

def main():
    parser=argparse.ArgumentParser();parser.add_argument('--compare-b',action='store_true');arg=parser.parse_args()
    assert arg.compare_b and (WORK/'reports/offline_A_initial/controller_selection.json').exists()
    samples_=samples(); groups={c:[s for s in samples_ if s['cohort']==c] for c in sorted({s['cohort'] for s in samples_})}
    checkpoints=[('initial_pretrain_reconstruction',Path(read_json(WORK/'protocol.json')['pretrain']),0,'initialization'),
                 ('E1_uniform',PROJECT/'revision_runs_20261006/runs/E1/checkpoints/best.ckpt',3200,'early'),
                 ('E1_uniform',PROJECT/'revision_runs_20261006/runs/E1/checkpoints/last.ckpt',3976,'early'),
                 ('Historical_Full',FULL/'checkpoints/best.ckpt',29600,'late'),('Historical_Full',FULL/'checkpoints/last.ckpt',30000,'late')]
    rows=[];identities=[];cfg=dict(model_family='beta_fft',trc='token',weight_mode='spatial',mix_mode='dynamic',run_root=str(WORK/'runs/offline_only'))
    mapping=read_json(WORK/'configs/ACDC_class_map.json'); fg=mapping['foreground_ids']; bg=mapping['background_id']
    for run,cp,step,phase in checkpoints:
        runtime.CFG=cfg; model=runtime.BCP_net(ema=True);ck=torch.load(cp,map_location='cpu',weights_only=False)
        if step==0:
            import v14_mask_adamix_components as parent
            parent.load_parent_pretrain(model,ck['net'],True)
            assert model.decoder.mask_token_router.gamma.item()==0
        else:model.load_state_dict(ck['ema_model'],strict=True)
        frozen={k:v.detach().clone() for k,v in model.state_dict().items()}; digest=model_digest(model);ch=sha(cp)
        identities.append(dict(run=run,checkpoint=str(cp),sha256=ch,step=step,phase=phase,
             scope='reconstructed SSL initial teacher; zero gate and exact historical pretrain, not saved teacher step0' if step==0 else 'existing post-step teacher; E1 is uniform CTW, not Full',
             middle_phase_available=False))
        for mode in ['eval','train_snapshot']:
            model.train(mode=='train_snapshot')
            for cohort,items in groups.items():
                for bi,start in enumerate(range(0,len(items),12)):
                    batch=items[start:start+12];assert len(batch)%2==0;n=len(batch)//2
                    model.load_state_dict(frozen);torch.manual_seed(4261100+bi);torch.cuda.manual_seed_all(4261100+bi)
                    x=torch.stack([s['x'] for s in batch]).cuda()
                    with torch.no_grad():
                        la=model(x[:n]);lb=model(x[n:]);ya=pseudolabel(la,nms=1);yb=pseudolabel(lb,nms=1)
                        st=signals(la,lb,ya,yb,fg,bg); pred=torch.cat([ya,yb]).cpu().numpy()
                        gt=np.stack([s['gt'] for s in batch]) if all(s['gt'] is not None for s in batch) else None
                        rows.append(row(st,run,cp,ch,step,mode,cohort,'batch',batch,pred,gt))
                        for i,item in enumerate(batch):
                            l=torch.cat([la,lb])[i:i+1];y=torch.cat([ya,yb])[i:i+1]
                            si=signals(l,l,y,y,fg,bg)
                            rows.append(row(si,run,cp,ch,step,mode,cohort,'single_sample_response',[item],pred[i],item['gt']))
                        # Directly verify actual mask areas and image/target/batch sharing for A.
                        mask,loss,ss=generate_mask(x[:n],la,lb,ya,yb,dict(controller_version='v2_A',foreground_ids=fg,background_id=bg))
                        assert torch.equal(mask,loss) and torch.equal(mask[0].expand_as(mask),mask)
                        assert int((mask[0]==0).sum())==ss['rectangle_h']*ss['rectangle_w']
                    model.load_state_dict(frozen)
        assert model_digest(model)==digest;identities[-1]['state_unchanged']=True
        print(f'offline {run} {step} complete',flush=True)
        del model,frozen,ck;torch.cuda.empty_cache()
    fields=list(dict.fromkeys(k for r in rows for k in r));write_csv(WORK/'control_diagnostics.csv',rows,fields)
    atomic_json(WORK/'reports/offline_checkpoint_inventory.json',identities)
    summary=[]
    for run,cp,step,phase in checkpoints:
        for mode in ['eval','train_snapshot']:
            for scope in ['batch','single_sample_response']:
                rr=[r for r in rows if r['run']==run and r['checkpoint_step']==step and r['mode']==mode and r['scope']==scope]
                for ver in ['old','A','B']:
                    counts=collections.Counter(r[ver+'_h'] for r in rr)
                    summary.append(dict(run=run,step=step,phase=phase,mode=mode,scope=scope,controller=ver,n=len(rr),n_sizes=len(counts),
                        size_counts=json.dumps(counts,sort_keys=True),most_common_fraction=max(counts.values())/len(rr),
                        size_min=min(counts),size_max=max(counts),q_min=min(r['old_q' if ver=='old' else 'q_'+ver] for r in rr),
                        q_max=max(r['old_q' if ver=='old' else 'q_'+ver] for r in rr),
                        min_endpoint_hits=sum(r[ver+'_h']==46 for r in rr),max_endpoint_hits=sum(r[ver+'_h']==97 for r in rr),
                        fallback_count=sum(r['fallback'] for r in rr) if ver!='old' else 0))
    write_csv(WORK/'analysis/offline_scale_distribution.csv',summary)
    # A is accepted by responsiveness/simplicity, never by segmentation/test performance.
    batch=[r for r in rows if r['scope']=='batch']; nonempty=[r for r in batch if not r['fallback']]
    paired=[]
    for run,cp,step,phase in checkpoints:
        rr=[r for r in batch if r['run']==run and r['checkpoint_step']==step and r['mode']=='train_snapshot']
        paired.append(dict(run=run,step=step,A_sizes=sorted({r['A_h'] for r in rr}),B_sizes=sorted({r['B_h'] for r in rr})))
    improved=[v for v in paired if len(v['A_sizes'])==1 and len(v['B_sizes'])>1]
    version='v2_B' if improved else 'v2_A'
    selection=dict(status='SELECTED',controller_version=version,formula='equal foreground-class mean of (P_c - max other P) on existing per-class LCC support' if improved else 'equal foreground-class mean of P_c on existing per-class LCC support',
        fallback=0.,mapping='unchanged .18+.20*q',B_evaluated=True,mapping_adjusted=False,
        B_trigger='A fixed train-snapshot batch geometry at E1 steps3200 and3976; one offline comparison permitted',
        basis='Use B only where actual same-checkpoint batch replay shows it resolves A remaining integer no-response; no GT/test score selection',
        same_checkpoint_response=paired,where_B_resolves_A_constant=improved,
        n_batches=len(batch),old_sizes=sorted({r['old_h'] for r in batch}),A_sizes=sorted({r['A_h'] for r in batch}),
        A_nonfallback_sizes=sorted({r['A_h'] for r in nonempty}),A_fallback_batches=sum(r['fallback'] for r in batch),
        B_sizes=sorted({r['B_h'] for r in batch}),
        formal_benefit='NA; replay uses historical teachers, not v2 training',test_access=False,time=now())
    assert len(selection['A_nonfallback_sizes'])>1
    atomic_json(WORK/'reports/controller_selection.json',selection)
    cor=[]
    for run,cp,step,phase in checkpoints:
        for mode in ['eval','train_snapshot']:
            for cohort in ['labeled_train','validation']:
                rr=[r for r in rows if r['run']==run and r['checkpoint_step']==step and r['mode']==mode and r['cohort']==cohort and r['scope']=='single_sample_response']
                for signal in ['old_q','q_A','q_B']:
                    rho=spearmanr([r[signal] for r in rr],[r['macro_dice'] for r in rr]).statistic
                    cor.append(dict(run=run,step=step,mode=mode,cohort=cohort,signal=signal,samples=len(rr),patients=len(set(p for r in rr for p in r['patients'].split(';'))),spearman=float(rho) if np.isfinite(rho) else 'NA',
                        interpretation='descriptive repeated slices/checkpoints; no independent-sample test, no selection by Dice'))
    write_csv(WORK/'analysis/control_quality_correlations.csv',cor)
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    fig,axs=plt.subplots(1,3,figsize=(12,3.7),layout='constrained')
    colors=['#0072B2','#56B4E9','#E69F00','#D55E00','#009E73']
    for (run,cp,step,phase),color in zip(checkpoints,colors):
        rr=[r for r in batch if r['run']==run and r['checkpoint_step']==step and r['mode']=='train_snapshot']
        qs='q_B' if version=='v2_B' else 'q_A';sz='B_h' if version=='v2_B' else 'A_h'
        axs[0].scatter([r['old_q'] for r in rr],[r[qs] for r in rr],c=color,s=20,label=f'{run} {step}')
        axs[1].scatter([r['predicted_background_fraction'] for r in rr],[r[sz] for r in rr],c=color,s=20)
        ss=[r for r in rows if r['run']==run and r['checkpoint_step']==step and r['mode']=='train_snapshot' and r['scope']=='single_sample_response' and r['cohort']=='validation']
        axs[2].scatter([r[qs] for r in ss],[r['macro_dice'] for r in ss],c=color,s=14,alpha=.7)
    axs[0].set(xlabel='Old all-pixel confidence',ylabel=f'{version} balanced foreground control');axs[0].legend(fontsize=6)
    axs[1].set(xlabel='Processed predicted background fraction',ylabel=f'{version} rectangle side (pixels)')
    axs[2].set(xlabel=f'{version} on fixed val middle slices',ylabel='Foreground macro Dice')
    fig.suptitle('Offline controller replay | original LCC supports | no test/no updates')
    for ext in ['png','svg']:fig.savefig(WORK/f'analysis/control_replay.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    atomic_json(WORK/'reports/offline_receipt.json',dict(status='DONE',time=now(),joint_updates=0,optimizer_step_calls=0,test_access=False,
        samples=len(samples_),batch_records=len(batch),rows=len(rows),checkpoint_count=len(checkpoints),
        modes=['eval','train_snapshot: buffers restored each batch'],support_source='literal frozen original get_ACDC_masks(nms=1)',
        controller_version=version,mid_training_checkpoint='NA: none retained for this Full; E1 early checkpoints belong to uniform CTW'))

if __name__=='__main__':torch.set_num_threads(1); main()
