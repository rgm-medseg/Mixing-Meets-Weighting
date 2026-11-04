"""Four frozen-loss conditions on aligned complementary mixed validation views."""
import argparse,ast,random
from collections import defaultdict
import h5py,numpy as np,torch
from common import *
from augmentation import recorded_aug,rng,restore,seed
from v21_3_acdc10_third_innovation_components import _norm01,_spatial_gradient,compute_v21_3_innovation_x

def pseudo(logits):
    # Literal main-Full per-class largest connected component pseudo-labels.
    from skimage.measure import label
    raw=logits.softmax(1).argmax(1).detach().cpu().numpy();result=np.zeros_like(raw)
    for i,a in enumerate(raw):
        for c in [1,2,3]:
            cc=label(a==c)
            if cc.max():result[i][cc==np.argmax(np.bincount(cc.flat)[1:])+1]=c
    return result

def prepare_inputs():
    planpath=W/'configs/joint_input_plan.json'
    if planpath.exists():return json.loads(planpath.read_text())
    samples=[];cache={}
    for frame in CFG['val_frames']:
        path=DATA/'data'/f'{frame}.h5'
        with h5py.File(path) as f:cache[frame]=(f['image'][:],f['label'][:])
        for z in range(len(cache[frame][0])):samples.append(dict(sample=f'{frame}:z{z}',patient=frame.split('_')[0],frame=frame,z=z,source=str(path)))
    samples.sort(key=lambda r:hashlib.sha256(('joint-fixed-v1:'+r['sample']).encode()).hexdigest())
    manifest=[];plans=[];compares=0
    for b in range((len(samples)+23)//24):
        slots=[]
        for slot in range(24):
            idx=(b*24+slot)%len(samples);r=samples[idx].copy();n=CFG['seed']+24*b+slot
            image,gt=(a[r['z']] for a in cache[r['frame']]);seed(n);before=rng()
            from dataloaders.dataset import WeakStrongAugment
            literal=WeakStrongAugment([256,256])(dict(image=image.copy(),label=gt.copy()));after=rng();restore(before)
            out,meta=recorded_aug(image,gt);after2=rng()
            assert all(torch.equal(literal[k],out[k]) for k in ['image','image_strong','label','label_strong'])
            assert after[0]==after2[0] and after[1][0]==after2[1][0] and np.array_equal(after[1][1],after2[1][1]) and after[1][2:]==after2[1][2:] and torch.equal(after[2],after2[2]);compares+=1
            p=W/'arrays/inputs'/f'b{b}_s{slot}.npz';p.parent.mkdir(exist_ok=True,parents=True)
            np.savez_compressed(p,weak=out['image'].numpy(),strong=out['image_strong'].numpy(),gt=meta['weak_true'].astype(np.uint8),anatomy=meta['anat'],geom_valid=meta['geom_valid'],occlusion=meta['occlusion'],strong_artificial_gt=out['label_strong'].numpy())
            r.update(batch=b,slot=slot,path=str(p),sha256=sha(p),seed=n,role=['img_a','img_b','uimg_a','uimg_b'][slot//6],geometry=meta['geometry'],cutout_rectangle=meta['cutout_rectangle'],color_jitter=meta.get('color_jitter'),repeat_exposure=(b*24+slot)>=len(samples))
            slots.append(r);manifest.append(r)
        plans.append(dict(batch=b,slots=slots,uy=((b*37+19)%997)/997,ux=((b*71+43)%991)/991))
    plan=dict(batches=plans,samples=len(samples),frames=len(CFG['val_frames']),patients=sorted({r['patient'] for r in samples}),exposures=len(manifest),repeat_exposures=len(manifest)-len(samples),figure_unit='batch0 pair0; fixed before outputs')
    js(planpath,plan);csvout(W/'results/joint_input_manifest.csv',manifest)
    js(W/'receipts/alignment.json',dict(status='PASS',literal_aug_comparisons=compares,outputs_and_rng_exact=True,shared_geometry=True,GT_use='offline statistics only',unique_slices=len(samples),exposures=len(manifest)))
    return plan

def regions(a):
    gv=a['geom_valid'];occ=a['occlusion'];cut=(occ|band(boundary(occ)))&gv
    seam=band(boundary(a['outside']))&gv&~cut
    anat=band(a['anatomy'])&gv&~cut&~seam
    other=gv&~cut&~seam&~anat
    rr=dict(all=np.ones(gv.shape,bool),valid=gv&~occ,anatomy=anat,seam=seam,cutout=cut,other_valid=other,invalid=~gv)
    assert np.all(sum(rr[k].astype(int) for k in ['anatomy','seam','cutout','other_valid','invalid'])==1)
    rr.update(pseudo_correct=a['pseudo_source']&(a['pseudo']==a['gt'])&gv&~occ,pseudo_error=a['pseudo_source']&(a['pseudo']!=a['gt'])&gv&~occ,
              RV=(a['gt']==1)&gv&~occ,MYO=(a['gt']==2)&gv&~occ,LV=(a['gt']==3)&gv&~occ,background=(a['gt']==0)&gv&~occ)
    return rr

def extract(meta,a,patients):
    rows=[];regs=regions(a)
    for pid in np.unique(a['patient_ids']):
        pm=a['patient_ids']==pid
        grad_denom=float(a['grad'][pm].sum(dtype=np.float64));loss_denom=float(a['we'][pm].sum(dtype=np.float64))
        for region,mask in regs.items():
            m=pm&mask;n=int(m.sum());ev=m&a['geom_valid']&~a['occlusion'];ne=int(ev.sum())
            row=dict(**meta,patient=patients[int(pid)],region=region,pixels=n,error_support=ne,student1_errors=int(((a['pred1']!=a['gt'])&ev).sum()),student2_errors=int(((a['pred2']!=a['gt'])&ev).sum()),disagreement_count=int(a['disagree'][m].sum()),source_patient_pixels=int(pm.sum()),grad_denominator=grad_denom,loss_denominator=loss_denom)
            for key in ['w','e','we','grad']:row[key+'_sum']=float(a[key][m].sum(dtype=np.float64))
            rows.append(row)
    return rows

def summarize(rows):
    by=defaultdict(list)
    for r in rows:by[(r['condition'],r['patient'],r['region'])].append(r)
    patients=[]
    for (condition,patient,region),v in sorted(by.items()):
        n=sum(r['pixels'] for r in v);ne=sum(r['error_support'] for r in v)
        row=dict(condition=condition,patient=patient,region=region,pixels=n,exposures=len(v),error_support=ne)
        for key in ['w','e','we','grad']:
            val=sum(r[key+'_sum'] for r in v);row[key+'_sum']=val;row['mean_'+key]=val/n if n else None
        gd=sum(r['grad_denominator'] for r in v);ld=sum(r['loss_denominator'] for r in v)
        row.update(grad_denominator=gd,loss_denominator=ld,gradient_fraction=row['grad_sum']/gd if gd else None,loss_fraction=row['we_sum']/ld if ld else None,
                   disagreement_rate=sum(r['disagreement_count'] for r in v)/n if n else None,student1_error_rate=sum(r['student1_errors'] for r in v)/ne if ne else None,student2_error_rate=sum(r['student2_errors'] for r in v)/ne if ne else None)
        patients.append(row)
    csvout(W/'results/joint_patient_region.csv',patients)
    aggregate=[]
    for condition,region in sorted({(r['condition'],r['region']) for r in patients}):
        v=[r for r in patients if r['condition']==condition and r['region']==region];row=dict(condition=condition,region=region,pixels=sum(r['pixels'] for r in v),support_patients=sum(r['pixels']>0 for r in v),aggregation='patient_equal')
        for k in ['mean_w','mean_e','mean_we','mean_grad','gradient_fraction','loss_fraction','disagreement_rate','student1_error_rate','student2_error_rate']:
            vals=[r[k] for r in v if r[k] is not None];row[k]=float(np.mean(vals)) if vals else None
        aggregate.append(row)
    csvout(W/'results/joint_region_summary.csv',aggregate)
    return aggregate

def run(smoke):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    plan=prepare_inputs();patients=plan['patients'];m1,r1=load('model_1');m2,r2=load('model_2');mt,rt=load('ema_model');models=[m1,m2,mt];before=[digest(m) for m in models]
    rows=[];geometries=[];checks=[]
    for bp in plan['batches'][:1 if smoke else len(plan['batches'])]:
        b=bp['batch'];a=[dict(np.load(s['path'])) for s in bp['slots']]
        weak=torch.from_numpy(np.stack([v['weak'] for v in a])).cuda();strong=torch.from_numpy(np.stack([v['strong'] for v in a])).cuda()
        with torch.no_grad():
            la=mt(weak[12:18]);lb=mt(weak[18:]);las=mt(strong[12:18]);lbs=mt(strong[18:]);pa=pseudo(la);pb=pseudo(lb);pas=pseudo(las);pbs=pseudo(lbs)
            q=float(.5*(la.softmax(1).max(1).values.mean()+lb.softmax(1).max(1).values.mean()))
        dynamic=max(1,min(255,round(256*(.18+.20*q))))
        for strategy,side in [('fixed',97),('dynamic',dynamic)]:
            y=int(bp['uy']*max(1,256-side));x=int(bp['ux']*max(1,256-side));outside=np.ones((256,256),bool);outside[y:y+side,x:x+side]=False
            mask=torch.from_numpy(outside)[None,None].cuda()
            with torch.no_grad():
                i1=[torch.where(mask,weak[12:18],weak[:6]),torch.where(mask,weak[6:12],weak[18:])]
                i2=[torch.where(mask,strong[12:18],strong[:6]),torch.where(mask,strong[6:12],strong[18:])]
                v1=torch.cat([m1(t) for t in i1]);v2=torch.cat([m2(t) for t in i2])
            # Actual network logits are autograd leaves; frozen parameters are not updated.
            z1=v1.detach().requires_grad_(True);z2=v2.detach().requires_grad_(True);p1=z1.softmax(1);p2=z2.softmax(1)
            w=(.25+_norm01(_spatial_gradient(.5*(p1.detach().max(1).values+p2.detach().max(1).values)))).clamp(.25,1.25).detach()
            e=(p1-p2.detach()).abs().mean(1)
            literal,_=compute_v21_3_innovation_x('V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',z1,z2,la,lb,z1,z2,29600,30000)
            assert torch.allclose(literal,(w*e).mean(),atol=1e-8,rtol=1e-6)
            pred1=p1.argmax(1).detach().cpu().numpy();pred2=p2.argmax(1).detach().cpu().numpy()
            for weighting in ['uniform','CTW']:
                ww=w.mean((1,2),keepdim=True).expand_as(w).detach() if weighting=='uniform' else w
                loss=CFG['lambda_ctw']*(ww*e).mean()
                g1,g2=torch.autograd.grad(loss,(z1,z2),retain_graph=True,allow_unused=True)
                assert g2 is None and not ww.requires_grad and torch.isfinite(g1).all() and g1.abs().sum()>0
                # Independent closed-form check, including batch/class reductions and lambda.
                sg=(p1.detach()-p2.detach()).sign();expected=CFG['lambda_ctw']*ww[:,None]*p1.detach()*(sg-(sg*p1.detach()).sum(1,keepdim=True))/(e.numel()*4)
                assert torch.allclose(g1,expected,atol=2e-14,rtol=2e-5)
                if weighting=='CTW':
                    gl=torch.autograd.grad(CFG['lambda_ctw']*literal,z1,retain_graph=True)[0]
                    assert torch.equal(g1,gl)
                grad=g1.detach().abs().sum(1).cpu().numpy();condition=strategy+'_'+weighting
                checks.append(dict(batch=b,condition=condition,lambda_weighted_loss=float(loss.detach()),student2_gradient=None,weight_detached=True,max_analytic_gradient_difference=float((g1-expected).abs().max()),literal_loss_agrees=True))
                for j in range(12):
                    group=j//6;k=j%6;o=12+k if group==0 else 6+k;ins=k if group==0 else 18+k
                    def compose(key):return np.where(outside,a[o][key],a[ins][key])
                    ps=outside if group==0 else ~outside
                    pseudo_mix=np.where(outside,pa[k],a[ins]['gt']) if group==0 else np.where(outside,a[o]['gt'],pb[k])
                    pseudo_strong=np.where(outside,pas[k],a[ins]['gt']) if group==0 else np.where(outside,a[o]['gt'],pbs[k])
                    arr=dict(image1=i1[group][k,0].cpu().numpy(),image2=i2[group][k,0].cpu().numpy(),gt=compose('gt'),anatomy=compose('anatomy'),geom_valid=compose('geom_valid'),occlusion=compose('occlusion'),outside=outside,pseudo_source=ps,pseudo=pseudo_mix,pseudo_strong=pseudo_strong,
                             patient_ids=np.where(outside,patients.index(bp['slots'][o]['patient']),patients.index(bp['slots'][ins]['patient'])).astype(np.uint8),
                             w=ww[j].detach().cpu().numpy(),e=e[j].detach().cpu().numpy(),we=(ww[j]*e[j]).detach().cpu().numpy(),grad=grad[j],grad_logits=g1[j].detach().cpu().numpy(),pred1=pred1[j],pred2=pred2[j],disagree=pred1[j]!=pred2[j])
                    path=W/'arrays/joint'/f'{condition}_b{b}_p{j}.npz';path.parent.mkdir(parents=True,exist_ok=True)
                    np.savez_compressed(path,**arr)
                    meta=dict(condition=condition,batch=b,pair=j,side=side,x=x,y=y,q=q,outside_sample=bp['slots'][o]['sample'],inside_sample=bp['slots'][ins]['sample'],array_path=str(path))
                    rows.extend(extract(meta,arr,patients))
            geometries.append(dict(batch=b,strategy=strategy,q=q,side=side,area=side*side/65536,y=y,x=x,position_uy=bp['uy'],position_ux=bp['ux']))
        # Geometry and predictions are allowed to differ by strategy; weight switch never reruns inference.
        after=[digest(m) for m in models];assert after==before and all(p.grad is None for m in models for p in m.parameters())
        print('joint batch',b,'q',q,'dynamic_side',dynamic,'state_unchanged',flush=True)
    name='joint_smoke' if smoke else 'joint_full'
    js(W/'receipts'/f'{name}.json',dict(status='PASS',batches=1 if smoke else len(plan['batches']),loads=[r1,r2,rt],model_hash_before=before,model_hash_after=after,optimizer_steps=0,parameter_gradients_none=True,scope='CTW consistency item only; not total training gradient',checks=checks))
    csvout(W/'results'/f'{name}_geometry.csv',geometries)
    if not smoke:
        csvout(W/'results/joint_image_region.csv',rows);agg=summarize(rows)
        print(json.dumps([r for r in agg if r['region'] in ['all','anatomy','seam','cutout','pseudo_error']]),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');run(p.parse_args().smoke)
