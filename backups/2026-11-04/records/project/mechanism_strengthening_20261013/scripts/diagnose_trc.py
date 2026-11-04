"""Transient gamma=0 intervention on main-Full paper student1, restored per slice."""
import argparse
from collections import defaultdict
import h5py,numpy as np,torch
from scipy.ndimage import zoom
from common import *

def run(smoke):
    torch.set_num_threads(4);torch.backends.cudnn.benchmark=False;torch.backends.cudnn.deterministic=True
    m,receipt=load('model_1');router=m.decoder.mask_token_router;before=digest(m);gamma=router.gamma.detach().clone();capture={}
    def feature_hook(mod,inputs,output):
        capture['feature']=inputs[0].detach().cpu().numpy();capture['conditioned']=output.detach().cpu().numpy()
    def residual_hook(mod,inputs,output):capture['residual']=output.detach().cpu().numpy()
    h1=router.register_forward_hook(feature_hook);h2=router.inverse_project.register_forward_hook(residual_hook)
    rows=[];frame_metrics=[];restored=[];features=[]
    for frame in CFG['val_frames'][:2 if smoke else len(CFG['val_frames'])]:
        with h5py.File(DATA/'data'/f'{frame}.h5') as f:image=f['image'][:];gt=f['label'][:]
        pn=np.zeros_like(gt);pz=np.zeros_like(gt)
        indices=[len(image)//2] if smoke else list(range(len(image)))
        for z in indices:
            orig=image[z];h,w=orig.shape;x=zoom(orig,(256/h,256/w),order=0);g=zoom(gt[z],(256/h,256/w),order=0).astype(np.uint8)
            inp=torch.from_numpy(x).float()[None,None].cuda()
            with torch.no_grad():
                normal=m(inp).detach();fn=capture['feature'][0].copy();res=capture['residual'][0].copy();cond=capture['conditioned'][0].copy()
                try:
                    router.gamma.zero_();zero=m(inp).detach();fz=capture['feature'][0].copy();cz=capture['conditioned'][0].copy()
                finally:router.gamma.copy_(gamma)
                restore=m(inp).detach()
            assert torch.equal(restore,normal) and np.array_equal(fn,fz) and np.array_equal(fz,cz)
            assert np.allclose(cond,fn+float(gamma.item())*res,atol=2e-6,rtol=2e-5)
            n=normal.argmax(1)[0].cpu().numpy().astype(np.uint8);q=zero.argmax(1)[0].cpu().numpy().astype(np.uint8)
            pn[z]=zoom(n,(h/256,w/256),order=0);pz[z]=zoom(q,(h/256,w/256),order=0)
            changed=n!=q;corrected=(q!=g)&(n==g);introduced=(q==g)&(n!=g)
            bothwrong_changed=changed&(q!=g)&(n!=g);assert np.array_equal(changed,corrected|introduced|bothwrong_changed)
            anat=band(boundary(g));interior=(g>0)&~anat;background=(g==0)&~anat
            rr=dict(all=np.ones(g.shape,bool),anatomy=anat,organ_interior=interior,background=background,RV=g==1,MYO=g==2,LV=g==3)
            assert np.all(anat.astype(int)+interior.astype(int)+background.astype(int)==1)
            residual_norm=np.linalg.norm(float(gamma.item())*res,axis=0);feature_norm=np.linalg.norm(fn,axis=0)
            path=W/'arrays/trc'/f'{frame}_z{z}.npz';path.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(path,image=x,gt=g,pred_normal=n,pred_gamma0=q,changed=changed,corrected=corrected,introduced=introduced,bothwrong_changed=bothwrong_changed,anatomy=anat,residual_norm=residual_norm,feature_norm=feature_norm,logit_change=np.linalg.norm(normal[0].cpu().numpy()-zero[0].cpu().numpy(),axis=0))
            # Preserve full pre-router feature, raw residual, and conditioned feature for every slice.
            fp=W/'arrays/trc_features'/f'{frame}_z{z}.npz';fp.parent.mkdir(parents=True,exist_ok=True)
            np.savez_compressed(fp,feature=fn,residual=res,conditioned=cond,gamma=gamma.cpu().numpy())
            features.append(dict(frame=frame,z=z,array_path=str(path),features_path=str(fp),gamma=float(gamma.item()),residual_to_feature=float(np.linalg.norm(float(gamma.item())*res)/max(np.linalg.norm(fn),1e-8))))
            restored.append(dict(frame=frame,z=z,max_abs_restore_difference=float((normal-restore).abs().max()),bitwise_equal=True))
            for region,mask in rr.items():
                pix=int(mask.sum());rows.append(dict(patient=frame.split('_')[0],frame=frame,z=z,region=region,pixels=pix,changed=int(changed[mask].sum()),corrected=int(corrected[mask].sum()),introduced=int(introduced[mask].sum()),bothwrong_changed=int(bothwrong_changed[mask].sum()),normal_errors=int((n[mask]!=g[mask]).sum()),gamma0_errors=int((q[mask]!=g[mask]).sum()),residual_sum=float(residual_norm[mask].sum(dtype=np.float64))))
            # Per-class one-vs-rest changes include class-specific false positives and negatives.
            for cls,name in [(1,'RV'),(2,'MYO'),(3,'LV')]:
                en=(n==cls)!=(g==cls);ez=(q==cls)!=(g==cls)
                rows.append(dict(patient=frame.split('_')[0],frame=frame,z=z,region=name+'_binary_all',pixels=g.size,changed=int(((n==cls)!=(q==cls)).sum()),corrected=int((ez&~en).sum()),introduced=int((en&~ez).sum()),bothwrong_changed=0,normal_errors=int(en.sum()),gamma0_errors=int(ez.sum()),residual_sum=float(residual_norm.sum(dtype=np.float64))))
        if not smoke:
            np.savez_compressed(W/'arrays/trc'/f'{frame}_native_predictions.npz',pred_normal=pn,pred_gamma0=pz)
            for cls,name in [(1,'RV'),(2,'MYO'),(3,'LV')]:frame_metrics.append(dict(patient=frame.split('_')[0],frame=frame,class_name=name,dice_normal=dice(pn==cls,gt==cls),dice_gamma0=dice(pz==cls,gt==cls),delta_dice=dice(pn==cls,gt==cls)-dice(pz==cls,gt==cls)))
        assert digest(m)==before and all(p.grad is None for p in m.parameters())
        print('TRC',frame,'slices',len(indices),'gamma',float(gamma.item()),'restored exactly',flush=True)
    h1.remove();h2.remove();after=digest(m);assert before==after and torch.equal(router.gamma,gamma)
    name='trc_smoke' if smoke else 'trc_full'
    js(W/'receipts'/f'{name}.json',dict(status='PASS',load=receipt,gamma=float(gamma.item()),slices=len(restored),model_hash_before=before,model_hash_after=after,gamma_restored=True,restored_output_bitwise_equal=True,checkpoint_written=False,optimizer_steps=0,restore_checks=restored))
    if not smoke:
        csvout(W/'results/trc_slice_region.csv',rows);csvout(W/'results/trc_feature_manifest.csv',features);csvout(W/'results/trc_frame_dice.csv',frame_metrics)
        by=defaultdict(list)
        for r in rows:by[(r['patient'],r['region'])].append(r)
        pr=[]
        for (patient,region),v in sorted(by.items()):
            row=dict(patient=patient,region=region)
            for k in ['pixels','changed','corrected','introduced','bothwrong_changed','normal_errors','gamma0_errors','residual_sum']:row[k]=sum(r[k] for r in v)
            for k in ['changed','corrected','introduced','normal_errors','gamma0_errors']:row[k+'_rate']=row[k]/row['pixels'] if row['pixels'] else None
            row['net_errors_removed']=row['corrected']-row['introduced'];pr.append(row)
        csvout(W/'results/trc_patient_region.csv',pr)
        ag=[]
        for region in sorted({r['region'] for r in pr}):
            v=[r for r in pr if r['region']==region];row=dict(region=region,aggregation='patient_equal',support_patients=sum(r['pixels']>0 for r in v))
            for k in ['pixels','changed','corrected','introduced','bothwrong_changed','normal_errors','gamma0_errors','net_errors_removed']:row[k]=sum(r[k] for r in v)
            for k in ['changed_rate','corrected_rate','introduced_rate','normal_errors_rate','gamma0_errors_rate']:row[k]=float(np.mean([r[k] for r in v if r[k] is not None]))
            ag.append(row)
        csvout(W/'results/trc_region_summary.csv',ag)
        pd=[]
        for pat,cl in sorted({(r['patient'],r['class_name']) for r in frame_metrics}):
            v=[r for r in frame_metrics if r['patient']==pat and r['class_name']==cl]
            pd.append(dict(patient=pat,class_name=cl,frames=len(v),**{k:float(np.mean([r[k] for r in v])) for k in ['dice_normal','dice_gamma0','delta_dice']}))
        csvout(W/'results/trc_patient_dice.csv',pd)
        da=[dict(class_name=cl,patients=10,**{k:float(np.mean([r[k] for r in pd if r['class_name']==cl])) for k in ['dice_normal','dice_gamma0','delta_dice']}) for cl in ['RV','MYO','LV']]
        da.append(dict(class_name='macro',patients=10,**{k:float(np.mean([r[k] for r in pd])) for k in ['dice_normal','dice_gamma0','delta_dice']}))
        csvout(W/'results/trc_dice_summary.csv',da);print(json.dumps(da),flush=True);print(json.dumps(ag),flush=True)
if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--smoke',action='store_true');run(p.parse_args().smoke)
