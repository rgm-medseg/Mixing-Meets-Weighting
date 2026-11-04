"""Protocol, real data, tensor gradient, shape, checkpoint and entry smoke gates."""
import argparse
import ast
import json
import os
import py_compile
import subprocess
import sys
import time
from pathlib import Path
from common import *

def static_checks():
    import h5py,numpy as np,torch
    bootstrap()
    import revision_runtime as r
    checks=[]
    for path in (WORK/'scripts').glob('*.py'):
        py_compile.compile(str(path),doraise=True)
    for path in (WORK/'scripts').glob('*.sh'):
        subprocess.run(['/bin/bash','-n',str(path)],check=True)
    checks.append('All Python and shell syntax')
    protocol=read_json(WORK/'protocol.json')
    manifest=read_json(WORK/'run_manifest.json')
    assert len(protocol['labeled_patients'])==7
    ids=(DATA/'train_slices.list').read_text().splitlines()
    for split in ('train_slices','val','test'):
        folder='data/slices' if split=='train_slices' else 'data'
        for case in (DATA/f'{split}.list').read_text().splitlines():
            with h5py.File(DATA/folder/f'{case}.h5','r') as f:
                assert set(('image','label'))<=set(f.keys())
                gt=f['label'][:]; img=f['image'][:]
                assert gt.shape==img.shape and np.isfinite(img).all()
                assert set(np.unique(gt))<={0,1,2,3}
    assert {x.split('_')[0] for x in ids[:136]}==set(protocol['labeled_patients'])
    checks.append('All train/val/test paths, image/label shapes, finite data, class IDs, split and seven labeled patient IDs')
    # A physical point displacement and an axis permutation independently check
    # distance units; they do not depend on reproducing a historical number.
    from train_COVID19.covid19_volume_evaluator import binary_volume_metrics
    pred=np.zeros((7,9,11),bool); gt=np.zeros_like(pred)
    pred[3,4,5]=True; gt[3,6,5]=True
    spacing=(3.,.7,1.1)
    metric=binary_volume_metrics(pred,gt,spacing)
    assert abs(metric['hd95_mm']-1.4)<1e-10 and abs(metric['asd_mm']-1.4)<1e-10
    permuted=binary_volume_metrics(pred.transpose(1,2,0),gt.transpose(1,2,0),(spacing[1],spacing[2],spacing[0]))
    assert all(abs(metric[k]-permuted[k])<1e-10 for k in metric)
    geometry=read_json(WORK/'reports/covid_reevaluation_receipt.json')
    assert geometry['geometry_policy_version']=='resize_spacing_v2_XYZ_reconstruction_verified'
    assert geometry['legacy_maximum_metric_difference']<1e-10 and geometry['case_records']==48
    checks.append('Independent anisotropic physical-distance/axis-permutation check and all16 exact COVID legacy replays; corrected policies stored separately')
    torch.manual_seed(741); out1=torch.randn(3,4,13,11,device='cuda',requires_grad=True); out2=torch.randn(3,4,13,11,device='cuda',requires_grad=True)
    raw,norm,orig,uniform,error=r.ctw_maps(out1,out2,True)
    assert not uniform.requires_grad and not orig.requires_grad
    assert torch.allclose(orig.mean((1,2)),uniform.mean((1,2)))
    assert (uniform.flatten(1).max(1).values-uniform.flatten(1).min(1).values).max()<1e-6
    (uniform*error).mean().backward()
    assert out1.grad is not None and out1.grad.abs().sum()>0 and out2.grad is None
    # Independent hand formula verifies the scientific CTW control.
    q=(out1.detach().softmax(1).max(1).values+out2.detach().softmax(1).max(1).values)/2
    grad=torch.zeros_like(q); grad[:,1:]+=(q[:,1:]-q[:,:-1]).abs(); grad[:,:,1:]+=(q[:,:,1:]-q[:,:,:-1]).abs()
    assert torch.equal(grad,raw)
    low=grad.flatten(1).min(1).values[:,None,None]; high=grad.flatten(1).max(1).values[:,None,None]
    expected=.25+((grad-low)/(high-low).clamp_min(1e-6)).clamp(0,1)
    assert torch.equal(orig,expected)
    checks.append('CTW backward differences, floor1e-6, per-image uniform means, detached weights, student2 stop-gradient and student1 gradient')
    for task in ('E2','E3'):
        path,cfg=config_for(task); r.configure(cfg)
        image=torch.zeros((6,1,256,256),device='cuda'); logits=torch.randn((6,4,256,256),device='cuda')
        m,l,s=r.generate_adamix_bcp_lite_v1_mask(image,logits,logits)
        assert torch.equal(m[0].expand_as(m),m)
        assert int((m[0]==0).sum())==97*97 and s['side_fraction']==read_json(WORK/'configs/fixed_scale.json')['fixed_side']
        assert torch.equal(l,m)
    checks.append('E2/E3 identical locked scale, shared mask, actual area97x97 and label mask')
    _,cfg=config_for('E4'); r.configure(cfg)
    model=r.BCP_net(); teacher=r.BCP_net(ema=True)
    router=model.decoder.mask_token_router
    x=torch.randn(2,16,256,256,device='cuda')
    assert torch.equal(router(x),x)
    assert not any('token' in name.split('.')[-1] for name,_ in router.named_parameters())
    assert router.gamma.item()==0 and sum(p.numel() for p in router.parameters())==647
    from v14_mask_adamix_components import SpatialMaskTokenRouter
    ref=SpatialMaskTokenRouter(16,4,16)
    assert sum(p.numel() for p in ref.parameters())==641
    with torch.no_grad():
        router.gamma.fill_(.7)
        new={key:.99*teacher.state_dict()[key]+.01*model.state_dict()[key] for key in model.state_dict()}
        teacher.load_state_dict(new)
    assert abs(teacher.decoder.mask_token_router.gamma.item()-.007)<1e-6
    assert all(not p.requires_grad for p in teacher.parameters())
    checks.append('E4 plain1x1/LN/GELU/1x1 zero-gate identity,647 vs TRC641 (+0.936%), teacher corresponding EMA parameters')
    for task in ('E1','E2','E3','E4','E5'):
        path,cfg=config_for(task)
        assert sha(path)==manifest['tasks'][task]['config_sha256']
        assert cfg['protocol_hash']==sha(WORK/'protocol.json')
    checks.append('All five config hashes and frozen protocol hashes')
    _,official=config_for('E5')
    commit=subprocess.run(['git','-C',str(WORK/'external/AdaMix'),'rev-parse','HEAD'],check=True,capture_output=True,text=True).stdout.strip()
    assert commit==official['official_commit']
    dirty=subprocess.run(['git','-C',str(WORK/'external/AdaMix'),'status','--porcelain','--untracked-files=no'],check=True,capture_output=True,text=True).stdout.strip()
    assert not dirty
    checks.append('Official AdaMix pinned commit verified; upstream tracked files unmodified')
    ck=torch.load(protocol['pretrain'],map_location='cpu',weights_only=False)
    assert 'net' in ck and 'opt' in ck
    groups=ck['opt']['param_groups']
    atomic_json(WORK/'reports/pretrain_inspection.json',dict(path=protocol['pretrain'],sha256=sha(protocol['pretrain']),keys=list(ck),
                optimizer_param_groups=[{k:v for k,v in group.items() if k!='params'} for group in groups],
                net_parameter_keys=len(ck['net']),historical_generation_provenance='NA: cannot infer generation seed or budget from weights alone'))
    checks.append('Historical initialization net/opt real keys and hashes; no invented generation provenance')
    del model,teacher,ck; torch.cuda.empty_cache()
    return checks

def smoke_checks(tasks=None):
    checks=[]
    env=os.environ.copy(); env.update(OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',PYTHONUNBUFFERED='1')
    for task in tasks or ('E1','E2','E3','E4','E5'):
        cfgpath,_=config_for(task)
        run=WORK/'runs/smoke'/task
        if run.exists():
            archive=WORK/'runs/smoke_archive'/f'{task}_{int(time.time())}'
            archive.parent.mkdir(parents=True,exist_ok=True)
            os.rename(run,archive)
        for mode in ('fresh','resume'):
            cmd=[str(PYTHON),str(WORK/'scripts/train_entry.py'),'--config',str(cfgpath),'--smoke']
            if mode=='resume': cmd+=['--resume']
            with (WORK/f'logs/preflight_{task}_{mode}.log').open('w') as f:
                result=subprocess.run(cmd,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT,timeout=300)
            if result.returncode: raise RuntimeError(f'{task} {mode} smoke failed exit={result.returncode}; inspect logs/preflight_{task}_{mode}.log')
        import torch
        ck=torch.load(run/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert ck['global_step']==3 and 'random_state' in ck and 'sampler_state' in ck
        assert 'optimizer1' in ck and 'model_2' in ck and (task=='E5' or 'ema_model' in ck)
        checks.append(f'{task} real batches forward/backward/save 2updates, restore +1update; full-state keys inspected')
        with (WORK/f'logs/preflight_{task}_evaluate.log').open('w') as f:
            subprocess.run([str(PYTHON),str(WORK/'scripts/evaluate.py'),'--task',task,'--split','val','--smoke'],cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT,check=True,timeout=180)
        checks.append(f'{task} restored final predictions passed unified evaluation on existing val frame volume, no test score selection')
    return checks

def main():
    p=argparse.ArgumentParser(); p.add_argument('--smoke',action='store_true'); p.add_argument('--reuse-smoke',action='store_true'); p.add_argument('--smoke-tasks',nargs='+',choices=['E1','E2','E3','E4','E5']); a=p.parse_args()
    out=WORK/'state/preflight.json'
    result=dict(status='RUNNING',time=now(),checks=[],smoke_requested=a.smoke)
    atomic_json(out,result)
    try:
        result['checks']=static_checks()
        if a.smoke: result['checks']+=smoke_checks(a.smoke_tasks)
        if a.smoke or a.reuse_smoke:
            import torch
            for task in ('E1','E2','E3','E4','E5'):
                ck=torch.load(WORK/f'runs/smoke/{task}/checkpoints/last.ckpt',map_location='cpu',weights_only=False)
                assert ck['global_step']==3
                receipt=read_json(WORK/f'runs/smoke/{task}/evaluation/EVALUATION_RECEIPT.json')
                assert receipt['smoke'] and receipt['checkpoint_step']==3 and receipt['split']=='val'
                assert receipt['checkpoint_sha256']==sha(WORK/f'runs/smoke/{task}/checkpoints/last.ckpt')
            result['checks'].append('All five fresh/restore smoke outputs and val-only evaluation receipts verified; unchanged successful entries reused')
        result.update(status='PASS',completed=now(),smoke_pass=a.smoke or a.reuse_smoke,
                script_sha256={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'scripts').glob('*.py'))},
                shell_sha256={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'scripts').glob('*.sh'))},
                source_sha256={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'source').rglob('*.py')) if '__pycache__' not in p.parts},
                config_sha256={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'configs').glob('*'))})
    except Exception as exc:
        import traceback
        result.update(status='FAIL',reason=str(exc),traceback=traceback.format_exc(),completed=now())
        atomic_json(out,result); raise
    atomic_json(out,result); print(json.dumps(result,ensure_ascii=False,indent=2))

if __name__=='__main__':
    main()
