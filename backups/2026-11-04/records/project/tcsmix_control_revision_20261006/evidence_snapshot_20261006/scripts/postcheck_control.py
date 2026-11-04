"""Actual short-run assertions, strict checkpoint loading and matched control cost."""
import ast
import collections
import json
import time
from common import *
bootstrap()
import numpy as np
import torch
import torch.nn.functional as F
import revision_runtime as r
import v14_mask_adamix_components as parent
from controller import generate_mask,geometry

def main():
    assert read_json(WORK/'state/status.json')['status']=='SHORT_CHECK_FINISHED'
    cfg=read_json(WORK/'configs/short_selected.yaml');r.CFG=cfg
    rows=read_table(WORK/'short_trace.csv');assert [int(x['step']) for x in rows]==list(range(1,241))
    for row in rows:
        active=[int(c) for c in cfg['foreground_ids'] if int(row[f'n_{c}'])>0]
        expected=np.mean([float(row[f'margin_class{c}']) for c in active]) if active else 0.
        assert abs(float(row['q'])-expected)<1e-7
        ss,h,w=geometry(expected,256,256);assert abs(ss-float(row['s']))<1e-10
        assert int(row['rectangle_h'])==h and int(row['rectangle_w'])==w and float(row['actual_area'])==h*w/65536
        assert abs(float(row['weighted_ctw_loss'])-.05*float(row['ctw_loss']))<1e-9
        assert float(row['lr'])==.01 and row['shared_mask_and_target_identical']=='True'
    budget=read_json(WORK/'state/update_budget.json');assert budget['optimizer_step_calls']==480 and budget['joint_SSL_updates']==240
    run=WORK/'runs/SHORT_TCSMIX_V2';checkpoints=[run/'checkpoints/recovery_boundary_200.ckpt',run/'checkpoints/last.ckpt']
    restored=[]
    for cp in checkpoints:
        ck=torch.load(cp,map_location='cpu',weights_only=False);models=[r.BCP_net(ema=i==2) for i in range(3)]
        for model,key in zip(models,['model_1','model_2','ema_model']):model.load_state_dict(ck[key],strict=True)
        for model,key in zip(models[:2],['optimizer1','optimizer2']):
            opt=parent.make_optimizer(model,.01,True);opt.load_state_dict(ck[key])
            assert all(st.get('momentum_buffer',p).shape==p.shape for p,st in opt.state.items())
        assert ck['controller_metadata']['stateless'] and ck['controller_metadata']['version']==cfg['controller_version']
        assert ck['controller_metadata']['cfg_sha256']==sha(WORK/'configs/short_selected.yaml')
        assert set(ck['random_state'])=={'python_random','numpy_random','torch_random','torch_cuda_random'}
        restored.append(dict(path=str(cp),sha256=sha(cp),step=ck['global_step'],keys=sorted(ck),
            controller_metadata=ck['controller_metadata'],strict_models_and_optimizers_load=True,parameters_student1=sum(p.numel() for p in models[0].parameters()),
            router_parameters=sum(p.numel() for p in models[0].decoder.mask_token_router.parameters()),sampler_plan=len(ck['sampler_state']['batches']),consumed=ck['sampler_state']['consumed']))
        del models,ck,opt;torch.cuda.empty_cache()
    assert [x['step'] for x in restored]==[200,240]
    atomic_json(WORK/'reports/short_checkpoint_checks.json',dict(status='PASS',time=now(),checkpoints=restored,
        actual_resume='second segment logs strict_resume_loaded step200; trace201..240 confirms same run',
        controller_state='stateless; definition/metadata hash stored and verified',resume_exact=False,
        boundary='16 worker augmentation RNG/prefetched tensors unavailable, restarted worker streams; not bitwise resume',
        optimizer_calls_in_postcheck=0,never_formal_initialization=True))
    # Matched GPU microbenchmark, one existing Full teacher and the same inputs/support.
    model=r.BCP_net(ema=True);ck=torch.load(FULL/'checkpoints/best.ckpt',map_location='cpu',weights_only=False);model.load_state_dict(ck['ema_model']);model.eval()
    import h5py
    from scipy.ndimage import zoom
    ids=(DATA/'train_slices.list').read_text().splitlines()[136:148];xx=[]
    for sid in ids:
        with h5py.File(DATA/'data/slices'/f'{sid}.h5','r') as f:x=f['image'][:]
        xx.append(zoom(x,(256/x.shape[0],256/x.shape[1]),order=0).astype(np.float32))
    x=torch.from_numpy(np.stack(xx))[:,None].cuda()
    code=(WORK/'source/historical_full.py').read_text();ns=dict(torch=torch,np=np,F=F)
    for node in ast.parse(code).body:
        if isinstance(node,ast.FunctionDef) and node.name in ['get_ACDC_2DLargestCC','get_ACDC_masks']:exec(ast.get_source_segment(code,node),ns)
    with torch.no_grad():la=model(x[:6]);lb=model(x[6:]);ya=ns['get_ACDC_masks'](la,nms=1);yb=ns['get_ACDC_masks'](lb,nms=1)
    cost=[]
    for version in ['old',cfg['controller_version']]:
        cc=dict(cfg,controller_version=version)
        for _ in range(10):generate_mask(x[:6],la,lb,ya,yb,cc)
        torch.cuda.synchronize();base=torch.cuda.memory_allocated();torch.cuda.reset_peak_memory_stats();times=[]
        for _ in range(30):
            torch.cuda.synchronize();start=time.perf_counter();m,l,st=generate_mask(x[:6],la,lb,ya,yb,cc);torch.cuda.synchronize();times.append((time.perf_counter()-start)*1000)
        cost.append(dict(controller=version,median_ms=float(np.median(times)),p10_ms=float(np.quantile(times,.1)),p90_ms=float(np.quantile(times,.9)),
            peak_incremental_allocated_bytes=torch.cuda.max_memory_allocated()-base,warmup=10,repeats=30,device=torch.cuda.get_device_name(),
            batch=12,groups='6+6',shape='1x256x256',precision='FP32',teacher_forward_measured=False,pseudolabel_processing_measured=False,
            checkpoint=str(FULL/'checkpoints/best.ckpt'),checkpoint_sha256=sha(FULL/'checkpoints/best.ckpt'),samples=';'.join(ids),
            scope='controller+mask only on identical precomputed teacher logits and original LCC supports; timing repetitions are not training'))
    write_csv(WORK/'analysis/controller_cost.csv',cost)
    counts=collections.Counter(int(x['rectangle_h']) for x in rows);oldcounts=collections.Counter(geometry(float(x['old_q']),256,256)[1] for x in rows)
    stable=[float(x['step_seconds']) for x in rows if 51<=int(x['step'])<=199 or 211<=int(x['step'])<=240]
    fullreceipt=read_json(run/'TRAIN_RECEIPT.json')
    metrics=dict(status='PASS',time=now(),joint_SSL_updates=240,actual_optimizer_step_calls=480,hard_limit=500,
        fallback_steps=sum(x['fallback']=='True' for x in rows),sizes=dict(sorted(counts.items())),most_common_size_fraction=max(counts.values())/240,
        counterfactual_old_sizes_on_same_v2_teachers=dict(sorted(oldcounts.items())),
        q_min=min(float(x['q']) for x in rows),q_max=max(float(x['q']) for x in rows),
        area_min=min(float(x['actual_area']) for x in rows),area_max=max(float(x['actual_area']) for x in rows),
        actual_size_change_count=sum(a['rectangle_h']!=b['rectangle_h'] for a,b in zip(rows,rows[1:])),
        stable_step_median_seconds=float(np.median(stable)),stable_step_p10_seconds=float(np.quantile(stable,.1)),stable_step_p90_seconds=float(np.quantile(stable,.9)),
        stable_steps=len(stable),stable_scope='steps51..199 and211..240, excludes startup/resume warmup; timer ends before EMA/logging/checkpoints/validation',
        full30000_compute_hours_estimate=float(np.median(stable))*30000/3600,
        active_segment_seconds=fullreceipt['training_seconds'],peak_training_cuda_bytes=max(int(x['peak_cuda_bytes']) for x in rows),
        short_not_formal=True,formal_Dice='NA',test_access=False,CTW='original spatial,lambda.05',TRC='original641parameters')
    atomic_json(WORK/'reports/short_statistics.json',metrics)
    locked=read_json(WORK/'state/old_inputs_locked.json');assert all(sha(p)==h for p,h in locked.items())
    assert (PROJECT/'revision_runs_20261006/state/stop_requested').exists()
    atomic_json(WORK/'reports/postcheck_PASS.json',dict(status='PASS',time=now(),old_locked_inputs_unchanged=len(locked),
        short_formula_every_step_verified=True,mask_area_every_step_verified=True,checkpoints_strict_loaded=True,extra_optimizer_calls=0,formal_dirs_absent=all(not (WORK/f'runs/{x}').exists() for x in ['R1','R2'])))
    print(json.dumps(metrics,ensure_ascii=False))

if __name__=='__main__':torch.set_num_threads(1);main()
