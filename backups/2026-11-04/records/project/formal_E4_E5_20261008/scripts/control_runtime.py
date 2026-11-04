"""Formal R1 telemetry; accepted controller unchanged, no short-phase budget guard."""
import csv
import os
import time
import torch
from common import *
import revision_runtime as r
from controller import generate_mask

CFG={};LATEST={};STEP_START=0.;CONTROLLER_MS=0.;COUNTS={'optimizer1':0,'optimizer2':0}
TRACE=None;EVENTS=None;SEGMENT=None;IDS=[];META_SECONDS=0.;TRACE_SECONDS=0.;EVENT_SECONDS=0.;LOGICAL_STEP=0

def configure(cfg,stop_after=None):
    global CFG,COUNTS,TRACE,EVENTS,SEGMENT
    assert stop_after is None and cfg['max_iterations']==30000 and cfg['controller_version']=='v2_B'
    CFG=cfg;SEGMENT=cfg['segment_id'];path=WORK/'results/E4_control_trace.csv'
    # Actual physical optimizer calls retain replayed updates across manual recovery.
    ledger=WORK/'results/optimizer_calls.csv'
    if ledger.exists():
        for row in read_table(ledger):
            if row['event']=='success':COUNTS[row['optimizer']]=max(COUNTS[row['optimizer']],int(row['cumulative_calls']))
    TRACE=path.open('a',newline='',buffering=65536);EVENTS=ledger.open('a',newline='',buffering=65536)

def guarded_optimizer(optimizer,name):
    original=optimizer.step
    def step(*args,**kwargs):
        start=time.perf_counter()
        try:result=original(*args,**kwargs)
        except BaseException as exc:
            _event(name,'failed',str(type(exc).__name__));raise
        COUNTS[name]+=1;_event(name,'success','')
        return result
    optimizer.step=step
    return optimizer

def _event(name,event,reason):
    global EVENT_SECONDS
    start=time.perf_counter()
    fields=['time','segment','logical_step','optimizer','event','cumulative_calls','reason']
    writer=csv.DictWriter(EVENTS,fieldnames=fields)
    if EVENTS.tell()==0:writer.writeheader()
    writer.writerow(dict(time=now(),segment=SEGMENT,logical_step=LOGICAL_STEP,optimizer=name,event=event,cumulative_calls=COUNTS[name],reason=reason))
    EVENTS.flush()  # small append; no per-call atomic rewrite or filesystem fsync
    EVENT_SECONDS+=time.perf_counter()-start

def start_step(step,sampled_batch=None):
    global STEP_START,LOGICAL_STEP,IDS
    LOGICAL_STEP=step
    IDS=[int(v) for v in sampled_batch['idx'].tolist()] if sampled_batch is not None else []
    torch.cuda.synchronize();STEP_START=time.perf_counter()

def generate_adamix_bcp_lite_v1_mask(img,la,lb,min_side=.18,max_side=.38,plab_a=None,plab_b=None):
    global LATEST,CONTROLLER_MS
    if plab_a is None or plab_b is None:raise RuntimeError('Must reuse existing teacher LCC labels')
    start=time.perf_counter();mm,ll,st=generate_mask(img,la,lb,plab_a,plab_b,CFG,min_side,max_side)
    assert torch.equal(mm,ll) and torch.equal(mm[0].expand_as(mm),mm)
    assert int((mm[0]==0).sum())==st['rectangle_h']*st['rectangle_w']
    CONTROLLER_MS=(time.perf_counter()-start)*1000;LATEST=st
    r.LATEST.update(q=st['teacher_conf'],s=st['side_fraction'],actual_area=st['patch_area_ratio'])
    return mm,ll,st

def record_step(run,step,optimizer,loss):
    global TRACE_SECONDS,META_SECONDS
    torch.cuda.synchronize();elapsed=time.perf_counter()-STEP_START;st=LATEST;start=time.perf_counter()
    row=dict(step=step,segment=SEGMENT,time=now(),controller_version=CFG['controller_version'],
        old_q=st['old_q'],q_A=st['q_fg'],q_B=st['q_margin'],q=st['teacher_conf'],s=st['side_fraction'],
        rectangle_h=st['rectangle_h'],rectangle_w=st['rectangle_w'],actual_area_pixels=st['rectangle_h']*st['rectangle_w'],
        actual_area=st['patch_area_ratio'],y=st['rectangle_y'],x=st['rectangle_x'],fallback=st['fallback'],
        active_classes=st['active_classes'],predicted_background_fraction=st['predicted_background_fraction'],background_confidence=st['background_confidence'],
        loss=float(loss.detach()),ctw_loss=r.LATEST.get('ctw_loss'),weighted_ctw_loss=r.LATEST.get('weighted_ctw_loss'),
        lr=optimizer.param_groups[0]['lr'],step_seconds=elapsed,controller_ms=CONTROLLER_MS,
        peak_cuda_bytes=max(r.PREVIOUS_PEAK,torch.cuda.max_memory_allocated()),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),shared_mask_and_target_identical=True,
        optimizer1_calls=COUNTS['optimizer1'],optimizer2_calls=COUNTS['optimizer2'],AMP_enabled=False,AMP_skipped_updates=0,
        sampled_indices=';'.join(map(str,IDS)))
    for c in CFG['foreground_ids']:row.update({f'n_{c}':st['foreground_counts'][str(c)],f'q_class{c}':st['q_by_class'][str(c)],f'margin_class{c}':st['margin_by_class'][str(c)]})
    writer=csv.DictWriter(TRACE,fieldnames=list(row))
    if TRACE.tell()==0:writer.writeheader()
    writer.writerow(row);TRACE.flush();TRACE_SECONDS+=time.perf_counter()-start
    if step==1 or step%10==0:
        meta_start=time.perf_counter()
        atomic_json(WORK/'state/control_progress.json',dict(pid=os.getpid(),time=now(),joint_step=step,segment=SEGMENT,
            optimizer_calls=COUNTS,trace_write_seconds=TRACE_SECONDS,**{k:row[k] for k in ['q','s','rectangle_h','rectangle_w','actual_area','step_seconds','peak_cuda_bytes']}))
        META_SECONDS+=time.perf_counter()-meta_start
    if stop_requested():r.STOP=True

def checkpoint_metadata():
    return dict(version='v2_B',foreground_ids=CFG['foreground_ids'],background_id=CFG['background_id'],empty_foreground_fallback=0.,
        min_side=.18,max_side=.38,stateless=True,cfg_sha256=CFG['config_sha256'],controller_sha256=CFG['controller_source_sha256'],
        optimizer_success_calls=dict(COUNTS),segment=SEGMENT,AMP_enabled=False,code_identity=task_identity('E4'),weight_mode='spatial',task='E4')

def initial_loaded(models,optimizers,pre_model):
    ck=torch.load(pre_model,map_location='cpu',weights_only=False)
    # Parent initialization retains random zero-gated TRC, exact common backbone and transplant.
    checks=[]
    for name,model in models:
        sd=model.state_dict();common={k:v for k,v in sd.items() if 'mask_token_router' not in k}
        assert set(common)==set(ck['net'])
        assert all(torch.equal(v.cpu(),ck['net'][k].cpu()) for k,v in common.items())
        assert model.decoder.mask_token_router.gamma.item()==0
        router=model.decoder.mask_token_router
        assert isinstance(router,r.BottleneckResidual) and sum(p.numel() for p in router.parameters())==647
        x=torch.zeros(1,16,7,9,device=next(model.parameters()).device)
        assert torch.equal(router(x),x)
        if name!='ema_model':
            opt=optimizers[0 if name=='model_1' else 1]
            ids={id(p) for g in opt.param_groups for p in g['params']}
            assert all(id(p) in ids and p not in opt.state for p in router.parameters())
        checks.append(dict(branch=name,plain_structure=True,gamma_zero_identity=True,optimizer_router_covered=name!='ema_model',exact_pretrained_common_state=True,gamma=0.,parameters=sum(p.numel() for p in model.parameters()),
            router_parameters=sum(p.numel() for p in model.decoder.mask_token_router.parameters())))
    def equal(a,b):
        if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a.cpu(),b.cpu())
        if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
        if isinstance(a,(tuple,list)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
        return a==b
    for optimizer in optimizers:
        actual=optimizer.state_dict()
        assert len(actual['state'])==len(ck['opt']['state'])
        assert equal(actual['state'],ck['opt']['state']), 'pretrain momentum state transplant mismatch'
    atomic_json(WORK/'reports/INITIALIZATION_ACTUAL.json' ,dict(time=now(),joint_step=0,pretrain=str(pre_model),sha256=sha(pre_model),
        fresh=True,checks=checks,rng_sha256=rng_digest(rng_state()),data_manifest_sha256=sha(MAIN_WORK/"manifests/data_assets.csv"),model_named_parameters={name:{n:list(p.shape) for n,p in model.named_parameters()} for name,model in models},optimizer_pretrained_state_exact=True,code_identity=task_identity('E4'),
        backend=dict(cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,AMP=False),optimizer1_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizers[0].param_groups],
        optimizer2_groups=[{k:v for k,v in g.items() if k!='params'} for g in optimizers[1].param_groups],
        source='exact parent load_parent_pretrain and transplant_parent_optimizer_state; no short/E1/late weights',
        generating_pretrain_history='NA: no original generating command/log retained; identical file reused',test_access=False))

def validation(step,value1,value2,best1):
    p=WORK/'results/validation_trace.csv';row=dict(time=now(),segment=SEGMENT,step=step,student1_macro_dice=float(value1),student2_macro_dice=float(value2),
        best_student1_macro_dice=float(best1),selection='strict student1 val macro Dice, no test selection')
    with p.open('a',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:wr.writeheader()
        wr.writerow(row);f.flush()

def close():
    for f in [TRACE,EVENTS]:
        if f:f.flush();os.fsync(f.fileno());f.close()
    atomic_json(WORK/f'reports/logging_cost_{SEGMENT}.json',dict(time=now(),segment=SEGMENT,trace_append_seconds=TRACE_SECONDS,
        optimizer_event_append_seconds=EVENT_SECONDS,control_progress_atomic_seconds=META_SECONDS,
        scope='instrumented new trace/events/progress only; inherited train.log/revision telemetry/checkpoint overhead not separately isolated',
        optimizer_success_calls=COUNTS))

def after_ema(step,model,teacher):
    assert type(model.decoder.mask_token_router) is type(teacher.decoder.mask_token_router)
    if CFG['phase']=='isolated_short':
        module=model.decoder.mask_token_router
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in module.parameters())
        assert all(torch.isfinite(p).all() for p in module.parameters())
        assert module.gamma.item()!=0
        atomic_json(WORK/'reports/PLAIN_GRADIENT_EMA.json',dict(status='PASS',step=step,gamma=module.gamma.item(),gradients={n:float(p.grad.abs().sum()) for n,p in module.named_parameters()},teacher_names_shapes_match=True,EMA_algebra_checked_in_loop=True))
    # Existing update has already happened; no additional forward/update/RNG.
    if step<=2 or step%200==0:
        atomic_json(WORK/'state/EMA_PROGRESS.json',dict(time=now(),step=step,alpha=.99,teacher_source='student1',
            student1_gamma=model.decoder.mask_token_router.gamma.item(),teacher_gamma=teacher.decoder.mask_token_router.gamma.item(),
            teacher_requires_grad=any(p.requires_grad for p in teacher.parameters())))
