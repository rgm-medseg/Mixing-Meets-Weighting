"""Control logging, scope assertions and durable 500 optimizer-call guard."""
import csv
import json
import os
import time
from pathlib import Path
import torch
from common import WORK,read_json,atomic_json,sha,now
import revision_runtime as r
from controller import generate_mask

CFG={};LATEST={};STEP_START=0.;STOP_AT=None;CALLS_START=0;CONTROLLER_MS=0.

def configure(cfg,stop_after=None):
    global CFG,STOP_AT,CALLS_START
    CFG=cfg;STOP_AT=stop_after
    CALLS_START=read_json(WORK/'state/update_budget.json')['optimizer_step_calls']

def generate_adamix_bcp_lite_v1_mask(img,la,lb,min_side=.18,max_side=.38,plab_a=None,plab_b=None):
    global LATEST,CONTROLLER_MS
    if plab_a is None or plab_b is None: raise RuntimeError('Reuse existing teacher LCC labels, do not regenerate')
    start=time.perf_counter()
    mm,ll,st=generate_mask(img,la,lb,plab_a,plab_b,CFG,min_side,max_side)
    assert torch.equal(mm,ll) and torch.equal(mm[0].expand_as(mm),mm)
    assert int((mm[0]==0).sum())==st['rectangle_h']*st['rectangle_w']
    CONTROLLER_MS=(time.perf_counter()-start)*1000
    LATEST=st
    r.LATEST.update(q=st['teacher_conf'],s=st['side_fraction'],actual_area=st['patch_area_ratio'])
    return mm,ll,st

def guarded_optimizer(optimizer,name):
    original=optimizer.step
    def step(*args,**kwargs):
        path=WORK/'state/update_budget.json'; budget=read_json(path)
        if budget['optimizer_step_calls']>=budget['limit_optimizer_step_calls']:
            raise RuntimeError('Hard cumulative limit500 optimizer.step calls reached; no formal handoff')
        # Reserve BEFORE step, so failure never permits uncounted retries beyond the limit.
        budget['optimizer_step_calls']+=1
        budget['last_call']=dict(optimizer=name,pid=os.getpid(),time=now(),run=CFG['task'])
        atomic_json(path,budget)
        return original(*args,**kwargs)
    optimizer.step=step
    return optimizer

def start_step(step):
    global STEP_START
    torch.cuda.synchronize();STEP_START=time.perf_counter()

def record_step(run,step,optimizer,loss):
    torch.cuda.synchronize();elapsed=time.perf_counter()-STEP_START
    st=LATEST; row=dict(step=step,controller_version=CFG['controller_version'],old_q=st['old_q'],q_A=st['q_fg'],q_B=st['q_margin'],q=st['teacher_conf'],
        s=st['side_fraction'],rectangle_h=st['rectangle_h'],rectangle_w=st['rectangle_w'],actual_area=st['patch_area_ratio'],
        y=st['rectangle_y'],x=st['rectangle_x'],fallback=st['fallback'],active_classes=st['active_classes'],
        predicted_background_fraction=st['predicted_background_fraction'],background_confidence=st['background_confidence'],
        loss=float(loss.detach()),ctw_loss=r.LATEST.get('ctw_loss'),weighted_ctw_loss=r.LATEST.get('weighted_ctw_loss'),
        lr=optimizer.param_groups[0]['lr'],step_seconds=elapsed,controller_ms=CONTROLLER_MS,
        cumulative_active_seconds=r.PREVIOUS_SECONDS+time.monotonic()-r.START,peak_cuda_bytes=torch.cuda.max_memory_allocated(),
        shared_mask_and_target_identical=True)
    for c in CFG['foreground_ids']:row.update({f'n_{c}':st['foreground_counts'][str(c)],f'q_class{c}':st['q_by_class'][str(c)],f'margin_class{c}':st['margin_by_class'][str(c)]})
    path=WORK/'short_trace.csv'
    with path.open('a',newline='') as f:
        wr=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:wr.writeheader()
        wr.writerow(row);f.flush();os.fsync(f.fileno())
    budget=read_json(WORK/'state/update_budget.json');budget['joint_SSL_updates']=step
    atomic_json(WORK/'state/update_budget.json',budget)
    atomic_json(WORK/'state/short_progress.json',dict(pid=os.getpid(),time=now(),**row))
    if STOP_AT is not None and step>=STOP_AT:r.STOP=True

def checkpoint_metadata():
    return dict(version=CFG['controller_version'],foreground_ids=CFG['foreground_ids'],background_id=CFG['background_id'],
        empty_foreground_fallback=0.,min_side=.18,max_side=.38,stateless=True,
        cfg_sha256=CFG['config_sha256'])
