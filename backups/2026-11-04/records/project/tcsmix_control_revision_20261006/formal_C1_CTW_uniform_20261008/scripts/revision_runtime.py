"""Explicit scientific controls and durable state for the frozen Full loop."""
import copy
import csv
import json
import os
import random
import signal
import time
import types
from pathlib import Path
import numpy as np
import torch
from torch import nn
from common import WORK, atomic_json, now, bootstrap
bootstrap()
import v14_mask_adamix_components as parent
from dataloaders.dataset import TwoStreamBatchSampler as ParentSampler

CFG = {}
STOP = False
SAMPLER = None
SAMPLER_RESTORE = None
LATEST = {}
START = time.monotonic()
PREVIOUS_SECONDS = 0.
PREVIOUS_PEAK = 0

def configure(cfg):
    global CFG, STOP, START, PREVIOUS_SECONDS, PREVIOUS_PEAK
    CFG = cfg
    STOP = False
    START = time.monotonic()
    PREVIOUS_SECONDS = 0.
    PREVIOUS_PEAK = 0
    progress=Path(cfg.get('run_root',WORK/'runs/not_a_training_run'))/'progress.json'
    if cfg.get('_resume') and progress.exists():
        old=json.loads(progress.read_text())
        PREVIOUS_SECONDS=float(old.get('wall_seconds',0))
        PREVIOUS_PEAK=int(old.get('peak_cuda_bytes',0))
    signal.signal(signal.SIGTERM, request_stop)
    signal.signal(signal.SIGINT, request_stop)

def request_stop(signum, frame):
    global STOP
    STOP = True

def atomic_torch(value, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp.' + str(os.getpid()))
    torch.save(value, tmp)
    with tmp.open('rb') as f:
        os.fsync(f.fileno())
    os.replace(tmp, path)

def rng_state():
    return dict(python_random=random.getstate(), numpy_random=np.random.get_state(),
                torch_random=torch.get_rng_state(),torch_cuda_random=torch.cuda.get_rng_state_all())

def restore_rng(s):
    random.setstate(s['python_random'])
    np.random.set_state(s['numpy_random'])
    torch.set_rng_state(s['torch_random'])
    torch.cuda.set_rng_state_all(s['torch_cuda_random'])

class StatefulSampler(ParentSampler):
    """Fresh sampling calls the exact parent iterator; records prefetched IDs.

    Mid-epoch restart replays saved remaining batch IDs. Worker augmentation
    RNG/prefetch state cannot be restored; this is explicitly not exact resume.
    """
    def __init__(self, *args, **kwargs):
        global SAMPLER
        super().__init__(*args, **kwargs)
        self.batches = []
        self.replay = None
        SAMPLER = self

    def __iter__(self):
        if self.replay is not None:
            replay = self.replay
            self.replay = None
            self.batches = list(replay['batches'])
            yield from self.batches[replay['consumed']:]
        else:
            self.batches = []
            for batch in super().__iter__():
                self.batches.append(tuple(int(x) for x in batch))
                yield batch

def sampler_state(step):
    if SAMPLER is None:
        return None
    consumed = step % len(SAMPLER)
    return dict(batches=SAMPLER.batches, consumed=consumed,
                boundary='worker RNG/prefetch unavailable; sampled IDs/main RNG saved')

def set_sampler_resume(state):
    global SAMPLER_RESTORE
    SAMPLER_RESTORE = state
    if SAMPLER is not None and state and state['consumed']:
        if len(state['batches']) < len(SAMPLER):
            raise RuntimeError('Incomplete saved batch plan; cannot resume safely')
        SAMPLER.replay = state

def resumed_iterator(loader, resumed, rng):
    iterator = iter(loader)
    # DataLoader consumes a base seed. Preserve saved main RNG on restart.
    if resumed:
        restore_rng(rng)
    return iterator

class BottleneckResidual(nn.Module):
    def __init__(self, in_channels=16, width=18):
        super().__init__()
        self.project = nn.Conv2d(in_channels,width,1,bias=True)
        self.norm = nn.LayerNorm(width)
        self.activation = nn.GELU()
        self.inverse_project = nn.Conv2d(width,in_channels,1,bias=True)
        self.gamma = nn.Parameter(torch.zeros(1))
        self.last_stats = {}

    def forward(self,x):
        y=self.project(x).permute(0,2,3,1)
        y=self.activation(self.norm(y)).permute(0,3,1,2)
        residual=self.inverse_project(y)
        return x+self.gamma*residual

def BCP_net(in_chns=1,class_num=4,ema=False,mask_token_enabled=True,mask_token_num=4,mask_token_dim=16):
    if CFG.get('trc')!='bottleneck':
        return parent.v14_bcp_net(in_chns,class_num,ema,mask_token_enabled,mask_token_num,mask_token_dim)
    model=parent._parent_bcp_net(in_chns,class_num,ema=False)
    decoder=model.decoder
    decoder.mask_token_router=BottleneckResidual(16,CFG['bottleneck_width']).cuda()
    def forward(self,features):
        x0,x1,x2,x3,x4=features
        x=self.up4(self.up3(self.up2(self.up1(x4,x3),x2),x1),x0)
        x=self.mask_token_router(x)
        return self.out_conv(x),x
    decoder.forward=types.MethodType(forward,decoder)
    if ema:
        for p in model.parameters():
            p.detach_()
    return model

def mask_token_endpoint(model):
    router=model.decoder.mask_token_router
    if isinstance(router,BottleneckResidual):
        return dict(type='plain_1x1_LN_GELU_1x1_residual',gamma=router.gamma.item(),
                    parameters=sum(p.numel() for p in router.parameters()))
    return parent.mask_token_endpoint(model)

def ctw_maps(out1,out2,uniform=False):
    p1,p2=out1.softmax(1),out2.softmax(1)
    with torch.no_grad():
        q=(p1.detach().max(1).values+p2.detach().max(1).values)*.5
        dx=torch.zeros_like(q); dy=torch.zeros_like(q)
        dx[:,1:,:]=(q[:,1:,:]-q[:,:-1,:]).abs()
        dy[:,:,1:]=(q[:,:,1:]-q[:,:,:-1]).abs()
        raw=dx+dy
        lo=raw.flatten(1).min(1).values[:,None,None]
        hi=raw.flatten(1).max(1).values[:,None,None]
        normalized=((raw-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
        original=(.25+normalized).clamp(.25,1.25)
        weight=original.mean((1,2),keepdim=True).expand_as(original) if uniform else original
    error=(p1-p2.detach()).abs().mean(1)
    return raw,normalized,original,weight,error

def compute_v21_3_innovation_x(candidate_id,out1,out2,*unused):
    raw,norm,original,w,error=ctw_maps(out1,out2,CFG['weight_mode']=='uniform')
    loss=(w*error).mean()
    stats=dict(raw_weight_mean=original.mean().item(),raw_weight_min=original.min().item(),raw_weight_max=original.max().item(),
               used_weight_mean=w.mean().item(),used_weight_min=w.min().item(),used_weight_max=w.max().item(),
               max_per_image_mean_error=(w.mean((1,2))-original.mean((1,2))).abs().max().item(),
               ctw_loss=loss.detach().item(),weighted_ctw_loss=.05*loss.detach().item())
    audit_start=time.perf_counter()
    spatial_mean=original.mean((1,2));used_mean=w.mean((1,2))
    used_range=(w.flatten(1).max(1).values-w.flatten(1).min(1).values).max().item()
    assert CFG['weight_mode']=='uniform' and used_range==0
    assert stats['max_per_image_mean_error']<=1e-6 and not w.requires_grad
    stats.update(weight_mode=CFG['weight_mode'],used_per_image_spatial_range_max=used_range,
        raw_per_image_means=json.dumps(spatial_mean.tolist()),used_per_image_means=json.dumps(used_mean.tolist()),
        weight_audit_seconds=time.perf_counter()-audit_start)
    LATEST.update(stats)
    return loss,stats

def generate_adamix_bcp_lite_v1_mask(img,la,lb,min_side=.18,max_side=.38):
    if CFG['mix_mode']=='dynamic':
        result=parent.generate_adamix_bcp_lite_v1_mask(img,la,lb,min_side,max_side)
    else:
        b,_,h,w=img.shape
        with torch.no_grad():
            q=.5*(la.detach().softmax(1).max(1).values.mean()+lb.detach().softmax(1).max(1).values.mean())
        s=CFG['fixed_side']; ph=max(1,min(h-1,round(h*s))); pw=max(1,min(w-1,round(w*s)))
        y=np.random.randint(0,max(1,h-ph)); x=np.random.randint(0,max(1,w-pw))
        mask=torch.ones((b,h,w),device=img.device,dtype=torch.long)
        mask[:,y:y+ph,x:x+pw]=0
        result=(mask,mask.clone(),dict(teacher_conf=q.item(),side_fraction=s,patch_area_ratio=ph*pw/(h*w)))
    LATEST.update(q=result[2]['teacher_conf'],s=result[2]['side_fraction'],actual_area=result[2]['patch_area_ratio'])
    return result

def record_step(run,step,optimizer,loss):
    row=dict(step=step,wall_seconds=PREVIOUS_SECONDS+time.monotonic()-START,lr=optimizer.param_groups[0]['lr'],total_loss=loss.detach().item(),**LATEST)
    path=Path(run)/'audit/revision_trace.csv'
    path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0: writer.writeheader()
        writer.writerow(row)
        f.flush()
    atomic_json(Path(run)/'progress.json',dict(task=CFG['task'],pid=os.getpid(),time=now(),peak_cuda_bytes=max(PREVIOUS_PEAK,torch.cuda.max_memory_allocated()),**row))

def completion(run,step,interrupted=False):
    atomic_json(Path(run)/'TRAIN_RECEIPT.json',dict(task=CFG['task'],status='INTERRUPTED' if interrupted else 'TRAINED',
                 global_step=step,time=now(),pid=os.getpid(),training_seconds=PREVIOUS_SECONDS+time.monotonic()-START,
                 wall_time_convention='Sum of active process segments; downtime excluded, unsaved final crash interval unavailable; replayed updates retained in trace',
                 resume_exact=False,resume_boundary='saved IDs and main RNG; worker augmentation streams restart',
                 test_access_during_training=0))
