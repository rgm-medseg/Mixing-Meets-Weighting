"""Stateless TCSMix control. Pseudolabel supports must be provided by the caller.

No GT, trainable parameters, temporal state, confidence threshold or added
postprocessing. Old mode is the literal old generator.
"""
import numpy as np
import torch
from common import bootstrap
bootstrap()
import v14_mask_adamix_components as parent

@torch.no_grad()
def signals(logits_a, logits_b, labels_a, labels_b, foreground_ids, background_id=0):
    for x,y in [(logits_a,labels_a),(logits_b,labels_b)]:
        if x.ndim!=4 or y.shape!=(x.shape[0],*x.shape[2:]): raise ValueError('Logit/support shape mismatch')
        if not torch.isfinite(x).all(): raise ValueError('Nonfinite teacher logits: fail before optimizer update')
        if not torch.isfinite(y).all(): raise ValueError('Nonfinite pseudolabel support')
        if y.min()<0 or y.max()>=x.shape[1]: raise ValueError('Support category outside model class axis')
    if logits_a.shape[1:]!=logits_b.shape[1:]: raise ValueError('Mismatched class/grid dimensions')
    C=logits_a.shape[1]; fg=tuple(int(c) for c in foreground_ids)
    if len(set(fg))!=len(fg) or not fg or background_id in fg or any(c<0 or c>=C for c in fg):
        raise ValueError('Invalid foreground class mapping')
    pa=logits_a.detach().softmax(1); pb=logits_b.detach().softmax(1)
    old=(.5*(pa.max(1).values.mean()+pb.max(1).values.mean())).clamp(0,1).item()
    p=torch.cat([pa,pb]); labels=torch.cat([labels_a,labels_b]).long()
    qcs=[]; mcs=[]; counts={}; q_by_class={}; margin_by_class={}
    for c in fg:
        support=labels==c; n=int(support.sum()); counts[str(c)]=n
        if n:
            qc=p[:,c][support].mean().item()
            other=p[:,[k for k in range(C) if k!=c]].max(1).values
            margin=(p[:,c]-other)[support].mean().item()
            if margin < -1e-6: raise ValueError('Processed support is not an argmax category')
            q_by_class[str(c)]=qc; margin_by_class[str(c)]=max(0.,margin)
            qcs.append(qc); mcs.append(max(0.,margin))
        else: q_by_class[str(c)]=None; margin_by_class[str(c)]=None
    # Class means combine BOTH groups by pixels before taking equal class mean.
    q_fg=float(np.mean(qcs)) if qcs else 0.
    q_margin=float(np.mean(mcs)) if mcs else 0.
    background=labels==background_id
    bg_conf=p.max(1).values[background].mean().item() if background.any() else None
    return dict(old_q=old,q_fg=q_fg,q_margin=q_margin,foreground_counts=counts,
                q_by_class=q_by_class,margin_by_class=margin_by_class,active_classes=len(qcs),
                foreground_pixels=sum(counts.values()),pixels=int(labels.numel()),background_confidence=bg_conf,
                predicted_background_fraction=float(background.float().mean()),fallback=not bool(qcs))

def geometry(q,h,w,min_side=.18,max_side=.38):
    min_side=max(.05,min(.75,float(min_side))); max_side=max(min_side,min(.90,float(max_side)))
    s=min_side+(max_side-min_side)*float(q)
    ph=max(1,min(h-1,int(round(h*s)))); pw=max(1,min(w-1,int(round(w*s))))
    return s,ph,pw

@torch.no_grad()
def generate_mask(img, logits_a, logits_b, labels_a, labels_b, cfg, min_side=.18,max_side=.38):
    version=cfg.get('controller_version','old')
    if version=='old': return parent.generate_adamix_bcp_lite_v1_mask(img,logits_a,logits_b,min_side,max_side)
    st=signals(logits_a,logits_b,labels_a,labels_b,cfg['foreground_ids'],cfg['background_id'])
    if version=='v2_A': q=st['q_fg']
    elif version=='v2_B': q=st['q_margin']
    elif version=='v2_fixed':
        if cfg.get('fixed_side') is None: raise ValueError('R2 fixed side not locked from R1 full trace')
        q=st['q_fg']
    else: raise ValueError('Unknown controller version')
    b,_,h,w=img.shape; s,ph,pw=geometry(q,h,w,min_side,max_side)
    if version=='v2_fixed':
        s=float(cfg['fixed_side']); ph=max(1,min(h-1,round(h*s))); pw=max(1,min(w-1,round(w*s)))
    y=np.random.randint(0,max(1,h-ph)); x=np.random.randint(0,max(1,w-pw))
    mask=torch.ones(h,w,device=img.device)
    mask[y:y+ph,x:x+pw]=0
    batch=mask.unsqueeze(0).expand(b,-1,-1).long(); loss=batch.clone()
    patch=logits_a.argmax(1)[:,y:y+ph,x:x+pw]
    hist=[float((patch==c).float().mean()) for c in range(logits_a.shape[1])]
    st.update(teacher_conf=q,side_fraction=s,patch_area_ratio=ph*pw/(h*w),rectangle_h=ph,rectangle_w=pw,
              rectangle_y=int(y),rectangle_x=int(x),controller_version=version,
              class_histogram=hist,foreground_coverage=1-hist[cfg['background_id']],finite=True)
    return batch,loss,st
