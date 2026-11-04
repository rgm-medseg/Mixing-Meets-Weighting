"""Controlled probability tests; zero optimizer updates."""
import torch
import numpy as np
from common import WORK,atomic_json,write_csv,now
from controller import signals,geometry,generate_mask

def case(values,classes):
    p=torch.tensor(values,dtype=torch.float32).T[None,:,None,:]
    return p.log(),torch.tensor(classes)[None,None,:]

def main():
    rows=[]; ids=[1,2,3]
    a,l=case([[.1,.7,.1,.1],[.1,.1,.6,.2]],[1,2])
    s=signals(a,a,l,l,ids); assert abs(s['q_fg']-.65)<1e-6
    # Add arbitrarily many background pixels with the SAME foreground probabilities.
    bg=[[.997,.001,.001,.001]]*100
    b,m=case([[.1,.7,.1,.1],[.1,.1,.6,.2]]+bg,[1,2]+[0]*100)
    t=signals(b,b,m,m,ids); assert abs(s['q_fg']-t['q_fg'])<1e-6 and abs(s['q_margin']-t['q_margin'])<1e-6 and t['old_q']>s['old_q']
    rows.extend([dict(test='background multiplicity',condition=k,**{v:x[v] for v in ['old_q','q_fg','q_margin','fallback']}) for k,x in [('two fg pixels',s),('same fg +100 bg',t)]])
    # Class imbalance: duplicate class1 by 100; class-balanced result unchanged.
    c,n=case([[.1,.7,.1,.1]]*100+[[.1,.1,.6,.2]],[1]*100+[2]); u=signals(c,c,n,n,ids)
    assert abs(s['q_fg']-u['q_fg'])<1e-6 and abs(s['q_margin']-u['q_margin'])<1e-6
    rows.append(dict(test='class imbalance 100:1',condition='same per-class P',old_q=u['old_q'],q_fg=u['q_fg'],q_margin=u['q_margin'],fallback=u['fallback']))
    bglog,bglab=case([[.997,.001,.001,.001]],[0]); v=signals(bglog,bglog,bglab,bglab,ids)
    assert v['fallback'] and v['q_fg']==0 and v['q_margin']==0 and geometry(0,256,256)[1:]==(46,46)
    rows.append(dict(test='all background',condition='fallback0',old_q=v['old_q'],q_fg=v['q_fg'],q_margin=v['q_margin'],fallback=v['fallback']))
    previous=(-1,-1,-1)
    for q in [.25,.4,.6,.8,.99,1.]:
        prob=[(1-q)/3,q,(1-q)/3,(1-q)/3]
        # finite extreme logits instead of log(0)
        x=torch.tensor(prob).clamp_min(1e-30).log()[None,:,None,None]; y=torch.ones(1,1,1,dtype=torch.long)
        st=signals(x,x,y,y,ids); side=geometry(st['q_fg'],256,256)[1]
        assert st['q_fg']>=previous[0] and st['q_margin']>=previous[1] and side>=previous[2]
        previous=(st['q_fg'],st['q_margin'],side)
        rows.append(dict(test='foreground monotonicity',condition=q,old_q=st['old_q'],q_fg=st['q_fg'],q_margin=st['q_margin'],fallback=st['fallback'],side_pixels=side))
    failures=[]
    for bad in [float('nan'),float('inf'),float('-inf')]:
        x=a.clone(); x[0,0,0,0]=bad
        try: signals(x,a,l,l,ids)
        except ValueError: failures.append(str(bad))
        else: raise AssertionError('Nonfinite teacher not rejected')
    x=a.clone().requires_grad_(); st=signals(x,a,l,l,ids); assert isinstance(st['q_fg'],float) and x.grad is None
    # Same final geometry/RNG gives exactly the old mask; tests image/target and batch sharing.
    img=torch.zeros(6,1,256,256); logits=torch.full((6,4,256,256),-8.); logits[:,1]=8.; lab=torch.ones(6,256,256,dtype=torch.long)
    masks=[]
    for ver in ['old','v2_A']:
        np.random.seed(42); mm,ll,_=generate_mask(img,logits,logits,lab,lab,dict(controller_version=ver,foreground_ids=ids,background_id=0))
        assert torch.equal(mm,ll) and torch.equal(mm[0].expand_as(mm),mm)
        masks.append(mm)
    assert torch.equal(*masks)
    # Generic class axis: the controller contains no fixed ACDC class count.
    for probs,fgs in [([.2,.8],[1]),([.05,.05,.05,.05,.8],[1,2,3,4])]:
        x=torch.tensor(probs).log()[None,:,None,None];y=torch.full((1,1,1),int(np.argmax(probs)))
        st=signals(x,x,y,y,fgs);assert st['active_classes']==1 and abs(st['q_fg']-.8)<1e-6
    np.random.seed(42)
    mm,ll,st=generate_mask(img,logits,logits,lab,lab,dict(controller_version='v2_fixed',signal_version='v2_B',fixed_side=90/256,foreground_ids=ids,background_id=0))
    assert st['teacher_conf']==st['q_margin'] and st['rectangle_h']==90
    write_csv(WORK/'analysis/controlled_probability_checks.csv',rows,list(dict.fromkeys(k for r in rows for k in r)))
    atomic_json(WORK/'reports/controller_checks.json',dict(status='PASS',time=now(),optimizer_updates=0,
        checks=['background multiplicity invariance','foreground class-size invariance','all background fallback0','monotonic probability/margin/size',
                'constant/extreme probabilities and missing classes','NaN/Inf fail closed','detach/no gradient','shared mask equals target mask','same-geometry old/new mask equality',
                'generic2/5class mappings','planned R2 keeps B signal logs while fixing geometry'],
        nonfinite_rejected=failures,scope='controlled tensors, not medical efficacy'))

if __name__=='__main__': torch.set_num_threads(1); main()
