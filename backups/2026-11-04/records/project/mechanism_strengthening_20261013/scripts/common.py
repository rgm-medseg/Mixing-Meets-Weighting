import csv,hashlib,json,os,sys
from pathlib import Path
import numpy as np, torch
W=Path(__file__).resolve().parents[1]; ROOT=W.parent
sys.path.insert(0,str(W/'scripts/frozen'))
CFG=json.loads((W/'configs/diagnostics.json').read_text()); DATA=Path(CFG['data_root'])
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def js(p,x):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True); tmp=p.with_suffix(p.suffix+'.tmp'); tmp.write_text(json.dumps(x,indent=2,allow_nan=False)+'\n'); os.replace(tmp,p)
def csvout(p,rows):
    p=Path(p); p.parent.mkdir(parents=True,exist_ok=True)
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fields); w.writeheader(); w.writerows(rows)
def digest(m):
    h=hashlib.sha256()
    for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
    return h.hexdigest()
def load(branch,checkpoint=None,freeze=True):
    from v14_mask_adamix_components import v14_bcp_net
    ck=torch.load(checkpoint or CFG['main_checkpoint'],map_location='cpu',weights_only=False)
    m=v14_bcp_net(1,4,ema=branch=='ema_model',mask_token_enabled=True,mask_token_num=4,mask_token_dim=16)
    m.load_state_dict(ck[branch],strict=True);m.eval()
    if freeze:
        for p in m.parameters():p.requires_grad_(False)
    return m,dict(branch=branch,step=ck['global_step'],strict=True,checkpoint_sha256=sha(checkpoint or CFG['main_checkpoint']))
def band(b,width=2):
    from scipy.ndimage import distance_transform_edt
    return distance_transform_edt(~b)<=width if b.any() else np.zeros_like(b,bool)
def boundary(g):
    b=np.zeros(g.shape,bool)
    for axis in [-2,-1]:
        i=[slice(None)]*g.ndim;j=i.copy();i[axis]=slice(1,None);j[axis]=slice(None,-1)
        delta=g[tuple(i)]!=g[tuple(j)];b[tuple(i)]|=delta;b[tuple(j)]|=delta
    return b
def dice(p,g):
    n=int(p.sum()+g.sum());return 2*int((p&g).sum())/n if n else 0.
