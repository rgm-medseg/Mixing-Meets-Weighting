"""Synthetic probability-to-mask probe only; no model or training involved."""
import argparse
import numpy as np
import torch
from pathlib import Path
from common import WORK, bootstrap, read_json, write_csv
bootstrap()
import revision_runtime as r

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args()
    dst=Path(a.output)/'synthetic_probability_mask_probe.csv'; assert not dst.exists()
    rows=[]; image=torch.zeros(1,1,256,256)
    for q in [.25,.5,.90,.97,.98,.984,.984765625,.985,.995,.99999]:
        prob=torch.tensor([q]+[(1-q)/3]*3,dtype=torch.float32)
        logits=prob.log()[None,:,None,None].expand(1,4,256,256)
        for task in ('E1','E2','E3'):
            r.CFG=read_json(next((WORK/'configs').glob(task+'_*.yaml')))
            np.random.seed(4261500)
            mask,loss,st=r.generate_adamix_bcp_lite_v1_mask(image,logits,logits)
            xy=(mask[0]==0).nonzero(); h=int(xy[:,0].max()-xy[:,0].min()+1); w=int(xy[:,1].max()-xy[:,1].min()+1)
            assert h*w==int((mask[0]==0).sum()) and torch.equal(mask,loss)
            if task in ('E2','E3'): assert h==w==97
            if task=='E1': assert h==w==round(256*(.18+.2*st['teacher_conf']))
            rows.append(dict(evidence='synthetic constant softmax distribution, not patient data or historical trajectory',
                q_requested=q,q_observed=st['teacher_conf'],task=task,s_used=st['side_fraction'],
                rectangle_h=h,rectangle_w=w,zero_pixels=h*w,actual_area=st['patch_area_ratio']))
    write_csv(dst,rows)
    print('Synthetic probability-to-mask checks passed; no model was loaded.')

if __name__=='__main__':
    torch.set_num_threads(1); main()
