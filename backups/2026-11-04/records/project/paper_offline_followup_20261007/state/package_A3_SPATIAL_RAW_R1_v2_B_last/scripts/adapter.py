"""Read-only adapters: frozen network, literal pseudo-label and controller functions."""
import ast,sys,hashlib
import numpy as np,torch
import torch.nn.functional as F
from scipy.ndimage import zoom
from common import *
sys.path.insert(0,str(W/'scripts/frozen/train_ACDC'))
import v14_mask_adamix_components as parent
def functions(source,names):
 ns=dict(torch=torch,np=np,F=F)
 for node in ast.parse(Path(source).read_text()).body:
  if isinstance(node,ast.FunctionDef) and node.name in names:
   exec(compile(ast.Module(body=[node],type_ignores=[]),str(source),'exec'),ns)
 return ns
PSEUDO={}
def pseudo(logits,method):
 s=method['source_path']
 if s not in PSEUDO:PSEUDO[s]=functions(s,['get_ACDC_2DLargestCC','get_ACDC_masks'])
 return PSEUDO[s]['get_ACDC_masks'](logits,nms=1).long()
ns=functions(W/'scripts/frozen/controller.py',['signals','geometry'])
signals=ns['signals'];geometry=ns['geometry']
def model(method,cp,branch):
 bind_gpu_pid()
 ck=torch.load(cp['path'],map_location='cpu',weights_only=False)
 m=parent.v14_bcp_net(1,4,ema=branch=='ema_model',mask_token_enabled=method['trc'])
 m.load_state_dict(ck[branch],strict=True);m.eval()
 assert int(ck['global_step'])==cp['step']
 receipt=dict(method_id=method['id'],checkpoint_sha256=cp['sha256'],global_step=ck['global_step'],branch=branch,strict=True,state_keys=len(m.state_dict()),parameters=sum(p.numel() for p in m.parameters()),trainable_parameters=sum(p.numel() for p in m.parameters() if p.requires_grad))
 del ck
 return m,receipt
def digest(m):
 h=hashlib.sha256()
 for k,v in m.state_dict().items():h.update(k.encode());h.update(v.detach().cpu().contiguous().numpy().tobytes())
 return h.hexdigest()
@torch.inference_mode()
def control(logits,labels,method):
 st=signals(logits,logits,labels,labels,[1,2,3],0)
 if method['controller']=='old':q=st['old_q'];s,h,w=geometry(q,256,256);definition='mean max class probability, all pixels'
 elif method['controller']=='none':q=None;s=h=w=None;definition='NA: baseline original random rectangle, no TCSMix control'
 else:
  q=st['q_margin'];s,h,w=geometry(q,256,256);definition='equal foreground-class mean probability margin on original per-class LCC support'
  if method['controller']=='fixed':s=92/256;h=w=92;definition+='; fixed geometry'
 return dict(sample_control_proxy=q if q is not None else 'NA',batch_control_value='NA: no historical unlabeled batch replay',control_definition=definition,sample_output_side=s if s is not None else 'NA',sample_rectangle_h=h if h is not None else 'NA',sample_rectangle_w=w if w is not None else 'NA',active_foreground_classes=st['active_classes'],fallback=st['fallback'],q_B_auxiliary=st['q_margin'],old_q_auxiliary=st['old_q'],support_counts={c:st['foreground_counts'][str(c)]//2 for c in [1,2,3]})
def volume(m,image):
 result=np.zeros(image.shape,dtype=np.uint8)
 with torch.inference_mode():
  for z,a in enumerate(image):
   h,w=a.shape;x=zoom(a,(256/h,256/w),order=0)
   out=m(torch.from_numpy(x).float()[None,None].cuda())
   y=out.softmax(1).argmax(1)[0].cpu().numpy();result[z]=zoom(y,(h/256,w/256),order=0)
 return result
