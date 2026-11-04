#!/usr/bin/env python3
"""Strict CPU model/optimizer loading from the freshly downloaded backup. No training or inference."""
import argparse,hashlib,importlib.util,json,sys
from pathlib import Path
from unittest.mock import patch
import torch
p=argparse.ArgumentParser(description=__doc__);p.add_argument('--catalog',type=Path,required=True);p.add_argument('--restored',type=Path,required=True);p.add_argument('--report',type=Path,required=True)
a=p.parse_args();cat=a.catalog.resolve();project=a.restored.resolve()/'project';torch.set_num_threads(2)
sys.path.insert(0,str(project/'train_ACDC'))
import v14_mask_adamix_components as acdc
spec=importlib.util.spec_from_file_location('remote_binary_unet',project/'train_promise12/networks/unet.py');binary=importlib.util.module_from_spec(spec);spec.loader.exec_module(binary)
weights=json.loads((cat/'weights.json').read_text());experiments=json.loads((cat/'experiments.json').read_text());rows=[]
for e in experiments:
    if e['priority']!='P0':continue
    refs=[w for w in weights if e['id'] in w['experiment_ids'] and
          (w['sha256']==e['expected_best_sha256'] or w['source']==e['last_training_state']['source'])]
    for w in refs:
        path=project/w['source'];h=hashlib.sha256(path.read_bytes()).hexdigest();assert h==w['sha256']
        state=torch.load(path,map_location='cpu',weights_only=False)
        models=state.get('model_states') or {k:state[k] for k in ['model_1','model_2','ema_model'] if k in state} or {'state_dict':state}
        opts=state.get('optimizer_states') or {k:state[k] for k in ['optimizer1','optimizer2'] if k in state}
        row={'experiment':e['id'],'source':w['source'],'sha256':h,'device':'cpu','models':{},'optimizers':{}}
        for name,sd in models.items():
            with patch.object(torch.nn.Module,'cuda',lambda self,*args,**kwargs:self):
                model=(acdc.v14_bcp_net(in_chns=1,class_num=4,mask_token_enabled=True) if e['dataset']=='ACDC' else
                       binary.UNet(in_chns=1,class_num=2,use_mask_token_prehead_router=True,mask_token_num=4,mask_token_dim=16))
            model.load_state_dict(sd,strict=True);row['models'][name]='PASS'
            optname='optimizer1' if name in {'model_1','model1'} else 'optimizer2' if name in {'model_2','model2'} else None
            if optname in opts:
                opt=(acdc.make_optimizer(model,0.01,True) if e['dataset']=='ACDC' else torch.optim.SGD(model.parameters(),lr=0.01,momentum=0.9,weight_decay=0.0001))
                opt.load_state_dict(opts[optname]);row['optimizers'][optname]='PASS'
                del opt
            del model
        row['status']='PASS';rows.append(row);print(e['id'],w['source'],'CPU MODELS / OPTIMIZERS PASS',flush=True)
report={'status':'PASS','weights_checked':len(rows),'strict_model_states_checked':sum(len(r['models']) for r in rows),
        'optimizer_states_checked':sum(len(r['optimizers']) for r in rows),'rows':rows,'training_started':False,'inference_started':False,
        'paper_metrics_reproduced':False,'limitation':'No data-loader worker RNG/prefetch execution or training-step equivalence test was performed.'}
a.report.write_text(json.dumps(report,ensure_ascii=False,indent=2)+'\n')
