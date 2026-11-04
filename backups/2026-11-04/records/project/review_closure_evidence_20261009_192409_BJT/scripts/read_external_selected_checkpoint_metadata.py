"""CPU metadata of already selected comparator checkpoints; no selection or inference."""
import csv
import gc
import hashlib
import json
import os
from pathlib import Path
os.environ['CUDA_VISIBLE_DEVICES']=''
import torch
torch.set_num_threads(1)
OUT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

rows=list(csv.DictReader((OUT/'TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv').open()))
meta=[]
for row in rows:
    if row['paper_method'] not in ['HiEn','PMEUL','UCAD']:continue
    p=Path(row['checkpoint']);d=torch.load(p,map_location='cpu',weights_only=False)
    m={'row_id':row['row_id'],'path':str(p),'sha256':sha(p),'keys':list(d),
       'loaded_device':'CPU','model_created':False,'forward_called':False,'checkpoint_selection_changed':False}
    for k in ['iteration','iter_num','iter','global_step','epoch','validation_dice','best_val_dice','best_validation_dice','selected_model_role','config_sha256','schema','run_id']:
        if k in d and isinstance(d[k],(str,int,float,bool)):m[k]=d[k]
    step=m.get('iteration',m.get('iter_num',m.get('iter',m.get('global_step',''))))
    row['checkpoint_selected_step']=step
    row['selected_checkpoint_metadata_source']='EXTERNAL_SELECTED_CHECKPOINT_CPU_METADATA.json'
    row['selected_model_role']=m.get('selected_model_role','')
    meta.append(m);del d;gc.collect()
    print(row['historical_run'],'selected',step,m.get('selected_model_role',''),flush=True)
(OUT/'EXTERNAL_SELECTED_CHECKPOINT_CPU_METADATA.json').write_text(json.dumps(meta,ensure_ascii=False,indent=2)+'\n')
with (OUT/'TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv').open('w',newline='') as f:
    fields=list(dict.fromkeys(k for r in rows for k in r));w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
