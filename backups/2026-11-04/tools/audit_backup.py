#!/usr/bin/env python3
"""CPU-only deserialization, state inventory, model compatibility, and scalar export."""
import csv, hashlib, importlib.util, json, math, os, platform, struct, subprocess, sys
from collections import Counter
from pathlib import Path
from unittest.mock import patch
import torch

TASK=Path(__file__).resolve().parents[1]; CAT=TASK/'catalog'; ROOT=Path('/home/renguanming/jupyterlab/rgm/beta-FFT-main')
torch.set_num_threads(2)
weights=json.loads((CAT/'weights.json').read_text()); experiments=json.loads((CAT/'experiments.json').read_text())
def save(name,data):
    p=CAT/name;p.parent.mkdir(parents=True,exist_ok=True);p.write_text(json.dumps(data,ensure_ascii=False,indent=2)+'\n')
def tensor_meta(x):return {'shape':list(x.shape),'dtype':str(x.dtype)}
def loadmod(name,path):
    spec=importlib.util.spec_from_file_location(name,path);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m);return m

audit={}; modelchecks=[]
for w in weights:
    sha=w['sha256']
    if sha in audit:continue
    p=TASK/'payload'/w['object']; row={'sha256':sha,'cpu_load':'NOT_RUN','source_references':[v['source'] for v in weights if v['sha256']==sha]}
    try:
        state=torch.load(p,map_location='cpu',weights_only=False)
        row.update(cpu_load='PASS',top_level_keys=list(state) if isinstance(state,dict) else [],root_type=type(state).__name__)
        if not isinstance(state,dict):raise ValueError('not a checkpoint dictionary')
        models=state.get('model_states',{})
        if not models:models={k:state[k] for k in ['model_1','model_2','ema_model','net','model'] if isinstance(state.get(k),dict)}
        if not models and all(isinstance(x,torch.Tensor) for x in state.values()):models={'state_dict':state}
        opts=state.get('optimizer_states',{}) or {k:state[k] for k in ['optimizer1','optimizer2','opt'] if k in state}
        row.update(model_states={k:{'tensor_count':sum(isinstance(v,torch.Tensor) for v in m.values()),'key_count':len(m)} for k,m in models.items()},
            optimizer_states={k:{'state_entries':len(o.get('state',{})),'param_groups':len(o.get('param_groups',[])),
                                  'lr':[g.get('lr') for g in o.get('param_groups',[])]} for k,o in opts.items() if isinstance(o,dict)},
            iteration=state.get('global_step',state.get('completed_self_train_updates')),
            scheduler_state=state.get('scheduler_state'),rng_keys=list((state.get('random_state') or state.get('rng_state') or {}).keys()),
            sampler_keys=(list(state['sampler_state']) if isinstance(state.get('sampler_state'),dict) else
                          ['list_state_entries='+str(len(state['sampler_state']))] if isinstance(state.get('sampler_state'),list) else []),
            amp_scaler_present=any('scaler' in k.lower() or k.lower()=='amp' for k in state),
            args=state.get('args'),lineage=state.get('lineage'),teacher_state=state.get('teacher_state'),amp_state=state.get('amp_state'),
            unexpected_image_keys=[k for k in state if any(s in k.lower() for s in ['image','volume','segmentation','ground_truth'])])
        has_two=('model_1' in models and 'model_2' in models) or ('model1' in models and 'model2' in models)
        full=has_two and 'ema_model' in models and len(opts)>=2 and bool(row['rng_keys']) and row['iteration'] is not None
        row['resume_status']=('TRAINING_STATE_PRESENT_WITH_WORKER_RNG_LIMITATION' if full and not row['sampler_keys'] else
                              'TRAINING_STATE_PRESENT_WITH_SAMPLER' if full else '可加载评估，恢复训练状态不完整')
        row['precision_note']='FP32 in retained main source; no AMP/GradScaler calls. For controls, inspect amp_state and saved configuration.'
        if full and not row['scheduler_state']:
            row['scheduler_note']='ACDC historical constant LR; optimizer group LR retained, no scheduler object used. Verify source for non-main candidates.'
        row['privacy_scan']='No image/ground-truth payload keys found; model, optimizer, RNG, sampler indices, args and lineage only.' if not row['unexpected_image_keys'] else 'REVIEW_REQUIRED'
    except Exception as ex:row.update(cpu_load='FAIL',error=type(ex).__name__+': '+str(ex)[:500])
    audit[sha]=row
    print('CPU',len(audit),w['source'],row['cpu_load'],flush=True)

# Exact model definitions; intercept only constructor .cuda() calls so this validation stays on CPU.
sys.path.insert(0,str(CAT/'records/project/train_ACDC'))
import v14_mask_adamix_components as acdc
binary=loadmod('backup_binary_unet',CAT/'records/project/train_promise12/networks/unet.py')
for e in experiments:
    if e['priority']!='P0':continue
    sources=[w for w in weights if e['id'] in w['experiment_ids'] and (w['sha256']==e['expected_best_sha256'] or 'last.ckpt' in w['source'] or 'full_state_final' in w['source'])]
    for w in sources:
        row={'experiment':e['id'],'source':w['source'],'sha256':w['sha256'],'device':'cpu','strict':True}
        try:
            d=torch.load(TASK/'payload'/w['object'],map_location='cpu',weights_only=False)
            states=d.get('model_states') or {k:d[k] for k in ['model_1','model_2','ema_model'] if k in d} or {'state_dict':d}
            checks={}
            for name,sd in states.items():
                with patch.object(torch.nn.Module,'cuda',lambda self,*a,**kw:self):
                    model=(acdc.v14_bcp_net(in_chns=1,class_num=4,mask_token_enabled=True) if e['dataset']=='ACDC' else
                           binary.UNet(in_chns=1,class_num=2,use_mask_token_prehead_router=True,mask_token_num=4,mask_token_dim=16))
                model.load_state_dict(sd,strict=True);checks[name]={'status':'PASS','keys':len(sd),'parameters':sum(p.numel() for p in model.parameters())}
                del model
            row.update(status='PASS',models=checks)
        except Exception as ex:row.update(status='FAIL',error=type(ex).__name__+': '+str(ex)[:1500])
        modelchecks.append(row);print('STRUCTURE',e['id'],w['source'],row['status'],flush=True)

save('reports/checkpoint_audit.json',audit);save('reports/model_compatibility.json',modelchecks)
for e in experiments:
    own=[w for w in weights if e['id'] in w['experiment_ids'] and w['source'].startswith(e['run_root']+'/')]
    e['checkpoint_inspection']=[{'source':w['source'],'sha256':w['sha256'],'iteration':audit[w['sha256']].get('iteration'),
                               'cpu_load':audit[w['sha256']]['cpu_load'],'resume_status':audit[w['sha256']].get('resume_status')} for w in own]
save('experiments.json',experiments)

# TensorBoardX protobuf decodes standard TFRecord Event messages. Never store encoded image/audio/tensor arrays.
from tensorboardX.proto.event_pb2 import Event
tb=[]; scalar_files=[]
for source in json.loads((CAT/'reports/tensorboard_sources.json').read_text()):
    p=ROOT/source['source']; counts=Counter(); nscalar=0; complete=True
    dest=CAT/'scalars'/(source['sha256']+'.csv');dest.parent.mkdir(exist_ok=True)
    try:
        with p.open('rb') as f,dest.open('w',newline='') as out:
            writer=csv.writer(out);writer.writerow(['wall_time','step','tag','value'])
            while True:
                header=f.read(12)
                if not header:break
                if len(header)!=12:complete=False;break
                length=struct.unpack('<Q',header[:8])[0]
                if length>2**30:raise ValueError('unreasonable event length')
                payload=f.read(length);crc=f.read(4)
                if len(payload)!=length or len(crc)!=4:complete=False;break
                event=Event();event.ParseFromString(payload);counts['records']+=1
                for val in event.summary.value:
                    kind=val.WhichOneof('value');counts[kind or 'unknown']+=1
                    value=None
                    if kind=='simple_value':value=float(val.simple_value)
                    elif kind=='tensor':
                        t=val.tensor
                        # A scalar tensor has no dimensions. Non-scalars, images and embeddings are excluded.
                        if not t.tensor_shape.dim:
                            for field in ['float_val','double_val','int_val','int64_val','bool_val']:
                                values=getattr(t,field,[])
                                if len(values)==1:value=float(values[0]);break
                            if value is None and t.tensor_content:
                                fmt={1:'f',2:'d',3:'i',9:'q'}.get(t.dtype)
                                if fmt and len(t.tensor_content)==struct.calcsize(fmt):value=struct.unpack('<'+fmt,t.tensor_content)[0]
                    if value is not None and math.isfinite(value):
                        writer.writerow([event.wall_time,event.step,val.tag,value]);nscalar+=1
        scalar_files.append({'backup_path':str(dest.relative_to(CAT)),'sha256':hashlib.sha256(dest.read_bytes()).hexdigest(),'size':dest.stat().st_size,
                             'experiment_ids':source['experiment_ids'],'source_event':source['source'],'kind':'tensorboard_scalars'})
        status='PASS' if complete else 'PARTIAL_TRAILING_RECORD'
    except Exception as ex:status='FAIL: '+str(ex)[:300]
    tb.append({**source,'summary_types':dict(counts),'scalar_rows':nscalar,'status':status,'raw_event_uploaded':False,
               'contains_image':bool(counts['image']),'scalar_file':str(dest.relative_to(CAT))})
    print('TB',source['source'],status,nscalar,dict(counts),flush=True)
save('reports/tensorboard_audit.json',tb)
files=json.loads((CAT/'files.json').read_text());files.extend(scalar_files);save('files.json',files)
env={'python':platform.python_version(),'pytorch':torch.__version__,'torch_cuda':torch.version.cuda,'cudnn':torch.backends.cudnn.version(),
     'platform':platform.platform(),'capture_kind':'Backup-time existing beta_fft environment; historical environment records are separately retained.'}
save('reports/environment_backup_time.json',env)
r=subprocess.run([sys.executable,'-m','pip','freeze'],capture_output=True,text=True)
(CAT/'reports/pip_freeze_backup_time.txt').write_text(r.stdout)
print('AUDIT COMPLETE',len(audit),len(modelchecks),len(tb),flush=True)
