import gc,threading
import numpy as np,torch,h5py
from scipy.ndimage import zoom
from medpy.metric import binary
from common import *
from deployment import load,deploy,digest,volume
def backend():
 torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 torch.backends.cuda.matmul.allow_tf32=False;torch.backends.cudnn.allow_tf32=True
def imagehash(x):return hashlib.sha256(np.ascontiguousarray(x).tobytes()).hexdigest()
def metric(a,b):
 if a.any() and b.any():return dict(Dice=float(binary.dc(a,b)),IoU=float(binary.jc(a,b)),HD95=float(binary.hd95(a,b)),ASD=float(binary.asd(a,b)))
 d=0. if not a.any() and not b.any() else float(np.sqrt(sum(s*s for s in a.shape)))
 return dict(Dice=0.,IoU=0.,HD95=d,ASD=d)
def preloaded():
 out={}
 for frame in config()['val_cases']:
  with h5py.File(DATA/'data'/f'{frame}.h5') as f:out[frame]=(f['image'][:],f['label'][:])
 return out
class Watch:
 def __enter__(self):
  self.records=[dict(time=now(),**gpu())];self.end=threading.Event()
  def run():
   while not self.end.wait(.4):self.records.append(dict(time=now(),**gpu()))
  self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
 def __exit__(self,*args):self.end.set();self.thread.join();self.records.append(dict(time=now(),**gpu()))
 @property
 def valid(self):return all(not(set(r['pids'])-OWN_GPU_PIDS) for r in self.records)
def unload():gc.collect();torch.cuda.empty_cache()
def equivalence():
 backend();data=preloaded();ud=unitdir('EQUIVALENCE');cfg=config()
 for method in cfg['methods']:
  cp=method['checkpoints'][0];pending=[f for f in cfg['val_cases'] if not(ud/f"{method['id']}_{f}.json").exists()]
  if not pending:continue
  wait_gpu(own=torch.cuda.is_initialized());m,receipt=load(method,cp);before=digest(m)
  original_router=m.decoder.mask_token_router.forward if method['trc'] else None;original_decoder=m.decoder.forward
  for frame in pending:
   stopped();wait_gpu(own=True);image,gt=data[frame];orig=[];pred0=np.zeros(image.shape,np.uint8);pred1=pred0.copy();errs=[]
   if method['trc']:m.decoder.mask_token_router.forward=original_router;m.decoder.forward=original_decoder
   with torch.inference_mode():
    for z,a in enumerate(image):
     h,w=a.shape;x=zoom(a,(256/h,256/w),order=0).astype(np.float32)
     log=m(torch.from_numpy(x)[None,None].cuda());orig.append(log.cpu().numpy());p=log.softmax(1).argmax(1)[0].cpu().numpy();pred0[z]=zoom(p,(h/256,w/256),order=0)
    deploy(m,'deployment_no_diagnostics')
    for z,a in enumerate(image):
     h,w=a.shape;x=zoom(a,(256/h,256/w),order=0).astype(np.float32);log=m(torch.from_numpy(x)[None,None].cuda());arr=log.cpu().numpy();ref=orig[z];diff=np.abs(arr-ref)
     p=log.softmax(1).argmax(1)[0].cpu().numpy();p0=ref.argmax(1)[0];pred1[z]=zoom(p,(h/256,w/256),order=0)
     errs.append(dict(method_id=method['id'],frame=frame,slice_index=z,input_sha256=imagehash(x),logits_exact=bool(np.array_equal(arr,ref)),max_abs=float(diff.max()),max_rel=float((diff/np.maximum(np.abs(ref),np.finfo(np.float32).tiny)).max()),allclose=bool(np.allclose(arr,ref,atol=cfg['equivalence_atol'],rtol=cfg['equivalence_rtol'])),mask_exact=bool(np.array_equal(p,p0)),mask_mismatch_pixels=int((p!=p0).sum())))
   metrics=[]
   for path,pred in [('original_runtime',pred0),('deployment_no_diagnostics',pred1)]:
    for c in [1,2,3]:metrics.append(dict(method_id=method['id'],frame=frame,path=path,class_id=c,unit='native grid pixels',**metric(pred==c,gt==c)))
   after=digest(m);assert before==after
   ok=all(r['allclose'] and r['mask_exact'] for r in errs) and imagehash(pred0)==imagehash(pred1)
   js(ud/f"{method['id']}_{frame}.json",dict(identity=identity(),status='PASS' if ok else 'FAIL',checkpoint_sha256=cp['sha256'],branch='model_1',load_receipt=receipt,model_state_before=before,model_state_after=after,source_shape=list(image.shape),errors=errs,metrics=metrics,original_prediction_sha256=imagehash(pred0),deployment_prediction_sha256=imagehash(pred1)))
   assert ok,'Deployment mismatch: retained failing unit, no timing authorized for altered output'
   progress('EQUIVALENCE',method=method['id'],frame=frame);print('equivalence',method['id'],frame,flush=True)
  del m;unload()
 vv=[read(p) for p in sorted(ud.glob('*.json'))];assert len(vv)==80
 csvout(W/'results/DEPLOYMENT_slice_equivalence.csv',[r for u in vv for r in u['errors']]);csvout(W/'results/DEPLOYMENT_volume_hashes.csv',[dict(method_id=u['load_receipt']['method_id'],frame=u['errors'][0]['frame'],original_prediction_sha256=u['original_prediction_sha256'],deployment_prediction_sha256=u['deployment_prediction_sha256'],status=u['status'],slices=len(u['errors'])) for u in vv]);csvout(W/'results/DEPLOYMENT_metrics.csv',[r for u in vv for r in u['metrics']])
 js(W/'results/DEPLOYMENT_EQUIVALENCE_RECEIPT.json',dict(status='PASS',methods=4,slices_per_method=210,method_slice_checks=840,method_volumes=80,all_logits_exact=all(r['logits_exact'] for u in vv for r in u['errors']),max_abs=max(r['max_abs'] for u in vv for r in u['errors']),all_masks_exact=True,all_native_hashes_exact=True,model_BN_states_unchanged=True,atol=cfg['equivalence_atol'],rtol=cfg['equivalence_rtol'],backend=dict(deterministic=True,benchmark=False,matmul_tf32=False,cudnn_tf32=True),training_updates=0,branch='model_1',empty_policy='both empty Dice/IoU/HD95/ASD0; single empty distance sqrt(sum(shape^2))',postprocessing='softmax argmax; no LCC; original order0 restore; no z-axis resize',metrics_unit='native grid pixels',identity=identity()))
def timing():
 backend();assert read(W/'results/DEPLOYMENT_EQUIVALENCE_RECEIPT.json')['status']=='PASS';cfg=config();ud=unitdir('TIMING');data=preloaded();methods=cfg['methods'];tuples=[(m,p) for m in methods for p in cfg['paths']]
 first=data[cfg['val_cases'][0]][0][0];real=zoom(first,(256/first.shape[0],256/first.shape[1]),order=0).astype(np.float32)
 for inputkind in cfg['inputs']:
  for turn in range(5):
   order=tuples[turn%8:]+tuples[:turn%8]
   if turn%2:order=list(reversed(order))
   for pos,(method,path) in enumerate(order):
    key=f"timing_{inputkind}_{turn}_{method['id']}_{path}";dst=ud/f'{key}.json'
    if dst.exists():continue
    attempt=0
    while True:
     stopped();before=wait_gpu(own=torch.cuda.is_initialized());cp=method['checkpoints'][0];m,receipt=load(method,cp,path=path);x=torch.zeros(1,1,256,256,device='cuda') if inputkind=='zeros' else torch.from_numpy(real)[None,None].cuda()
     with torch.inference_mode():
      for _ in range(50):m(x)
      torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();astart=torch.cuda.memory_allocated();rstart=torch.cuda.memory_reserved();pairs=[]
      with Watch() as watch:
       t=time.perf_counter()
       for j in range(100):
        a=torch.cuda.Event(enable_timing=True);b=torch.cuda.Event(enable_timing=True);a.record();m(x);b.record();pairs.append((a,b))
       torch.cuda.synchronize();elapsed=time.perf_counter()-t
      ms=[a.elapsed_time(b) for a,b in pairs]
     value=dict(identity=identity(),method_id=method['id'],checkpoint_sha256=cp['sha256'],branch='model_1',path=path,input_kind=inputkind,input_source='resident all zeros' if inputkind=='zeros' else cfg['val_cases'][0]+':z0',input_sha256=imagehash(np.zeros((256,256),np.float32) if inputkind=='zeros' else real),round=turn,order_index=pos,rotating_order=[f"{m['id']}:{p}" for m,p in order],warmup=50,cuda_event_ms=ms,host_block_seconds=elapsed,valid=watch.valid,attempt=attempt,resource_before=before,resource_samples=watch.records,memory=dict(start_allocated=astart,start_reserved=rstart,peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()),load_receipt=receipt,scope='resident input model call only; original path retains telemetry; resource monitor start/exit/query outside host boundary',GPU_uuid=before['uuid'],backend=dict(deterministic=True,benchmark=False,matmul_tf32=False,cudnn_tf32=True,precision='FP32 storage; no AMP'))
     del m,x;unload()
     js(dst if value['valid'] else ud/f"invalid_{key}_{attempt}_{time.time_ns()}.json",value)
     if value['valid']:break
     attempt+=1
    progress('TIMING',unit=key);print(key,float(np.mean(ms)),flush=True)
 for turn in range(3):
  order=tuples[turn:]+tuples[:turn]
  if turn%2:order=list(reversed(order))
  for pos,(method,path) in enumerate(order):
   pending=[f for f in cfg['val_cases'] if not(ud/f"volume_{turn}_{method['id']}_{path}_{f}.json").exists()]
   if not pending:continue
   wait_gpu(own=torch.cuda.is_initialized());m,_=load(method,method['checkpoints'][0],path=path)
   with torch.inference_mode():
    x=torch.from_numpy(real)[None,None].cuda()
    for _ in range(50):m(x)
    torch.cuda.synchronize();del x
   for frame in pending:
    dst=ud/f"volume_{turn}_{method['id']}_{path}_{frame}.json";attempt=0
    while True:
     stopped();wait_gpu(own=True);torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats()
     with Watch() as watch:
      t=time.perf_counter();pred=volume(m,data[frame][0]);torch.cuda.synchronize();elapsed=time.perf_counter()-t
     expected=read(unitdir('EQUIVALENCE')/f"{method['id']}_{frame}.json")['original_prediction_sha256'];assert imagehash(pred)==expected
     value=dict(identity=identity(),method_id=method['id'],checkpoint_sha256=method['checkpoints'][0]['sha256'],path=path,round=turn,order_index=pos,frame=frame,patient=frame.split('_')[0],slices=len(pred),seconds=elapsed,ms_per_slice=1000*elapsed/len(pred),prediction_sha256=imagehash(pred),valid=watch.valid,attempt=attempt,warmup_on_load=50,memory=dict(peak_allocated=torch.cuda.max_memory_allocated(),peak_reserved=torch.cuda.max_memory_reserved()),resource_samples=watch.records,scope='preloaded CPU native image -> native mask; order0 in-plane resize/H2D/model/softmax argmax/D2H/order0 restore; no file IO/save/GT metrics; z unchanged')
     js(dst if watch.valid else ud/f"invalid_{dst.stem}_{attempt}_{time.time_ns()}.json",value)
     if watch.valid:break
     attempt+=1
    progress('VOLUME_TIMING',method=method['id'],path=path,frame=frame,round=turn)
   del m;unload()
 summarize()
def summarize():
 ud=unitdir('TIMING');vv=[read(p) for p in sorted(ud.glob('timing_*.json'))];iv=[read(p) for p in sorted(ud.glob('invalid_timing*.json'))];raw=[];summary=[];memory=[]
 for u in vv+iv:
  for j,t in enumerate(u['cuda_event_ms']):raw.append(dict(**{k:u[k] for k in ['method_id','path','input_kind','round','order_index','warmup','valid','attempt','input_sha256','checkpoint_sha256']},repeat=j,forward_ms=t,GPU_uuid=u['GPU_uuid']))
  memory.append(dict(method_id=u['method_id'],path=u['path'],input_kind=u['input_kind'],round=u['round'],valid=u['valid'],allocated_start_MiB=u['memory']['start_allocated']/2**20,reserved_start_MiB=u['memory']['start_reserved']/2**20,peak_allocated_MiB=u['memory']['peak_allocated']/2**20,peak_reserved_MiB=u['memory']['peak_reserved']/2**20,whole_card_peak_used_MiB=max(g['used_mib'] for g in u['resource_samples']),reset='after load+50 warmup; includes model and input'))
 for method in config()['methods']:
  for path in config()['paths']:
   for ik in config()['inputs']:
    uu=[u for u in vv if u['method_id']==method['id'] and u['path']==path and u['input_kind']==ik];ms=np.array([t for u in uu for t in u['cuda_event_ms']]);assert len(uu)==5 and len(ms)==500
    summary.append(dict(method_id=method['id'],path=path,input_kind=ik,valid_blocks=5,warmups=250,forwards=500,mean_ms=float(ms.mean()),median_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),event_total_ms=float(ms.sum()),throughput_slices_per_second=float(500/(ms.sum()/1000)),host_total_seconds=sum(u['host_block_seconds'] for u in uu),invalid_blocks=sum(u['method_id']==method['id'] and u['path']==path and u['input_kind']==ik for u in iv)))
 for r in summary:r['relative_to_baseline']=r['mean_ms']/next(b['mean_ms'] for b in summary if b['method_id']=='baseline' and b['path']==r['path'] and b['input_kind']==r['input_kind'])
 v=[read(p) for p in sorted(ud.glob('volume_*.json'))];vi=[read(p) for p in sorted(ud.glob('invalid_volume*.json'))];vs=[]
 for m in config()['methods']:
  for path in config()['paths']:
   ss=[u for u in v if u['method_id']==m['id'] and u['path']==path];assert len(ss)==60;tot=sum(u['seconds'] for u in ss);n=sum(u['slices'] for u in ss)
   vs.append(dict(method_id=m['id'],path=path,rounds=3,unique_frames=20,unique_patients=10,repeated_slices=n,total_seconds=tot,mean_ms_per_slice=1000*tot/n,throughput_slices_per_second=n/tot))
 for r in vs:r['relative_to_baseline']=r['mean_ms_per_slice']/next(b['mean_ms_per_slice'] for b in vs if b['method_id']=='baseline' and b['path']==r['path'])
 csvout(W/'results/A5_timing_raw.csv',raw);csvout(W/'results/A5_timing_summary.csv',summary);csvout(W/'results/A5_memory.csv',memory);csvout(W/'results/A5_volume_runtime.csv',[{k:x for k,x in u.items() if k not in ['resource_samples','memory']} for u in v+vi]);csvout(W/'results/A5_volume_summary.csv',vs)
 js(W/'results/TIMING_RECEIPT.json',dict(status='DONE',blocks=len(vv),volume_units=len(v),invalid_timing_blocks=len(iv),invalid_volume_units=len(vi),training_updates=0,identity=identity()))
if __name__=='__main__':
 if sys.argv[1]=='equivalence':equivalence()
 else:timing()
