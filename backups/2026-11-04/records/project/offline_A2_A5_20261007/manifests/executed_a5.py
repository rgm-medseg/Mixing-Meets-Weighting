import gc,threading,shutil
from collections import Counter
import h5py,numpy as np,torch
from torch import nn
from common import *
from adapter import model,volume,parent
class Watch:
 def __enter__(self):
  self.records=[];self.end=threading.Event()
  def run():
   while not self.end.is_set():
    self.records.append(dict(time=now(),**gpu()));self.end.wait(.75)
  self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
 def __exit__(self,*args):
  self.end.set();self.thread.join();self.records.append(dict(time=now(),**gpu()))
 @property
 def valid(self):return all(not(set(r['pids'])-OWN_GPU_PIDS) for r in self.records)
def release(m):
 del m;gc.collect();torch.cuda.empty_cache()
def operator_count(m,mid):
 rr=[];handles=[];router=[]
 def hook(name,mod,inp,out):
  if isinstance(mod,nn.Conv2d):mac=int(out.numel()*mod.in_channels//mod.groups*np.prod(mod.kernel_size));formula='N*Cout*Hout*Wout*(Cin/groups)*Kh*Kw'
  else:mac=int(out.numel()*mod.in_features);formula='output_elements*in_features'
  rr.append(dict(method_id=mid,module=name,type=type(mod).__name__,input_shape=str(tuple(inp[0].shape)),output_shape=str(tuple(out.shape)),MACs=mac,MAC_FLOPs=2*mac,formula=formula,bias_elementwise_adds=int(out.numel()) if mod.bias is not None else 0))
 for name,mod in m.named_modules():
  if isinstance(mod,(nn.Conv2d,nn.Linear)):handles.append(mod.register_forward_hook(lambda mm,i,o,n=name:hook(n,mm,i,o)))
  if name.endswith('mask_token_router'):
   handles.append(mod.register_forward_pre_hook(lambda mm,i:router.append(dict(B=int(i[0].shape[0]),H=int(i[0].shape[2]),W=int(i[0].shape[3]),D=mm.token_dim,T=mm.token_num,C=mm.in_channels))))
 x=torch.zeros(1,1,256,256,device='cuda')
 with torch.inference_mode(),torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.CPU,torch.profiler.ProfilerActivity.CUDA],record_shapes=True,with_flops=True) as prof:m(x);torch.cuda.synchronize()
 for h in handles:h.remove()
 native=[dict(method_id=mid,operator=e.key,calls=e.count,native_FLOPs=int(e.flops),input_shapes=str(e.input_shapes),native_coverage='reported' if e.flops else 'zero/uncovered/nonarithmetic; inspect definition') for e in prof.key_averages(group_by_input_shape=True)]
 csvout(W/'results'/f'A5_operators_{mid}.csv',native);csvout(W/'results'/f'A5_modules_{mid}.csv',rr)
 conv=sum(r['MACs'] for r in rr);matrix=0;formula=[]
 if router:
  r=router[0];B,H,WW,D,T,C=(r[k] for k in ['B','H','W','D','T','C']);matrix=2*B*H*WW*T*D
  for expression in ['bdhw,nd->bnhw','bnhw,nd->bdhw']:formula.append(dict(method_id=mid,operation='TRC einsum '+expression,formula='B*H*W*T*D',MACs=B*H*WW*T*D,FLOPs=2*B*H*WW*T*D,coverage='manual MAC count; torch profiler bmm agrees; add once to convolution hooks'))
  for op,f,n in [('logit scale','B*H*W*T',B*H*WW*T),('residual gamma multiply and add','2*B*H*W*C',2*B*H*WW*C),('softmax convention','B*H*W*(4*T-1)',B*H*WW*(4*T-1)),('LayerNorm convention','B*H*W*(7*D+2)',B*H*WW*(7*D+2))]:formula.append(dict(method_id=mid,operation='TRC '+op,formula=f,MACs=0,FLOPs=n,coverage='supplementary analytic arithmetic convention; exp/sqrt each1, no hardware-cycle interpretation; outside MAC FLOPs'))
 nativebmm=sum(r['native_FLOPs'] for r in native if r['operator']=='aten::bmm');assert nativebmm==2*matrix
 result=dict(method_id=mid,scope='single deployed student1',parameters_total=sum(p.numel() for p in m.parameters()),parameters_trainable=sum(p.numel() for p in m.parameters() if p.requires_grad),TRC_parameters=sum(p.numel() for n,p in m.named_parameters() if 'mask_token_router' in n),input_shape='1x1x256x256',precision='FP32',conv_linear_MACs=conv,einsum_MACs=matrix,MACs_covered=conv+matrix,MAC_FLOPs_covered=2*(conv+matrix),native_profiler_FLOPs=sum(r['native_FLOPs'] for r in native),complete_inference_FLOPs='NA',compute_status='PARTIAL: all convolution/linear/einsum MACs covered; complete non-MAC arithmetic and telemetry not covered',tool='torch.profiler '+torch.__version__+'; Conv/Linear hooks; exact einsum formula',conversion='1 MAC=2 FLOPs; additional elemental formulas separate, no duplicate profiler bmm addition',full_training_FLOPs='NA: backward/FFT/loss/teacher paths not comprehensively counted')
 return result,formula
def counts():
 dst=unitdir('A5')/'parameters.json'
 if dst.exists():return
 table=[];formulas=[]
 for method in config()['methods']:
  wait_gpu();cp=method['checkpoints'][0];m,receipt=model(method,cp,'model_1');r,ff=operator_count(m,method['id']);r.update(checkpoint_sha256=cp['sha256'],checkpoint_branch='model_1',architecture_source=str(W/'scripts/frozen/train_ACDC'),state='loaded strict',structure_shared_with='Full_main,R1_v2_B,R2_fixed92' if method['trc'] else 'baseline');table.append(r);formulas.extend(ff)
  n=r['parameters_total'];table.append(dict(method_id=method['id'],scope='training system',parameters_student1=n,parameters_student2=n,parameters_EMA_teacher=n,parameters_total=3*n,parameters_trainable=2*n,teacher_trainable_parameters=0,teacher_freeze_source='parent.v14_bcp_net ema=True detach_(); runtime teacher definition verified',system_networks=3,complete_inference_FLOPs='NA',full_training_FLOPs='NA',compute_status='parameter count only; deployment excludes student2/EMA teacher'))
  del m;gc.collect();torch.cuda.empty_cache()
 arch=dict(next(r for r in table if r['method_id']=='Full_main' and r['scope']=='single deployed student1'));arch.update(method_id='TRC_architecture_only',state='architecture-only, no corresponding trained TRC-only checkpoint',checkpoint_sha256='NA',checkpoint_branch='NA',structure_shared_with='Full_main; same parent UNet + TRC router; TCSMix/CTW do not add deploy modules');table.append(arch)
 csvout(W/'results/A5_parameters_compute.csv',table);csvout(W/'results/A5_TRC_analytic_operations.csv',formulas);js(dst,dict(status='PARTIAL',parameters='DONE',MACs='DONE within declared conv/linear/einsum convention',complete_FLOPs='PARTIAL',rows=table))
 atomic(W/'reports/A5_OPERATOR_COVERAGE.md','''# 算子覆盖\n\n输入1×1×256×256，FP32 eval。torch.profiler(with_flops=True,record_shapes=True)原生事件保存在A5_operators_*.csv。PyTorch版本见环境。模块hooks统计Conv2d/Linear：输出元素数×输入通道(分组修正)×卷积核面积；偏置加法另列。einsum的原生bmm计数逐项核验，人工公式仅补至模块MAC表，未再将原生profile总数相加。统一1 MAC=2 FLOPs。\n\nTRC project/inverse_project由Conv hooks覆盖，各16×16×256² MAC；两次einsum各256²×4×16 MAC；softmax、logit尺度、残差融合、LayerNorm的算式与数值另见A5_TRC_analytic_operations.csv。LayerNorm公式按一次共享中心化、均值/方差及仿射的7D+2标量操作约定；softmax减max、exp、求和、除法为4T−1，加上max比较另不计。这是明确的标量约定，exp/sqrt各记1不代表硬件cycle。\n\n原生profile未完整报告BatchNorm、LayerNorm、ReLU、MaxPool比较、双线性上采样、softmax、exp/log/norm、isfinite/比较/规约以及复制/CPU同步的运算成本；argmax及原forward遥测也保留。完整推理FLOPs=NA/PARTIAL，主表提供已覆盖MAC及其2倍FLOPs，不将遗漏静默记0。辅助TRC标量公式不会将主表伪装为完整计数。训练FFT、全部损失、反向、EMA运算未完整计数，训练总FLOPs=NA。\n\nTCSMix/CTW在混合、损失阶段执行，真实student1部署结构只有UNet或UNet+TRC；训练系统参数为两学生加冻结EMA教师，单模型计时不包含第二学生/教师。\n''')
def inference():
 methods=config()['methods'];ud=unitdir('A5');raw=[]
 for block in range(5):
  order=methods[block%len(methods):]+methods[:block%len(methods)]
  for pos,method in enumerate(order):
   p=ud/f"inference_{block}_{method['id']}.json"
   if p.exists() and read(p)['valid']:continue
   attempt=0
   while True:
    stopped();before=wait_gpu();cp=method['checkpoints'][0];m,receipt=model(method,cp,'model_1');x=torch.zeros(1,1,256,256,device='cuda')
    with torch.inference_mode():
     if block==0:
      for _ in range(50):m(x)
     torch.cuda.synchronize();torch.cuda.reset_peak_memory_stats();allocated_start=torch.cuda.memory_allocated();reserved_start=torch.cuda.memory_reserved()
     pairs=[];wall=time.perf_counter()
     with Watch() as watch:
      for j in range(100):
       start=torch.cuda.Event(enable_timing=True);end=torch.cuda.Event(enable_timing=True);start.record();m(x);end.record();pairs.append((start,end))
      torch.cuda.synchronize()
     wall=time.perf_counter()-wall;ms=[a.elapsed_time(b) for a,b in pairs]
    memory=dict(allocated_start_bytes=allocated_start,reserved_start_bytes=reserved_start,peak_allocated_bytes=torch.cuda.max_memory_allocated(),peak_reserved_bytes=torch.cuda.max_memory_reserved(),reset='after model/input load and first-block50 warmups; includes one loaded model/input; other methods not resident')
    value=dict(identity=identity(),method_id=method['id'],checkpoint_sha256=cp['sha256'],branch='model_1',block=block,order_index=pos,rotating_order=[m['id'] for m in order],attempt=attempt,valid=watch.valid,warmup_this_block=50 if block==0 else 0,cuda_event_ms=ms,host_block_seconds=wall,memory=memory,resource_before=before,resource_samples=watch.records,sampling_scope='pure original model forward on resident GPU input; includes original router forward telemetry; no softmax/argmax outside model, IO/H2D excluded',GPU_uuid=before['uuid'],backend=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32,cudnn_allow_tf32=torch.backends.cudnn.allow_tf32))
    del x,m;gc.collect();torch.cuda.empty_cache()
    if value['valid']:js(p,value);break
    js(ud/f"invalid_inference_{block}_{method['id']}_{now().replace(':','')}.json",value);attempt+=1
   progress('A5_INFERENCE',block=block,method=method['id'],valid=value['valid']);print('A5 timing',block,method['id'],np.mean(ms),flush=True)
 values=[read(p) for p in sorted(ud.glob('inference_*.json'))]
 invalid=[read(p) for p in sorted(ud.glob('invalid_inference_*.json'))]
 for u in values+invalid:
  for j,ms in enumerate(u['cuda_event_ms']):raw.append(dict(method_id=u['method_id'],checkpoint_sha256=u['checkpoint_sha256'],branch=u['branch'],block=u['block'],order_index=u['order_index'],rotating_order=';'.join(u['rotating_order']),attempt=u['attempt'],valid=u['valid'],repeat=j,forward_ms=ms,precision='FP32',batch=1,unit='ms',warmup_block=u['warmup_this_block'],GPU_uuid=u['GPU_uuid']))
 csvout(W/'results/A5_inference_raw.csv',raw);summary=[]
 for method in methods:
  vv=[v for v in values if v['method_id']==method['id']];ms=np.concatenate([v['cuda_event_ms'] for v in vv]);assert len(ms)==500
  memory=[v['memory'] for v in vv];summary.append(dict(method_id=method['id'],checkpoint_sha256=method['checkpoints'][0]['sha256'],branch='model_1',valid_blocks=len(vv),invalid_blocks=sum(v['method_id']==method['id'] for v in invalid),measured_forwards=len(ms),warmup_total=sum(v['warmup_this_block'] for v in vv),mean_ms=float(ms.mean()),median_ms=float(np.median(ms)),p95_ms=float(np.percentile(ms,95)),event_total_ms=float(ms.sum()),throughput_slices_per_second=float(len(ms)/(ms.sum()/1000)),throughput_basis='500 / sum of 500 individual CUDA-event forward durations seconds',host_block_total_seconds=sum(v['host_block_seconds'] for v in vv),peak_allocated_MiB=max(v['peak_allocated_bytes'] for v in memory)/2**20,peak_reserved_MiB=max(v['peak_reserved_bytes'] for v in memory)/2**20,nvidia_smi_peak_used_MiB=max(g['used_mib'] for v in vv for g in v['resource_samples']),GPU_uuid=vv[0]['GPU_uuid'],precision='FP32 storage; existing backend TF32 flags in raw unit JSON',timing_scope=vv[0]['sampling_scope'],memory_reset=memory[0]['reset']))
 csvout(W/'results/A5_inference_summary.csv',summary)
def volumes():
 cfg=config();preloaded={}
 for case in cfg['val_cases']:
  with h5py.File(DATA/'data'/f'{case}.h5') as f:preloaded[case]=f['image'][:]
 assert len(preloaded)==20
 ud=unitdir('A5');out=[]
 for turn in range(3):
  methods=cfg['methods'][turn:]+cfg['methods'][:turn]
  for pos,method in enumerate(methods):
   pending=[c for c in cfg['val_cases'] if not(ud/f"volume_{turn}_{method['id']}_{c}.json").exists()]
   if not pending:continue
   wait_gpu();m,_=model(method,method['checkpoints'][0],'model_1')
   # Same predetermined first validation frame for all methods, excluded warmup.
   volume(m,preloaded[cfg['val_cases'][0]]);torch.cuda.synchronize()
   for case in pending:
    while True:
     stopped();wait_gpu(own=True);torch.cuda.synchronize()
     with Watch() as watch:
      t=time.perf_counter();pred=volume(m,preloaded[case]);torch.cuda.synchronize();elapsed=time.perf_counter()-t
     r=dict(identity=identity(),method_id=method['id'],checkpoint_sha256=method['checkpoints'][0]['sha256'],branch='model_1',round=turn,order_index=pos,frame=case,patient=case.split('_')[0],slices=preloaded[case].shape[0],seconds=elapsed,ms_per_slice=1000*elapsed/preloaded[case].shape[0],valid=watch.valid,input_shape=str(preloaded[case].shape),prediction_sha256=hashlib.sha256(pred.tobytes()).hexdigest(),resource_samples=watch.records,time_scope='preloaded CPU image -> native-size mask; order0 resize,H2D,forward,softmax+argmax,D2H,order0 restore; excludes disk IO/save and GT metrics',batch=1,precision='FP32')
     if r['valid']:js(ud/f"volume_{turn}_{method['id']}_{case}.json",r);break
     js(ud/f"invalid_volume_{turn}_{method['id']}_{case}_{now().replace(':','')}.json",r)
    progress('A5_VOLUME',method=method['id'],round=turn,frame=case)
   del m;gc.collect();torch.cuda.empty_cache()
 vv=[read(p) for p in sorted(ud.glob('volume_*.json'))];iv=[read(p) for p in sorted(ud.glob('invalid_volume_*.json'))]
 csvout(W/'results/A5_volume_runtime.csv',[{k:v for k,v in r.items() if k!='resource_samples'} for r in vv+iv]);ss=[]
 for method in cfg['methods']:
  rr=[r for r in vv if r['method_id']==method['id']];assert len(rr)==60
  total=sum(r['seconds'] for r in rr);sl=sum(r['slices'] for r in rr)
  ss.append(dict(method_id=method['id'],valid_rounds=3,frames=20,patients=10,total_repeated_volumes=len(rr),total_repeated_slices=sl,total_seconds=total,mean_seconds_per_volume=total/len(rr),median_seconds_per_volume=float(np.median([r['seconds'] for r in rr])),ms_per_slice=1000*total/sl,throughput_slices_per_second=sl/total,time_scope=rr[0]['time_scope']))
 csvout(W/'results/A5_volume_summary.csv',ss)
def training_cost():
 out=[];forward=[]
 for method in config()['methods']:
  if method['id'] in ['R1_v2_B','R2_fixed92']:
   root=R1 if method['id']=='R1_v2_B' else R2;prefix='R1' if root==R1 else 'R2'
   segments=read(root/'state/train_segments.json');history=read(root/'state/stage_history.json');trace=rows(root/f'results/{prefix}_control_trace.csv');events=rows(root/'results/optimizer_calls.csv');runtime=read(root/'analysis/runtime_cost.json');post=read(root/'reports/TRAIN_POSTCHECK.json')
   successful=sum(s['elapsed_seconds'] for s in segments if s['exit_code']==0);failed=sum(s['elapsed_seconds'] for s in segments if s['exit_code']!=0)
   assert len(trace)==30000 and len({int(r['step']) for r in trace})==30000
   assert post['optimizer_success_calls']=={'optimizer1':30000,'optimizer2':30000}
   # Preserve exact timing, ledger and boundary evidence without modifying parents.
   for p in [root/'state/train_segments.json',root/'state/stage_history.json',root/f'results/{prefix}_control_trace.csv',root/'results/optimizer_calls.csv',root/'analysis/runtime_cost.json',root/'reports/TRAIN_POSTCHECK.json']:
    dest=W/'legacy/training_cost'/prefix/p.name;dest.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,dest)
   local=[float(r['step_seconds']) for r in trace];stable=[float(r['step_seconds']) for r in trace if int(r['step'])>50 and int(r['step'])%200 not in range(11)]
   out.append(dict(method_id=method['id'],status='DONE',host='n1',GPU='A10040GB GPU0',precision='FP32',batch=24,labeled_batch=12,joint_steps=30000,optimizer1_success=30000,optimizer2_success=30000,replayed_updates=post['replayed_physical_updates'],optimizer_event_rows=len(events),successful_training_seconds=successful,successful_training_hours=successful/3600,failed_startup_seconds=failed,failed_startup_updates=0,recorded_runtime_cost_total_seconds=runtime['training_active_segment_seconds'],runtime_total_includes_failure=(failed>0),training_scope='successful Popen launch -> detected exit, includes import/loading/data wait/EMA/validation/checkpoints; polling detection <=5s',validation_save_frequency='200; full last also on completion',local_timer_mean_seconds=float(np.mean(local)),local_timer_median_seconds=float(np.median(local)),stable_local_timer_median_seconds=float(np.median(stable)),local_timer_scope='after loader yields through control trace; excludes following EMA/validation/checkpoint and loader wait',evaluation_extra_seconds=sum(s['elapsed_seconds'] for s in history if s['stage']=='EVALUATE' and s['exit_code']==0),analysis_extra_seconds=sum(s['elapsed_seconds'] for s in history if s['stage'] in ['ANALYSIS','ERROR_GEOMETRY'] and s['exit_code']==0),peak_allocated_GiB=max(int(r['peak_cuda_bytes']) for r in trace)/2**30,peak_reserved_GiB='NA: not retained',median_controller_ms=float(np.median([float(r['controller_ms']) for r in trace])),controller_scope='local synchronous signal/mask generation, not paired total method overhead',training_FLOPs='NA',comparison='R1/R2 same hardware/precision/budget/validation; geometry differs; single runs descriptive only',source=str(root/'state/train_segments.json'),source_sha256=sha(root/'state/train_segments.json')))
  else:out.append(dict(method_id=method['id'],status='PARTIAL',successful_training_seconds='NA',successful_training_hours='NA',failed_startup_seconds='NA',training_FLOPs='NA',reason='retained train.log lacks equivalent launcher/exit segment boundary and GPU runtime environment; no matched complete total cost',source=str(Path(method['run'])/'train.log')))
  forward.append(dict(method_id=method['id'],path='teacher weak group a,b + strong group a,b',network='EMA teacher',forward_calls_per_joint_iteration=4,input_batch=6,source=method['source_path'],full_training_FLOPs='NA'))
  # The literal inherited loop calls each student on four mixed labeled/unlabeled views.
  source=Path(method['source_path']).read_text();calls=[]
  import ast
  for n in ast.walk(ast.parse(source)):
   if isinstance(n,ast.Call) and isinstance(n.func,ast.Name) and n.func.id in ['model_1','model_2','ema_model']:calls.append((n.func.id,n.lineno,ast.get_source_segment(source,n)))
  csvout(W/'results'/f'A5_forward_source_{method["id"]}.csv',[dict(network=n,line=line,call=call,source=method['source_path']) for n,line,call in calls])
 csvout(W/'results/A5_training_cost.csv',out);csvout(W/'results/A5_training_forward_audit.csv',forward)
def main():
 torch.set_num_threads(1);torch.backends.cudnn.deterministic=True;torch.backends.cudnn.benchmark=False
 unitdir('A5').mkdir(parents=True,exist_ok=True)
 counts();inference();volumes();training_cost()
 js(W/'results/A5_RECEIPT.json',dict(status='PARTIAL',time=now(),parameters='DONE',covered_MACs='DONE',complete_FLOPs='PARTIAL',inference='DONE',volume_runtime='DONE',R1_R2_training_cost='DONE',historical_training_cost='NA',optimizer_updates=0,unit_identity=identity(),invalid_timing_retained=len(list(unitdir('A5').glob('invalid*.json')))))
if __name__=='__main__':main()
