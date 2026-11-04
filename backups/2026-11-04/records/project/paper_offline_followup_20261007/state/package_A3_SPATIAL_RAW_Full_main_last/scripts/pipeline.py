import fcntl
from common import *
def main():
 fd=(W/'state/controller.lock').open('w')
 try:fcntl.flock(fd,fcntl.LOCK_EX|fcntl.LOCK_NB)
 except BlockingIOError:print('Existing controller reused');return
 stages=[('A2_MAPPING','a2_mapping.py',[]),('INPUTS','a3_inputs.py',[]),('EQUIVALENCE','measurement.py',['equivalence']),('TIMING','measurement.py',['timing']),('A3','a3.py',[]),('REVIEW','finish.py',[]),('ARCHIVE','delivery.py',[])]
 prior=read(W/'state/controller.json') if (W/'state/controller.json').exists() else {}
 state=dict(status='RUNNING',controller_pid=os.getpid(),identity=identity(),start=now(),training_updates=0,optimizer_updates=0,EMA_updates=0,completed=[],reused_stages=[],old_queue='INTERRUPTED/E1=3976 E2-E5=PENDING')
 scientific=read(W/'manifests/identity.json')
 assert all(sha(W/p)==h for p,h in scientific['sources'].items()),'Scientific source identity changed; prepare a new version before resuming'
 receipt_paths={'A2_MAPPING':unitdir('A2')/'complete.json','INPUTS':unitdir('INPUTS')/'complete.json','EQUIVALENCE':W/'results/DEPLOYMENT_EQUIVALENCE_RECEIPT.json','TIMING':W/'results/TIMING_RECEIPT.json','A3':W/'results/A3_RECEIPT.json','REVIEW':W/'results/REPORTS_RECEIPT.json'}
 try:
  for stage,script,args in stages:
   stopped();state.update(stage=stage,heartbeat=now());js(W/'state/controller.json',state)
   rp=receipt_paths.get(stage)
   if prior.get('identity')==identity() and stage in prior.get('completed',[]) and rp and rp.exists() and read(rp).get('identity')==identity():
    value=read(rp)
    producer_ok=stage!='REVIEW' or (value.get('producer_sha256')==sha(W/'scripts/finish.py') and value.get('presentation_sha256')==sha(W/'scripts/presentation_addendum.py'))
    if value.get('status') in ['DONE','PASS'] and producer_ok:
     state['completed'].append(stage);state['reused_stages'].append(stage);print('REUSED COMPLETED STAGE',stage,flush=True);continue
   log=W/'logs'/f'{stage}_{time.time_ns()}.log';t=time.monotonic()
   with log.open('w') as f:
    proc=subprocess.Popen([PY,str(W/'scripts'/script),*args],cwd=W,stdout=f,stderr=subprocess.STDOUT,env={**os.environ,'CUDA_VISIBLE_DEVICES':'0','PYTHONUNBUFFERED':'1','MPLBACKEND':'Agg'})
    while proc.poll() is None:
     state.update(child_pid=proc.pid,heartbeat=now(),elapsed_stage_seconds=time.monotonic()-t,log=str(log));js(W/'state/controller.json',state);time.sleep(2)
   hist=read(W/'state/stage_history.json') if (W/'state/stage_history.json').exists() else [];hist.append(dict(stage=stage,start=state['heartbeat'],end=now(),seconds=time.monotonic()-t,exit_code=proc.returncode,log=str(log)));js(W/'state/stage_history.json',hist)
   if proc.returncode:
    if (W/'state/stop_requested').exists():raise InterruptedError('safe unit boundary')
    raise RuntimeError(f'{stage} exited {proc.returncode}: {log}')
   state['completed'].append(stage)
  state.update(status='DONE',stage='FINISHED',child_pid=None,finished=now())
 except InterruptedError as e:state.update(status='STOPPED',child_pid=None,reason=str(e),stopped=now())
 except Exception as e:state.update(status='FAILED',child_pid=None,reason=str(e),failed=now())
 js(W/'state/controller.json',state);print(json.dumps(state,ensure_ascii=False),flush=True)
if __name__=='__main__':main()
