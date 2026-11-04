import fcntl,signal,subprocess,traceback
from common import *
def main():
 lock=(W/'state/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=dict(status='RUNNING',controller_pid=os.getpid(),start=now(),stage='INITIALIZE',completed=[],pending=['A2','A5','REVIEW','PACKAGE'],training_updates=0,old_queue='INTERRUPTED',identity=identity())
 js(W/'state/controller.json',state)
 for stage,file in [('A2','a2.py'),('A5','a5.py'),('REVIEW','finish.py')]:
  stopped();state.update(stage=stage,heartbeat=now());js(W/'state/controller.json',state)
  log=W/'logs'/f'{stage}_{now().replace(":","")}.log';t=time.monotonic()
  with log.open('w') as f:
   child=subprocess.Popen([PY,str(W/'scripts'/file)],stdout=f,stderr=subprocess.STDOUT,cwd=W,env=dict(os.environ,PYTHONDONTWRITEBYTECODE='1',CUDA_VISIBLE_DEVICES='0'))
   state.update(child_pid=child.pid,log=str(log));js(W/'state/controller.json',state)
   while child.poll() is None:
    state.update(heartbeat=now(),elapsed_stage_seconds=time.monotonic()-t);js(W/'state/controller.json',state);time.sleep(3)
   if child.returncode:
    if (W/'state/stop_requested').exists():raise InterruptedError('graceful stop; completed units retained')
    raise RuntimeError(f'{stage} exit {child.returncode}; evidence retained, see {log}')
  state['completed'].append(stage);state['pending'].remove(stage);state['child_pid']=None
 state.update(status='DONE',stage='FINISHED',finished=now(),pending=[],archive=str(W/'A2_A5_offline_evidence_20261007.tar.gz'));js(W/'state/controller.json',state)
if __name__=='__main__':
 try:main()
 except BaseException as e:
  p=W/'state/controller.json';s=read(p) if p.exists() else {};s.update(status='STOPPED' if isinstance(e,InterruptedError) else 'FAILED',failure=repr(e),failure_time=now());js(p,s);traceback.print_exc();sys.exit(1)
