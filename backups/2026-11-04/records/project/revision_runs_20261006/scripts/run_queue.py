"""Persistent ordinary process controller; no agent or network session required."""
import argparse
import fcntl
import json
import os
import signal
import subprocess
import sys
import time
from common import *

STOP=False
CHILD=None

def signal_stop(signum,frame):
    global STOP
    STOP=True

def alive(pid):
    try:
        os.kill(int(pid),0)
        return True
    except (OSError,TypeError,ValueError): return False

def gpu_free():
    result=subprocess.run(['nvidia-smi','--query-gpu=memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits','--id=0'],capture_output=True,text=True)
    if result.returncode: return False,result.stderr.strip()
    used,free,util=[int(x.strip()) for x in result.stdout.strip().split(',')]
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader,nounits'],capture_output=True,text=True)
    pids=[x.strip() for x in processes.stdout.splitlines() if x.strip().isdigit()]
    return used<2000 and free>28000 and not pids,dict(used_mib=used,free_mib=free,utilization_percent=util,pids=pids)

class Controller:
    def __init__(self,test_mode=False):
        self.test_mode=test_mode
        self.manifest=read_json(WORK/'run_manifest.json')
        self.state_path=WORK/('state/queue_selftest_runtime.json' if test_mode else 'state/queue.json')
        self.state=read_json(self.state_path) if self.state_path.exists() else dict(tasks=self.manifest['tasks'],analyses={},archive_status='PENDING')
        self.state.update(controller_pid=os.getpid(),controller_started=now(),active_task=None,status='RUNNING')
        self.env=os.environ.copy()
        self.env.update(CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',MKL_NUM_THREADS='1',WANDB_MODE='disabled',PYTHONUNBUFFERED='1',MPLBACKEND='Agg')

    def persist(self):
        self.state.update(heartbeat=now(),heartbeat_epoch=time.time())
        atomic_json(self.state_path,self.state)
        self.manifest.update(tasks=self.state['tasks'],analyses=self.state['analyses'],archive_status=self.state['archive_status'],controller_pid=os.getpid(),heartbeat=self.state['heartbeat'])
        if not self.test_mode: atomic_json(WORK/'run_manifest.json',self.manifest)

    def stopped(self):
        return STOP or (WORK/'state/stop_requested').exists()

    def wait_resource(self,task):
        while not self.stopped():
            free,info=gpu_free()
            self.state['resource']=info
            if free: return True
            if task in self.state['tasks']: self.state['tasks'][task].update(status='WAITING_RESOURCE',reason='GPU0 occupied; unchanged configuration queued')
            self.persist()
            for _ in range(15):
                if self.stopped(): return False
                time.sleep(2)
                self.persist()
        return False

    def child(self,command,label,log,task=None):
        global CHILD
        self.state['active_task']=label
        log=Path(log); log.parent.mkdir(parents=True,exist_ok=True)
        with log.open('a') as f:
            f.write('\n'+now()+' command='+json.dumps(command)+'\n'); f.flush()
            CHILD=subprocess.Popen(command,cwd=WORK,env=self.env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
            if task:
                self.state['tasks'][task].update(pid=CHILD.pid,status='RUNNING',started=now(),log=str(log),actual_command=command)
            self.state['child_pid']=CHILD.pid; self.persist()
            sent=False
            while CHILD.poll() is None:
                if task:
                    progress=WORK/f'runs/{task}/progress.json'
                    if progress.exists(): self.state['tasks'][task]['latest_step']=read_json(progress)['step']
                if self.stopped() and not sent:
                    CHILD.send_signal(signal.SIGTERM); sent=True
                    self.state['stopping_since']=now()
                self.persist(); time.sleep(2)
            code=CHILD.returncode
        CHILD=None; self.state['child_pid']=None; self.state['active_task']=None
        if task: self.state['tasks'][task].update(exit_code=code,process_exited=now())
        self.persist(); return code

    def train_task(self,task):
        t=self.state['tasks'][task]
        if t['status'] in ('DONE','BLOCKED','FAILED'): return
        _,cfg=config_for(task); run=Path(cfg['run_root']); latest=run/'checkpoints/last.ckpt'
        receipt=run/'TRAIN_RECEIPT.json'
        is_trained=receipt.exists() and read_json(receipt).get('status')=='TRAINED' and read_json(receipt)['global_step']==cfg['max_iterations']
        if not is_trained:
            t['status']='READY'; self.persist()
            if not self.wait_resource(task): return
            command=list(t['command'])
            if latest.exists(): command+=['--resume']
            code=self.child(command,task,WORK/f'logs/{task}.log',task)
            if self.stopped() or code==75:
                t.update(status='INTERRUPTED',reason='Safe stop, full-state checkpoint available' if latest.exists() else 'Interrupted before first checkpoint')
                self.persist(); return
            if code:
                t.update(status='FAILED',reason=f'Training exit={code}; traceback retained; no automatic protocol change or retry')
                self.persist(); return
            if not receipt.exists() or read_json(receipt).get('global_step')!=cfg['max_iterations']:
                t.update(status='FAILED',reason='Training exit0 without locked budget completion receipt')
                self.persist(); return
        t.update(status='TRAINED',training_completed=now()); self.persist()
        best=run/'checkpoints/best.ckpt'
        if not best.exists():
            t.update(status='FAILED',reason='No validation-selected best checkpoint'); self.persist(); return
        freeze=run/'BEST_FROZEN.json'
        payload=dict(checkpoint=str(best),sha256=sha(best),selection='validation student1 foreground macro Dice, strict improvement',time=now(),test_reselection=False)
        if freeze.exists() and read_json(freeze)['sha256']!=payload['sha256']:
            t.update(status='FAILED',reason='Frozen checkpoint hash changed'); self.persist(); return
        if not freeze.exists(): atomic_json(freeze,payload)
        if not self.wait_resource(task): return
        code=self.child([str(PYTHON),str(WORK/'scripts/evaluate.py'),'--task',task],task+'_evaluate',WORK/f'logs/{task}_evaluate.log')
        if code:
            t.update(status='FAILED',reason=f'Evaluation exit={code}; completed training retained')
        else:
            t.update(status='EVALUATED',evaluation_completed=now(),checkpoint=str(best),checkpoint_sha256=payload['sha256'])
            self.persist()
            t.update(status='DONE',finished=now())
        self.persist()

    def finish(self):
        for i in range(1,7):
            task=f'A{i}'
            if self.stopped(): return
            if task in ('A2','A3','A5') and not self.wait_resource(task): return
            command=[str(PYTHON),str(WORK/f'scripts/analyze_{task}.py')]
            code=self.child(command,task,WORK/f'logs/{task}_final.log')
            status=WORK/f'analysis/{task}/status.json'
            self.state['analyses'][task]=read_json(status) if status.exists() and not code else dict(status='FAILED',reason=f'Analysis exit={code}',log=str(WORK/f'logs/{task}_final.log'))
            self.persist()
        if self.stopped(): return
        self.state.update(status='FINISHED_WITH_FAILURES' if any(t['status']!='DONE' for t in self.state['tasks'].values()) else 'FINISHED',finished=now(),archive_status='PACKAGING')
        self.persist()
        packaging_code=None
        for name in ('summarize_results.py','package_results.py'):
            code=self.child([str(PYTHON),str(WORK/'scripts'/name)],name,WORK/f'logs/{name}.log')
            self.state.setdefault('delivery_commands',{})[name]=dict(exit_code=code,time=now())
            if name=='package_results.py': packaging_code=code
            if code: self.state['archive_status']='FAILED'; self.persist()
        archive=WORK/'revision_evidence_20261006.tar.gz'
        self.state['archive_status']='DONE' if packaging_code==0 and archive.exists() else 'FAILED'
        self.state.update(status='FINISHED_WITH_FAILURES' if any(t['status']!='DONE' for t in self.state['tasks'].values()) else 'FINISHED',finished=now())
        self.persist()

    def run(self):
        # Completed script/config gate must match what is about to run.
        gate=read_json(WORK/'state/preflight.json')
        if gate['status']!='PASS' or not gate['smoke_pass']: raise RuntimeError('Full preflight/smoke gate required')
        locked={**gate['script_sha256'],**gate.get('shell_sha256',{}),**gate.get('source_sha256',{})}
        for rel,digest in locked.items():
            if sha(WORK/rel)!=digest: raise RuntimeError('Script changed after preflight: '+rel)
        for rel,digest in gate['config_sha256'].items():
            if sha(WORK/rel)!=digest: raise RuntimeError('Config changed after preflight: '+rel)
        for task,t in self.state['tasks'].items():
            if t.get('status')=='RUNNING' and alive(t.get('pid')):
                raise RuntimeError(f'Prior task still alive: {task} pid {t["pid"]}; refusing duplicate')
            if t.get('status')=='RUNNING': t['status']='INTERRUPTED'
        self.persist()
        for task in ('E1','E2','E3','E4','E5'):
            if self.stopped(): break
            self.train_task(task)
        if self.stopped():
            self.state.update(status='INTERRUPTED',stopped=now()); self.persist()
        else: self.finish()

def selftest():
    path=WORK/'state/queue_failure_selftest.json'
    c=Controller(test_mode=True)
    code=c.child([str(PYTHON),'-c','raise RuntimeError("intentional queue capture fixture")'],'fixture_failure',WORK/'logs/queue_fixture_failure.log')
    assert code!=0
    c.state['fixture_failure']=dict(status='FAILED',exit_code=code)
    independent=c.child([str(PYTHON),'-c','print("independent fixture executed")'],'fixture_independent',WORK/'logs/queue_fixture_independent.log')
    assert independent==0
    c.persist()
    atomic_json(path,dict(status='PASS',failed_child_status='FAILED',exit_code=code,dependent_status='BLOCKED',independent_task_exit=independent,time=now()))
    assert read_json(path)['exit_code']==code
    print('Queue exit-code and atomic-status fixture PASS')

def main():
    p=argparse.ArgumentParser(); p.add_argument('--selftest',action='store_true'); a=p.parse_args()
    if a.selftest: return selftest()
    for sig in (signal.SIGINT,signal.SIGTERM): signal.signal(sig,signal_stop)
    with (WORK/'state/controller.lock').open('a') as lock:
        try: fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        except BlockingIOError: raise SystemExit('Queue controller already holds lock; no duplicate started')
        controller=Controller()
        try: controller.run()
        except BaseException as exc:
            import traceback
            controller.state.update(status='FAILED',reason=str(exc),traceback=traceback.format_exc(),failed=now())
            controller.persist(); raise

if __name__=='__main__':
    main()
