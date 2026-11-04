"""One bounded short run, interrupted once to verify same-run recovery."""
import os
import subprocess
from common import *

def main():
    # This phase permits one bounded run only. Refuse before touching status/logs.
    if (WORK/'short_trace.csv').exists() or read_json(WORK/'state/update_budget.json').get('optimizer_step_calls',0):
        raise SystemExit('Short run already executed; preserve its artifacts and WAITING_FOR_FULL_APPROVAL. No restart permitted.')
    cfg=WORK/'configs/short_selected.yaml'
    env=os.environ.copy();env.update(OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',CUDA_VISIBLE_DEVICES='0',PYTHONUNBUFFERED='1')
    commands=[('fresh',[str(PYTHON),str(WORK/'scripts/train_control_entry.py'),'--config',str(cfg),'--stop-after','200'],75),
              ('resume',[str(PYTHON),str(WORK/'scripts/train_control_entry.py'),'--config',str(cfg),'--resume'],0)]
    atomic_json(WORK/'state/status.json',dict(status='SHORT_RUN_RUNNING',pid=os.getpid(),formal_training_authorized=False,time=now()))
    receipts=[]
    for name,cmd,expected in commands:
        with (WORK/f'logs/short_{name}.log').open('x') as f:
            res=subprocess.run(cmd,cwd=WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
        receipts.append(dict(segment=name,command=cmd,exit_code=res.returncode,expected_exit_code=expected,time=now()))
        atomic_json(WORK/'reports/short_segments.json',receipts)
        if res.returncode!=expected:
            atomic_json(WORK/'state/status.json',dict(status='SHORT_FAILED',segment=name,exit_code=res.returncode,time=now(),formal_training_authorized=False))
            raise SystemExit(res.returncode or 1)
    budget=read_json(WORK/'state/update_budget.json');assert budget['optimizer_step_calls']==480 and budget['joint_SSL_updates']==240
    atomic_json(WORK/'state/status.json',dict(status='SHORT_CHECK_FINISHED',time=now(),optimizer_step_calls=480,joint_SSL_updates=240,formal_training_authorized=False,
        old_queue='INTERRUPTED',next='reports and five review rounds, then WAITING_FOR_FULL_APPROVAL; no training handoff'))
    print('Short check finished: 240 joint updates / 480 optimizer calls. No formal training launched.')

if __name__=='__main__':main()
