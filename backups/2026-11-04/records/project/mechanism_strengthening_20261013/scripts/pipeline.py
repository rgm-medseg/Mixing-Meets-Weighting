"""Durable new-run driver; wait for our diagnostics, train once, then evaluate once."""
import json,os,subprocess,sys,time
from pathlib import Path
W=Path(__file__).resolve().parents[1]
def state(status,**kw):
    p=W/'receipts/pipeline_status.json';q=p.with_suffix('.tmp');q.write_text(json.dumps(dict(status=status,pid=os.getpid(),time=time.time(),**kw),indent=2)+'\n');q.replace(p)
def call(name,log):
    with (W/'logs'/log).open('w') as out:
        p=subprocess.Popen([sys.executable,str(W/'scripts'/name)],stdout=out,stderr=subprocess.STDOUT,env=dict(os.environ,OMP_NUM_THREADS='4',MKL_NUM_THREADS='4'))
        state('RUNNING',stage=name,child_pid=p.pid)
        code=p.wait()
    if code:state('FAILED',stage=name,exit_code=code);raise SystemExit(code)
def main():
    state('WAITING_FOR_DIAGNOSTICS')
    while not all((W/'receipts'/f'{n}_full.json').exists() for n in ['joint','trc']):time.sleep(5)
    for n in ['joint','trc']:assert json.loads((W/'receipts'/f'{n}_full.json').read_text())['status']=='PASS'
    state('WAITING_FOR_FREE_GPU')
    while subprocess.check_output(['nvidia-smi','--query-compute-apps=pid','--format=csv,noheader'],text=True).strip():time.sleep(5)
    call('train_fixed.py','fixed_training.log')
    call('evaluate_fixed.py','fixed_test.log')
    # Reporting reads real CSV/receipts and can also be rerun independently.
    if (W/'scripts/report.py').exists():call('report.py','report.log')
    state('DONE')
if __name__=='__main__':main()
