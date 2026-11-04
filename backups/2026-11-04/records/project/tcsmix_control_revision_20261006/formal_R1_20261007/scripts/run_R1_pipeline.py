"""Server-owned, bounded single-run R1 train→evaluate→analyze→summary→archive."""
import argparse
import csv
import fcntl
import signal
import subprocess
import time
import traceback
from datetime import datetime,timedelta
from common import *

CHILD=None;STOP=False;STARTED=now();STAGES=[];ERRORS=[];LAST_RESOURCE=0.;RESOURCE={}

def stop_signal(signum,frame):
    global STOP
    STOP=True;(WORK/'state/stop_requested').touch()

def last_trace():
    p=WORK/'results/R1_control_trace.csv'
    if not p.exists():return {}
    with p.open('rb') as f:
        header=f.readline().decode().strip().split(',');f.seek(0,2);size=f.tell();f.seek(max(0,size-16384));tail=f.read().decode(errors='replace')
    lines=tail.splitlines()
    if tail and not tail.endswith('\n'):lines=lines[:-1]
    for line in reversed(lines):
        values=next(csv.reader([line]))
        if len(values)==len(header) and values[0].isdigit():return dict(zip(header,values))
    return {}

def resource():
    global RESOURCE,LAST_RESOURCE
    if time.monotonic()-LAST_RESOURCE<30 and RESOURCE:return RESOURCE
    gpu=subprocess.run(['nvidia-smi','--id=0','--query-gpu=uuid,name,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().split(', ')
    processes=subprocess.run(['nvidia-smi','--query-compute-apps=pid,gpu_uuid,used_gpu_memory','--format=csv,noheader,nounits'],capture_output=True,text=True,check=True).stdout.strip().splitlines()
    rows=[line.split(', ') for line in processes if line.strip()];pids=[int(row[0]) for row in rows if len(row)>1 and row[1]==gpu[0]]
    RESOURCE=dict(time=now(),gpu0_name=gpu[1],used_mib=int(gpu[2]),free_mib=int(gpu[3]),utilization=int(gpu[4]),pids=pids,
        admission_threshold='no GPU0 compute PIDs, used<2048MiB, free>=30000MiB, utilization<=10%')
    LAST_RESOURCE=time.monotonic();return RESOURCE

def state(status,stage,extra=None):
    previous=read_json(WORK/'state/status.json');r=last_trace();step=int(r.get('step',0))
    elapsed=time.monotonic()-extra['child_start'] if extra and 'child_start' in extra else None
    priorstep=extra.get('resume_step',0) if extra else 0
    remaining=24156*(30000-step)/30000;basis='short-stage planning estimate'
    if stage=='TRAIN' and elapsed and step-priorstep>=50:
        remaining=elapsed/(step-priorstep)*(30000-step);basis='current active formal segment elapsed / progressed joint iterations, includes validation/loading'
    obj=dict(status=status,stage=stage,heartbeat=now(),heartbeat_epoch=time.time(),pipeline_pid=os.getpid(),child_pid=CHILD.pid if CHILD and CHILD.poll() is None else None,
        host=socket.gethostname(),gpu=0,started=previous.get('started',STARTED),joint_step=step,budget_joint_iterations=30000,
        optimizer1_calls=int(r.get('optimizer1_calls',0)),optimizer2_calls=int(r.get('optimizer2_calls',0)),optimizer_counter_scope='actual successful physical calls in latest complete trace row; optimizer_events.csv additionally retains partial failure evidence',
        ETA_training_finish=(datetime.now().astimezone()+timedelta(seconds=remaining)).isoformat() if step<30000 else None,
        ETA_remaining_hours=remaining/3600,ETA_basis=basis,latest_checkpoint=str(WORK/'runs/R1_full_v2B/checkpoints/last.ckpt'),
        latest_training_log=str(WORK/'runs/R1_full_v2B/train.log'),resource=resource(),formal_R1_authorized=True,R2_started=False,old_queue='INTERRUPTED',
        stage_errors=ERRORS,archive_status='VERIFIED' if (WORK/'final_archive_receipt.json').exists() else 'PENDING')
    if extra:obj.update({k:v for k,v in extra.items() if k not in ['child_start']})
    atomic_json(WORK/'state/status.json',obj)

def admission(stage):
    while True:
        if STOP or (WORK/'state/stop_requested').exists():return False
        rr=resource()
        if not rr['pids'] and rr['used_mib']<2048 and rr['free_mib']>=30000 and rr['utilization']<=10:return True
        state('WAITING_RESOURCE',stage);time.sleep(5)

def run(stage,script,args=None,status='RUNNING'):
    global CHILD
    cmd=(["/bin/bash",str(WORK/'scripts/train_R1.sh')] if stage=='TRAIN' else [str(PYTHON),str(WORK/'scripts'/script)])+(args or [])
    stamp=now().replace(':','').replace('.','');log=WORK/'logs'/f'{stage.lower()}_{stamp}.log';start=time.monotonic()
    step=int(last_trace().get('step',0));entry=dict(stage=stage,command=cmd,start=now(),log=str(log),resume_step=step)
    with log.open('x') as f:
        CHILD=subprocess.Popen(cmd,cwd=WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled'),stdout=f,stderr=subprocess.STDOUT)
        entry['pid']=CHILD.pid;state(status,stage,dict(child_start=start,resume_step=step,current_log=str(log),current_command=cmd))
        while CHILD.poll() is None:
            state(status,stage,dict(child_start=start,resume_step=step,current_log=str(log),current_command=cmd));time.sleep(5)
        entry.update(end=now(),exit_code=CHILD.returncode,elapsed_seconds=time.monotonic()-start,
            measurement='Popen launch to detected subprocess completion; <=5s polling detection lag, imports/validation/checkpoints included')
    CHILD=None;STAGES.append(entry);atomic_json(WORK/'state/stage_history.json',STAGES)
    if stage=='TRAIN':
        segments=read_json(WORK/'state/train_segments.json') if (WORK/'state/train_segments.json').exists() else [];segments.append(entry);atomic_json(WORK/'state/train_segments.json',segments)
    if entry['exit_code'] not in [0,75]:
        ERRORS.append(entry);atomic_json(WORK/'state/stage_errors.json',ERRORS)
    return entry['exit_code']

def main():
    global STAGES,ERRORS,STOP
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=(WORK/'state/pipeline.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:raise SystemExit('Existing R1 pipeline owns lock; no second controller')
    signal.signal(signal.SIGTERM,stop_signal);signal.signal(signal.SIGINT,stop_signal)
    if (WORK/'state/stage_history.json').exists():STAGES=read_json(WORK/'state/stage_history.json')
    if (WORK/'state/stage_errors.json').exists():ERRORS=read_json(WORK/'state/stage_errors.json')
    prior=read_json(WORK/'state/status.json')
    if prior['status']=='DONE':print('R1 already DONE; no retraining');return
    if a.resume:(WORK/'state/stop_requested').unlink(missing_ok=True)
    elif (WORK/'state/stop_requested').exists():raise RuntimeError('Stopped R1 requires explicit same-run resume')
    try:
        if run('PREFLIGHT','preflight.py')!=0:raise RuntimeError('Definition/input preflight failed; training not started')
        receipt=WORK/'runs/R1_full_v2B/TRAIN_RECEIPT.json';trained=receipt.exists() and read_json(receipt).get('status')=='TRAINED' and read_json(receipt).get('global_step')==30000
        if not trained:
            cp=WORK/'runs/R1_full_v2B/checkpoints/last.ckpt';has_started=(WORK/'runs/R1_full_v2B/train.log').exists()
            if has_started and (not a.resume or not cp.exists()):raise RuntimeError('Existing incomplete R1 requires --resume and its full last checkpoint; never restart from0')
            if not admission('TRAIN'):
                state('INTERRUPTED','TRAIN');return
            code=run('TRAIN','train_R1_entry.py',['--resume'] if has_started else [])
            if code==75 or STOP or (WORK/'state/stop_requested').exists():
                state('INTERRUPTED','TRAIN');run('SUMMARY','summarize_R1.py',status='INTERRUPTED');run('PACKAGE','package_R1.py',status='INTERRUPTED');state('INTERRUPTED','STOPPED');return
            if code!=0:raise RuntimeError(f'Training exited{code}; no automatic retry or parameter change')
        state('TRAINED','POSTCHECK')
        if run('POSTCHECK','train_postcheck.py',status='TRAINED')!=0:raise RuntimeError('Completed-training postcheck failed; test evaluation not entered')
        evalreceipt=WORK/'results/EVALUATION_RECEIPT.json'
        if not evalreceipt.exists():
            if not admission('EVALUATE'):state('INTERRUPTED','EVALUATE');return
            evalcode=run('EVALUATE','evaluate_R1.py',status='TRAINED')
        else:evalcode=0
        state('EVALUATED' if evalcode==0 else 'FAILED','ANALYSIS')
        if STOP or (WORK/'state/stop_requested').exists():state('INTERRUPTED','ANALYSIS');return
        if admission('ANALYSIS'):analysiscode=run('ANALYSIS','analyze_R1.py',status='EVALUATED' if evalcode==0 else 'FAILED')
        else:state('INTERRUPTED','ANALYSIS');return
        state('ANALYZED' if analysiscode==0 else 'FAILED','SUMMARY')
        summarycode=run('SUMMARY','summarize_R1.py',status='ANALYZED')
        packcode=run('PACKAGE','package_R1.py',status='ANALYZED')
        if packcode!=0:state('FAILED','ARCHIVE',dict(archive_status='FAILED'))
        else:
            review=read_json(WORK/'reports/FINAL_FIVE_ROUND_CHECK.json')
            state('DONE' if evalcode==analysiscode==summarycode==0 and review['status']!='FAIL' else 'FAILED','FINISHED',dict(archive_status='VERIFIED',scientific_review=review['status'],results=str(WORK/'results'),archive=str(WORK/'TCSMix_v2B_R1_final_evidence.tar.gz')))
    except BaseException as exc:
        text=traceback.format_exc();(WORK/f'reports/pipeline_failure_{int(time.time())}.txt').write_text(text)
        ERRORS.append(dict(stage='PIPELINE',time=now(),error=str(exc)));atomic_json(WORK/'state/stage_errors.json',ERRORS)
        state('FAILED','RECOVERY_REQUIRED')
        for stage,script in [('PARTIAL_ANALYSIS','analyze_R1.py'),('SUMMARY','summarize_R1.py'),('PACKAGE','package_R1.py')]:
            try:run(stage,script,status='FAILED')
            except BaseException:(WORK/f'reports/{stage}_exception_{int(time.time())}.txt').write_text(traceback.format_exc())
        state('FAILED','RECOVERY_REQUIRED',dict(archive_status='VERIFIED' if (WORK/'final_archive_receipt.json').exists() else 'FAILED'))
        print(text,flush=True)

if __name__=='__main__':main()
