"""Ordinary server controller: serial GPU tasks; no model/tool/client dependency."""
import argparse,fcntl,signal,subprocess,time,traceback,shutil
from datetime import datetime,timedelta
from common import *
from resources import gpu
SESSION='beta_fft_E4_E5_20261008'
CHILD=None;TASK=None;CURRENT={};GPU_CACHE={};GPU_TIME=0
def process_identity(pid):
    try:
        p=Path('/proc')/str(pid);cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode();st=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,command=cmd,start_ticks=st[19])
    except (FileNotFoundError,ProcessLookupError):return None
def live(identity):return bool(identity and process_identity(identity['pid'])==identity)
def taskroot(t):return MAIN_WORK/'tasks'/t
def last_trace(t):
    p=taskroot(t)/'results'/f'{t}_control_trace.csv'
    if not p.exists():return {}
    with p.open('rb') as f:
        head=next(csv.reader([f.readline().decode().strip()]));f.seek(0,2);f.seek(max(0,f.tell()-65536));tail=f.read().decode(errors='replace')
    lines=tail.splitlines();lines=lines if tail.endswith('\n') else lines[:-1]
    for line in reversed(lines):
        v=next(csv.reader([line]))
        if len(v)==len(head) and v[0].isdigit():return dict(zip(head,v))
    return {}
def state_tasks():return read_json(MAIN_WORK/'state/tasks.json') if (MAIN_WORK/'state/tasks.json').exists() else {t:dict(status='PENDING',step=0) for t in ['E4','E5']}
def set_task(t,status,**kw):
    tasks=state_tasks();tasks[t].update(status=status,time=now(),**kw);atomic_json(MAIN_WORK/'state/tasks.json',tasks)
def verify_completed(t):
    root=taskroot(t);a=read_json(root/'reports/ANALYSIS_RECEIPT.json');assert all(sha(root/p)==h for p,h in a['outputs'].items())
    e=read_json(root/'results/EVALUATION_RECEIPT.json');s=read_json(root/'reports/SELECTED_CHECKPOINT.json')
    assert e['checkpoint_sha256']==s['sha256']==sha(s['checkpoint']) and e['status']=='EVALUATED'
def update(status,stage,**kw):
    global GPU_CACHE,GPU_TIME
    if time.monotonic()-GPU_TIME>30 or not GPU_CACHE:
        GPU_CACHE=gpu();GPU_TIME=time.monotonic()
        procs=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
        own=None
        binding=taskroot(TASK)/'reports'/f'GPU_PID_BINDING_{CHILD.pid}.json' if TASK and CHILD else None
        if binding and binding.exists():own=read_json(binding)['NVML_pid']
        matched=[x for x in procs if x and x.split(', ')[0]==GPU_CACHE['uuid'] and int(x.split(', ')[1])==own]
        mem=int(matched[0].split(', ')[2]) if matched else 'NA'
        path=MAIN_WORK/'results/resource_samples.csv'
        with path.open('a',newline='') as f:
            wr=csv.DictWriter(f,fieldnames=['time','task','stage','GPU_uuid','whole_card_used_mib','own_NVML_pid','own_NVML_process_mib','all_NVML_PIDs'])
            if f.tell()==0:wr.writeheader()
            wr.writerow(dict(time=now(),task=TASK or 'queue',stage=stage,GPU_uuid=GPU_CACHE['uuid'],whole_card_used_mib=GPU_CACHE['used_mib'],own_NVML_pid=own or 'NA',own_NVML_process_mib=mem,all_NVML_PIDs=json.dumps(GPU_CACHE['pids'])))
    tasks=state_tasks()
    for t in tasks:
        row=last_trace(t);tasks[t]['step']=int(row.get('step',0));tasks[t]['optimizer_calls']={n:int(row.get(n+'_calls',0)) for n in ['optimizer1','optimizer2']}
    eta=None;basis='pending preparation'
    if TASK:
        cfg=read_json(next((MAIN_WORK/'configs').glob(TASK+'_*.yaml')));step=tasks[TASK]['step'];base=CURRENT.get('resume_step',0)
        short=taskroot(TASK)/'reports/SHORT_CHECK.json';seconds=cfg.get('estimated_seconds') or (read_json(short)['mean_step_seconds']*30000*1.35 if short.exists() else None)
        if seconds:eta=(30000-step)*seconds/30000;basis='R1 actual training wall' if TASK=='E4' else 'isolated official short throughput +35% validation/IO allowance'
        if stage=='TRAIN' and step-base>=200:
            eta=(30000-step)*(time.monotonic()-CURRENT['monotonic_start'])/(step-base);basis='current active process wall/progress including validation and IO'
        wr=taskroot(TASK)/'state/resource_wait.json';bound=CHILD and (taskroot(TASK)/'reports'/f'GPU_PID_BINDING_{CHILD.pid}.json').exists()
        if wr.exists() and CHILD and stage in ['TRAIN','EVALUATE'] and not bound and GPU_CACHE['pids']:status='WAITING_FOR_RESOURCE'
    obj=dict(status=status,stage=stage,task=TASK,heartbeat=now(),heartbeat_epoch=time.time(),host=socket.gethostname(),GPU=0,resource=GPU_CACHE,tmux_session=SESSION,pipeline_identity=process_identity(os.getpid()),child_identity=process_identity(CHILD.pid) if CHILD and CHILD.poll() is None else None,current_command=CURRENT.get('command'),current_log=CURRENT.get('log'),tasks=tasks,ETA_current_training_remaining_hours=eta/3600 if eta is not None else None,ETA_basis=basis,formal_budget=60000,E5_pretrain_budget=0,E3_executed=False,old_queue='INTERRUPTED',automatic_retries=0,worker_resume_exact=False,**kw)
    atomic_json(MAIN_WORK/'state/status.json',obj)
def stop(signum,frame):(MAIN_WORK/'state/stop_requested').touch()
def first_checkpoint(t):
    root=taskroot(t);receipt=root/'reports/FIRST_200_CHECKPOINT.json'
    if receipt.exists():return
    val=root/'results/validation_trace.csv'
    if not val.exists() or not any(int(r['step'])==200 for r in read_table(val)):return
    cfg=read_json(next((MAIN_WORK/'configs').glob(t+'_*.yaml')));cp=Path(cfg['run_root'])/'checkpoints/last.ckpt';snap=root/'state/first200_snapshot.ckpt'
    shutil.copyfile(cp,snap)
    cmd=[str(PYTHON),str(MAIN_WORK/'scripts/check_checkpoint.py'),'--task',t,'--path',str(snap),'--step','200','--receipt',str(receipt)]
    with (root/'logs/first200_checkpoint_audit.log').open('a') as f:subprocess.run(cmd,env=dict(os.environ,BETA_TASK_WORK=str(root),BETA_TASK=t),stdout=f,stderr=subprocess.STDOUT,check=True)
def run(stage,script,args=(),task=None):
    global CHILD,CURRENT,TASK
    TASK=task;root=taskroot(task) if task else MAIN_WORK;stamp=now().replace(':','').replace('.','');log=root/'logs'/f'{stage}_{stamp}.log'
    cmd=[str(PYTHON),str(MAIN_WORK/'scripts'/script),*args];start=time.monotonic()
    entry=dict(task=task,stage=stage,command=cmd,start=now(),log=str(log),resume_step=int(last_trace(task).get('step',0)) if task else 0)
    CURRENT=dict(entry,monotonic_start=start)
    env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    if task:env.update(BETA_TASK_WORK=str(root),BETA_TASK=task)
    with log.open('x') as f:
        CHILD=subprocess.Popen(cmd,cwd=MAIN_WORK,env=env,stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        time.sleep(.2);entry['process_identity']=process_identity(CHILD.pid);atomic_json(MAIN_WORK/'state/active_stage.json',entry)
        while CHILD.poll() is None:
            update('RUNNING',stage)
            if stage=='TRAIN':
                try:first_checkpoint(task)
                except BaseException:
                    (root/'reports/FIRST_CHECKPOINT_FAILURE.txt').write_text(traceback.format_exc())
                    (root/'state/stop_requested').touch()
            time.sleep(5)
        code=CHILD.returncode
    entry.update(end=now(),exit_code=code,elapsed_seconds=time.monotonic()-start)
    for dest in [MAIN_WORK]+([root] if task else []):
        path=dest/'state/stage_history.json';h=read_json(path) if path.exists() else [];h.append(entry);atomic_json(path,h)
    if stage=='TRAIN':
        path=root/'state/train_segments.json';h=read_json(path) if path.exists() else [];h.append(entry);atomic_json(path,h)
    CHILD=None;CURRENT={}
    if (root/'reports/FIRST_CHECKPOINT_FAILURE.txt').exists() and code==75:raise RuntimeError('First200 audit failed; this task stopped at safe boundary; independent task may continue')
    if code==75 or (MAIN_WORK/'state/stop_requested').exists():raise InterruptedError(f'{task} {stage} safe stop')
    if code!=0:raise RuntimeError(f'{task} {stage} exited {code}; retained {log}; no automatic retry')
    return entry
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=(MAIN_WORK/'state/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    old=read_json(MAIN_WORK/'state/status.json') if (MAIN_WORK/'state/status.json').exists() else {}
    if old.get('status')=='DONE':
        from preflight import verify_frozen
        verify_frozen()
        for t in ['E4','E5']:verify_completed(t)
        print('Both tasks DONE and outputs verified; no duplicate run');return
    if a.resume:(MAIN_WORK/'state/stop_requested').unlink(missing_ok=True)
    elif (MAIN_WORK/'state/stop_requested').exists():raise RuntimeError('Explicit resume required')
    try:
        run('PREFLIGHT','preflight.py')
        for t in ['E4','E5']:
            if state_tasks()[t]['status']=='DONE':verify_completed(t);continue
            try:run('SHORT_CHECK','short_check.py',['--task',t],t);set_task(t,'READY')
            except InterruptedError:raise
            except BaseException:
                path=taskroot(t)/'reports'/f'PREP_FAILURE_{time.time_ns()}.txt';path.write_text(traceback.format_exc());set_task(t,'FAILED',reason=str(path))
        for t in ['E4','E5']:
            if state_tasks()[t]['status']=='DONE':continue
            if state_tasks()[t]['status']!='READY':continue
            try:
                cfg=read_json(next((MAIN_WORK/'configs').glob(t+'_*.yaml')));root=taskroot(t);r=Path(cfg['run_root']);receipt=r/'TRAIN_RECEIPT.json'
                completed=receipt.exists() and read_json(receipt)['status']=='TRAINED' and read_json(receipt)['global_step']==30000
                if not completed:
                    started=(root/'reports/INITIALIZATION_ACTUAL.json').exists() or (root/'results'/f'{t}_control_trace.csv').exists()
                    if started and (not a.resume or not(r/'checkpoints/last.ckpt').exists()):raise RuntimeError('Existing task requires own complete last checkpoint and explicit resume; never restart0')
                    set_task(t,'RUNNING',stage='TRAIN');run('TRAIN',f'train_{t}_entry.py',['--resume'] if started else [],t)
                first_checkpoint(t);run('POSTCHECK','train_postcheck.py',task=t)
                run('EVALUATE','evaluate.py',task=t);run('ANALYSIS','analyze.py',task=t)
                set_task(t,'DONE',stage='ANALYZED',step=30000)
            except InterruptedError:
                set_task(t,'INTERRUPTED',stage='STOPPED');raise
            except BaseException:
                path=taskroot(t)/'reports'/f'FAILURE_{time.time_ns()}.txt';path.write_text(traceback.format_exc());set_task(t,'FAILED',reason=str(path));print(traceback.format_exc(),flush=True)
        run('SUMMARY_REVIEW','package_results.py',['review']);run('ARCHIVE','package_results.py',['package'])
        good=all(v['status']=='DONE' for v in state_tasks().values())
        update('DONE' if good else 'PARTIAL_DONE','FINISHED',archive=str(MAIN_WORK/'E4_E5_final_evidence_20261008.tar.gz'),training_queue_stopped=True)
    except InterruptedError as e:update('INTERRUPTED','STOPPED',reason=str(e))
    except BaseException as e:
        (MAIN_WORK/'reports'/f'CONTROLLER_FAILURE_{time.time_ns()}.txt').write_text(traceback.format_exc());update('FAILED','RECOVERY_REQUIRED',reason=str(e));print(traceback.format_exc(),flush=True)
if __name__=='__main__':main()
