"""Server-owned one-run pipeline. Failure/stop never retries from zero."""
import argparse,csv,fcntl,signal,subprocess,time,traceback,shutil
from datetime import datetime,timedelta
from common import *
from resources import gpu
CHILD=None;CURRENT={};START=time.monotonic();GPU_CACHE={};GPU_TIME=0
def process_identity(pid):
    try:
        p=Path('/proc')/str(pid);cmd=(p/'cmdline').read_bytes().replace(b'\x00',b' ').decode();st=(p/'stat').read_text().rsplit(')',1)[1].split()
        return dict(pid=pid,command=cmd,start_ticks=st[19])
    except (FileNotFoundError,ProcessLookupError):return None
def live(identity):
    return bool(identity and process_identity(identity['pid'])==identity)
def last_trace():
    p=WORK/'results/C1_control_trace.csv'
    if not p.exists():return {}
    with p.open('rb') as f:
        header=next(csv.reader([f.readline().decode().strip()]));f.seek(0,2);f.seek(max(0,f.tell()-65536));tail=f.read().decode(errors='replace')
    ls=tail.splitlines();ls=ls if tail.endswith('\n') else ls[:-1]
    for line in reversed(ls):
        v=next(csv.reader([line]))
        if len(v)==len(header) and v[0].isdigit():return dict(zip(header,v))
    return {}
def update(status,stage,**kw):
    global GPU_CACHE,GPU_TIME
    if time.monotonic()-GPU_TIME>30 or not GPU_CACHE:GPU_CACHE=gpu();GPU_TIME=time.monotonic()
    row=last_trace();step=int(row.get('step',0));base=CURRENT.get('resume_step',0);elapsed=time.monotonic()-CURRENT.get('monotonic_start',time.monotonic())
    eta=(30000-step)*26440.0130477/30000;basis='R1 actual successful Popen segment 26440.013s'
    if stage=='TRAIN' and step-base>=100:eta=(30000-step)*elapsed/(step-base);basis='current active C1 process elapsed/progress; includes startup and validation'
    wait=WORK/'state/resource_wait.json'
    if CHILD and stage in ['TRAIN','SHORT_CHECK','EVALUATE'] and wait.exists() and GPU_CACHE['pids'] and not any((WORK/'reports').glob(f'GPU_PID_BINDING_{CHILD.pid}.json')):
        status='WAITING_FOR_RESOURCE' if wait.stat().st_mtime>=CURRENT.get('epoch_start',0) else status
    obj=dict(status=status,stage=stage,heartbeat=now(),heartbeat_epoch=time.time(),pipeline_identity=process_identity(os.getpid()),child_identity=process_identity(CHILD.pid) if CHILD and CHILD.poll() is None else None,
        host=socket.gethostname(),GPU=0,resource=GPU_CACHE,task='C1_CTW_mean_uniform_v2B',tmux_session='beta_fft_C1_CTW_uniform_20261008',joint_step=step,optimizer1_calls=int(row.get('optimizer1_calls',0)),optimizer2_calls=int(row.get('optimizer2_calls',0)),
        ETA_remaining_hours=eta/3600,ETA_training_finish=(datetime.now().astimezone()+timedelta(seconds=eta)).isoformat() if step<30000 else None,ETA_basis=basis,
        latest_checkpoint=str(WORK/'runs/C1_CTW_mean_uniform_v2B/checkpoints/last.ckpt'),budget_joint=30000,current_command=CURRENT.get('command'),current_log=CURRENT.get('log'),old_queue='INTERRUPTED',automatic_retries=0,worker_resume_exact=False,**kw)
    atomic_json(WORK/'state/status.json',obj)
    first=read_json(WORK/'reports/FIRST_200_CHECKPOINT.json') if (WORK/'reports/FIRST_200_CHECKPOINT.json').exists() else 'PENDING'
    (WORK/'START_REPORT_C1_CN.md').write_text(f'C1当前 {status} / {stage}；时间 {now()}。\n\n服务器n1，GPU0 {GPU_CACHE["uuid"]}，{GPU_CACHE["name"]}。tmux：beta_fft_C1_CTW_uniform_20261008。控制器/子进程身份及完整命令见state/status.json。\n\n正式逻辑step={step}/30000，两优化器成功物理调用={obj["optimizer1_calls"]}/{obj["optimizer2_calls"]}。预计余下训练{eta/3600:.2f}小时，依据{basis}；评价分析归档另计。\n\n首个200步完整检查点：{json.dumps(first,ensure_ascii=False)}。初始化、短程、绑定和梯度收据见reports；未出现的阶段不宣称已通过。\n\n唯一科学干预：逐图自身整图原权重均值均匀CTW；TRC和TCSMix v2_B保留。E1保持3976暂停，E2–E5未启动。完成C1后停止。\n\n查看：scripts/status.sh；安全停止：scripts/stop.sh；原状态恢复：scripts/resume.sh。进入：tmux attach -t beta_fft_C1_CTW_uniform_20261008；detach：Ctrl-b d。服务器重启后需执行resume.sh；tmux不跨重启保活。\n')
def stop(signum,frame):(WORK/'state/stop_requested').touch()
def first_checkpoint():
    if (WORK/'reports/FIRST_200_CHECKPOINT.json').exists():return
    p=WORK/'results/validation_trace.csv'
    if not p.exists() or not any(int(r['step'])==200 for r in read_table(p)):return
    cp=WORK/'runs/C1_CTW_mean_uniform_v2B/checkpoints/last.ckpt';snapshot=WORK/'state/first200_snapshot.ckpt'
    shutil.copyfile(cp,snapshot)
    # CPU load in an independent process; leaves live training model/RNG untouched.
    cmd=[str(PYTHON),str(WORK/'scripts/check_checkpoint.py'),'--path',str(snapshot),'--step','200','--receipt',str(WORK/'reports/FIRST_200_CHECKPOINT.json')]
    with (WORK/'logs/first200_checkpoint_audit.log').open('a') as f:subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT,check=True)
def run(stage,script,args=()):
    global CHILD,CURRENT
    command=[str(PYTHON),str(WORK/'scripts'/script),*args];stamp=now().replace(':','').replace('.','');log=WORK/'logs'/f'{stage}_{stamp}.log'
    start=time.monotonic();entry=dict(stage=stage,command=command,start=now(),log=str(log),resume_step=int(last_trace().get('step',0)))
    CURRENT=dict(entry,monotonic_start=start,epoch_start=time.time())
    with log.open('x') as f:
        CHILD=subprocess.Popen(command,cwd=WORK,env=dict(os.environ,CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled'),stdout=f,stderr=subprocess.STDOUT,start_new_session=True)
        entry['process_identity']=process_identity(CHILD.pid);entry['process_group']=CHILD.pid
        atomic_json(WORK/'state/active_stage.json',entry)
        while CHILD.poll() is None:
            update('RUNNING',stage)
            if stage=='TRAIN':first_checkpoint()
            time.sleep(5)
        code=CHILD.returncode
    entry.update(exit_code=code,end=now(),elapsed_seconds=time.monotonic()-start,scope='Popen launch to observed completion, <=5s poll lag; stage imports/IO included')
    hist=read_json(WORK/'state/stage_history.json') if (WORK/'state/stage_history.json').exists() else [];hist.append(entry);atomic_json(WORK/'state/stage_history.json',hist)
    if stage=='TRAIN':
        h=read_json(WORK/'state/train_segments.json') if (WORK/'state/train_segments.json').exists() else [];h.append(entry);atomic_json(WORK/'state/train_segments.json',h)
    CHILD=None;CURRENT={}
    if code==75 or (WORK/'state/stop_requested').exists():raise InterruptedError(f'{stage} stopped; no further stage')
    if code!=0:raise RuntimeError(f'{stage} exit {code}; see {log}; no automatic retry')
    return entry
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=(WORK/'state/pipeline.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    signal.signal(signal.SIGTERM,stop);signal.signal(signal.SIGINT,stop)
    previous=read_json(WORK/'state/status.json') if (WORK/'state/status.json').exists() else {}
    if previous.get('status')=='DONE':
        from preflight import verify_frozen
        verify_frozen();print('C1 DONE; no second run');return
    if a.resume:(WORK/'state/stop_requested').unlink(missing_ok=True)
    elif (WORK/'state/stop_requested').exists():raise RuntimeError('Explicit resume required')
    try:
        run('PREFLIGHT','preflight.py');run('SHORT_CHECK','short_check.py')
        receipt=WORK/'runs/C1_CTW_mean_uniform_v2B/TRAIN_RECEIPT.json';complete=receipt.exists() and read_json(receipt).get('status')=='TRAINED' and read_json(receipt).get('global_step')==30000
        if not complete:
            runroot=receipt.parent;started=(runroot/'train.log').exists()
            if started and (not a.resume or not(runroot/'checkpoints/last.ckpt').exists()):raise RuntimeError('Existing incomplete C1 must resume own complete last checkpoint')
            run('TRAIN','train_C1_entry.py',['--resume'] if started else [])
        first_checkpoint()
        run('POSTCHECK','train_postcheck.py')
        run('EVALUATE','evaluate_C1.py')
        run('ANALYSIS','finalize.py',['analysis'])
        run('REVIEW','finalize.py',['review'])
        run('ARCHIVE','finalize.py',['package'])
        update('DONE','FINISHED',archive=str(WORK/'C1_CTW_uniform_v2B_final_evidence.tar.gz'),archive_status='VERIFIED',training_queue_stopped=True)
    except InterruptedError as e:update('INTERRUPTED','STOPPED',reason=str(e))
    except BaseException as e:
        (WORK/'reports'/f'FAILURE_{int(time.time())}.txt').write_text(traceback.format_exc());update('FAILED','RECOVERY_REQUIRED',reason=str(e));print(traceback.format_exc(),flush=True)
if __name__=='__main__':main()
