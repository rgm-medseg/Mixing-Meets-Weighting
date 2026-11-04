"""Own file-stop, lock and guarded tmux launch; no foreign-process signalling."""
import argparse,fcntl,shlex,subprocess,time
from common import *
from run_queue import SESSION,process_identity,live,last_trace
def lock_available(path):
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        except BlockingIOError:return False
def selftest():
    d=MAIN_WORK/'runs/isolated_control_test';d.mkdir(exist_ok=True);flag=d/'stop';flag.unlink(missing_ok=True)
    with (d/'test.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not lock_available(d/'test.lock')
    assert lock_available(d/'test.lock')
    code='import pathlib,time,sys; p=pathlib.Path(sys.argv[1]);\nwhile not p.exists(): time.sleep(.02)\n(p.parent/"INTERRUPTED").write_text("saved boundary")'
    child=subprocess.Popen([str(PYTHON),'-c',code,str(flag)],start_new_session=True)
    try:
        time.sleep(.2);identity=process_identity(child.pid);assert live(identity) and not live(dict(identity,start_ticks='wrong'))
        flag.touch();assert child.wait(timeout=10)==0 and (d/'INTERRUPTED').exists()
    finally:flag.touch();child.wait(timeout=10)
    assert not live(identity);flag.unlink()
    atomic_json(MAIN_WORK/'reports/ISOLATED_CONTROL_TEST.json',dict(status='PASS',time=now(),exclusive_lock=True,real_file_stop=True,stale_PID_identity_rejected=True,scope='isolated CPU; real trainer safe stop/resume separately in each2-step short'))
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','resume','stop','status','self-test']);a=p.parse_args()
    if a.action=='self-test':selftest();return
    s=read_json(MAIN_WORK/'state/status.json') if (MAIN_WORK/'state/status.json').exists() else {}
    if a.action=='status':
        s.update(pipeline_alive=live(s.get('pipeline_identity')),child_alive=live(s.get('child_identity')),latest_trace={t:last_trace(t) for t in ['E4','E5']});print(json.dumps(s,ensure_ascii=False,indent=2));return
    if a.action=='stop':
        if not live(s.get('pipeline_identity')):print('No live owned controller; no PID signalled');return
        (MAIN_WORK/'state/stop_requested').touch();atomic_json(MAIN_WORK/'reports/LAST_STOP_REQUEST.json',dict(time=now(),scope='this new queue stop file only; safe boundary; no signal/kill'));print('Safe stop requested; wait for INTERRUPTED');return
    launcher=(MAIN_WORK/'state/launcher.lock').open('a');fcntl.flock(launcher,fcntl.LOCK_EX)
    if not lock_available(MAIN_WORK/'state/pipeline.lock') or live(s.get('pipeline_identity')):print('Existing owned queue reused; no duplicate');return
    for task in ['E4','E5']:
        if not lock_available(MAIN_WORK/'tasks'/task/'state/training.lock'):raise RuntimeError('Existing task trainer lock; no duplicate')
    if s.get('status')=='DONE':
        from run_queue import verify_completed
        for t in ['E4','E5']:verify_completed(t)
        print('Both DONE, outputs verified; no second run');return
    if s.get('status') in ['FAILED','INTERRUPTED','PARTIAL_DONE'] and a.action!='resume':raise RuntimeError('Use resume.sh for same-run recovery')
    stamp=now().replace(':','').replace('.','');log=MAIN_WORK/'logs'/f'controller_{stamp}.log'
    cmd='exec '+shlex.quote(str(PYTHON))+' '+shlex.quote(str(MAIN_WORK/'scripts/run_queue.py'))+(' --resume' if a.action=='resume' else '')+' >>'+shlex.quote(str(log))+' 2>&1'
    exists=subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0
    if exists:
        owner=subprocess.run(['tmux','show-option','-qv','-t',SESSION,'@E4_E5_owner'],capture_output=True,text=True).stdout.strip();assert owner==str(MAIN_WORK),'Unverified existing tmux session; preserve it'
        index=subprocess.check_output(['tmux','new-window','-P','-F','#{window_index}','-t',SESSION,'-n','resume_'+stamp[-12:],'-c',str(MAIN_WORK),cmd],text=True).strip();target=SESSION+':'+index
    else:
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'-n','pipeline','-c',str(MAIN_WORK),cmd],check=True);target=SESSION+':pipeline'
    subprocess.run(['tmux','set-option','-t',SESSION,'@E4_E5_owner',str(MAIN_WORK)],check=True);subprocess.run(['tmux','set-window-option','-t',target,'remain-on-exit','on'],check=True)
    atomic_json(MAIN_WORK/f'reports/TMUX_LAUNCH_{stamp}.json',dict(time=now(),session=SESSION,command=cmd,host=socket.gethostname(),cwd=str(MAIN_WORK),python=str(PYTHON),resume=a.action=='resume',client_independent=True,log=str(log)))
    print('Independent tmux queue started:',SESSION)
if __name__=='__main__':main()
