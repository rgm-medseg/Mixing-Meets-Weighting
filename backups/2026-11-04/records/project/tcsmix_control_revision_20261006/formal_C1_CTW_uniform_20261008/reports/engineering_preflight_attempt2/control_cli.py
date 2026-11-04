import argparse,fcntl,shlex,subprocess,time,tempfile
from common import *
from run_pipeline import process_identity,live,last_trace
SESSION='beta_fft_C1_CTW_uniform_20261008'
def lock_available(path):
    with path.open('a') as f:
        try:fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);return True
        except BlockingIOError:return False
def isolated_test():
    # Real isolated file-stop process and exclusive-lock semantics, no training/GPU.
    d=WORK/'runs/isolated_control_test';d.mkdir(exist_ok=True)
    flag=d/'stop';flag.unlink(missing_ok=True)
    with (d/'test.lock').open('a') as f:
        fcntl.flock(f,fcntl.LOCK_EX|fcntl.LOCK_NB);assert not lock_available(d/'test.lock')
    assert lock_available(d/'test.lock')
    code='import pathlib,time,sys; p=pathlib.Path(sys.argv[1]);\nwhile not p.exists(): time.sleep(.02)\n(p.parent/"INTERRUPTED").write_text("saved boundary")'
    child=subprocess.Popen([str(PYTHON),'-c',code,str(flag)],start_new_session=True)
    try:
        time.sleep(.2);ident=process_identity(child.pid);assert live(ident)
        wrong=dict(ident,start_ticks='wrong');assert not live(wrong)
        flag.touch();assert child.wait(timeout=10)==0 and (d/'INTERRUPTED').read_text()=='saved boundary'
    finally:
        flag.touch();child.wait(timeout=10)
    assert not live(ident);flag.unlink()
    atomic_json(WORK/'reports/ISOLATED_CONTROL_TEST.json',dict(status='PASS',time=now(),real_stop_flag=True,exclusive_lock=True,stale_identity_rejected=True,scope='isolated CPU controller fixture; actual trainer stop/resume independently exercised in 1+1 short check',formal_interrupted=False))
def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['start','resume','status','stop','self-test']);a=p.parse_args()
    if a.action=='self-test':isolated_test();return
    s=read_json(WORK/'state/status.json') if (WORK/'state/status.json').exists() else {}
    if a.action=='status':
        s.update(pipeline_alive=live(s.get('pipeline_identity')),child_alive=live(s.get('child_identity')),latest_complete_trace=last_trace());print(json.dumps(s,ensure_ascii=False,indent=2));return
    if a.action=='stop':
        if s.get('status') in ['DONE','FAILED','INTERRUPTED'] or not live(s.get('pipeline_identity')):print('No live C1 controller; no PID signalled');return
        (WORK/'state/stop_requested').touch();atomic_json(WORK/'reports/LAST_STOP_REQUEST.json',dict(time=now(),identity=s['pipeline_identity'],scope='C1 own stop file only; trainer saves at joint boundary; no PID/group kill'));print('C1 safe stop requested; inspect status for INTERRUPTED');return
    launcher=(WORK/'state/launcher.lock').open('a');fcntl.flock(launcher,fcntl.LOCK_EX)
    if not lock_available(WORK/'state/pipeline.lock') or not lock_available(WORK/'state/training.lock') or live(s.get('pipeline_identity')):print('Existing C1 process/lock reused; no duplicate');return
    if s.get('status')=='DONE':print('C1 already DONE');return
    if s.get('status') in ['FAILED','INTERRUPTED'] and a.action!='resume':raise RuntimeError('Use resume.sh for same-run recovery')
    stamp=now().replace(':','').replace('.','');log=WORK/'logs'/f'controller_{stamp}.log';command='exec '+shlex.quote(str(PYTHON))+' '+shlex.quote(str(WORK/'scripts/run_pipeline.py'))+(' --resume' if a.action=='resume' else '')+' >>'+shlex.quote(str(log))+' 2>&1'
    exists=subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0
    if exists:
        panes=subprocess.check_output(['tmux','list-panes','-t',SESSION,'-F','#{pane_start_path}'],text=True).splitlines();assert all(x==str(WORK) for x in panes)
        target=SESSION+':'+subprocess.check_output(['tmux','new-window','-P','-F','#{window_index}','-t',SESSION,'-n','resume_'+stamp[-12:],'-c',str(WORK),command],text=True).strip()
    else:
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'-n','pipeline','-c',str(WORK),command],check=True);target=SESSION+':pipeline'
    subprocess.run(['tmux','set-window-option','-t',target,'remain-on-exit','on'],check=True)
    atomic_json(WORK/f'reports/TMUX_LAUNCH_{stamp}.json',dict(time=now(),session=SESSION,command=command,host=socket.gethostname(),cwd=str(WORK),python=str(PYTHON),resume=a.action=='resume',client_independent=True,log=str(log)))
    print('Detached server tmux launched:',SESSION)
if __name__=='__main__':main()
