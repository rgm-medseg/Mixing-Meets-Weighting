"""Idempotent session-local R2 launcher, no global tmux changes."""
import argparse
import fcntl
import shlex
import subprocess
import time
from common import *

SESSION='beta_fft_R2_fixed92_20261007'

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=(WORK/'state/launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    from preflight import verify_frozen
    verify_frozen();state=read_json(WORK/'state/status.json')
    if state['status']=='DONE':print('R2 DONE:',WORK/'TCSMix_R2_fixed92_final_evidence.tar.gz');return
    pid=state.get('pipeline_pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists():
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        if str(WORK/'scripts/run_R2_pipeline.py') in cmd:
            print(f'Reuse live R2 controller PID{pid} session{SESSION}; no duplicate launch');return
    if state['status'] in ['FAILED','INTERRUPTED'] and not a.resume:raise SystemExit('Same-run recovery required: use resume.sh; no restart')
    command='/bin/bash '+shlex.quote(str(WORK/'scripts/pipeline_tmux.sh'))+(' --resume' if a.resume else '')
    exists=subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0
    if exists:
        # tmux 3.0a has no pane_start_path; dead panes also lose current cwd.
        original_command=subprocess.run(['tmux','display-message','-p','-t',SESSION+':pipeline','#{pane_start_command}'],capture_output=True,text=True,check=True).stdout.strip()
        if str(WORK/'scripts/pipeline_tmux.sh') not in shlex.split(original_command):
            raise RuntimeError('Existing named tmux has another pipeline command; no mutation')
        dead=subprocess.run(['tmux','display-message','-p','-t',SESSION+':pipeline','#{pane_dead}'],capture_output=True,text=True,check=True).stdout.strip()
        if dead=='0':
            print('Reuse existing live R2 pipeline pane; awaiting heartbeat, no duplicate or respawn');return
        subprocess.run(['tmux','respawn-pane','-k','-t',SESSION+':pipeline','-c',str(WORK),command],check=True)
    else:
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'-n','pipeline','-c',str(WORK),command],check=True)
        for name,mode in [('monitor','monitor'),('results','results')]:
            cmd=shlex.quote(str(PYTHON))+' '+shlex.quote(str(WORK/'scripts/monitor_R2.py'))+' --mode '+mode
            subprocess.run(['tmux','new-window','-t',SESSION,'-n',name,'-c',str(WORK),cmd],check=True)
    # On resume, wait for the new controller heartbeat before reopening dead monitors.
    for _ in range(20):
        current=read_json(WORK/'state/status.json')
        if current.get('pipeline_pid')!=pid:break
        time.sleep(.25)
    for name,mode in [('monitor','monitor'),('results','results')]:
        dead=subprocess.run(['tmux','display-message','-p','-t',SESSION+':'+name,'#{pane_dead}'],capture_output=True,text=True,check=True).stdout.strip()
        if dead=='1':
            monitor=shlex.quote(str(PYTHON))+' '+shlex.quote(str(WORK/'scripts/monitor_R2.py'))+' --mode '+mode
            subprocess.run(['tmux','respawn-pane','-k','-t',SESSION+':'+name,'-c',str(WORK),monitor],check=True)
    for window in ['pipeline','monitor','results']:subprocess.run(['tmux','set-window-option','-t',SESSION+':'+window,'remain-on-exit','on'],check=True)
    subprocess.run(['tmux','select-window','-t',SESSION+':monitor'],check=True)
    atomic_json(WORK/'reports/TMUX_LAUNCH.json',dict(time=now(),session=SESSION,command=command,cwd=str(WORK),resume=a.resume,
        ownership='server tmux ordinary bash/Python subprocess pipeline',client_independent=True,scope='authorized R2 only; old queue paused; R1 untouched'))
    print('Server detached tmux launched:',SESSION)

if __name__=='__main__':main()
