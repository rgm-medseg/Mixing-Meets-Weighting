"""Idempotent session-local R1 launcher, no global tmux changes."""
import argparse
import fcntl
import shlex
import subprocess
from common import *

SESSION='beta_fft_R1_v2B_20261007'

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    lock=(WORK/'state/launcher.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX)
    from preflight import verify_frozen
    verify_frozen();state=read_json(WORK/'state/status.json')
    if state['status']=='DONE':print('R1 DONE:',WORK/'TCSMix_v2B_R1_final_evidence.tar.gz');return
    pid=state.get('pipeline_pid')
    if pid and Path(f'/proc/{pid}/cmdline').exists():
        cmd=Path(f'/proc/{pid}/cmdline').read_bytes().decode(errors='replace')
        if str(WORK/'scripts/run_R1_pipeline.py') in cmd:
            print(f'Reuse live R1 controller PID{pid} session{SESSION}; no duplicate launch');return
    if state['status'] in ['FAILED','INTERRUPTED'] and not a.resume:raise SystemExit('Same-run recovery required: use resume.sh; no restart')
    command='/bin/bash '+shlex.quote(str(WORK/'scripts/pipeline_tmux.sh'))+(' --resume' if a.resume else '')
    exists=subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0
    if exists:
        cwd=subprocess.run(['tmux','display-message','-p','-t',SESSION+':pipeline','#{pane_start_path}'],capture_output=True,text=True,check=True).stdout.strip()
        if cwd!=str(WORK):raise RuntimeError('Existing named tmux belongs to another directory; no mutation')
        subprocess.run(['tmux','respawn-pane','-k','-t',SESSION+':pipeline','-c',str(WORK),command],check=True)
    else:
        subprocess.run(['tmux','new-session','-d','-s',SESSION,'-n','pipeline','-c',str(WORK),command],check=True)
        for name,mode in [('monitor','monitor'),('results','results')]:
            cmd=shlex.quote(str(PYTHON))+' '+shlex.quote(str(WORK/'scripts/monitor_R1.py'))+' --mode '+mode
            subprocess.run(['tmux','new-window','-t',SESSION,'-n',name,'-c',str(WORK),cmd],check=True)
    for window in ['pipeline','monitor','results']:subprocess.run(['tmux','set-window-option','-t',SESSION+':'+window,'remain-on-exit','on'],check=True)
    subprocess.run(['tmux','select-window','-t',SESSION+':monitor'],check=True)
    atomic_json(WORK/'reports/TMUX_LAUNCH.json',dict(time=now(),session=SESSION,command=command,cwd=str(WORK),resume=a.resume,
        ownership='server tmux ordinary bash/Python subprocess pipeline',client_independent=True,R2_started=False))
    print('Server detached tmux launched:',SESSION)

if __name__=='__main__':main()
