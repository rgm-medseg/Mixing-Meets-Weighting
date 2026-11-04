import shlex
from common import *
SESSION='beta_fft_paper_offline_followup_20261007'
def alive(pid):
 try:os.kill(pid,0);return True
 except (OSError,TypeError):return False
def main():
 action=sys.argv[1];state=read(W/'state/controller.json') if (W/'state/controller.json').exists() else {}
 if action=='status':print(json.dumps(dict(controller=state,progress=read(W/'state/progress.json') if (W/'state/progress.json').exists() else {},gpu=gpu()),ensure_ascii=False,indent=2));return
 if action=='stop':atomic(W/'state/stop_requested',now());print('Safe unit stop requested; no process killed');return
 if state.get('status')=='RUNNING' and alive(state.get('controller_pid')):
  js(W/'reports/DUPLICATE_START_RECEIPT.json',dict(status='PASS',time=now(),reused_pid=state['controller_pid'],second_controller_created=False));print('REUSED',state['controller_pid']);return
 if state.get('status')=='DONE' and action=='start':print('DONE reused; use resume only for missing-unit/review regeneration');return
 if action=='resume':(W/'state/stop_requested').unlink(missing_ok=True)
 cmd=shlex.join([PY,str(W/'scripts/pipeline.py')]);exists=subprocess.run(['tmux','has-session','-t',SESSION],stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL).returncode==0
 subprocess.check_call(['tmux','new-window' if exists else 'new-session','-d','-t' if exists else '-s',SESSION,'-n','offline-'+str(time.time_ns()),cmd])
 subprocess.run(['tmux','set-option','-t',SESSION,'remain-on-exit','on'],check=True);print('STARTED',SESSION)
if __name__=='__main__':main()
