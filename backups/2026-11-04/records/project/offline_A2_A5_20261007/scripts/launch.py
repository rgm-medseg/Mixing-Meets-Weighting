import fcntl,subprocess,shutil,hashlib,json
from common import *
SESSION='beta_fft_A2_A5_20261007'
def main():
 f=(W/'state/launch.lock').open('a');fcntl.flock(f,fcntl.LOCK_EX)
 p=W/'state/controller.json'
 if p.exists():
  s=read(p);pid=s.get('controller_pid')
  if pid and Path(f'/proc/{pid}/cmdline').exists() and any(str(W/'scripts'/n).encode() in Path(f'/proc/{pid}/cmdline').read_bytes() for n in ['pipeline.py','finalize_delivery.py']):print('Reuse existing controller',pid);return
  child=s.get('child_pid')
  if child and Path(f'/proc/{child}/cmdline').exists():raise RuntimeError('Existing child still live; avoid duplicate controller')
 current=dict(config=sha(W/'configs/analysis.json'),fixed_samples=sha(W/'manifests/fixed_samples.csv'),inputs=protected(),scripts={n:sha(W/'scripts'/n) for n in ['common.py','adapter.py','a2.py','a5.py','finish.py']})
 fingerprint=hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest()
 ip=W/'manifests/identity.json'
 report_fingerprint=fingerprint
 if ip.exists():
  old=read(ip)['fingerprint'];comparison=json.loads(json.dumps(old));comparison['scripts']['finish.py']=current['scripts']['finish.py']
  if comparison==current:
   fingerprint=read(ip)['id']
   if old['scripts']['finish.py']!=current['scripts']['finish.py']:
    js(W/'reports/REPORT_ONLY_AMENDMENT.json',dict(time=now(),reason='verification cohort selector only; source/input/config/checkpoint and all GPU analysis scripts byte-identical; completed GPU units retained',analysis_identity=fingerprint,previous_finish_sha256=old['scripts']['finish.py'],active_finish_sha256=current['scripts']['finish.py'],report_fingerprint=report_fingerprint))
 if ip.exists() and read(ip)['id']!=fingerprint:
  prior=read(ip)['id'];v=W/'versions'/prior;v.mkdir(parents=True,exist_ok=True)
  for n in ['configs','manifests','results','reports','figures','legacy']:
   if (W/n).exists():shutil.copytree(W/n,v/n,dirs_exist_ok=True)
  for artifact in [*W.glob('*.md'),*W.glob('*.tar.gz'),*W.glob('*.sha256')]:shutil.copy2(artifact,v/artifact.name)
  if current['inputs']!=read(W/'manifests/protected.json'):
   subprocess.run([PY,str(W/'scripts/prepare.py')],check=True)
   current.update(config=sha(W/'configs/analysis.json'),fixed_samples=sha(W/'manifests/fixed_samples.csv'),inputs=protected())
   fingerprint=hashlib.sha256(json.dumps(current,sort_keys=True).encode()).hexdigest()
 js(ip,dict(id=fingerprint,created=now(),fingerprint=current,active_report_fingerprint=report_fingerprint,identity_scope='original analysis identity retained for report-only amendments; source/input/checkpoint/analysis GPU scripts must match exactly'))
 if p.exists() and read(p).get('status')=='DONE' and read(p).get('identity')==fingerprint:print('Completed identical analysis; no rerun');return
 for n in ['common.py','adapter.py','a2.py','a5.py','finish.py']:shutil.copy2(W/'scripts'/n,W/'manifests'/('executed_'+n))
 (W/'state/stop_requested').unlink(missing_ok=True)
 cmd=f'cd {W} && exec env PYTHONDONTWRITEBYTECODE=1 {PY} -u {W}/scripts/pipeline.py >> {W}/logs/controller.log 2>&1'
 exists=subprocess.run(['tmux','has-session','-t',SESSION],capture_output=True).returncode==0
 if exists:subprocess.run(['tmux','respawn-pane','-k','-t',SESSION+':0.0',cmd],check=True)
 else:subprocess.run(['tmux','new-session','-d','-s',SESSION,'-c',str(W),cmd],check=True)
 subprocess.run(['tmux','set-option','-t',SESSION,'remain-on-exit','on'],check=True)
 print('Started independent offline pipeline',SESSION)
if __name__=='__main__':main()
