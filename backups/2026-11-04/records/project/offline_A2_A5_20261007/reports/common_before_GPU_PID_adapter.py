import csv,hashlib,json,os,subprocess,time,sys
from pathlib import Path
from datetime import datetime,timezone,timedelta
W=Path(__file__).resolve().parents[1]
ROOT=W.parent
DATA=Path('/home/renguanming/jupyterlab/rgm/KnowSAM-main/ACDC')
PY='/home/renguanming/.conda/envs/beta_fft/bin/python'
R1=ROOT/'tcsmix_control_revision_20261006/formal_R1_20261007'
R2=ROOT/'tcsmix_control_revision_20261006/formal_R2_fixed92_20261007'
def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def sha(p):
 h=hashlib.sha256()
 with Path(p).open('rb') as f:
  for b in iter(lambda:f.read(1048576),b''):h.update(b)
 return h.hexdigest()
def atomic(p,s):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 t.write_text(s);os.replace(t,p)
def js(p,o):atomic(p,json.dumps(o,ensure_ascii=False,indent=2,allow_nan=False)+'\n')
def read(p):return json.loads(Path(p).read_text())
def rows(p):
 with Path(p).open(newline='') as f:return list(csv.DictReader(f))
def csvout(p,rr):
 p=Path(p);p.parent.mkdir(parents=True,exist_ok=True);t=p.with_name(p.name+'.tmp.'+str(os.getpid()))
 fields=list(dict.fromkeys(k for r in rr for k in r))
 with t.open('w',newline='') as f:
  w=csv.DictWriter(f,fields);w.writeheader();w.writerows(rr)
 os.replace(t,p)
def config():return read(W/'configs/analysis.json')
def identity():return read(W/'manifests/identity.json')['id']
def unitdir(task):return W/'state/units'/identity()/task
def progress(task,**kw):js(W/'state/progress.json',dict(stage=task,time=now(),**kw))
def stopped():
 if (W/'state/stop_requested').exists():raise InterruptedError('stop requested; completed units retained')
def gpu():
 s=subprocess.check_output(['nvidia-smi','--query-gpu=index,uuid,memory.used,memory.free,utilization.gpu','--format=csv,noheader,nounits'],text=True).strip().split(',')
 apps=subprocess.check_output(['nvidia-smi','--query-compute-apps=gpu_uuid,pid,process_name,used_gpu_memory','--format=csv,noheader,nounits'],text=True).strip().splitlines()
 pids=[int(a.split(',')[1]) for a in apps if a and s[1].strip() in a]
 return dict(index=int(s[0]),uuid=s[1].strip(),used_mib=int(s[2]),free_mib=int(s[3]),utilization=int(s[4]),pids=pids)
def wait_gpu(own=False):
 while True:
  stopped();g=gpu();others=[p for p in g['pids'] if not(own and p==os.getpid())]
  if not others and (own or (g['used_mib']<2048 and g['free_mib']>30000 and g['utilization']<=10)):return g
  progress('WAIT_GPU',resource=g);time.sleep(5)
def protected():
 return {p:sha(p) for p in read(W/'manifests/protected.json')}
