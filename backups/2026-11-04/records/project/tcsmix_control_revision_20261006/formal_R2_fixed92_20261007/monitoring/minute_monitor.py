"""Read-only R2 observer, one sample per minute; no training controls or GPU forward."""
from pathlib import Path
import csv,fcntl,json,os,socket,time
from datetime import datetime,timezone,timedelta

ROOT=Path('/home/renguanming/jupyterlab/rgm/beta-FFT-main/tcsmix_control_revision_20261006/formal_R2_fixed92_20261007')
FIELDS=['observed_at','status','stage','joint_step','optimizer1_calls','optimizer2_calls','ETA_training_finish','ETA_remaining_minutes','heartbeat_age_seconds','pipeline_pid','pipeline_live','child_pid','child_live','gpu_used_mib','gpu_utilization','error_records','new_error_records','analysis_status','geometry_status','archive_status','old_queue','alert']

def now():return datetime.now(timezone(timedelta(hours=8))).isoformat()
def read(path):return json.loads(path.read_text()) if path.exists() else {}
def atomic(path,text):
    temp=path.with_name(path.name+'.tmp.'+str(os.getpid()))
    with temp.open('w') as f:f.write(text);f.flush();os.fsync(f.fileno())
    os.replace(temp,path)
def dump(path,value):atomic(path,json.dumps(value,ensure_ascii=False,indent=2)+'\n')
def live(pid,entry):
    if not pid:return False
    path=Path(f'/proc/{pid}/cmdline')
    try:return entry in path.read_bytes().decode(errors='replace')
    except OSError:return False

def main():
    lock=(ROOT/'monitoring/observer.lock').open('w')
    try:fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:print('Existing minute observer reused; no second instance',flush=True);return
    statefile=ROOT/'monitoring/status.json';previous=read(statefile)
    if previous.get('status')=='COMPLETE':print('R2 observation already complete',flush=True);return
    csvpath=ROOT/'reports/MINUTE_MONITOR.csv';rows=[]
    if csvpath.exists():
        with csvpath.open() as f:rows=list(csv.DictReader(f))
    baseline=len(read(ROOT/'state/status.json').get('stage_errors',[]));initial_step=int(read(ROOT/'state/status.json').get('joint_step',0));last_step=initial_step;stalled=0;start=time.monotonic();next_sample=start
    dump(statefile,dict(status='MONITORING',pid=os.getpid(),started=now(),interval_seconds=60,host=socket.gethostname(),initial_step=initial_step,scope='readonly status/process/receipt observation only; no signals/resume/retraining'))
    while True:
        s=read(ROOT/'state/status.json');analysis=read(ROOT/'analysis/ANALYSIS_RECEIPT.json');geometry=read(ROOT/'analysis/error_R1_R2/ERROR_ANALYSIS_RECEIPT.json');arch=read(ROOT/'final_archive_receipt.json');resource=s.get('resource',{})
        step=int(s.get('joint_step',0));p_live=live(s.get('pipeline_pid'),'run_R2_pipeline.py');c_live=live(s.get('child_pid'),str(ROOT))
        errors=len(s.get('stage_errors',[]));age=max(0,time.time()-s.get('heartbeat_epoch',time.time()))
        stalled=stalled+1 if s.get('stage')=='TRAIN' and step==last_step else 0
        alerts=[]
        if age>90 and s.get('status') not in ['DONE','FAILED','INTERRUPTED']:alerts.append('heartbeat_stale')
        if errors>baseline:alerts.append('new_stage_errors')
        if s.get('stage')=='TRAIN' and stalled>=3:alerts.append('training_step_stalled_3_samples')
        if s.get('status') not in ['DONE','FAILED','INTERRUPTED'] and not p_live:alerts.append('pipeline_not_live')
        row=dict(observed_at=now(),status=s.get('status'),stage=s.get('stage'),joint_step=step,optimizer1_calls=s.get('optimizer1_calls'),optimizer2_calls=s.get('optimizer2_calls'),ETA_training_finish=s.get('ETA_training_finish'),ETA_remaining_minutes=round(60*s.get('ETA_remaining_hours',0),2),heartbeat_age_seconds=round(age,2),pipeline_pid=s.get('pipeline_pid'),pipeline_live=p_live,child_pid=s.get('child_pid'),child_live=c_live,gpu_used_mib=resource.get('used_mib'),gpu_utilization=resource.get('utilization'),error_records=errors,new_error_records=max(0,errors-baseline),analysis_status=analysis.get('status','PENDING'),geometry_status=geometry.get('status','PENDING'),archive_status=arch.get('status','PENDING'),old_queue=s.get('old_queue'),alert=';'.join(alerts))
        rows.append(row)
        import io
        output=io.StringIO();writer=csv.DictWriter(output,fieldnames=FIELDS);writer.writeheader();writer.writerows(rows);atomic(csvpath,output.getvalue())
        dump(ROOT/'reports/MINUTE_MONITOR_LATEST.json',row);print(json.dumps(row,ensure_ascii=False),flush=True)
        if s.get('status')=='DONE' and analysis.get('status')=='ANALYZED' and geometry.get('status')=='DONE' and arch.get('status')=='VERIFIED':
            dump(statefile,dict(status='COMPLETE',pid=os.getpid(),completed=now(),interval_seconds=60,samples=len(rows),last=row,final_archive=arch.get('archive'),final_archive_sha256=arch.get('sha256'),scope='observed train/evaluate/analyze/geometry/package completion; no training changes'))
            dump(ROOT/'reports/MINUTE_MONITOR_COMPLETION.json',read(statefile));return
        dump(statefile,dict(status='ALERT' if alerts else 'MONITORING',pid=os.getpid(),updated=now(),interval_seconds=60,samples=len(rows),last=row,scope='readonly; user/agent handles any genuine failure, no automatic retries'))
        last_step=step;next_sample+=60
        if next_sample<time.monotonic():next_sample=time.monotonic()+60
        while time.monotonic()<next_sample:time.sleep(min(1,next_sample-time.monotonic()))

if __name__=='__main__':main()
