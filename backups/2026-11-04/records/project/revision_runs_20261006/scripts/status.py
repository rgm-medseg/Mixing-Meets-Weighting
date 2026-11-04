import json
import os
import time
from common import *

def main():
    path=WORK/'state/queue.json'
    state=read_json(path) if path.exists() else read_json(WORK/'run_manifest.json')
    pid=state.get('controller_pid')
    living=False
    if pid:
        try:
            os.kill(pid,0)
            living=str(WORK/'scripts/run_queue.py').encode() in Path(f'/proc/{pid}/cmdline').read_bytes()
        except OSError: pass
    age=time.time()-state.get('heartbeat_epoch',0) if 'heartbeat_epoch' in state else None
    print(f"host={read_json(WORK/'resolved_paths.json')['host']} controller_pid={pid} alive={living} heartbeat_age_seconds={age} status={state.get('status','NOT_STARTED')}")
    if age is not None and age>60 and state.get('status')=='RUNNING': print('STALE_HEARTBEAT: inspect controller process/log; launch is not completion')
    if state.get('status')=='RUNNING' and not living: print('CONTROLLER_EXITED: inspect logs and checkpoint before resume; RUNNING is stale')
    for task,row in state['tasks'].items():
        progress=WORK/f'runs/{task}/progress.json'
        step=read_json(progress)['step'] if progress.exists() else row.get('latest_step',0)
        print(f"{task}: {row['status']} pid={row.get('pid')} step={step} reason={row.get('reason','')} log={row.get('log',str(WORK/'logs'/f'{task}.log'))}")
    print(f"active={state.get('active_task')} archive={state.get('archive_status','PENDING')} results={WORK}")

if __name__=='__main__': main()
