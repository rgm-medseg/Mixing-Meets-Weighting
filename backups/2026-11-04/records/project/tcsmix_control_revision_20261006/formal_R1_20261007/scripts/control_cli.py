"""Only this R1 status and safe-stop controls; no other process signals."""
import argparse
import time
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('action',choices=['status','stop']);a=p.parse_args()
    s=read_json(WORK/'state/status.json')
    if a.action=='status':print(json.dumps(s,ensure_ascii=False,indent=2));return
    if s['status'] in ['DONE','INTERRUPTED','FAILED']:
        print('No active R1 to stop:',s['status']);return
    (WORK/'state/stop_requested').touch()
    atomic_json(WORK/'reports/LAST_STOP_REQUEST.json',dict(time=now(),scope='this R1 only',request='file flag; trainer saves at next full iteration boundary; no forced kill',state=s))
    print('R1 safe stop requested; waiting for completed checkpoint and process exit.',flush=True)
    for _ in range(24):
        time.sleep(5);s=read_json(WORK/'state/status.json')
        if s['status'] in ['INTERRUPTED','DONE','FAILED'] and not s.get('child_pid'):
            print(json.dumps(s,ensure_ascii=False,indent=2));return
    print('Stop remains pending; no forced kill. Inspect status/log; next safe boundary will save.')

if __name__=='__main__':main()
