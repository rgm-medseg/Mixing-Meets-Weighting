import argparse
import time
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=['monitor','results'],default='monitor');a=p.parse_args()
    while True:
        s=read_json(WORK/'state/status.json')
        if a.mode=='monitor':print(json.dumps({k:s.get(k) for k in ['heartbeat','status','stage','pipeline_pid','child_pid','joint_step','optimizer1_calls','optimizer2_calls','ETA_remaining_hours','ETA_training_finish']},ensure_ascii=False),flush=True)
        else:
            print('R1 results:',WORK/'results','archive:',WORK/'TCSMix_v2B_R1_final_evidence.tar.gz',flush=True)
            for name in ['R1_results.csv','R1_class_metrics.csv']:
                if (WORK/'results'/name).exists():print((WORK/'results'/name).read_text(),flush=True)
        if s['status'] in ['DONE','FAILED','INTERRUPTED']:return
        time.sleep(15)

if __name__=='__main__':main()
