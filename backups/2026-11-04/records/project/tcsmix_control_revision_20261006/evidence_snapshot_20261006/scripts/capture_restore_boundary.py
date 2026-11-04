"""Preserve the completed short segment's checkpoint before its later final save."""
import argparse
import time
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--fresh-pid',required=True,type=int);a=p.parse_args()
    deadline=time.monotonic()+600
    while Path(f'/proc/{a.fresh_pid}').exists():
        if time.monotonic()>deadline:raise RuntimeError('Fresh segment still live; no checkpoint copied')
        time.sleep(1)
    run=WORK/'runs/SHORT_TCSMIX_V2';cp=run/'checkpoints/last.ckpt'
    import torch
    ck=torch.load(cp,map_location='cpu',weights_only=False)
    if ck['global_step']!=200:raise RuntimeError('Missed completed200 boundary; do not fabricate it')
    dst=run/'checkpoints/recovery_boundary_200.ckpt';assert not dst.exists()
    raw=cp.read_bytes();dst.write_bytes(raw)
    assert sha(dst)==sha(cp)
    atomic_json(WORK/'reports/recovery_boundary_200.json',dict(time=now(),checkpoint=str(dst),sha256=sha(dst),
        original_path=str(cp),global_step=200,epoch=ck['epoch'],controller_metadata=ck['controller_metadata'],
        keys=sorted(ck),sampler_consumed=ck['sampler_state']['consumed'],n_saved_batch_plans=len(ck['sampler_state']['batches']),
        captured_after_fresh_process_exited=True,scope='same short run diagnostic boundary, not formal initialization'))
    receipt=read_json(run/'TRAIN_RECEIPT.json')
    atomic_json(WORK/'reports/short_first_segment_TRAIN_RECEIPT.json',receipt)
    print('Preserved actual final200 short checkpoint; no optimizer calls.')

if __name__=='__main__':main()
