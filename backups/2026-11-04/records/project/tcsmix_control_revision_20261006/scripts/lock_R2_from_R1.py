"""Future area-matched fixed config; training trajectory only, never test scores."""
import argparse
import math
from common import *

def main():
    p=argparse.ArgumentParser();p.add_argument('--run',default=str(WORK/'runs/R1'));a=p.parse_args();run=Path(a.run)
    receipt=read_json(run/'TRAIN_RECEIPT.json')
    if receipt.get('task')!='R1' or receipt.get('status')!='TRAINED' or receipt.get('global_step')!=30000:raise RuntimeError('Require complete R1 run; offline/short estimate forbidden')
    trace=run/'control_trace.csv';rows=read_table(trace)
    latest={int(r['step']):r for r in rows};assert set(latest)==set(range(1,30001))
    assert all(r['controller_version']=='v2_B' for r in latest.values()), 'R1 must use selected v2_B controller'
    mean_area=sum(float(latest[i]['actual_area']) for i in range(1,30001))/30000
    target_pixels=mean_area*256*256
    # Original square aspect ratio. Pick nearest representable square area, not round(mean(s)).
    best=min(range(1,256),key=lambda d:(abs(d*d-target_pixels),d))
    cfg=read_json(WORK/'configs/R2_planned.yaml');cfg.update(fixed_side=best/256,run_root=str(WORK/'runs/R2'),phase='formal',
        fixed_scale_source=str(trace),fixed_scale_source_sha256=sha(trace),fixed_scale_locked=True)
    dst=WORK/'configs/R2_fixed_locked.yaml';assert not dst.exists();atomic_json(dst,cfg)
    actual=best*best/(256*256)
    atomic_json(WORK/'reports/R2_area_lock.json',dict(source_run=str(run),trace_sha256=sha(trace),source_receipt_sha256=sha(run/'TRAIN_RECEIPT.json'),
        n_raw_records=len(rows),n_final_logical_steps=30000,duplicate_records=len(rows)-30000,
        duplicate_policy='retain all raw rows; latest occurrence per logical step belongs to final resumed trajectory',
        mean_actual_area=mean_area,fixed_h=best,fixed_w=best,fixed_area=actual,absolute_area_error=actual-mean_area,
        relative_area_error=(actual-mean_area)/mean_area,area_percentage_point_error=100*(actual-mean_area),
        config=str(dst),config_sha256=sha(dst),test_metrics_read=False,time=now()))
    print('R2 locked from complete R1 trace; no training started')

if __name__=='__main__':main()
