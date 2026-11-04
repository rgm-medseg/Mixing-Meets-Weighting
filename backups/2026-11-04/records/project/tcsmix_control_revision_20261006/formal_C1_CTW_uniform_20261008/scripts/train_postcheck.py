"""Freeze val-best identity and verify completed training before any test access."""
import numpy as np
import torch
from common import *

def main():
    cfg=read_json(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml');run=Path(cfg['run_root'])
    receipt=read_json(run/'TRAIN_RECEIPT.json');assert receipt['status']=='TRAINED' and receipt['global_step']==30000
    rows=read_table(WORK/'results/C1_control_trace.csv');latest={int(r['step']):r for r in rows}
    assert set(latest)==set(range(1,30001))
    for r in latest.values():
        active=[c for c in cfg['foreground_ids'] if int(r[f'n_{c}'])>0]
        q=np.mean([float(r[f'margin_class{c}']) for c in active]) if active else 0
        assert abs(q-float(r['q']))<1e-7 and abs((.18+.20*q)-float(r['s']))<1e-10
        d=max(1,min(255,round(256*(.18+.20*q))));assert int(r['rectangle_h'])==int(r['rectangle_w'])==d
        assert int(r['actual_area_pixels'])==d*d and float(r['actual_area'])==d*d/65536
        assert r['controller_version']=='v2_B' and r['shared_mask_and_target_identical']=='True'
        assert float(r['lr'])==.01 and abs(float(r['weighted_ctw_loss'])-.05*float(r['ctw_loss']))<1e-9
        assert np.isfinite(float(r['loss']))
    last=torch.load(run/'checkpoints/last.ckpt',map_location='cpu',weights_only=False);assert last['global_step']==30000
    needed={'model_1','model_2','ema_model','optimizer1','optimizer2','random_state','sampler_state','lr_state','amp_state','controller_metadata'}
    assert needed<=set(last) and last['amp_state'] is None
    assert last['controller_metadata']['cfg_sha256']==sha(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml') and last['controller_metadata']['stateless']
    bestpath=run/'checkpoints/best.ckpt';best=torch.load(bestpath,map_location='cpu',weights_only=False)
    assert best['controller_metadata']['cfg_sha256']==sha(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml')
    vals=read_table(WORK/'results/validation_trace.csv');finalvals={int(r['step']):r for r in vals};assert set(finalvals)==set(range(200,30001,200))
    bestvalue=max(float(r['student1_macro_dice']) for r in finalvals.values())
    assert abs(bestvalue-best['best_metrics']['model1_best_val_dice'])<1e-10
    assert abs(float(finalvals[best['global_step']]['student1_macro_dice'])-bestvalue)<1e-10
    events=read_table(WORK/'results/optimizer_calls.csv');counts={k:sum(e['optimizer']==k and e['event']=='success' for e in events) for k in ['optimizer1','optimizer2']}
    assert counts['optimizer1']>=30000 and counts['optimizer2']>=30000
    initial=read_json(WORK/'reports/INITIALIZATION_ACTUAL.json');assert initial['joint_step']==0 and initial['fresh']
    assert initial['sha256']==read_json(WORK/'protocol.json')['pretrain_sha256']
    atomic_json(WORK/'reports/SELECTED_CHECKPOINT.json',dict(status='FROZEN_BEFORE_TEST',time=now(),checkpoint=str(bestpath),sha256=sha(bestpath),
        student='model_1',selected_step=best['global_step'],selected_val_macro_dice=bestvalue,selection='original strict improvement of student1 validation macro Dice',
        validation_records=len(vals),test_metrics_consulted=False))
    atomic_json(WORK/'reports/TRAIN_POSTCHECK.json',dict(status='PASS',time=now(),logical_joint_steps=30000,raw_trace_rows=len(rows),
        duplicate_trace_records=len(rows)-30000,optimizer_success_calls=counts,normal_calls_each_student=30000,
        replayed_physical_updates=count if (count:=sum(counts.values())-60000) else 0,full_checkpoint_keys=sorted(last),
        last_sha256=sha(run/'checkpoints/last.ckpt'),best_sha256=sha(bestpath),worker_resume_exact=False,test_access_before_selection=False))
    print('Completed C1 training checked; validation-selected student1 checkpoint frozen before final test.')
    assert all(float(r['used_per_image_spatial_range_max'])==0 and float(r['max_per_image_mean_error'])<=1e-6 for r in rows)
    assert all(r['weight_mode']=='uniform' for r in rows)

if __name__=='__main__':main()
