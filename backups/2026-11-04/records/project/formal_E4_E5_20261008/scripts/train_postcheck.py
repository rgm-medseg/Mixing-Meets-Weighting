"""Complete trajectory and own validation-best identity frozen before test."""
import numpy as np
from common import *
def main():
    from preflight import verify_frozen
    from check_checkpoint import check
    import torch
    task=os.environ['BETA_TASK'];verify_frozen(task);cfgpath,cfg=config_for(task);run=Path(cfg['run_root'])
    rec=read_json(run/'TRAIN_RECEIPT.json');assert rec['status']=='TRAINED' and rec['global_step']==30000
    rows=read_table(WORK/'results'/f'{task}_control_trace.csv');latest={int(r['step']):r for r in rows};assert set(latest)==set(range(1,30001))
    for n,r in latest.items():
        assert np.isfinite(float(r['loss']))
        if task=='E4':
            active=[c for c in cfg['foreground_ids'] if int(r[f'n_{c}'])>0];q=np.mean([float(r[f'margin_class{c}']) for c in active]) if active else 0
            assert abs(q-float(r['q']))<1e-7 and abs((.18+.20*q)-float(r['s']))<1e-10
            d=max(1,min(255,round(256*(.18+.20*q))));assert int(r['rectangle_h'])==int(r['rectangle_w'])==d and int(r['actual_area_pixels'])==d*d and float(r['actual_area'])==d*d/65536
            assert float(r['lr'])==.01 and abs(float(r['weighted_ctw_loss'])-.05*float(r['ctw_loss']))<1e-9
        else:
            assert int(r['cur_step'])==n-1 and int(r['curriculum_total_steps'])==7500 and int(r['total_images'])==24 and int(r['visible_GT_images'])==12 and float(r['lr'])==1e-4
    check(run/'checkpoints/last.ckpt',30000,task)
    bestpath=run/'checkpoints/best.ckpt';best=torch.load(bestpath,map_location='cpu',weights_only=False)
    vals=read_table(WORK/'results/validation_trace.csv');unique={int(v['step']):v for v in vals};assert set(unique)==set(range(200,30001,200))
    bv=max(float(v['student1_macro_dice']) for v in unique.values());assert abs(best['best_metrics']['model1_best_val_dice']-bv)<1e-10 and abs(float(unique[best['global_step']]['student1_macro_dice'])-bv)<1e-10
    # Last surviving lineage uses its own first strict improvement for the maximal score.
    assert best['global_step']==min(n for n,v in unique.items() if float(v['student1_macro_dice'])==bv)
    meta=best['controller_metadata' if task=='E4' else 'metadata'];assert meta['cfg_sha256']==sha(cfgpath) and meta['code_identity']==task_identity(task)
    events=read_table(WORK/'results/optimizer_calls.csv');counts={k:sum(e['optimizer']==k and e['event']=='success' for e in events) for k in ['optimizer1','optimizer2']};assert all(v>=30000 for v in counts.values())
    init=read_json(WORK/'reports/INITIALIZATION_ACTUAL.json');assert init['joint_step']==0 and init['fresh']
    selection_path=WORK/'reports/SELECTED_CHECKPOINT.json'
    if selection_path.exists():
        prior=read_json(selection_path);assert prior['sha256']==sha(bestpath) and prior['selected_step']==best['global_step']
    else:
        atomic_json(selection_path,dict(status='FROZEN_BEFORE_TEST',time=now(),checkpoint=str(bestpath),sha256=sha(bestpath),student='model_1',selected_step=best['global_step'],selected_val_macro_dice=bv,selection='own strict student1 validation foreground macro Dice every200',test_metrics_consulted=False,validation_records=len(vals)))
    atomic_json(WORK/'reports/TRAIN_POSTCHECK.json',dict(status='PASS',time=now(),logical_joint_steps=30000,raw_trace_rows=len(rows),duplicate_trace_records=len(rows)-30000,optimizer_success_calls=counts,replayed_physical_updates=sum(counts.values())-60000,last_sha256=sha(run/'checkpoints/last.ckpt'),best_sha256=sha(bestpath),worker_resume_exact=False))
    print(task,'postcheck PASS; selection frozen before test')
if __name__=='__main__':main()
