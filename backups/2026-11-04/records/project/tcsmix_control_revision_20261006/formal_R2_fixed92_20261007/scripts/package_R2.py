"""Static actual evidence archive, including unsuccessful execution evidence."""
import argparse
import shutil
import tarfile
import hashlib
import csv
from common import *

def final_reviews():
    import numpy as np
    reviews=[];cfg=read_json(WORK/'configs/R2_fixed92.yaml')
    try:
        from preflight import verify_frozen
        verify_frozen()
        assert cfg['controller_version']=='v2_fixed' and cfg['fixed_side']==.359375 and cfg['rectangle_h']==cfg['rectangle_w']==92
        scope='PASS'
    except Exception as exc:scope='FAIL: '+repr(exc)
    reviews.append(dict(round=1,topic='唯一R2定义、范围与旧结果保护',status=scope,controller='v2_fixed',signal='v2_B',fixed_h=92,fixed_w=92,CTW='spatial .05',TRC='token',R1_retrained=False))
    post=read_json(WORK/'reports/TRAIN_POSTCHECK.json') if (WORK/'reports/TRAIN_POSTCHECK.json').exists() else None
    selection=read_json(WORK/'reports/SELECTED_CHECKPOINT.json') if (WORK/'reports/SELECTED_CHECKPOINT.json').exists() else None
    initial=read_json(WORK/'reports/INITIALIZATION_ACTUAL.json') if (WORK/'reports/INITIALIZATION_ACTUAL.json').exists() else None
    evaluation=read_json(WORK/'results/EVALUATION_RECEIPT.json') if (WORK/'results/EVALUATION_RECEIPT.json').exists() else None
    wiring='PARTIAL'
    if post and selection and initial:
        try:
            assert post['status']=='PASS' and post['logical_joint_steps']==30000
            assert initial['joint_step']==0 and initial['fresh']
            assert all(r['exact_pretrained_common_state'] and r['gamma']==0 for r in initial['checks'])
            assert all(r['base_state_exact_match'] and r['router_state_empty'] for r in initial['optimizer_state_checks'])
            assert not evaluation or selection['time']<evaluation['time']
            assert not evaluation or selection['sha256']==evaluation['checkpoint_sha256']
            assert (WORK/'reports/expected_effective_source.py').read_bytes()==(PARENT_R1/'runs/R1_full_v2B/v2_effective_source.py').read_bytes()
            wiring='PASS'
        except Exception as exc:wiring='FAIL: '+repr(exc)
    reviews.append(dict(round=2,topic='接线、初始化、损失和预算及测试时序',status=wiring,postcheck=post,selection=selection,initializer=initial))
    numeric='PARTIAL'
    if evaluation:
        try:
            rows=read_table(WORK/'results/case_class_metrics.csv');assert len(rows)==120
            for k in ['dice','iou','hd95','asd']:assert abs(np.mean([float(r[k]) for r in rows])-evaluation['metrics'][k])<1e-10
            for level,idfield,file in [('class','class_name','R2_class_metrics.csv'),('frame','case','R2_case_metrics.csv'),('patient','patient','R2_patient_metrics.csv')]:
                for r in read_table(WORK/'results'/file):
                    subset=[x for x in rows if x[idfield]==r[idfield]]
                    for k in ['dice','iou','hd95','asd']:assert abs(np.mean([float(x[k]) for x in subset])-float(r[k]))<1e-10
            trace=read_table(WORK/'results/R2_control_trace.csv');assert len({int(r['step']) for r in trace})==30000
            assert all(r['rectangle_h']=='92' and r['rectangle_w']=='92' and float(r['actual_area'])==8464/65536 and float(r['s_actual'])==.359375 for r in trace)
            assert evaluation['protocol']=='ACDC_v22_grid_directed_ASD' and evaluation['postprocessing']=='none'
            differences=read_table(WORK/'analysis/paired_case_class_differences.csv')
            for d in differences:
                for k in ['dice','iou','hd95','asd']:assert abs(float(d[k+'_method'])-float(d[k+'_reference'])-float(d[k+'_delta']))<1e-10
            assert len(differences)==480
            numeric='PASS'
        except Exception as exc:numeric='FAIL: '+repr(exc)
    reviews.append(dict(round=3,topic='固定逐步轨迹、面积、指标汇总和配对单位',status=numeric,unit='grid, not mm',ASD='directed prediction-to-GT',HD95='concatenated bidirectional surface distances'))
    errors=read_json(WORK/'state/stage_errors.json') if (WORK/'state/stage_errors.json').exists() else []
    geometry=WORK/'analysis/error_R1_R2/ERROR_ANALYSIS_RECEIPT.json'
    audit=read_json(geometry) if geometry.exists() else dict(status='PARTIAL',reason='frozen prediction geometry unavailable')
    diffs=read_table(WORK/'analysis/paired_case_class_differences.csv') if (WORK/'analysis/paired_case_class_differences.csv').exists() else []
    reviews.append(dict(round=4,topic='误差解释的mask证据及全部不利结果',status='PASS' if audit.get('status')=='DONE' and audit.get('recomputed_metrics_match_frozen_CSV') else 'PARTIAL',geometry=audit,paired_rows=len(diffs),unfavorable_or_equal_Dice_rows=sum(float(r['dice_delta'])<=0 for r in diffs),all_stage_errors=errors,all_cases_retained=True,no_test_based_retraining=True,causal_claim='descriptive only, no information/variation/curriculum/CTW synergy decomposition'))
    old=read_json(WORK/'state/old_inputs_locked.json');parent=read_json(WORK/'state/parent_R1_inputs_locked.json')
    preserved=all(sha(p)==h for p,h in {**old,**parent}.items())
    queue=read_json(PROJECT/'revision_runs_20261006/state/queue.json')
    assert not any((WORK/'runs'/name).exists() for name in ['R1','R3','E1','E2','E3','E4','E5'])
    reviews.append(dict(round=5,topic='独立后台、恢复边界、归档可追溯与旧队列',status='PASS' if preserved and queue['status']=='INTERRUPTED' else 'FAIL',preserved_input_files=len(old)+len(parent),server_tmux='beta_fft_R2_fixed92_20261007',resume_exact=False,worker_RNG_boundary='16-worker augment/prefetch not bitwise restorable',archive_validation='compressed readback every member against internal SHA256SUMS, then outer SHA256; final receipt written AFTER snapshot'))
    status='FAIL' if any(r['status'].startswith('FAIL') for r in reviews) else 'PASS' if all(r['status']=='PASS' for r in reviews) else 'PARTIAL'
    atomic_json(WORK/'reports/FINAL_FIVE_ROUND_CHECK.json',dict(time=now(),reviews=reviews,training_updates_triggered=0,status=status))
    (WORK/'reports/FINAL_FIVE_ROUND_CHECK_CN.md').write_text('# 最终五轮核查\n\n'+'\n'.join(f"{r['round']}. {r['topic']}：{r['status']}。" for r in reviews)+'\n\n核查实际协议、接线/初始化、预算与数值、病例mask证据、独立归档；不触发训练或改变方法，PARTIAL/失败和不利结果保留。归档校验完成后另写回执，包内是此前一致快照。\n')

def package(start=False):
    if not start:final_reviews()
    label='startup' if start else 'final';stamp=now().replace(':','').replace('.','')
    snapshot=WORK/'archives'/f'{label}_snapshot_{stamp}';snapshot.mkdir(parents=True)
    allowed={'.py','.sh','.md','.json','.yaml','.csv','.tsv','.txt','.patch','.png','.svg','.log'}
    captured=read_json(WORK/'state/status.json');cutoff=int(captured.get('joint_step',0))
    files=[p for p in WORK.iterdir() if p.is_file() and p.suffix in allowed and p.name!='final_archive_receipt.json']
    roots=['scripts','source','configs','reports','handoff_sources','results','analysis']
    for root in roots:files.extend(p for p in (WORK/root).rglob('*') if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts and 'predictions' not in p.parts)
    cfg=read_json(WORK/'configs/R2_fixed92.yaml');run=Path(cfg['run_root'])
    files.extend(p for p in run.rglob('*') if p.is_file() and p.suffix in {'.py','.csv','.json','.log'} and 'checkpoints' not in p.parts)
    # Immutable finished phase logs, or finite startup excerpts; never copy a live log in a race.
    for p in (WORK/'logs').glob('*.log'):
        if p.name.startswith('package') or p.name=='pipeline.log':continue
        if not start:files.append(p)
    for p in sorted(set(files)):
        dst=snapshot/p.relative_to(WORK);dst.parent.mkdir(parents=True,exist_ok=True)
        if start and p.suffix in ['.log','.csv']:
            raw=p.read_bytes()
            if raw and not raw.endswith(b'\n'):raw=raw[:raw.rfind(b'\n')+1]
            if p.suffix=='.csv' and ('results' in p.parts or 'audit' in p.parts):
                import io
                rows=list(csv.DictReader(io.StringIO(raw.decode())))
                field='logical_step' if rows and 'logical_step' in rows[0] else 'step'
                if rows and field in rows[0]:rows=[r for r in rows if int(r[field])<=cutoff]
                fields=next(csv.reader(io.StringIO(raw.decode())),[])
                write_csv(dst,rows,fields)
            else:dst.write_bytes(raw)
        else:shutil.copyfile(p,dst)
    # Atomic JSON read captures one coherent live heartbeat; scientific outputs are phase-stable.
    for name in ['status.json','stage_errors.json','stage_history.json','train_segments.json','old_inputs_locked.json','parent_R1_inputs_locked.json']:
        p=WORK/'state'/name
        if p.exists():atomic_json(snapshot/'state'/name,captured if name=='status.json' else read_json(p))
    checkpoints=[]
    for p in (run/'checkpoints').glob('*.ckpt'):checkpoints.append(dict(path=str(p),sha256=sha(p),large_weight_not_packaged=True))
    atomic_json(snapshot/'CHECKPOINT_INDEX.json',checkpoints)
    atomic_json(snapshot/'SNAPSHOT_RECEIPT.json',dict(time=now(),label=label,cutoff_joint_step=cutoff,startup_trace_and_optimizer_ledger_truncated_to_common_completed_step=start,medical_raw_arrays_and_predictions_excluded=True,
        rule='static copied bytes and captured atomic heartbeat, no large weights/credentials; qualitative PNG/SVG follow existing anonymous ACDC render rules'))
    sums={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    (snapshot/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name,h in sums.items()))
    name='TCSMix_R2_fixed92_start_evidence.tar.gz' if start else 'TCSMix_R2_fixed92_final_evidence.tar.gz'
    archive=WORK/name
    if archive.exists():
        preserved=WORK/'archives'/f'previous_{stamp}_{name}';shutil.move(archive,preserved)
        outer=Path(str(archive)+'.sha256')
        if outer.exists():shutil.move(outer,Path(str(preserved)+'.sha256'))
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(snapshot,arcname='TCSMix_R2_fixed92_'+label)
    with tarfile.open(archive,'r:gz') as tar:
        prefix='TCSMix_R2_fixed92_'+label+'/'
        members={m.name[len(prefix):]:m for m in tar.getmembers() if m.isfile()}
        assert set(members)==set(sums)|{'SHA256SUMS'}
        for p,h in sums.items():assert hashlib.sha256(tar.extractfile(members[p]).read()).hexdigest()==h
    h=sha(archive);Path(str(archive)+'.sha256').write_text(f'{h}  {name}\n')
    atomic_json(WORK/f'{label}_archive_receipt.json',dict(status='VERIFIED',time=now(),archive=str(archive),sha256=h,bytes=archive.stat().st_size,files_verified=len(sums),snapshot=str(snapshot)))
    print(json.dumps(read_json(WORK/f'{label}_archive_receipt.json'),ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--startup',action='store_true');a=p.parse_args();package(a.startup)

if __name__=='__main__':main()
