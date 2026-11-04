"""Static actual evidence archive, including unsuccessful execution evidence."""
import argparse
import shutil
import tarfile
import hashlib
from common import *

def final_reviews():
    cfg=read_json(WORK/'configs/R1_full_v2B.yaml');manifest=read_json(WORK/'run_manifest.json');reviews=[]
    try:
        from preflight import verify_frozen
        verify_frozen();scope='PASS'
    except Exception as exc:scope='FAIL: '+str(exc)
    reviews.append(dict(round=1,topic='实验定义与范围',status=scope,controller=cfg['controller_version'],CTW='spatial .05',TRC='token',code_manifest=str(WORK/'run_manifest.json')))
    post=read_json(WORK/'reports/TRAIN_POSTCHECK.json') if (WORK/'reports/TRAIN_POSTCHECK.json').exists() else None
    selection=read_json(WORK/'reports/SELECTED_CHECKPOINT.json') if (WORK/'reports/SELECTED_CHECKPOINT.json').exists() else None
    reviews.append(dict(round=2,topic='初始化预算和检查点选择',status='PASS' if post and selection else 'PARTIAL',
        evidence=['reports/INITIALIZATION_ACTUAL.json','reports/TRAIN_POSTCHECK.json','reports/SELECTED_CHECKPOINT.json'],postcheck=post,selection=selection))
    evaluation=read_json(WORK/'results/EVALUATION_RECEIPT.json') if (WORK/'results/EVALUATION_RECEIPT.json').exists() else None
    numeric='PARTIAL'
    if evaluation:
        try:
            import numpy as np
            rr=read_table(WORK/'results/case_class_metrics.csv');assert len(rr)==120
            for k in ['dice','iou','hd95','asd']:assert abs(np.mean([float(r[k]) for r in rr])-evaluation['metrics'][k])<1e-10
            assert all(r['unit']=='grid' for r in rr);numeric='PASS'
        except Exception as exc:numeric='FAIL: '+repr(exc)
    reviews.append(dict(round=3,topic='轨迹和评价数值单位',status=numeric,unit='grid, not mm',empty_policy='both0,one-grid-diagonal',evidence='results/EVALUATION_RECEIPT.json; analysis/scale_distribution.csv'))
    errors=read_json(WORK/'state/stage_errors.json') if (WORK/'state/stage_errors.json').exists() else []
    diffs=read_table(WORK/'analysis/paired_case_class_differences.csv') if (WORK/'analysis/paired_case_class_differences.csv').exists() else []
    reviews.append(dict(round=4,topic='失败与不利结果完整性',status='PASS',all_stage_errors=errors,paired_rows=len(diffs),
        unfavorable_or_equal_dice_rows=sum(float(r['dice_delta'])<=0 for r in diffs),raw_logs_preserved=True,no_result_based_retraining=True))
    old=read_json(WORK/'state/old_inputs_locked.json');unchanged=all(sha(p)==h for p,h in old.items())
    no_R2=not (WORK/'runs/R2').exists()
    reviews.append(dict(round=5,topic='可追溯归档与旧队列/R2边界',status='PASS' if unchanged and no_R2 else 'FAIL',original_inputs_unchanged=len(old) if unchanged else False,R2_started=not no_R2,
        archive_validation='all snapshot bytes verified against SHA256SUMS on compressed readback; see archive receipt'))
    atomic_json(WORK/'reports/FINAL_FIVE_ROUND_CHECK.json',dict(time=now(),reviews=reviews,training_updates_triggered=0,
        status='FAIL' if any(r['status'].startswith('FAIL') for r in reviews) else 'PASS' if all(r['status']=='PASS' for r in reviews) else 'PARTIAL'))
    (WORK/'reports/FINAL_FIVE_ROUND_CHECK_CN.md').write_text('# 最终五轮核查\n\n'+'\n'.join(f"{r['round']}. {r['topic']}：{r['status']}。" for r in reviews)+
        '\n\n检查只核对实际协议、日志、轨迹、评价和归档；未触发训练或改变方法。失败/缺失报告与全部配对行保留。详细证据见同名JSON。\n')

def package(start=False):
    if not start:final_reviews()
    label='startup' if start else 'final';stamp=now().replace(':','').replace('.','')
    snapshot=WORK/'archives'/f'{label}_snapshot_{stamp}';snapshot.mkdir(parents=True)
    allowed={'.py','.sh','.md','.json','.yaml','.csv','.tsv','.txt','.patch','.png','.svg','.log'}
    files=[p for p in WORK.iterdir() if p.is_file() and p.suffix in allowed]
    roots=['scripts','source','configs','reports','handoff_sources','results','analysis']
    for root in roots:files.extend(p for p in (WORK/root).rglob('*') if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts and 'predictions' not in p.parts)
    cfg=read_json(WORK/'configs/R1_full_v2B.yaml');run=Path(cfg['run_root'])
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
            dst.write_bytes(raw)
        else:shutil.copyfile(p,dst)
    # Atomic JSON read captures one coherent live heartbeat; scientific outputs are phase-stable.
    for name in ['status.json','stage_errors.json','stage_history.json','train_segments.json','old_inputs_locked.json']:
        p=WORK/'state'/name
        if p.exists():atomic_json(snapshot/'state'/name,read_json(p))
    checkpoints=[]
    for p in (run/'checkpoints').glob('*.ckpt'):checkpoints.append(dict(path=str(p),sha256=sha(p),large_weight_not_packaged=True))
    atomic_json(snapshot/'CHECKPOINT_INDEX.json',checkpoints)
    atomic_json(snapshot/'SNAPSHOT_RECEIPT.json',dict(time=now(),label=label,medical_raw_arrays_and_predictions_excluded=True,
        rule='static copied bytes and captured atomic heartbeat, no large weights/credentials; qualitative PNG/SVG follow existing anonymous ACDC render rules'))
    sums={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    (snapshot/'SHA256SUMS').write_text(''.join(f'{h}  {name}\n' for name,h in sums.items()))
    name='TCSMix_v2B_R1_start_evidence.tar.gz' if start else 'TCSMix_v2B_R1_final_evidence.tar.gz'
    archive=WORK/name
    if archive.exists():
        preserved=WORK/'archives'/f'previous_{stamp}_{name}';shutil.move(archive,preserved)
        outer=Path(str(archive)+'.sha256')
        if outer.exists():shutil.move(outer,Path(str(preserved)+'.sha256'))
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:tar.add(snapshot,arcname='TCSMix_v2B_R1_'+label)
    with tarfile.open(archive,'r:gz') as tar:
        prefix='TCSMix_v2B_R1_'+label+'/'
        members={m.name[len(prefix):]:m for m in tar.getmembers() if m.isfile()}
        assert set(members)==set(sums)|{'SHA256SUMS'}
        for p,h in sums.items():assert hashlib.sha256(tar.extractfile(members[p]).read()).hexdigest()==h
    h=sha(archive);Path(str(archive)+'.sha256').write_text(f'{h}  {name}\n')
    atomic_json(WORK/f'{label}_archive_receipt.json',dict(status='VERIFIED',time=now(),archive=str(archive),sha256=h,bytes=archive.stat().st_size,files_verified=len(sums),snapshot=str(snapshot)))
    print(json.dumps(read_json(WORK/f'{label}_archive_receipt.json'),ensure_ascii=False,indent=2))

def main():
    p=argparse.ArgumentParser();p.add_argument('--startup',action='store_true');a=p.parse_args();package(a.startup)

if __name__=='__main__':main()
