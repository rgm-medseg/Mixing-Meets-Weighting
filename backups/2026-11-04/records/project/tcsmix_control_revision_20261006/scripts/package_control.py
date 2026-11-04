"""Freeze a small, fully verified evidence snapshot; no weights or raw medical arrays."""
import difflib
import shutil
import tarfile
import hashlib
from common import *

def main():
    assert read_json(WORK/'state/status.json')['status']=='WAITING_FOR_FULL_APPROVAL'
    assert read_json(WORK/'reports/five_round_review.json')['status']=='PASS'
    assert read_json(WORK/'state/update_budget.json')['optimizer_step_calls']==480
    locked=read_json(WORK/'state/old_inputs_locked.json');assert all(sha(p)==h for p,h in locked.items())
    # Honest source chronology: actual execution files are retained separately.
    atomic_json(WORK/'reports/short_wrapper_postrun_guard.json',dict(time=now(),
        executed_source='reports/run_short_check_executed.py',executed_sha256=sha(WORK/'reports/run_short_check_executed.py'),
        prepared_source='scripts/run_short_check.py',prepared_sha256=sha(WORK/'scripts/run_short_check.py'),
        change='after run, refuse repeat before status/log writes; negative check passed without updates',
        optimizer_calls_after_change=0))
    atomic_json(WORK/'reports/offline_source_metadata_alignment.json',dict(time=now(),
        executed_source='reports/offline_diagnostics_executed.py',executed_sha256=sha(WORK/'reports/offline_diagnostics_executed.py'),
        prepared_source='scripts/offline_diagnostics.py',prepared_sha256=sha(WORK/'scripts/offline_diagnostics.py'),
        changes=['unique-pixel support counts in single response','explicit Full versus E1 inventory scope'],
        teacher_predictions_probabilities_geometry_quality_changed=False,new_model_forwards=0,new_optimizer_calls=0))
    (WORK/'reports/new_controller.patch').write_text(''.join(difflib.unified_diff([], (WORK/'scripts/controller.py').read_text().splitlines(True),fromfile='/dev/null',tofile='scripts/controller.py')))
    added=['controller.py','control_runtime.py','train_control_entry.py','run_short_check.py','lock_R2_from_R1.py']
    (WORK/'reports/new_control_support.patch').write_text(''.join(''.join(difflib.unified_diff([], (WORK/'scripts'/name).read_text().splitlines(True),fromfile='/dev/null',tofile='scripts/'+name)) for name in added))
    (WORK/'reports/SOURCE_CHRONOLOGY_CN.md').write_text('''# 实际执行与后续准备源文件

历史source/23个Python文件原SHA保留，原历史入口未改。新增controller是独立模块，new_controller.patch表示新增文件，不删除旧TRC或CTW。control_loop.patch是相对保留Full科学定义及耐久工程层的真实控制接线差异。

短程实际controller见controller_executed_short.py；之后只修正未运行R2固定分支的B信号日志，signals/geometry和B路径未改，R2_logging_alignment.json给出hash。
短程实际调度wrapper见run_short_check_executed.py；之后添加已运行检查，阻止重复调用在失败前改写status，short_wrapper_postrun_guard.json及final_negative_checks.json给出证据。
离线实际入口见offline_diagnostics_executed.py；之后修正单切片重复像素计数与Full/E1来源说明，保留修正前CSV/inventory。预测、控制均值、几何、质量和方案取舍不变，metadata修正为0次模型前向/优化。
原receipt计时194.075s按兼容规则保留；测量报告220.242562s来自实际两个入口timestamp至子进程结束。旧规则恢复偏移遗漏先前验证尾部，不能只展示较小计时。
所有五轮检查后的配置与脚本版本列入code_identity.json和本包SHA256SUMS。阶段之后没有训练或额外optimizer.step。
''')
    identity=read_json(WORK/'reports/code_identity.json')
    identity.update(time=now(),script_files={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'scripts').glob('*.py'))})
    atomic_json(WORK/'reports/code_identity.json',identity)
    allowed={'.md','.json','.csv','.tsv','.txt','.log','.py','.patch','.png','.svg','.yaml'}
    files=[p for p in WORK.iterdir() if p.is_file() and p.suffix in allowed]
    for directory in ['source','scripts','configs','reports','analysis','logs','state','runs/SHORT_TCSMIX_V2']:
        files.extend(p for p in (WORK/directory).rglob('*') if p.is_file() and p.suffix in allowed and '__pycache__' not in p.parts)
    files=sorted(set(files));assert not any(p.suffix in {'.ckpt','.pth','.h5','.npy','.npz'} for p in files)
    original_hashes={str(p.relative_to(WORK)):sha(p) for p in files}
    snapshot=WORK/'evidence_snapshot_20261006';assert not snapshot.exists();snapshot.mkdir()
    for p in files:
        dst=snapshot/p.relative_to(WORK);dst.parent.mkdir(parents=True,exist_ok=True);shutil.copyfile(p,dst)
        assert sha(dst)==original_hashes[str(p.relative_to(WORK))]
    assert all(sha(WORK/p)==h for p,h in original_hashes.items()),'Source changed during snapshot'
    atomic_json(snapshot/'SNAPSHOT_RECEIPT.json',dict(time=now(),status='WAITING_FOR_FULL_APPROVAL',
        copies_consistent=True,n_source_files=len(files),optimizer_calls=480,joint_steps=240,
        old_inputs_unchanged=len(locked),weights_and_raw_medical_arrays_excluded=True,
        convention='manifest hashes static copied bytes; originals and actual executed pre-correction files retained in archive'))
    manifest={str(p.relative_to(snapshot)):sha(p) for p in sorted(snapshot.rglob('*')) if p.is_file()}
    (snapshot/'SHA256SUMS').write_text(''.join(f'{h}  {p}\n' for p,h in manifest.items()))
    archive=WORK/'TCSMIX_control_evidence_20261006.tar.gz';assert not archive.exists()
    with tarfile.open(archive,'w:gz',compresslevel=6) as tar:
        tar.add(snapshot,arcname='TCSMIX_control_evidence_20261006')
    # Read back actual compressed bytes, verify every file against the embedded manifest.
    verified=0
    with tarfile.open(archive,'r:gz') as tar:
        root='TCSMIX_control_evidence_20261006/'
        contents={m.name[len(root):]:m for m in tar.getmembers() if m.isfile()}
        embedded=tar.extractfile(contents['SHA256SUMS']).read().decode()
        entries={line.split('  ',1)[1]:line.split('  ',1)[0] for line in embedded.splitlines()}
        assert set(contents)==set(entries)|{'SHA256SUMS'}
        for name,h in entries.items():
            f=tar.extractfile(contents[name]);actual=hashlib.sha256(f.read()).hexdigest();assert actual==h;verified+=1
        required=['TCSMIX_CONTROL_DESIGN_CN.md','SCOPE_DIFF_AUDIT.md','SHORT_RUN_REPORT_CN.md','FULL_READINESS_CN.md','EDITOR_HANDOFF_CN.md',
            'control_diagnostics.csv','short_trace.csv','FIVE_ROUND_REVIEW_CN.md','reports/control_loop.patch','reports/new_controller.patch',
            'analysis/control_replay.png','analysis/short_control_trace.png','analysis/controlled_signal_response.png',
            'reports/five_round_review.json','reports/offline_metadata_correction.json','state/status.json']
        assert all(name in contents for name in required)
        assert all(Path(name).suffix in allowed or name=='SHA256SUMS' for name in contents)
        archived_status=json.loads(tar.extractfile(contents['state/status.json']).read());assert archived_status['status']=='WAITING_FOR_FULL_APPROVAL'
    h=sha(archive);(WORK/(archive.name+'.sha256')).write_text(f'{h}  {archive.name}\n')
    atomic_json(WORK/'archive_delivery.json',dict(time=now(),status='VERIFIED',archive=str(archive),sha256=h,
        bytes=archive.stat().st_size,MiB=archive.stat().st_size/2**20,files_verified=verified,
        manifest=str(snapshot/'SHA256SUMS'),formal_training_started=False,optimizer_calls=480,old_queue='INTERRUPTED'))
    print(json.dumps(read_json(WORK/'archive_delivery.json'),ensure_ascii=False,indent=2))

if __name__=='__main__':main()
