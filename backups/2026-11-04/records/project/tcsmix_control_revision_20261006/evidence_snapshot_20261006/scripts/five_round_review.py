"""Five independent evidence gates. Read-only model/trace review, no optimizer step."""
import ast
import collections
import hashlib
import math
import os
import py_compile
import subprocess
from common import *
from train_control_entry import build_source
from legacy_train_builder import build_beta_source

def main():
    old=PROJECT/'revision_runs_20261006'
    cfg=read_json(WORK/'configs/short_selected.yaml');rounds=[]
    before_budget=sha(WORK/'state/update_budget.json')
    # 1: Inspect scientific blocks and the actual executed source, not switch labels.
    source=build_source(cfg);base=build_beta_source(cfg)
    executed=WORK/'runs/SHORT_TCSMIX_V2/v2_effective_source.py'
    assert source==executed.read_text()
    blocks=[]
    for start,end in [('            consistency_weight =','            optimizer1.zero_grad()'),
                      ('def update_model_ema(', '\n\ndef get_ACDC_2DLargestCC'),
                      ('def mix_loss(', '\n\ndef patients_to_slices')]:
        new=source[source.index(start):source.index(end)];parent=base[base.index(start):base.index(end)]
        assert new==parent;blocks.append(dict(start=start,sha256=hashlib.sha256(new.encode()).hexdigest()))
    assert source.count('ema_model(')==base.count('ema_model(')
    frozen=list((WORK/'source').rglob('*.py'))
    assert len(frozen)==23
    assert all(sha(p)==sha(old/'source'/p.relative_to(WORK/'source')) for p in frozen)
    assert sha(WORK/'scripts/evaluate.py')==sha(old/'scripts/evaluate.py')
    cp=read_json(WORK/'reports/short_checkpoint_checks.json')
    assert all(c['parameters_student1']==1827469 and c['router_parameters']==641 and c['strict_models_and_optimizers_load'] for c in cp['checkpoints'])
    assert cfg['weight_mode']=='spatial' and cfg['trc']=='token'
    pre=read_json(WORK/'reports/control_preflight.json')
    assert pre['original_spatial_CTW_loss_and_update_direction_equal'] and pre['optimizer_step_calls']==0
    rounds.append(dict(round=1,topic='范围与实际计算',status='PASS',frozen_source_files=23,
        identical_scientific_blocks=blocks,teacher_forward_count=source.count('ema_model('),new_parameters=0,
        evidence=['reports/control_loop.patch','reports/control_preflight.json','reports/short_checkpoint_checks.json'],
        boundary='historical imported versions/pretrain-generating budget not all retained; explicitly recorded'))
    # 2: Reconstruct each signal from stored class means and integer geometry independently.
    diagnostics=read_table(WORK/'control_diagnostics.csv');trace=read_table(WORK/'short_trace.csv')
    for row in diagnostics:
        active=[c for c in [1,2,3] if int(row[f'n_{c}'])>0]
        aa=sum(float(row[f'q_class{c}']) for c in active)/len(active) if active else 0
        bb=sum(float(row[f'margin_class{c}']) for c in active)/len(active) if active else 0
        assert abs(aa-float(row['q_A']))<1e-7 and abs(bb-float(row['q_B']))<1e-7
        assert (row['fallback']=='True')==(not active)
        assert len(active)==int(row['active_classes'])
        assert int(row['pixels'])==int(row['n_samples'])*65536
        foreground=sum(int(row[f'n_{c}']) for c in [1,2,3]);assert foreground==int(row['foreground_pixels'])
        background=int(row['pixels'])-foreground
        assert abs(background/int(row['pixels'])-float(row['predicted_background_fraction']))<1e-7
        old_reconstruction=(background*float(row['background_confidence'])+
            sum(int(row[f'n_{c}'])*float(row[f'q_class{c}']) for c in active))/int(row['pixels'])
        assert abs(old_reconstruction-float(row['old_q']))<3e-7
        for name,q in [('old',float(row['old_q'])),('A',aa),('B',bb)]:
            ss=.18+.20*q;d=max(1,min(255,round(256*ss)))
            assert abs(ss-float(row[name+'_s']))<1e-10
            assert int(row[name+'_h'])==int(row[name+'_w'])==d
            assert float(row[name+'_area'])==d*d/65536
    assert [int(r['step']) for r in trace]==list(range(1,241))
    for row in trace:
        active=[c for c in [1,2,3] if int(row[f'n_{c}'])>0]
        q=sum(float(row[f'margin_class{c}']) for c in active)/len(active) if active else 0
        d=round(256*(.18+.20*q))
        assert row['controller_version']=='v2_B' and abs(q-float(row['q']))<1e-7
        assert d==int(row['rectangle_h'])==int(row['rectangle_w'])
        assert float(row['actual_area'])==d*d/65536
        assert row['shared_mask_and_target_identical']=='True'
        assert abs(float(row['weighted_ctw_loss'])-.05*float(row['ctw_loss']))<1e-9
        assert float(row['lr'])==.01
        assert all(math.isfinite(float(row[k])) for k in ['loss','q','s','ctw_loss','weighted_ctw_loss'])
    checks=read_json(WORK/'reports/controller_checks.json')
    assert checks['status']=='PASS' and checks['optimizer_updates']==0
    assert set(checks['nonfinite_rejected'])=={'nan','inf','-inf'}
    # Only prepared future fixed-geometry branch differs from the executed controller.
    old_controller=ast.parse((WORK/'reports/controller_executed_short.py').read_text())
    new_controller=ast.parse((WORK/'scripts/controller.py').read_text())
    for name in ['signals','geometry']:
        fn=lambda tree:next(n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name==name)
        assert ast.dump(fn(old_controller))==ast.dump(fn(new_controller))
    diff=read_json(WORK/'reports/R2_logging_alignment.json')
    assert diff['executed_controller_sha256']==sha(WORK/'reports/controller_executed_short.py')
    assert diff['prepared_controller_sha256']==sha(WORK/'scripts/controller.py')
    rounds.append(dict(round=2,topic='公式、实现与数值',status='PASS',offline_rows_verified=len(diagnostics),
        short_steps_verified=len(trace),checks=checks['checks'],
        old_q_reconstructed_from_background_and_foreground=True,
        correction='single-response unique pixel counts repaired; no q/geometry/quality changes',
        evidence=['reports/controller_checks.json','reports/offline_metadata_correction.json','control_diagnostics.csv','short_trace.csv']))
    # 3: Split membership, GT exclusion, source attribution, and immutable old inputs.
    samples=read_table(WORK/'analysis/diagnostic_samples.csv')
    train=(DATA/'train_slices.list').read_text().splitlines();val=(DATA/'val.list').read_text().splitlines();test=(DATA/'test.list').read_text().splitlines()
    test_patients={v.split('_')[0] for v in test};patients=collections.defaultdict(set)
    for sample in samples:
        sid=sample['sample'];cohort=sample['cohort'];patient=sample['patient']
        patients[cohort].add(patient);assert patient not in test_patients
        assert sha(sample['source'])==sample['sha256']
        if cohort.startswith('unlabeled'):
            assert sid in train[136:] and sample['GT_used']=='False'
        elif cohort=='labeled_train':assert sid in train[:136] and sample['GT_used']=='True'
        else:assert cohort=='validation' and sid.split(':')[0] in val and sample['GT_used']=='True'
    assert len(samples)==70 and len(patients['labeled_train'])==7 and len(patients['validation'])==10
    locked=read_json(WORK/'state/old_inputs_locked.json');assert all(sha(p)==h for p,h in locked.items())
    identities=read_json(WORK/'reports/offline_checkpoint_inventory.json')
    assert sorted(v['step'] for v in identities)==[0,3200,3976,29600,30000]
    for item in identities:
        assert sha(item['checkpoint'])==item['sha256'] and item['state_unchanged']
        assert (item['step']==0)==(item['run']=='initial_pretrain_reconstruction')
    for row in diagnostics:
        assert row['samples'] and int(row['n_patients'])==len(set(row['patients'].split(';')))
        if row['cohort'].startswith('unlabeled'):assert not row['macro_dice'] and row['GT_present_classes']=='NA'
    handoff=(WORK/'EDITOR_HANDOFF_CN.md').read_text();assert '90.35%' in handoff and 'NA' in handoff and '仍只属于old' in handoff
    rounds.append(dict(round=3,topic='数据使用与版本归属',status='PASS',unique_input_samples=70,
        independent_patients={k:len(v) for k,v in patients.items()},locked_old_inputs_unchanged=len(locked),
        checkpoint_steps=[0,3200,3976,29600,30000],test_model_inputs=0,
        evidence=['analysis/diagnostic_samples.csv','reports/offline_checkpoint_inventory.json','reports/data_manifest_hashes.csv','state/old_inputs_locked.json'],
        limits='E1 early is uniform CTW; initial teacher reconstructed; Full middle absent; no new formal metrics'))
    # 4: Completeness, executed recovery, actual budget, timing and interpretation.
    required=['TCSMIX_CONTROL_DESIGN_CN.md','SCOPE_DIFF_AUDIT.md','SHORT_RUN_REPORT_CN.md','FULL_READINESS_CN.md','EDITOR_HANDOFF_CN.md',
        'control_diagnostics.csv','short_trace.csv','analysis/offline_scale_distribution.csv','analysis/controlled_probability_checks.csv',
        'analysis/control_quality_correlations.csv','analysis/controller_cost.csv','analysis/diagnostic_GT_summary.csv']
    for f in required:assert (WORK/f).is_file() and (WORK/f).stat().st_size>0
    for figure in ['control_replay','controlled_signal_response','short_control_trace']:
        for ext in ['png','svg']:assert (WORK/f'analysis/{figure}.{ext}').stat().st_size>1000
    for p in WORK.glob('*.md'):
        content=p.read_text();assert not any(ord(c)<32 and c not in '\n\r\t' for c in content),p
    segments=read_json(WORK/'reports/short_segments.json');assert [s['exit_code'] for s in segments]==[75,0]
    assert [s['segment'] for s in segments]==['fresh','resume']
    assert 'strict_resume_loaded' in (WORK/'logs/short_resume.log').read_text()
    assert [c['step'] for c in cp['checkpoints']]==[200,240]
    for c in cp['checkpoints']:assert sha(c['path'])==c['sha256']
    budget=read_json(WORK/'state/update_budget.json');assert budget['optimizer_step_calls']==480<=500 and budget['joint_SSL_updates']==240
    stat=read_json(WORK/'reports/short_statistics.json')
    counts=collections.Counter(int(r['rectangle_h']) for r in trace);assert {str(k):v for k,v in counts.items()}==stat['sizes']
    assert stat['actual_size_change_count']==176 and stat['fallback_steps']==0
    measurements=read_json(WORK/'reports/measurement_summary.json')
    assert abs(sum(x['seconds'] for x in measurements['segments'])-measurements['actual_measured_segment_seconds'])<1e-6
    assert measurements['actual_measured_segment_seconds']>measurements['raw_compatibility_receipt_seconds']
    rounds.append(dict(round=4,topic='证据完整与解释边界',status='PASS',joint_steps=240,optimizer_calls=480,
        actual_exit_codes=[75,0],saved_restored_steps=[200,240],required_files=required,
        figures_visually_inspected=['control_replay.png','controlled_signal_response.png','short_control_trace.png'],
        identified_fixed_reporting_issues=['escaped math control characters','duplicated single-sample pixel counts','ambiguous Full inventory scope'],
        disclosed_measurement_limit='raw compatibility receipt undercounts prior validation tail; actual segment timestamps 220.242562s used',
        no_automatic_efficacy_claim=True,no_extra_training_after_repairs=True))
    # 5: Negative admission checks, actual live state, and unchanged paused queue.
    status_before=sha(WORK/'state/status.json')
    refusals=[]
    commands=[('R1_formal',[str(PYTHON),str(WORK/'scripts/train_control_entry.py'),'--config',str(WORK/'configs/R1_planned.yaml')]),
              ('R2_formal',[str(PYTHON),str(WORK/'scripts/train_control_entry.py'),'--config',str(WORK/'configs/R2_planned.yaml')]),
              ('repeat_short',[str(PYTHON),str(WORK/'scripts/run_short_check.py')]),
              ('lock_R2_from_short',[str(PYTHON),str(WORK/'scripts/lock_R2_from_R1.py'),'--run',str(WORK/'runs/SHORT_TCSMIX_V2')])]
    for name,cmd in commands:
        res=subprocess.run(cmd,capture_output=True,text=True,cwd=WORK)
        assert res.returncode!=0
        refusals.append(dict(task=name,command=cmd,exit_code=res.returncode,stderr=res.stderr.strip(),stdout=res.stdout.strip(),optimizer_updates=0))
    assert status_before==sha(WORK/'state/status.json') and before_budget==sha(WORK/'state/update_budget.json')
    assert all(not (WORK/f'runs/{name}').exists() for name in ['R1','R2'])
    assert not (WORK/'configs/R2_fixed_locked.yaml').exists()
    queue=read_json(old/'state/queue.json');assert queue['status']=='INTERRUPTED' and (old/'state/stop_requested').exists()
    assert queue['tasks']['E1']['status']=='INTERRUPTED'
    assert all(queue['tasks'][f'E{i}']['status']=='PENDING' and queue['tasks'][f'E{i}']['latest_step']==0 for i in [2,3,4,5])
    pids=[queue['controller_pid'],queue['tasks']['E1']['pid']]+[s['pid'] for s in [read_json(WORK/'runs/SHORT_TCSMIX_V2/TRAIN_RECEIPT.json'),read_json(WORK/'reports/short_first_segment_TRAIN_RECEIPT.json')]]
    for pid in pids:assert not Path(f'/proc/{pid}').exists(),f'Unexpected live historical/short training PID {pid}'
    live=[]
    for p in Path('/proc').iterdir():
        if not p.name.isdigit():continue
        try:cmdline=(p/'cmdline').read_bytes().replace(b'\0',b' ').decode(errors='replace')
        except (OSError,PermissionError):continue
        if any(str(WORK/f'scripts/{name}') in cmdline for name in ['train_control_entry.py','run_short_check.py']):live.append(dict(pid=int(p.name),cmdline=cmdline))
    assert not live
    session=subprocess.run(['tmux','list-sessions','-F','#{session_name}:#{session_windows}'],capture_output=True,text=True,check=True).stdout.strip()
    assert session=='beta_fft_revision_20261006:3'
    gpu=subprocess.run(['nvidia-smi','--query-gpu=index,name,memory.total,memory.used,utilization.gpu','--format=csv,noheader'],capture_output=True,text=True,check=True).stdout.strip()
    atomic_json(WORK/'reports/final_negative_checks.json',refusals)
    rounds.append(dict(round=5,topic='正式训练前停止',status='PASS',host=socket.gethostname(),gpu=gpu,
        old_queue_status=queue['status'],old_E1_final_checkpoint_step=3976,
        original_queue_sampled_step=queue['tasks']['E1']['latest_step'],original_queue_sample_is_stale=True,
        pending=['E2','E3','E4','E5'],tmux_sessions=session,no_new_tmux=True,
        live_short_training_processes=live,formal_runs_absent=True,R2_scale_still_unlocked=True,
        negative_checks=refusals,unchanged_counter=480))
    for p in (WORK/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True)
    atomic_json(WORK/'reports/five_round_review.json',dict(status='PASS',time=now(),rounds=rounds,extra_optimizer_calls=0))
    (WORK/'FIVE_ROUND_REVIEW_CN.md').write_text('# 提交前五轮检查\n\n'+
        '\n'.join(f"{x['round']}. **{x['topic']}：PASS。** 详见reports/five_round_review.json对应轮次的逐项证据。" for x in rounds)+
        '\n\n范围检查针对实际执行源码、loss/EMA/混合块、参数及原文件hash；公式检查逐行重算770条离线记录与240个真实步骤；数据检查实际病例归属和GT使用；交付检查日志/计时/恢复及图；边界检查真实运行四项拒绝命令、进程/tmux/原队列。不是仅比较开关或复述结论。\n\n修正：报告公式转为明确文本公式避免转义控制字符；单切片支持计数从重复双组改为唯一像素，原表与执行脚本保留，概率/矩形/质量不变；旧Full空间CTW与E1均匀CTW来源明确分开。原兼容训练receipt计时低估已披露并改用实际阶段timestamp。没有因这些修正追加训练，实际总480次optimizer调用。原worker恢复非位级与正式效果未知继续保留。\n\n正式R1/R2、重复short和用short面积锁R2均在写状态/启动训练前拒绝。用户明确要求本阶段停在正式训练前，状态WAITING_FOR_FULL_APPROVAL。\n')
    atomic_json(WORK/'state/status.json',dict(status='WAITING_FOR_FULL_APPROVAL',time=now(),
        phase='mechanism_offline_short_evidence_completed',controller_version='v2_B',optimizer_step_calls=480,joint_SSL_updates=240,
        optimizer_call_limit=500,formal_training_authorized=False,formal_training_started=False,
        old_queue='INTERRUPTED',old_E1_checkpoint_step=3976,old_E2_E5='PENDING',
        new_tmux_created=False,review='five rounds PASS',R1='configuration ready; formal entry/queue only after explicit user confirmation',
        R2='planned, fixed geometry deferred until complete R1 actual area trajectory',
        next='deliver static evidence package; no training handoff'))
    print('Five rounds PASS; WAITING_FOR_FULL_APPROVAL, optimizer calls unchanged at480.')

if __name__=='__main__':main()
