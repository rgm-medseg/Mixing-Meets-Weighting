"""Definition, input and frozen-code admission. No training or optimizer update."""
import argparse
import ast
import platform
import py_compile
import shutil
import subprocess
from common import *

def verify_frozen():
    frozen=read_json(WORK/'run_manifest.json')
    for p,h in frozen['frozen_files'].items():
        if sha(WORK/p)!=h:raise RuntimeError(f'Frozen R1 file changed: {p}')
    for p,h in frozen['external_inputs'].items():
        if sha(p)!=h:raise RuntimeError(f'External input changed: {p}')
    queue=read_json(PROJECT/'revision_runs_20261006/state/queue.json')
    assert queue['status']=='INTERRUPTED' and (PROJECT/'revision_runs_20261006/state/stop_requested').exists()
    assert all(queue['tasks'][f'E{i}']['status']=='PENDING' for i in [2,3,4,5])
    assert not (WORK/'runs/R2').exists()
    import torch
    assert torch.__version__==frozen['environment']['torch']

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    assert socket.gethostname()=='n1'
    cfg=read_json(WORK/'configs/R1_full_v2B.yaml');accepted=WORK.parent
    assert cfg['controller_version']=='v2_B' and cfg['weight_mode']=='spatial' and cfg['trc']=='token'
    assert cfg['max_iterations']==30000 and cfg['seed']==42 and cfg['batch_size']==24 and cfg['labeled_bs']==12 and cfg['num_workers']==16
    assert cfg['pretrained_model']==read_json(WORK/'protocol.json')['pretrain']
    assert cfg['protocol_hash']==sha(WORK/'protocol.json')
    assert sha(cfg['pretrained_model'])=='477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3'
    assert sha(WORK/'scripts/controller.py')==sha(accepted/'scripts/controller.py')==cfg['controller_source_sha256']
    assert sha(WORK/'scripts/accepted_short_wiring.py')==sha(accepted/'scripts/train_control_entry.py')
    assert all(sha(p)==sha(accepted/'source'/p.relative_to(WORK/'source')) for p in (WORK/'source').rglob('*.py'))
    assert sha(WORK/'scripts/unified_evaluation.py')==sha(accepted/'scripts/evaluate.py')
    for p in (WORK/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True)
    from train_R1_entry import build_source
    from legacy_train_builder import build_beta_source
    src=build_source(cfg);base=build_beta_source(cfg);ast.parse(src)
    for begin,end in [('            consistency_weight =','            optimizer1.zero_grad()'),('def update_model_ema(', '\n\ndef get_ACDC_2DLargestCC'),('def mix_loss(', '\n\ndef patients_to_slices')]:
        assert src[src.index(begin):src.index(end)]==base[base.index(begin):base.index(end)]
    assert src.count('ema_model(')==base.count('ema_model(')
    (WORK/'reports/expected_effective_source.py').write_text(src)
    assert shutil.disk_usage(WORK).free>4*2**30
    import torch,h5py,numpy,scipy,matplotlib,medpy
    ck=torch.load(cfg['pretrained_model'],map_location='cpu',weights_only=False);assert 'net' in ck and 'opt' in ck
    env=subprocess.run([str(PYTHON),'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (WORK/'reports/environment_freeze.txt').write_text(env)
    inputs={str(DATA/f'{x}.list'):sha(DATA/f'{x}.list') for x in ['train_slices','val','test']};inputs[cfg['pretrained_model']]=sha(cfg['pretrained_model'])
    old_archive=accepted/'TCSMIX_control_evidence_20261006.tar.gz'
    assert sha(old_archive)==read_json(accepted/'archive_delivery.json')['sha256']
    inputs[str(old_archive)]=sha(old_archive)
    split_lists={x:(DATA/f'{x}.list').read_text().splitlines() for x in ['train_slices','val','test']}
    patient_sets={x:{v.split('_')[0] for v in vv} for x,vv in split_lists.items()}
    assert [len(patient_sets[x]) for x in ['train_slices','val','test']]==[70,10,20]
    assert not (patient_sets['train_slices']&patient_sets['val'] or patient_sets['train_slices']&patient_sets['test'] or patient_sets['val']&patient_sets['test'])
    assert {v.split('_')[0] for v in split_lists['train_slices'][:136]}==set(read_json(WORK/'protocol.json')['labeled_patients'])
    assert all((DATA/'data/slices'/f'{v}.h5').exists() for v in split_lists['train_slices'])
    assert all((DATA/'data'/f'{v}.h5').exists() for split in ['val','test'] for v in split_lists[split])
    assert all(sha(p)==h for p,h in read_json(WORK/'state/old_inputs_locked.json').items())
    if a.freeze:
        assert not (WORK/'run_manifest.json').exists()
        files=list((WORK/'scripts').glob('*.py'))+list((WORK/'scripts').glob('*.sh'))+list((WORK/'source').rglob('*.py'))+[WORK/'protocol.json',WORK/'configs/R1_full_v2B.yaml']
        atomic_json(WORK/'run_manifest.json',dict(created=now(),task='R1',host='n1',gpu=0,formal_authorized=True,R2_authorized=False,
            frozen_files={str(p.relative_to(WORK)):sha(p) for p in sorted(files)},external_inputs=inputs,
            accepted_phase_archive=str(accepted/'TCSMIX_control_evidence_20261006.tar.gz'),accepted_phase_archive_sha256=sha(accepted/'TCSMIX_control_evidence_20261006.tar.gz'),
            scientific_changes='only accepted TCSMix v2_B teacher confidence path; original spatial CTW/TRC/loss/EMA',
            training_budget_joint_iterations=30000,optimizer_calls_normal_each_student=30000,precision='FP32 no AMP',automatic_retries=0,
            resume_exact=False,resume_boundary='main RNG and sampler IDs saved; 16-worker augmentation RNG/prefetch not restored',python=str(PYTHON),
            environment=dict(torch=torch.__version__,python=platform.python_version(),pip_freeze_sha256=sha(WORK/'reports/environment_freeze.txt'))))
    assert sha(WORK/'reports/environment_freeze.txt')==read_json(WORK/'run_manifest.json')['environment']['pip_freeze_sha256']
    verify_frozen()
    for mod in ['train_R1_entry','evaluate_R1','analyze_R1','summarize_R1','package_R1','run_R1_pipeline']:__import__(mod)
    atomic_json(WORK/'reports/PREFLIGHT.json',dict(status='PASS',time=now(),host=socket.gethostname(),python=str(PYTHON),torch=torch.__version__,
        controller_sha256=sha(WORK/'scripts/controller.py'),config_sha256=sha(WORK/'configs/R1_full_v2B.yaml'),pretrain_sha256=sha(cfg['pretrained_model']),
        reused_short_joint_steps=240,new_smoke_optimizer_calls=0,source_unchanged=True,loss_blocks_identical=True,teacher_forward_count_unchanged=True,
        fresh_initializer_step=0,budget_joint=30000,optimizer_calls_each_student=30000,formal_R2=False,imports_pass=True))
    print('R1 preflight PASS: locked accepted v2_B, spatial CTW/TRC, fresh parent initialization, 30000 joint iterations; zero new smoke updates.')

if __name__=='__main__':main()
