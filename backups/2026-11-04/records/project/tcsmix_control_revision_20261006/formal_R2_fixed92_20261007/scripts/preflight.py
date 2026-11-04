"""R2 source/area/data admission and synthetic forward checks; zero training updates."""
import argparse,ast,platform,py_compile,shutil,subprocess,difflib
from common import *

UNCHANGED=['controller.py','accepted_short_wiring.py','accepted_diagnostic_inputs.py','legacy_train_builder.py','revision_runtime.py','unified_evaluation.py']
def verify_frozen():
    frozen=read_json(WORK/'run_manifest.json')
    for p,h in frozen['frozen_files'].items():
        if sha(WORK/p)!=h:raise RuntimeError(f'Frozen R2 file changed: {p}')
    for p,h in frozen['external_inputs'].items():
        if sha(p)!=h:raise RuntimeError(f'External input changed: {p}')
    for name in ['old_inputs_locked.json','parent_R1_inputs_locked.json']:
        for p,h in read_json(WORK/'state'/name).items():
            if sha(p)!=h:raise RuntimeError(f'Preserved parent/old input changed: {p}')
    queue=read_json(PROJECT/'revision_runs_20261006/state/queue.json')
    assert queue['status']=='INTERRUPTED' and (PROJECT/'revision_runs_20261006/state/stop_requested').exists()
    assert all(queue['tasks'][f'E{i}']['status']=='PENDING' for i in [2,3,4,5])
    assert read_json(PARENT_R1/'state/status.json')['status']=='DONE'
    import torch
    assert torch.__version__==frozen['environment']['torch']

def synthetic(cfg):
    import numpy as np,torch
    from controller import generate_mask
    result=[]
    for number,magnitude in enumerate([.2,2.,8.,None]):
        logits=torch.zeros(6,4,256,256);labels=torch.zeros(6,256,256,dtype=torch.long)
        for c in [1,2,3]:
            if magnitude is not None:
                labels[:,32*c:32*(c+1),60:100]=c
                logits[:,c,32*c:32*(c+1),60:100]=magnitude
        if magnitude is None:logits[:,0]=4.
        seed=74200+number;np.random.seed(seed)
        ey=int(np.random.randint(0,164));ex=int(np.random.randint(0,164));expected_next=int(np.random.randint(0,1000000))
        np.random.seed(seed)
        mask,target,st=generate_mask(torch.zeros(6,1,256,256),logits,logits,labels,labels,cfg)
        assert mask.shape==(6,256,256) and torch.equal(mask,target) and torch.equal(mask[0].expand_as(mask),mask)
        assert st['rectangle_h']==st['rectangle_w']==92 and st['patch_area_ratio']==8464/65536 and st['side_fraction']==.359375
        assert int((mask[0]==0).sum())==8464
        assert (st['rectangle_y'],st['rectangle_x'])==(ey,ex) and int(np.random.randint(0,1000000))==expected_next
        # Parent complementary image/target equations, same single mask.
        mix1=mask+2*(1-mask);mix2=2*mask+(1-mask)
        assert torch.equal(mix1+mix2,torch.full_like(mask,3))
        result.append(dict(logit_magnitude=magnitude,q_B=st['q_margin'],s_dynamic_B=.18+.20*st['q_margin'],s_actual=.359375,
            h=92,w=92,pixels=8464,y=ey,x=ex,fallback=st['fallback'],same_image_target_mask=True,two_original_randint_calls=True))
    assert len({round(x['q_B'],6) for x in result})==4
    atomic_json(WORK/'reports/SYNTHETIC_FIXED_MASK_CHECK.json',dict(status='PASS',time=now(),cases=result,device='CPU',optimizer_calls=0,
        controller_source_exact_R1=True,no_extra_random_calls=True,scientific_loop_same_R1=True))

def main():
    p=argparse.ArgumentParser();p.add_argument('--freeze',action='store_true');a=p.parse_args()
    assert socket.gethostname()=='n1'
    cfg=read_json(WORK/'configs/R2_fixed92.yaml');protocol=read_json(WORK/'protocol.json');lock=read_json(WORK/'R2_AREA_LOCK.json')
    assert cfg['task']=='R2' and cfg['phase']=='formal_R2' and cfg['formal_authorized']
    assert cfg['controller_version']=='v2_fixed' and cfg['signal_version']=='v2_B' and cfg['mix_mode']=='fixed'
    assert cfg['fixed_side']==.359375 and cfg['rectangle_h']==cfg['rectangle_w']==92
    assert cfg['weight_mode']=='spatial' and cfg['trc']=='token' and cfg['max_iterations']==30000
    assert cfg['seed']==42 and cfg['batch_size']==24 and cfg['labeled_bs']==12 and cfg['num_workers']==16
    assert cfg['run_root']==str(WORK/'runs/R2_fixed92') and cfg['protocol_hash']==sha(WORK/'protocol.json')
    assert cfg['area_lock_sha256']==sha(WORK/'R2_AREA_LOCK.json') and lock['source_sha256']==sha(lock['source_trace'])
    assert lock['source_sha256']=='315139c2d54a89205988e0dd92eb28855ac3c42b84198b216d4c49fe2a35cb95'
    assert cfg['pretrained_model']==protocol['pretrain'] and sha(cfg['pretrained_model'])=='477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3'
    for name in UNCHANGED:assert sha(WORK/'scripts'/name)==sha(PARENT_R1/'scripts'/name),name
    assert sha(WORK/'scripts/controller.py')==cfg['controller_source_sha256']
    for p in (WORK/'source').rglob('*.py'):assert sha(p)==sha(PARENT_R1/'source'/p.relative_to(WORK/'source'))
    for p in (WORK/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True)
    from train_R2_entry import build_source
    from legacy_train_builder import build_beta_source
    source=build_source(cfg);base=build_beta_source(cfg);ast.parse(source)
    parent_source=(PARENT_R1/'runs/R1_full_v2B/v2_effective_source.py').read_text()
    assert source==parent_source,'R2 scientific loop differs from actual R1 beyond separate controller configuration'
    assert source.count('ema_model(')==base.count('ema_model(')
    (WORK/'reports/expected_effective_source.py').write_text(source)
    changes=[]
    for path in (WORK/'scripts').glob('*.py'):
        original=PARENT_R1/'scripts'/path.name.replace('R2','R1')
        if original.exists() and original.read_text()!=path.read_text():
            changes.append(''.join(difflib.unified_diff(original.read_text().splitlines(True),path.read_text().splitlines(True),fromfile='R1/scripts/'+original.name,tofile='R2/scripts/'+path.name)))
    (WORK/'reports/R1_to_R2_engineering.patch').write_text(''.join(changes))
    synthetic(cfg)
    assert shutil.disk_usage(WORK).free>4*2**30
    import torch,h5py,numpy,scipy,matplotlib,medpy
    ck=torch.load(cfg['pretrained_model'],map_location='cpu',weights_only=False);assert {'net','opt'}<=set(ck)
    env=subprocess.run([str(PYTHON),'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (WORK/'reports/environment_freeze.txt').write_text(env)
    inputs={str(DATA/f'{x}.list'):sha(DATA/f'{x}.list') for x in ['train_slices','val','test']};inputs[cfg['pretrained_model']]=sha(cfg['pretrained_model'])
    lists={x:(DATA/f'{x}.list').read_text().splitlines() for x in ['train_slices','val','test']}
    patients={x:{v.split('_')[0] for v in vv} for x,vv in lists.items()}
    assert [len(patients[x]) for x in ['train_slices','val','test']]==[70,10,20]
    assert not (patients['train_slices']&patients['val'] or patients['train_slices']&patients['test'] or patients['val']&patients['test'])
    assert {v.split('_')[0] for v in lists['train_slices'][:136]}==set(protocol['labeled_patients'])
    if a.freeze:
        assert not (WORK/'run_manifest.json').exists()
        data=[]
        for split,ids in lists.items():
            for number,v in enumerate(ids):
                path=DATA/('data/slices' if split=='train_slices' else 'data')/f'{v}.h5'
                assert path.exists();h=sha(path);inputs[str(path)]=h
                data.append(dict(split=split,index=number,id=v,patient=v.split('_')[0],labeled=split=='train_slices' and number<136,path=str(path),sha256=h))
        write_csv(WORK/'reports/DATA_FILE_INDEX.csv',data)
        files=list((WORK/'scripts').glob('*.py'))+list((WORK/'scripts').glob('*.sh'))+list((WORK/'source').rglob('*.py'))
        files += [WORK/'protocol.json',WORK/'configs/R2_fixed92.yaml',WORK/'R2_AREA_LOCK.json',WORK/'reports/DATA_FILE_INDEX.csv']
        for path in (WORK/'reports').glob('Parent_R1_*'):inputs[str(path)]=sha(path)
        atomic_json(WORK/'run_manifest.json',dict(created=now(),task='R2',host='n1',gpu=0,formal_authorized=True,R1_retrained=False,
            frozen_files={str(p.relative_to(WORK)):sha(p) for p in sorted(files)},external_inputs=inputs,
            parent_R1_manifest=str(PARENT_R1/'run_manifest.json'),parent_R1_manifest_sha256=sha(PARENT_R1/'run_manifest.json'),parent_R1_protocol_sha256=sha(PARENT_R1/'protocol.json'),
            scientific_changes='ONLY rectangle dimensions fixed92; q_B retained for diagnostic logging; literal R1 effective loop and scientific dependencies unchanged',
            area_lock_sha256=sha(WORK/'R2_AREA_LOCK.json'),training_budget_joint_iterations=30000,optimizer_calls_normal_each_student=30000,precision='FP32 no AMP',automatic_retries=0,
            resume_exact=False,resume_boundary='main RNG and sampler IDs saved; 16-worker augmentation RNG/prefetch not restored',python=str(PYTHON),
            environment=dict(torch=torch.__version__,python=platform.python_version(),pip_freeze_sha256=sha(WORK/'reports/environment_freeze.txt'))))
    assert sha(WORK/'reports/environment_freeze.txt')==read_json(WORK/'run_manifest.json')['environment']['pip_freeze_sha256']
    verify_frozen()
    for mod in ['train_R2_entry','evaluate_R2','analyze_R2','summarize_R2','package_R2','run_R2_pipeline','error_analysis']:__import__(mod)
    atomic_json(WORK/'reports/PREFLIGHT.json',dict(status='PASS',time=now(),host=socket.gethostname(),python=str(PYTHON),torch=torch.__version__,
        controller_sha256=sha(WORK/'scripts/controller.py'),config_sha256=sha(WORK/'configs/R2_fixed92.yaml'),pretrain_sha256=sha(cfg['pretrained_model']),
        new_smoke_optimizer_calls=0,scientific_loop_byte_identical_R1=True,loss_blocks_identical=True,teacher_forward_count_unchanged=True,
        fresh_initializer_step=0,budget_joint=30000,optimizer_calls_each_student=30000,controller_version='v2_fixed',signal_version='v2_B',fixed_h=92,fixed_w=92,
        imports_pass=True,data_file_count=len(read_table(WORK/'reports/DATA_FILE_INDEX.csv')),R1_preserved=True,old_queue_paused=True))
    print('R2 preflight PASS: fixed92, exact R1 loop/loss/EMA/initializer, zero new training smoke updates.',flush=True)

if __name__=='__main__':main()
