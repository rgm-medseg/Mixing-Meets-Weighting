"""Freeze inputs/scientific sources and audit the two independent tasks."""
import ast,difflib,inspect,platform,py_compile,subprocess
from common import *
def verify_frozen(task=None):
    for t in ([task] if task else ['E4','E5']):
        m=read_json(MAIN_WORK/'manifests'/f'{t}_frozen.json')
        for p,h in m['frozen_files'].items():assert sha(MAIN_WORK/p)==h,('code/config changed',t,p)
        for p,h in m['external_inputs'].items():assert sha(p)==h,('input changed',p)
    for p,h in read_json(MAIN_WORK/'manifests/protected.json').items():assert sha(p)==h,('protected historical file changed',p)
    q=read_json(PROJECT/'revision_runs_20261006/state/queue.json')
    assert q['status']=='INTERRUPTED' and all(q['tasks'][f'E{i}']['status']=='PENDING' for i in [2,3,4,5])
def freeze():
    assert socket.gethostname()=='n1' and sys.executable==str(PYTHON)
    e4=read_json(MAIN_WORK/'configs/E4_plain_residual_v2B.yaml');parent=read_json(PARENT/'configs/R1_full_v2B.yaml')
    allowed={'task','trc','run_root','protocol_hash','budget_note','estimated_seconds','phase','controller_source','parent_protocol_hash'}
    assert set(e4)==set(parent) and all(e4[k]==parent[k] for k in parent if k not in allowed)
    assert e4['trc']=='bottleneck' and e4['weight_mode']=='spatial'
    assert sha(e4['pretrained_model'])=='477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3'
    assert sha(MAIN_WORK/'scripts/controller.py')==sha(PARENT/'scripts/controller.py')==e4['controller_source_sha256']
    for p in (MAIN_WORK/'source').rglob('*.py'):assert sha(p)==sha(PARENT/'source'/p.relative_to(MAIN_WORK/'source'))
    assert sha(MAIN_WORK/'scripts/unified_evaluation.py')==sha(PARENT/'scripts/unified_evaluation.py')
    for p in (MAIN_WORK/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True)
    import torch,numpy as np,h5py
    bootstrap();import revision_runtime as r
    from train_E4_entry import build_source
    source=build_source(e4);ast.parse(source);original=(PARENT/'runs/R1_full_v2B/v2_effective_source.py').read_text()
    for start,end in [('            consistency_weight =','            optimizer1.zero_grad()'),('def update_model_ema(', '\n\ndef get_ACDC_2DLargestCC'),('def mix_loss(', '\n\ndef patients_to_slices')]:
        assert source[source.index(start):source.index(end)]==original[original.index(start):original.index(end)]
    def node(s,n):return ast.dump(next(x for x in ast.parse(s).body if isinstance(x,(ast.FunctionDef,ast.ClassDef)) and x.name==n),include_attributes=False)
    for n in ['ctw_maps','compute_v21_3_innovation_x','BottleneckResidual','BCP_net']:
        assert node((MAIN_WORK/'scripts/revision_runtime.py').read_text(),n)==node((PARENT/'scripts/revision_runtime.py').read_text(),n)
    (MAIN_WORK/'reports/E4_expected_effective_source.py').write_text(source)
    (MAIN_WORK/'reports/E4_effective_source.patch').write_text(''.join(difflib.unified_diff(original.splitlines(True),source.splitlines(True),fromfile='R1/effective.py',tofile='E4/effective.py')))
    g=torch.Generator().manual_seed(129);x=torch.randn(3,4,17,19,generator=g,requires_grad=True);y=torch.randn(3,4,17,19,generator=g,requires_grad=True)
    raw,norm,orig,w,error=r.ctw_maps(x,y,False);loss=(w*error).mean();grad=torch.autograd.grad(.05*loss,(x,y),allow_unused=True)
    assert torch.equal(w,orig) and not w.requires_grad and grad[0].abs().sum()>0 and grad[1] is None
    atomic_json(MAIN_WORK/'reports/E4_FUNCTION_BINDING.json',dict(status='PASS',controller_sha256=sha(MAIN_WORK/'scripts/controller.py'),constructor=inspect.getsourcefile(r.BCP_net),plain_class=inspect.getsourcefile(r.BottleneckResidual),ctw_ast_identical=True,loss_EMA_blocks_identical=True,weight_mode='spatial',CTW_lambda=.05,student2_CTWs_stop_gradient=True,teacher_forward_count_unchanged=source.count('ema_model(')==original.count('ema_model(')))
    lists={n:(DATA/(n+'.list')).read_text().splitlines() for n in ['train_slices','val','test']};sets={k:{c.split('_')[0] for c in v} for k,v in lists.items()}
    assert [len(sets[k]) for k in lists]==[70,10,20] and not(sets['train_slices']&sets['val'] or sets['train_slices']&sets['test'] or sets['val']&sets['test'])
    labels=set(read_json(PARENT/'protocol.json')['labeled_patients']);assert len(lists['train_slices'])==1312 and {c.split('_')[0] for c in lists['train_slices'][:136]}==labels
    assert not labels&{c.split('_')[0] for c in lists['train_slices'][136:]}
    inputs=[Path(e4['pretrained_model'])]+[DATA/(k+'.list') for k in lists];assets=[];intensities=[]
    for split,ids in lists.items():
        for i,c in enumerate(ids):
            p=DATA/('data/slices' if split=='train_slices' else 'data')/(c+'.h5');assert p.exists();inputs.append(p)
            assets.append(dict(split=split,id=c,patient=c.split('_')[0],role='labeled' if split=='train_slices' and i<136 else 'unlabeled' if split=='train_slices' else split,source=str(p),current_sha256=sha(p),historical_training_content_hash='NA: original per-H5 historical manifest absent'))
            if split=='train_slices':
                with h5py.File(p,'r') as f:im=f['image'][:]
                intensities.append(dict(id=c,min=float(im.min()),max=float(im.max()),dtype=str(im.dtype),shape='x'.join(map(str,im.shape))))
                assert im.min()>=0 and im.max()<=1
    valmapping=[]
    for case in lists['val']:
        with h5py.File(DATA/'data'/(case+'.h5'),'r') as f:nz=f['image'].shape[0]
        valmapping.extend(dict(case=case,z=z,source=str(DATA/'data'/(case+'.h5')),array_axes='z,y,x',processing='native slice; official PIL *255 and bilinear256; GTnearest') for z in range(nz))
    write_csv(MAIN_WORK/'manifests/E5_original_selection_val_slice_mapping.csv',valmapping)
    write_csv(MAIN_WORK/'manifests/data_assets.csv',assets);write_csv(MAIN_WORK/'reports/training_intensity_audit.csv',intensities)
    c1=PROJECT/'tcsmix_control_revision_20261006/formal_C1_CTW_uniform_20261008';protected=read_json(c1/'manifests/protected.json')
    for p in c1.rglob('*'):
        if p.is_file() and 'archives' not in p.parts and '__pycache__' not in p.parts and ('runs' not in p.parts or p.name in ['best.ckpt','last.ckpt','TRAIN_RECEIPT.json']) and p.suffix in ['.py','.sh','.yaml','.json','.csv','.md','.ckpt','.gz','.sha256']:protected[str(p)]=sha(p)
    atomic_json(MAIN_WORK/'manifests/protected.json',protected)
    ref=read_table(PARENT/'results/R1_results.csv')[0];assert ref['checkpoint_sha256']=='5df2de9b8df16e4d89df9265f104b4b8c7c321245d9eaa0479971672b4d7a1a9' and sha(ref['checkpoint'])==ref['checkpoint_sha256'] and int(ref['selected_step'])==29000
    er=read_json(PARENT/'results/EVALUATION_RECEIPT.json');assert er['protocol']=='ACDC_v22_grid_directed_ASD' and er['test_manifest_sha256']==sha(DATA/'test.list')
    rr=read_table(PARENT/'results/case_class_metrics.csv');assert len(rr)==120 and {z['case'] for z in rr}==set(lists['test'])
    for k in ['dice','iou','hd95','asd']:assert abs(np.mean([float(z[k]) for z in rr])-float(ref[k]))<1e-12
    for row in read_table(PARENT/'results/prediction_index.csv'):assert sha(row['prediction'])==row['prediction_sha256'] and sha(row['GT_source'])==row['GT_source_sha256']
    atomic_json(MAIN_WORK/'reports/R1_REFERENCE_VERIFIED.json',dict(status='PASS',time=now(),reference=ref,evaluation=er,rows=120))
    official=MAIN_WORK/'external/official_original';adapted=MAIN_WORK/'external/AdaMix'
    for p in official.rglob('*.py'):assert sha(p)==sha(adapted/p.relative_to(official))
    assert sha(official/'exp_acdc/train_adamix_ct.py')=='c47fbdba2fa503eb5c8e0767238079db1a1fa3437ded697ddec43d98dac1c773'
    from official_adapter import load_official,make_loaders
    ns=load_official();e5=read_json(MAIN_WORK/'configs/E5_AdaMix_CT_common_protocol.yaml');loaders,samplers,v=make_loaders(ns,e5)
    assert all(len(s)==196 for s in samplers)
    atomic_json(MAIN_WORK/'reports/E5_OFFICIAL_SOURCE.json',dict(status='PASS',time=now(),commit=e5['official_commit'],repository=e5['official_url'],entry='exp_acdc/train_adamix_ct.py',entry_sha256=sha(official/'exp_acdc/train_adamix_ct.py'),method_files_byte_identical=True,pretrain_budget=0,batch_streams=[6,6,6,6],total_images=24,GT_images=12,pseudo_pool='1176 slices;63 strictly unlabeled patients; GT unread',epoch_batches=196,val_slices=len(v.dataset),dependencies_local=True))
    (MAIN_WORK/'reports/environment_freeze.txt').write_text(subprocess.check_output([str(PYTHON),'-m','pip','freeze'],text=True))
    atomic_json(MAIN_WORK/'reports/environment.json',dict(python=platform.python_version(),torch=torch.__version__,numpy=np.__version__,CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version(),executable=sys.executable))
    rows=[dict(item=k,R1=json.dumps(parent[k],ensure_ascii=False),E4=json.dumps(e4[k],ensure_ascii=False),category='预期改变' if k=='trc' else '保持一致' if parent[k]==e4[k] else '其他差异',evidence='config + E4 effective source + function binding') for k in parent]
    for k in ['optimizer','base_lr','lr_schedule','ema','ctw_lambda','evaluation','labeled_patients','image_size','selection','resume_boundary']:
        z=read_json(PARENT/'protocol.json')[k];rows.append(dict(item=k,R1=json.dumps(z,ensure_ascii=False),E4=json.dumps(z,ensure_ascii=False),category='保持一致',evidence='identical scientific source and inherited protocol'))
    rows.append(dict(item='pretrain generating history / original per-training-H5 hash',R1='NA',E4='same asset; current hashes frozen',category='无法核验',evidence='historical provenance absent'))
    write_csv(MAIN_WORK/'E4_PROTOCOL_COMPARISON.csv',rows)
    differences=[
      ('data','70/10/20;7fixed patients','same explicit lists','预期改变','official shuffle overridden by common lists'),
      ('pseudo pool','1176 strict unlabeled','1176 strict unlabeled; officialdefaultall1312incllabeled','预期改变','common label visibility adapter'),
      ('batch','24input;12GT','4streams6each=24;main6+aux6GT','预期改变','official4x16=64 adapted4x6'),
      ('architecture','BCP U-Net+TRC','official UNet default16/32/64/128/256','其他差异','native constructors'),
      ('initialization','pretrained shared net/SGDstate','official random+classifierkaiming;pretrain0','其他差异','official initializer'),
      ('optimizer','SGD .01 momentum.9 wd1e-4','AdamW1e-4 betas.9/.999 wd.01 constant','其他差异','official definition'),
      ('teacher','EMA.99student1','none;2students','其他差异','officialCT'),
      ('loss','originalFFT/CPS/ICT+CTW.05','official(loss_s+loss_u)/2;CE+Dice;conf.95','其他差异','literal official step'),
      ('augmentation','R1 frozen transforms','official flips/rotation/ColorJitter/autocontrast/equalize/blur;A2fromA1','其他差异','official dataset methods'),
      ('input processing','R1 frozen256','official*255 PILL/bilinear;GTnearest;evalcommonnearest','其他差异','all training range[0,1] verified'),
      ('workers','16','2each4trainloaders=8;2val','其他差异','official defaults'),
      ('budget/seed/precision','30000/42/FP32','same','保持一致','locked configs and backend'),
      ('curriculum','v2_B teacher margin','AdaMix total_steps7500;cur_step0..29999','其他差异','relative budget adapted'),
      ('primary selection','student1 strict valDice every200','same;originalval_loss<=every196also saved','预期改变','common selector + original record'),
      ('evaluation','ACDC_v22_grid_directed_ASD','identical;official published eval differs','预期改变','nearest/nativegrid/noLCC/directedASD'),
      ('source','R1 frozen','Senyh/AdaMix85361fb exact CTstep explicitwrapper','其他差异','pinned official source'),
      ('historical generating logs','NA','fromscratch command saved','无法核验','R1 historical absence retained')]
    write_csv(MAIN_WORK/'E5_PROTOCOL_COMPARISON.csv',[dict(item=i,R1=a,E5=b,category=c,evidence=d) for i,a,b,c,d in differences])
    write_csv(MAIN_WORK/'budget_and_differences.csv',[dict(task=t,formal_joint_budget=30000,normal_calls_each_student=30000,pretrain_budget=0,short_joint_max_including_failures=4,seed=42,additional_seeds=0,E3=False,science='TRC to plain residual' if t=='E4' else 'official AdaMix CT common protocol') for t in ['E4','E5']])
    for task in ['E4','E5']:
        names=['common.py','resources.py','preflight.py','unified_evaluation.py','check_checkpoint.py']+(['revision_runtime.py','controller.py','control_runtime.py','legacy_train_builder.py','accepted_short_wiring.py','train_E4_entry.py'] if task=='E4' else ['official_adapter.py','train_E5_entry.py'])
        files=[MAIN_WORK/'scripts'/n for n in names]+list((MAIN_WORK/'configs').glob(task+'_*.yaml'))+list((MAIN_WORK/'configs').glob('*.list'))+[MAIN_WORK/'tasks'/task/'protocol.json']
        files+=list((MAIN_WORK/'source').rglob('*.py')) if task=='E4' else [p for p in (MAIN_WORK/'external').rglob('*') if p.is_file() and '__pycache__' not in p.parts]
        frozen={str(p.relative_to(MAIN_WORK)):sha(p) for p in sorted(files)};identity=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
        atomic_json(MAIN_WORK/'manifests'/f'{task}_frozen.json',dict(task=task,time=now(),frozen_files=frozen,code_identity=identity,external_inputs={str(p):sha(p) for p in inputs},formal_budget=30000,short_max_joint=4,pretrain_budget=0,resume_exact=False))
    atomic_json(MAIN_WORK/'reports/STATIC_PREFLIGHT.json',dict(status='PASS',time=now(),data_rows=len(assets),protected_files=len(protected),formal_budget=60000,E3=False,short_checks='PENDING'))
    verify_frozen();print('Static preflight PASS')
if __name__=='__main__':
    if (MAIN_WORK/'manifests/E4_frozen.json').exists():verify_frozen();print('Frozen identities verified')
    else:freeze()
