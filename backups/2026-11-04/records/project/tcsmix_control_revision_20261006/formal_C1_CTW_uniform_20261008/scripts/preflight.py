"""Freeze current inputs/code, verify parent protocol and isolated algebra."""
import ast,difflib,inspect,platform,py_compile,subprocess
from common import *
CFG_PATH=MAIN_WORK/'configs/C1_CTW_mean_uniform_v2B.yaml'
def verify_frozen():
    m=read_json(MAIN_WORK/'run_manifest.json')
    for p,h in m['frozen_files'].items():assert sha(MAIN_WORK/p)==h,('C1 code/config changed',p)
    for p,h in m['external_inputs'].items():assert sha(p)==h,('input changed',p)
    for p,h in read_json(MAIN_WORK/'manifests/protected.json').items():assert sha(p)==h,('protected parent changed',p)
    q=read_json(PROJECT/'revision_runs_20261006/state/queue.json')
    assert q['status']=='INTERRUPTED' and all(q['tasks'][f'E{i}']['status']=='PENDING' for i in [2,3,4,5])
def freeze():
    assert socket.gethostname()=='n1' and sys.executable==str(PYTHON)
    cfg=read_json(CFG_PATH);old=read_json(PARENT/'configs/R1_full_v2B.yaml');protocol=read_json(MAIN_WORK/'protocol.json')
    assert cfg['weight_mode']=='uniform' and old['weight_mode']=='spatial'
    allowed={'task','weight_mode','run_root','protocol_hash','budget_note','estimated_seconds','phase','controller_source','parent_protocol_hash'}
    assert set(cfg)==set(old)
    assert all(cfg[k]==old[k] for k in old if k not in allowed)
    assert cfg['protocol_hash']==sha(MAIN_WORK/'protocol.json') and cfg['parent_protocol_hash']==sha(PARENT/'protocol.json')
    assert sha(cfg['pretrained_model'])=='477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3'
    assert sha(MAIN_WORK/'scripts/controller.py')==sha(PARENT/'scripts/controller.py')==cfg['controller_source_sha256']=='f923e7f9bf0ef9bdae94bbd400d54f52fbbc6064efac9bcece365465657f795c'
    assert all(sha(p)==sha(PARENT/'source'/p.relative_to(MAIN_WORK/'source')) for p in (MAIN_WORK/'source').rglob('*.py'))
    assert sha(MAIN_WORK/'scripts/unified_evaluation.py')==sha(PARENT/'scripts/unified_evaluation.py')
    for p in (MAIN_WORK/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True)
    import torch,numpy,scipy,h5py,matplotlib,medpy,torchvision,skimage,tensorboardX
    bootstrap();import revision_runtime as r
    from train_C1_entry import build_source
    source=build_source(cfg);ast.parse(source);parent=(PARENT/'runs/R1_full_v2B/v2_effective_source.py').read_text()
    for start,end in [('            consistency_weight =','            optimizer1.zero_grad()'),('def update_model_ema(', '\n\ndef get_ACDC_2DLargestCC'),('def mix_loss(', '\n\ndef patients_to_slices')]:
        assert source[source.index(start):source.index(end)]==parent[parent.index(start):parent.index(end)]
    def fn(s,name):
        node=next(n for n in ast.parse(s).body if isinstance(n,ast.FunctionDef) and n.name==name)
        return ast.dump(node,include_attributes=False)
    assert fn((MAIN_WORK/'scripts/revision_runtime.py').read_text(),'ctw_maps')==fn((PARENT/'scripts/revision_runtime.py').read_text(),'ctw_maps')
    assert source.count('ema_model(')==parent.count('ema_model(')
    assert 'from revision_runtime import BCP_net, compute_v21_3_innovation_x, mask_token_endpoint' in source
    (MAIN_WORK/'reports/expected_effective_source.py').write_text(source)
    changes=[]
    for p in sorted((MAIN_WORK/'scripts').glob('*.py')):
        op=PARENT/'scripts'/p.name
        if op.exists():changes.append(''.join(difflib.unified_diff(op.read_text().splitlines(True),p.read_text().splitlines(True),fromfile='R1/'+p.name,tofile='C1/'+p.name)))
        else:changes.append(''.join(difflib.unified_diff([],p.read_text().splitlines(True),fromfile='/dev/null',tofile='C1/'+p.name)))
    changes.append(''.join(difflib.unified_diff(parent.splitlines(True),source.splitlines(True),fromfile='R1/effective_source.py',tofile='C1/effective_source.py')))
    (MAIN_WORK/'reports/C1_engineering_and_telemetry.patch').write_text('\n'.join(changes))
    (MAIN_WORK/'reports/C1_config.patch').write_text(''.join(difflib.unified_diff((PARENT/'configs/R1_full_v2B.yaml').read_text().splitlines(True),CFG_PATH.read_text().splitlines(True),fromfile='R1/config',tofile='C1/config')))
    # Independent CPU synthetic logits; does not touch training RNG/model state.
    g=torch.Generator().manual_seed(192);x=torch.randn(3,4,17,19,generator=g,requires_grad=True);y=torch.randn(3,4,17,19,generator=g,requires_grad=True)
    raw,norm,original,w,e=r.ctw_maps(x,y,True);loss=(w*e).mean();grad=torch.autograd.grad(.05*loss,(x,y),allow_unused=True)
    err=(original.mean((1,2))-w.mean((1,2))).abs().max().item();rg=(w.flatten(1).amax(1)-w.flatten(1).amin(1)).max().item()
    assert err<=1e-6 and rg==0 and not w.requires_grad and grad[0].abs().sum()>0 and grad[1] is None
    expected=(original.mean((1,2),keepdim=True).expand_as(original)*(x.softmax(1)-y.softmax(1).detach()).abs().mean(1)).mean()
    assert torch.equal(loss,expected)
    atomic_json(MAIN_WORK/'reports/WEIGHT_GRADIENT.json',dict(status='PASS',time=now(),mean_error=err,max_within_image_range=rg,weight_requires_grad=False,student1_gradient_L1=grad[0].abs().sum().item(),student2_gradient=None,lambda_ctw=.05,formula_exact=True,tolerance=1e-6,scope='independent CPU synthetic logits; no optimizer call'))
    atomic_json(MAIN_WORK/'reports/FUNCTION_BINDING.json',dict(status='PASS',time=now(),ctw_maps_file=inspect.getsourcefile(r.ctw_maps),ctw_maps_line=inspect.getsourcelines(r.ctw_maps)[1],compute_file=inspect.getsourcefile(r.compute_v21_3_innovation_x),compute_line=inspect.getsourcelines(r.compute_v21_3_innovation_x)[1],actual_import='last binding from revision_runtime overrides historical innovation',weight_mode=cfg['weight_mode'],science_change='spatial to per-image full HxW mean uniform only',loss_EMA_blocks_identical=True,teacher_forward_count_unchanged=True))
    lists={name:(DATA/f'{name}.list').read_text().splitlines() for name in ['train_slices','val','test']}
    sets={k:{c.split('_')[0] for c in v} for k,v in lists.items()}
    assert [len(sets[k]) for k in lists]==[70,10,20] and not(sets['train_slices']&sets['val'] or sets['train_slices']&sets['test'] or sets['val']&sets['test'])
    assert {c.split('_')[0] for c in lists['train_slices'][:136]}==set(protocol['labeled_patients'])
    input_paths=[Path(cfg['pretrained_model'])]+[DATA/f'{k}.list' for k in lists]
    dataset=[]
    for split,cs in lists.items():
        for i,c in enumerate(cs):
            p=DATA/('data/slices' if split=='train_slices' else 'data')/(c+'.h5');assert p.exists();input_paths.append(p)
            dataset.append(dict(split=split,id=c,patient=c.split('_')[0],role='labeled' if split=='train_slices' and i<136 else ('unlabeled' if split=='train_slices' else split),source=str(p),current_sha256=sha(p),historical_train_content_hash='NA: R1 retained list hashes; no original per-training-H5 content manifest'))
    write_csv(MAIN_WORK/'manifests/data_assets.csv',dataset)
    protected=[]
    for root in [PARENT,PROJECT/'tcsmix_control_revision_20261006/formal_R2_fixed92_20261007']:
        protected += [p for p in root.rglob('*') if p.is_file() and '__pycache__' not in p.parts and (p.suffix in ['.py','.yaml','.json','.csv','.md','.sh','.ckpt','.gz'] or p.name.endswith('.sha256')) and 'archives' not in p.parts]
    protected += [PROJECT/'revision_runs_20261006/state/queue.json',PROJECT/'revision_runs_20261006/runs/E1/checkpoints/last.ckpt',PROJECT/'revision_runs_20261006/runs/E1/TRAIN_RECEIPT.json',PROJECT/'paper_offline_followup_20261007/WRITING_HANDOFF_CN.md']
    atomic_json(MAIN_WORK/'manifests/protected.json',{str(p):sha(p) for p in protected})
    rows=[]
    for k in old:
        rows.append(dict(item=k,R1=old[k],C1=cfg[k],category='预期改变' if k=='weight_mode' else ('保持一致' if old[k]==cfg[k] else '其他差异'),evidence='parent/copy configs; engineering paths and identity separated'))
    for k in ['optimizer','base_lr','lr_schedule','ema','ctw_lambda','evaluation','labeled_patients','labeled_slice_count','image_size','selection','resume_boundary']:
        v=read_json(PARENT/'protocol.json')[k];assert protocol[k]==v;rows.append(dict(item=k,R1=json.dumps(v,ensure_ascii=False),C1=json.dumps(v,ensure_ascii=False),category='保持一致',evidence='actual effective code and parent protocol'))
    for k in ['original_pretrain_generating_command','historical_training_H5_content_hash']:
        rows.append(dict(item=k,R1='NA',C1='current exact file reused; hashes now frozen',category='无法核验',evidence='parent generating logs/per-H5 historical manifests absent; same current asset not retroactive provenance'))
    write_csv(MAIN_WORK/'C1_PROTOCOL_COMPARISON.csv',rows)
    ref=read_table(PARENT/'results/R1_results.csv')[0];assert ref['checkpoint_sha256']=='5df2de9b8df16e4d89df9265f104b4b8c7c321245d9eaa0479971672b4d7a1a9' and sha(ref['checkpoint'])==ref['checkpoint_sha256'] and int(ref['selected_step'])==29000
    er=read_json(PARENT/'results/EVALUATION_RECEIPT.json');assert er['test_manifest_sha256']==sha(DATA/'test.list') and er['protocol']=='ACDC_v22_grid_directed_ASD'
    rr=read_table(PARENT/'results/case_class_metrics.csv');assert len(rr)==120 and {r['case'] for r in rr}==set(lists['test']) and len({(r['case'],r['class_id']) for r in rr})==120
    import numpy as np
    for k in ['dice','iou','hd95','asd']:assert abs(np.mean([float(r[k]) for r in rr])-float(ref[k]))<1e-12
    for row in read_table(PARENT/'results/prediction_index.csv'):
        assert sha(row['prediction'])==row['prediction_sha256'] and sha(row['GT_source'])==row['GT_source_sha256'] and row['checkpoint_sha256']==ref['checkpoint_sha256']
    atomic_json(MAIN_WORK/'reports/R1_REFERENCE_VERIFIED.json',dict(status='PASS',time=now(),reference=ref,protocol=er,per_case_rows=120,source_lists_verified=True,prediction_GT_hashes_verified=True))
    pip=subprocess.check_output([str(PYTHON),'-m','pip','freeze'],text=True);(MAIN_WORK/'reports/environment_freeze.txt').write_text(pip)
    files=list((MAIN_WORK/'scripts').glob('*.py'))+list((MAIN_WORK/'scripts').glob('*.sh'))+list((MAIN_WORK/'source').rglob('*.py'))+[CFG_PATH,MAIN_WORK/'protocol.json']+list((MAIN_WORK/'configs').glob('*.list'))
    frozen={str(p.relative_to(MAIN_WORK)):sha(p) for p in sorted(files)}
    import hashlib
    codeid=hashlib.sha256(json.dumps(frozen,sort_keys=True).encode()).hexdigest()
    atomic_json(MAIN_WORK/'run_manifest.json',dict(created=now(),task='C1_CTW_mean_uniform_v2B',frozen_files=frozen,code_identity=codeid,external_inputs={str(p):sha(p) for p in input_paths},parent_manifest_sha256=sha(PARENT/'run_manifest.json'),parent_config_sha256=sha(PARENT/'configs/R1_full_v2B.yaml'),parent_protocol_sha256=sha(PARENT/'protocol.json'),environment=dict(torch=torch.__version__,python=platform.python_version(),CUDA=torch.version.cuda,cudnn=torch.backends.cudnn.version()),formal_budget=30000,short_budget_max_joint=2,seed=42,science='CTW spatial -> per-image mean uniform',automatic_retries=0,resume_exact=False))
    verify_frozen()
    atomic_json(MAIN_WORK/'reports/STATIC_PREFLIGHT.json',dict(status='PASS',time=now(),data_rows=len(dataset),code_identity=codeid,scientific_change='CTW spatial allocation only',source_unchanged=True,controller_unchanged=True,short_check='PENDING',initialization_actual='PENDING before formal first optimizer call'))
    (MAIN_WORK/'PREFLIGHT_C1_CN.md').write_text('静态协议、数据/预训练/父运行身份、控制器、CTW函数绑定和合成梯度检查PASS。唯一科学干预为weight_mode spatial→uniform；路径/身份/统计属于工程适配。\n\n真实短程检查与正式初始化尚待执行，见自动更新收据。历史预训练生成命令及原训练H5逐文件哈希无法追溯；当前资产及清单已完整锁定，不据此补造历史。原Full仍NEEDS_MAPPING，90.35%保留原归属。\n')
def main():
    if (MAIN_WORK/'run_manifest.json').exists():verify_frozen();print('Existing frozen identity verified')
    else:freeze();print('Static C1 preflight PASS; no training update')
if __name__=='__main__':main()
