"""One 97x97 main-signal condition: literal Full loop, only rectangle dimensions replaced."""
import argparse,ast,csv,json,os,sys,time
from pathlib import Path
import numpy as np,torch
from common import *

def run():
    p=argparse.ArgumentParser();p.add_argument('--resume',default='');a=p.parse_args()
    cfg=json.loads((W/'configs/fixed_training_args.json').read_text())
    if a.resume:cfg['resume_ckpt']=str(Path(a.resume).resolve())
    root=Path(cfg['run_root']);root.mkdir(parents=True,exist_ok=True)
    if not a.resume and ((root/'checkpoints').exists() or (W/'receipts/fixed_launch.json').exists()):
        raise RuntimeError('This fixed condition already exists. Refusing a duplicate start or overwrite.')
    torch.set_num_threads(4)
    # Require completed diagnostic technical checks before admitting training.
    for name in ['joint_smoke','trc_smoke']:
        assert json.loads((W/'receipts'/f'{name}.json').read_text())['status']=='PASS'
    original=W/'scripts/main_full_source.py';source=original.read_text()
    sys.argv=[str(original)]
    for k,v in cfg.items():
        sys.argv.append('--'+k)
        sys.argv.extend(map(str,v if isinstance(v,list) else [str(v).lower() if isinstance(v,bool) else v]))
    ns=dict(__name__='frozen_main_training',__file__=str(original),V21_3_EFFECTIVE_SOURCE=source)
    exec(compile(source,str(original),'exec'),ns)
    import v14_mask_adamix_components as parent
    fn=[n for n in ast.parse((W/'scripts/frozen/v14_mask_adamix_components.py').read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='generate_adamix_bcp_lite_v1_mask'][0]
    replaced=[]
    for n in fn.body:
        if isinstance(n,ast.Assign) and isinstance(n.targets[0],ast.Name) and n.targets[0].id in ['patch_x','patch_y']:
            n.value=ast.Constant(value=97);replaced.append(n.targets[0].id)
    assert replaced==['patch_x','patch_y']
    fixed_code=ast.unparse(ast.Module(body=[fn],type_ignores=[]))+'\n'
    (W/'scripts/fixed_rectangle_function.py').write_text('import numpy as np, torch\n'+fixed_code)
    fnns=dict(np=np,torch=torch);exec(compile(fixed_code,'fixed_rectangle_function.py','exec'),fnns)
    fixed=fnns['generate_adamix_bcp_lite_v1_mask'];trace=root/'rectangle_trace.csv';start=time.monotonic();count=0
    if not trace.exists():
        with trace.open('w',newline='') as f:csv.writer(f).writerow(['execution_index','q','dynamic_side_fraction','fixed_h','fixed_w','fixed_area_ratio','elapsed_seconds'])
    def rectangle(*args,**kwargs):
        nonlocal count
        mask,lm,stats=fixed(*args,**kwargs);count+=1
        with trace.open('a',newline='') as f:csv.writer(f).writerow([count,stats['teacher_conf'],stats['side_fraction'],97,97,97*97/65536,time.monotonic()-start])
        stats.update(fixed_h=97,fixed_w=97,region_size_rule='fixed97; original full-image teacher q recorded')
        return mask,lm,stats
    ns['generate_adamix_bcp_lite_v1_mask']=rectangle
    # Record strict reuse of parent pretraining and optimizer inheritance, not a new pretrain.
    pre=Path(cfg['pretrained_model'])/'unet_best_model.pth';pre_ck=torch.load(pre,map_location='cpu',weights_only=False)
    assert set(pre_ck)>={'net','opt'}
    init_checks=[];optimizer_checks=[]
    original_load=ns['load_parent_pretrain'];original_transplant=ns['transplant_parent_optimizer_state']
    def verified_load(model,state,enabled):
        original_load(model,state,enabled)
        loaded=model.state_dict();assert all(torch.equal(loaded[k].cpu(),v.cpu()) for k,v in state.items())
        init_checks.append(dict(base_pretrain_tensors_exact=True,base_tensor_count=len(state),router_initialized_separately=enabled))
        js(W/'receipts/fixed_initialization_checks.json',dict(model_checks=init_checks,optimizer_checks=optimizer_checks))
    def verified_transplant(optimizer,state,enabled):
        original_transplant(optimizer,state,enabled)
        group=optimizer.param_groups[0];parent_group=state['param_groups'][0]
        assert all(group[k]==v for k,v in parent_group.items() if k!='params')
        for param,old_id in zip(group['params'],parent_group['params']):
            for k,v in state['state'].get(old_id,{}).items():
                got=optimizer.state[param][k]
                assert torch.equal(got.cpu(),v.cpu()) if torch.is_tensor(v) else got==v
        optimizer_checks.append(dict(base_hyperparameters_exact=True,base_state_tensors_exact=True,base_parameters=len(group['params']),router_group_preserved=len(optimizer.param_groups)==2))
        js(W/'receipts/fixed_initialization_checks.json',dict(model_checks=init_checks,optimizer_checks=optimizer_checks))
    ns['load_parent_pretrain']=verified_load;ns['transplant_parent_optimizer_state']=verified_transplant
    js(W/'receipts/fixed_launch.json',dict(status='RUNNING',run_name='F97_main_signal_20261013',args=cfg,pretrain=str(pre),pretrain_sha256=sha(pre),parent_source_sha256=sha(original),fixed_function_sha256=sha(W/'scripts/fixed_rectangle_function.py'),changed_scientific_variable='rectangle dimensions only; patch_x=patch_y=97',optimizer_inheritance='literal load_parent_pretrain / transplant_parent_optimizer_state from original Full',budget_per_student=30000,validation_selection='student1 native-volume mean 3-class Dice every200; same original validator',resume=bool(a.resume),start_time_local=time.strftime('%Y-%m-%dT%H:%M:%S%z')))
    del pre_ck
    ns['train']()
    last=torch.load(root/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
    assert last['global_step']==30000
    best=torch.load(root/'checkpoints/best.ckpt',map_location='cpu',weights_only=False)
    js(W/'receipts/fixed_training_complete.json',dict(status='DONE',updates_per_student=last['global_step'],best_step=best['global_step'],best_sha256=sha(root/'checkpoints/best.ckpt'),last_sha256=sha(root/'checkpoints/last.ckpt'),best_validation=best['best_metrics'],elapsed_seconds=time.monotonic()-start,rectangle_calls=count))
if __name__=='__main__':run()
