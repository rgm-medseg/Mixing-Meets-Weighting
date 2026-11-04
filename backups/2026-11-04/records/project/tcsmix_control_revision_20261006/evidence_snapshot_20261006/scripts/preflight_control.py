"""Scope and restoration prerequisites before the only bounded short run."""
import ast
import hashlib
import py_compile
from common import *

def main():
    bootstrap()
    import torch
    import revision_runtime as r
    import v21_3_acdc10_third_innovation_components as original
    from train_control_entry import build_source
    from legacy_train_builder import build_beta_source
    cfg=read_json(WORK/'configs/short_selected.yaml')
    assert read_json(WORK/'reports/controller_checks.json')['status']=='PASS'
    assert read_json(WORK/'reports/offline_receipt.json')['status']=='DONE'
    for p in (WORK/'scripts').glob('*.py'):py_compile.compile(str(p),doraise=True)
    s=build_source(cfg);base=build_beta_source(cfg);ast.parse(s)
    assert s.count('ema_model(')==base.count('ema_model(')
    for start,end in [('            consistency_weight =','            optimizer1.zero_grad()'),
                      ('def update_model_ema(', '\n\ndef get_ACDC_2DLargestCC'),
                      ('def mix_loss(', '\n\ndef patients_to_slices')]:
        assert s[s.index(start):s.index(end)]==base[base.index(start):base.index(end)]
    assert cfg['weight_mode']=='spatial' and cfg['trc']=='token' and cfg['batch_size']==24 and cfg['labeled_bs']==12 and cfg['num_workers']==16
    assert sha(cfg['pretrained_model'])==read_json(WORK/'protocol.json')['pretrain_sha256']
    torch.manual_seed(420600);o1=torch.randn(2,4,11,13,requires_grad=True);o2=torch.randn_like(o1,requires_grad=True)
    raw,norm,w,used,error=r.ctw_maps(o1,o2,False);assert torch.equal(w,used) and not used.requires_grad
    reference,st=original.compute_v21_3_innovation_x('V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',o1,o2,o1,o2,o1,o2,0,30000)
    actual=(used*error).mean();assert torch.equal(actual,reference)
    (.05*actual).backward();assert o1.grad is not None and o2.grad is None
    # Read-only restoration prerequisite: no parameter/optimizer step here.
    ck=torch.load(cfg['pretrained_model'],map_location='cpu',weights_only=False);assert 'net' in ck and 'opt' in ck
    old=read_json(WORK/'state/old_inputs_locked.json');assert all(sha(p)==h for p,h in old.items())
    atomic_json(WORK/'reports/control_preflight.json',dict(status='PASS',time=now(),optimizer_step_calls=0,
        selected_controller=cfg['controller_version'],teacher_forward_count_unchanged=True,loss_blocks_byte_identical=True,
        original_spatial_CTW_loss_and_update_direction_equal=True,TRC='original4x16token',EMA=.99,ctw_lambda=.05,
        original_inputs_unchanged=True,short_updates=240,strict_optimizer_call_budget=500,
        generated_source_sha256=hashlib.sha256(s.encode()).hexdigest(),
        config_sha256=sha(WORK/'configs/short_selected.yaml')))
    (WORK/'reports/planned_effective_source.py').write_text(s)
    print('Preflight PASS; no optimizer calls; original Full scientific blocks preserved.')

if __name__=='__main__':main()
