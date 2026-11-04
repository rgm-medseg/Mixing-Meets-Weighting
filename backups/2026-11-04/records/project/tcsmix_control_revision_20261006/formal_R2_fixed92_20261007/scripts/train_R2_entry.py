"""Authorized single Full R2, using the actual accepted short-phase scientific wiring."""
import argparse
import difflib
import fcntl
import sys
from common import *
from accepted_short_wiring import build_source as accepted_build_source
from legacy_train_builder import build_beta_source,replace

def build_source(cfg):
    s=accepted_build_source(cfg)
    s=replace(s,'            control.start_step(iter_num + 1)','            control.start_step(iter_num + 1, sampled_batch)')
    anchor='            transplant_parent_optimizer_state(optimizer2, load_state["opt"], args.mask_token_enabled)'
    s=replace(s,anchor,anchor+'\n        control.initial_loaded([("model_1",model_1),("model_2",model_2),("ema_model",ema_model)], [optimizer1,optimizer2], pre_model)')
    anchor='                model_2.train()'
    s=replace(s,anchor,anchor+'\n                control.validation(iter_num, performance1, performance2, best_performance1)')
    return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');a=p.parse_args()
    cfg=read_json(WORK/'configs/R2_fixed92.yaml')
    assert cfg['formal_authorized'] and cfg['phase']=='formal_R2' and cfg['task']=='R2'
    assert cfg['max_iterations']==30000 and cfg['weight_mode']=='spatial' and cfg['trc']=='token' and cfg['controller_version']=='v2_fixed'
    assert cfg['signal_version']=='v2_B' and cfg['fixed_side']==.359375 and cfg['mix_mode']=='fixed'
    lock=(WORK/'state/training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    from preflight import verify_frozen
    verify_frozen()
    run=Path(cfg['run_root']);cfg=dict(cfg,_resume=a.resume,config_sha256=sha(WORK/'configs/R2_fixed92.yaml'),segment_id=now().replace(':','').replace('.',''))
    if not a.resume and ((run/'train.log').exists() or (WORK/'results/R2_control_trace.csv').exists()):raise RuntimeError('Existing R2 must be resumed, never restarted')
    if a.resume:
        import torch
        ck=torch.load(run/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert ck['global_step']<30000 and ck['controller_metadata']['cfg_sha256']==cfg['config_sha256']
        assert ck['controller_metadata']['version']=='v2_fixed' and ck['controller_metadata']['stateless']
        assert ck['controller_metadata']['area_lock_sha256']==cfg['area_lock_sha256']
        assert {'model_1','model_2','ema_model','optimizer1','optimizer2','random_state','sampler_state','lr_state','amp_state'}<=set(ck)
        del ck
    os.environ.update(CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled');bootstrap()
    import revision_runtime as r
    import control_runtime as control
    r.configure(cfg);control.configure(cfg)
    # Correct resumed active-process time from completed segment receipts, including validation tails.
    segments=read_json(WORK/'state/train_segments.json') if (WORK/'state/train_segments.json').exists() else []
    r.PREVIOUS_SECONDS=sum(x.get('elapsed_seconds',0) for x in segments)
    source=build_source(cfg)
    assert sha(WORK/'scripts/controller.py')==cfg['controller_source_sha256']
    (run/'v2_effective_source.py').write_text(source)
    parent=build_beta_source(cfg)
    (WORK/'reports/R2_control_wiring.patch').write_text(''.join(difflib.unified_diff(parent.splitlines(True),source.splitlines(True),fromfile='retained_Full_durable.py',tofile='R2_fixed92_scientific_loop.py')))
    argv=[str(WORK/'source/historical_full.py'),'--root_path',cfg['dataset_root'],'--run_root',str(run),
        '--pretrained_model',str(Path(cfg['pretrained_model']).parent),'--seed','42','--labelnum','7','--gpu','0','--max_iterations','30000','--save_interval','200',
        '--mask_token_enabled','true','--adamix_enabled','true','--innovation_x_id','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',
        '--enable_v21_3_innovation_x','true','--v21_3_innovation_x_weight','0.05','--mechanism_log_interval','200']
    if a.resume:argv+=['--resume_ckpt',str(run/'checkpoints/last.ckpt')]
    atomic_json(WORK/f'reports/command_{cfg["segment_id"]}.json',dict(time=now(),pid=os.getpid(),resume=a.resume,
        actual_entry=sys.argv,compiled_parser_argv=argv,cfg_sha256=cfg['config_sha256'],initial_joint_step=0 if not a.resume else 'checkpoint global_step',
        initializer=cfg['pretrained_model'],initializer_sha256=sha(cfg['pretrained_model'])))
    sys.argv=argv
    try:exec(compile(source,str(run/'v2_effective_source.py'),'exec'),{'__name__':'__main__','__file__':str(WORK/'source/historical_full.py'),'V21_3_EFFECTIVE_SOURCE':source})
    finally:control.close()

if __name__=='__main__':main()
