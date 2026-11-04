"""Bounded short-run entry. Formal configs are always refused in this phase."""
import argparse
import difflib
import fcntl
import os
import sys
from pathlib import Path
from common import *
from legacy_train_builder import build_beta_source,replace

def build_source(cfg):
    s=build_beta_source(cfg)
    s=replace(s,'from revision_runtime import BCP_net, generate_adamix_bcp_lite_v1_mask, compute_v21_3_innovation_x, mask_token_endpoint',
        'from revision_runtime import BCP_net, compute_v21_3_innovation_x, mask_token_endpoint\nimport control_runtime as control\nfrom control_runtime import generate_adamix_bcp_lite_v1_mask')
    s=replace(s,'                        img_a, pre_a, pre_b, args.adamix_min_side, args.adamix_max_side)',
        '                        img_a, pre_a, pre_b, args.adamix_min_side, args.adamix_max_side, plab_a=plab_a, plab_b=plab_b)')
    s=replace(s,'            volume_batch = sampled_batch["image"].cuda()',
        '            control.start_step(iter_num + 1)\n            volume_batch = sampled_batch["image"].cuda()')
    s=replace(s,'            revision.record_step(run_root, iter_num, optimizer1, loss)',
        '            revision.record_step(run_root, iter_num, optimizer1, loss)\n            control.record_step(run_root, iter_num, optimizer1, loss)')
    for name in ['optimizer1','optimizer2']:
        idx='1' if name=='optimizer1' else '2'
        anchor=f'    {name} = make_optimizer(model_{idx}, args.base_lr, args.mask_token_enabled)'
        s=replace(s,anchor,anchor+f'\n    control.guarded_optimizer({name}, "{name}")')
    s=replace(s,'        "resume_exact": False,','        "resume_exact": False,\n        "controller_metadata": control.checkpoint_metadata(),')
    return s

def main():
    p=argparse.ArgumentParser();p.add_argument('--config',required=True);p.add_argument('--resume',action='store_true');p.add_argument('--stop-after',type=int)
    a=p.parse_args();cfg=read_json(a.config)
    if cfg.get('phase')!='short' or cfg.get('max_iterations',30000)>250:
        raise SystemExit('WAITING_FOR_FULL_APPROVAL: this entry refuses formal training; no tmux or queue handoff')
    if cfg['weight_mode']!='spatial' or cfg['trc']!='token':raise RuntimeError('Short check must use original Full spatial CTW/TRC')
    if cfg['controller_version']!=read_json(WORK/'reports/controller_selection.json')['controller_version']:raise RuntimeError('Selected version mismatch')
    lock=(WORK/'state/short_training.lock').open('w');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    assert read_json(PROJECT/'revision_runs_20261006/state/queue.json')['status']=='INTERRUPTED'
    assert (PROJECT/'revision_runs_20261006/state/stop_requested').exists()
    cfg=dict(cfg,_resume=a.resume,config_sha256=sha(a.config));run=Path(cfg['run_root'])
    if run.exists() and not a.resume:raise RuntimeError('Refusing to overwrite existing short run')
    if a.resume:
        cp=run/'checkpoints/last.ckpt'
        import torch
        ck=torch.load(cp,map_location='cpu',weights_only=False)
        assert ck['controller_metadata']['cfg_sha256']==cfg['config_sha256'] and ck['controller_metadata']['stateless']
        assert ck['global_step']<cfg['max_iterations']
        del ck
    run.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']='0';os.environ['WANDB_MODE']='disabled';bootstrap()
    import revision_runtime as r
    import control_runtime as c
    r.configure(cfg);c.configure(cfg,a.stop_after)
    source=build_source(cfg);(run/'v2_effective_source.py').write_text(source)
    old=build_beta_source(cfg)
    (WORK/'reports/control_loop.patch').write_text(''.join(difflib.unified_diff(old.splitlines(True),source.splitlines(True),fromfile='frozen_Full_with_durable_state.py',tofile='v2_Full_control_only.py')))
    argv=[str(WORK/'source/historical_full.py'),'--root_path',cfg['dataset_root'],'--run_root',str(run),'--pretrained_model',str(Path(cfg['pretrained_model']).parent),
        '--seed','42','--labelnum','7','--gpu','0','--max_iterations',str(cfg['max_iterations']),'--save_interval','200',
        '--mask_token_enabled','true','--adamix_enabled','true','--innovation_x_id','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',
        '--enable_v21_3_innovation_x','true','--v21_3_innovation_x_weight','0.05','--mechanism_log_interval','200']
    if a.resume:argv+=['--resume_ckpt',str(run/'checkpoints/last.ckpt')]
    atomic_json(WORK/f'reports/short_command_{"resume" if a.resume else "fresh"}.json',dict(time=now(),actual_wrapper_command=sys.argv,compiled_parser_argv=argv,cfg_sha256=cfg['config_sha256'],stop_after=a.stop_after))
    sys.argv=argv
    exec(compile(source,str(run/'v2_effective_source.py'),'exec'),{'__name__':'__main__','__file__':str(WORK/'source/historical_full.py'),'V21_3_EFFECTIVE_SOURCE':source})

if __name__=='__main__':main()
