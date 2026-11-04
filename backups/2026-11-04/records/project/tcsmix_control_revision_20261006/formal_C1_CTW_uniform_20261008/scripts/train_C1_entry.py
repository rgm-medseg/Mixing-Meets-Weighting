"""One fresh C1 run; isolated two-step stop/resume exercise has separate state."""
import argparse,difflib,fcntl,sys,time
from common import *
from accepted_short_wiring import build_source as accepted_build_source
from legacy_train_builder import replace
def build_source(cfg,short_target=None,resume_probe=False):
    s=accepted_build_source(cfg)
    s=replace(s,'            control.start_step(iter_num + 1)','            control.start_step(iter_num + 1, sampled_batch)')
    anchor='            transplant_parent_optimizer_state(optimizer2, load_state["opt"], args.mask_token_enabled)'
    s=replace(s,anchor,anchor+'\n        control.initial_loaded([("model_1",model_1),("model_2",model_2),("ema_model",ema_model)], [optimizer1,optimizer2], pre_model)')
    s=replace(s,'                model_2.train()','                model_2.train()\n                control.validation(iter_num, performance1, performance2, best_performance1)')
    s=replace(s,'            update_model_ema(model_1, ema_model, 0.99)','            update_model_ema(model_1, ema_model, 0.99)\n            control.after_ema(iter_num, model_1, ema_model)')
    s=replace(s,'        "resume_exact": False,','        "resume_exact": False,\n        "scheduler_state": None,\n        "lr_state_optimizer2": [g["lr"] for g in optimizer2.param_groups],')
    if short_target is not None:
        assert cfg['phase']=='isolated_short' and short_target in [1,2]
        # No shortened scientific budget argument/ramp-up; isolated termination only.
        s=replace(s,'            if iter_num > 0 and iter_num % 200 == 0:', '            if iter_num == 1:')
        s=replace(s,'            if revision.STOP:',f'            if iter_num >= {short_target}:\n                revision.STOP = True\n            if revision.STOP:')
        # Verify real EMA algebra using temporary snapshots in short process only.
        anchor='            update_model_ema(model_1, ema_model, 0.99)'
        s=replace(s,anchor,'            old_teacher = {k:v.detach().clone() for k,v in ema_model.state_dict().items()}\n'+anchor+'\n            assert all(torch.equal(v, (.99*old_teacher[k]+(1-.99)*model_1.state_dict()[k]).to(v.dtype)) for k,v in ema_model.state_dict().items())\n            del old_teacher')
    if resume_probe:
        assert cfg['phase']=='isolated_short'
        s=replace(s,'    start_batch_offset = 0', '    assert args.resume_ckpt and iter_num == 1\n    assert revision.SAMPLER.replay == resume_state["sampler_state"]\n    for branch, model in [("model_1",model_1),("model_2",model_2),("ema_model",ema_model)]:\n        assert all(torch.equal(v.cpu(),resume_state[branch][k].cpu()) for k,v in model.state_dict().items())\n    revision.atomic_json(revision.WORK/"reports/RESUME_ZERO_UPDATE.json", dict(status="PASS",global_step=iter_num,optimizer_calls=0,models_exact=True,sampler_replay_loaded=True,main_RNG_loaded=True,worker_exact=False))\n    writer.close()\n    raise SystemExit(0)\n    start_batch_offset = 0')
    return s
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--short-target',type=int,choices=[1,2]);p.add_argument('--resume-probe',action='store_true');a=p.parse_args()
    from preflight import verify_frozen
    verify_frozen()
    cfg=read_json(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml')
    assert cfg['task']=='C1' and cfg['max_iterations']==30000 and cfg['weight_mode']=='uniform' and cfg['trc']=='token' and cfg['controller_version']=='v2_B'
    assert (cfg['phase']=='isolated_short')==(a.short_target is not None)
    lock=(WORK/'state/training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    run=Path(cfg['run_root']);run.mkdir(parents=True,exist_ok=True)
    cfg=dict(cfg,_resume=a.resume,config_sha256=sha(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml'),segment_id=now().replace(':','').replace('.',''))
    if not a.resume and ((run/'train.log').exists() or (WORK/'results/C1_control_trace.csv').exists()):raise RuntimeError('Existing C1 must resume; fresh restart refused')
    if a.resume:
        import torch
        ck=torch.load(run/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert ck['global_step']<30000 and ck['controller_metadata']['cfg_sha256']==cfg['config_sha256']
        assert ck['controller_metadata']['code_identity']==read_json(MAIN_WORK/'run_manifest.json')['code_identity']
        assert {'model_1','model_2','ema_model','optimizer1','optimizer2','random_state','sampler_state','lr_state','amp_state'}<=set(ck)
        del ck
    os.environ.update(CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled');bootstrap()
    from resources import bind_gpu_pid
    bind_gpu_pid()
    import revision_runtime as r,control_runtime as control
    r.configure(cfg);control.configure(cfg)
    segments=read_json(WORK/'state/train_segments.json') if (WORK/'state/train_segments.json').exists() else []
    r.PREVIOUS_SECONDS=sum(x.get('elapsed_seconds',0) for x in segments)
    source=build_source(cfg,a.short_target,a.resume_probe);(run/'v2_effective_source.py').write_text(source)
    argv=[str(SOURCE_WORK/'source/historical_full.py'),'--root_path',cfg['dataset_root'],'--run_root',str(run),'--pretrained_model',str(Path(cfg['pretrained_model']).parent),'--seed','42','--labelnum','7','--gpu','0','--max_iterations','30000','--save_interval','200','--mask_token_enabled','true','--adamix_enabled','true','--innovation_x_id','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT','--enable_v21_3_innovation_x','true','--v21_3_innovation_x_weight','0.05','--mechanism_log_interval','200']
    if a.resume:argv+=['--resume_ckpt',str(run/'checkpoints/last.ckpt')]
    atomic_json(WORK/f'reports/command_{cfg["segment_id"]}.json',dict(time=now(),pid=os.getpid(),resume=a.resume,actual_entry=sys.argv,compiled_parser_argv=argv,cfg_sha256=cfg['config_sha256'],initializer=cfg['pretrained_model'],initializer_sha256=sha(cfg['pretrained_model']),short_target=a.short_target))
    sys.argv=argv
    try:exec(compile(source,str(run/'v2_effective_source.py'),'exec'),{'__name__':'__main__','__file__':str(SOURCE_WORK/'source/historical_full.py'),'V21_3_EFFECTIVE_SOURCE':source})
    finally:control.close()
if __name__=='__main__':main()
