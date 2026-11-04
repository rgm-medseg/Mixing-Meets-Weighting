import argparse,fcntl
from common import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--resume',action='store_true');p.add_argument('--short-target',type=int,choices=[1,2]);p.add_argument('--resume-probe',action='store_true');a=p.parse_args()
    from preflight import verify_frozen
    verify_frozen('E5');cp,cfg=config_for('E5');assert (cfg['phase']=='isolated_short')==(a.short_target is not None)
    lock=(WORK/'state/training.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    cfg=dict(cfg,config_sha256=sha(cp),segment_id=now().replace(':','').replace('.',''))
    if not a.resume and (WORK/'results/E5_control_trace.csv').exists():raise RuntimeError('Existing E5 must resume; fresh restart refused')
    os.environ.update(CUDA_VISIBLE_DEVICES='0',WANDB_MODE='disabled')
    from resources import bind_gpu_pid
    bind_gpu_pid()
    from official_adapter import run_official
    atomic_json(WORK/f'reports/command_{cfg["segment_id"]}.json',dict(time=now(),pid=os.getpid(),actual_entry=sys.argv,resume=a.resume,config_sha256=cfg['config_sha256'],official_commit=cfg['official_commit'],official_entry=str(MAIN_WORK/'external/AdaMix/exp_acdc/train_adamix_ct.py'),short_target=a.short_target))
    run_official(cfg,a.resume,a.short_target,a.resume_probe)
if __name__=='__main__':main()
