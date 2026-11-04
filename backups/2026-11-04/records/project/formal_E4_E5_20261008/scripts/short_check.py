"""Two isolated physical updates (1+1), real resume and independent reload."""
import argparse,subprocess,time,shutil
from common import *
def main():
    p=argparse.ArgumentParser();p.add_argument('--task',choices=['E4','E5'],required=True);a=p.parse_args();task=a.task
    from preflight import verify_frozen
    verify_frozen(task)
    dest=MAIN_WORK/'runs'/f'{task}_isolated_short';receipt=MAIN_WORK/'tasks'/task/'reports/SHORT_CHECK.json'
    if receipt.exists():
        v=read_json(receipt);assert v['status']=='PASS' and v['code_identity']==task_identity(task);print('Short receipt reused');return
    for d in ['configs','reports','logs','state','results','runs']:(dest/d).mkdir(parents=True,exist_ok=True)
    cfgpath=next((MAIN_WORK/'configs').glob(task+'_*.yaml'));cfg=read_json(cfgpath);cfg.update(phase='isolated_short',run_root=str(dest/'runs/short'))
    targetcfg=dest/'configs'/cfgpath.name
    if targetcfg.exists():assert read_json(targetcfg)==cfg
    else:atomic_json(targetcfg,cfg)
    env=dict(os.environ,BETA_TASK_WORK=str(dest),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    last=Path(cfg['run_root'])/'checkpoints/last.ckpt';history=read_json(dest/'state/history.json') if (dest/'state/history.json').exists() else []
    if history and any(x['exit_code']!=75 for x in history if x['stage']=='update'):raise RuntimeError('Prior short failure retained; review physical budget before retry')
    import torch
    step=torch.load(last,map_location='cpu',weights_only=False)['global_step'] if last.exists() else 0
    for target in range(step+1,3):
        events=read_table(dest/'results/optimizer_calls.csv') if (dest/'results/optimizer_calls.csv').exists() else []
        counts={n:sum(r['event']=='success' and r['optimizer']==n for r in events) for n in ['optimizer1','optimizer2']}
        assert counts==dict(optimizer1=target-1,optimizer2=target-1) and max(counts.values(),default=0)<4
        cmd=[str(PYTHON),str(MAIN_WORK/'scripts'/f'train_{task}_entry.py'),'--short-target',str(target)]+(['--resume'] if target>1 else [])
        log=dest/'logs'/f'update_{target}_{time.time_ns()}.log';t=time.monotonic()
        with log.open('x') as f:code=subprocess.call(cmd,cwd=MAIN_WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
        history.append(dict(stage='update',target=target,command=cmd,log=str(log),exit_code=code,elapsed_seconds=time.monotonic()-t));atomic_json(dest/'state/history.json',history)
        assert code==75,'Short failed; no automatic extra updates'
        subprocess.run([str(PYTHON),str(MAIN_WORK/'scripts/check_checkpoint.py'),'--task',task,'--path',str(last),'--step',str(target),'--receipt',str(dest/'reports'/f'CHECKPOINT_{target}.json')],env=env,check=True)
        shutil.copyfile(last,dest/'runs'/f'step{target}.ckpt')
    cmd=[str(PYTHON),str(MAIN_WORK/'scripts'/f'train_{task}_entry.py'),'--resume','--short-target','2','--resume-probe']
    with (dest/'logs/resume_zero_update.log').open('w') as f:subprocess.run(cmd,env=env,stdout=f,stderr=subprocess.STDOUT,check=True)
    subprocess.run([str(PYTHON),str(MAIN_WORK/'scripts/check_checkpoint.py'),'--task',task,'--path',str(last),'--step','2','--independent','--receipt',str(dest/'reports/INDEPENDENT_RELOAD.json')],env=env,check=True)
    c1=torch.load(dest/'runs/step1.ckpt',map_location='cpu',weights_only=False);c2=torch.load(last,map_location='cpu',weights_only=False)
    branches=['model_1','model_2']+(['ema_model'] if task=='E4' else [])
    assert all(any(not torch.equal(v,c2[b][k]) for k,v in c1[b].items()) for b in branches)
    events=read_table(dest/'results/optimizer_calls.csv');counts={n:sum(r['event']=='success' and r['optimizer']==n for r in events) for n in ['optimizer1','optimizer2']};assert counts==dict(optimizer1=2,optimizer2=2)
    vals=read_table(dest/'results/validation_trace.csv');assert len(vals)==1 and int(vals[0]['step'])==1
    trace=read_table(dest/'results'/f'{task}_control_trace.csv');assert [int(r['step']) for r in trace]==[1,2]
    atomic_json(receipt,dict(status='PASS',time=now(),task=task,code_identity=task_identity(task),physical_joint_updates=2,optimizer_success_calls=counts,failed_attempts=0,history=history,short_root=str(dest),validation_all20_frames=True,independent_reload=True,real_resume=True,zero_update_probe=True,worker_resume_exact=False,formal_reuses_short_state=False,mean_step_seconds=sum(float(r['step_seconds']) for r in trace)/2))
    print('Short PASS',task)
if __name__=='__main__':main()
