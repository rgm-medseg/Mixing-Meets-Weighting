"""Exactly two isolated physical joint updates, split 1+1 for recovery check."""
import subprocess,time,shutil
from common import *
def main():
    from preflight import verify_frozen
    verify_frozen();dest=MAIN_WORK/'runs/isolated_short';receipt=MAIN_WORK/'reports/SHORT_CHECK.json'
    if receipt.exists():
        r=read_json(receipt);assert r['status']=='PASS' and r['code_identity']==read_json(MAIN_WORK/'run_manifest.json')['code_identity'];return
    for name in ['configs','reports','logs','state','results','runs']: (dest/name).mkdir(parents=True,exist_ok=True)
    cfg=read_json(MAIN_WORK/'configs/C1_CTW_mean_uniform_v2B.yaml');cfg.update(phase='isolated_short',run_root=str(dest/'runs/short'))
    cp=dest/'configs/C1_CTW_mean_uniform_v2B.yaml'
    if cp.exists():assert read_json(cp)==cfg
    else:atomic_json(cp,cfg)
    env=dict(os.environ,BETA_C1_WORK=str(dest),CUDA_VISIBLE_DEVICES='0',OMP_NUM_THREADS='1',MKL_NUM_THREADS='1',OPENBLAS_NUM_THREADS='1',PYTHONUNBUFFERED='1',WANDB_MODE='disabled')
    start=time.monotonic();hist=read_json(dest/'state/history.json') if (dest/'state/history.json').exists() else []
    import torch
    last=Path(cfg['run_root'])/'checkpoints/last.ckpt';step=torch.load(last,map_location='cpu',weights_only=False)['global_step'] if last.exists() else 0
    events=read_table(dest/'results/optimizer_calls.csv') if (dest/'results/optimizer_calls.csv').exists() else []
    # Never replay a consumed short update after failure; budget counts physical calls.
    counts={n:sum(r['event']=='success' and r['optimizer']==n for r in events) for n in ['optimizer1','optimizer2']}
    assert counts==dict(optimizer1=step,optimizer2=step),'Short partial update/rollback: budget exhausted or review required'
    failedroot=MAIN_WORK/'runs/isolated_short_failed_attempt1'
    prior=read_table(failedroot/'results/optimizer_calls.csv') if failedroot.exists() else []
    priorcounts={n:sum(r['event']=='success' and r['optimizer']==n for r in prior) for n in ['optimizer1','optimizer2']}
    assert priorcounts==dict(optimizer1=1,optimizer2=1)
    for target in range(step+1,2):
        cmd=[str(PYTHON),str(MAIN_WORK/'scripts/train_C1_entry.py'),'--short-target',str(target)]+(['--resume'] if target>1 else [])
        stamp=now().replace(':','');log=dest/'logs'/f'update_{target}_{stamp}.log';t=time.monotonic()
        with log.open('x') as f:code=subprocess.call(cmd,cwd=MAIN_WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
        hist.append(dict(target=target,command=cmd,exit_code=code,elapsed_seconds=time.monotonic()-t,log=str(log)));atomic_json(dest/'state/history.json',hist)
        assert code==75,'Short stop failed; no additional training attempt'
        checkcmd=[str(PYTHON),str(MAIN_WORK/'scripts/check_checkpoint.py'),'--path',str(last),'--step',str(target),'--receipt',str(dest/'reports'/f'CHECKPOINT_{target}.json')]
        subprocess.run(checkcmd,env=env,check=True)
        shutil.copy2(last,dest/'runs'/f'step{target}.ckpt')
    # Remaining budget is one successful step; exercise real resume entry with zero additional updates.
    cmd=[str(PYTHON),str(MAIN_WORK/'scripts/train_C1_entry.py'),'--resume','--short-target','1','--resume-probe']
    t=time.monotonic()
    with (dest/'logs/resume_zero_update_probe.log').open('w') as f:code=subprocess.call(cmd,cwd=MAIN_WORK,env=env,stdout=f,stderr=subprocess.STDOUT)
    assert code==0
    hist.append(dict(stage='zero_update_resume_probe',command=cmd,exit_code=code,elapsed_seconds=time.monotonic()-t));atomic_json(dest/'state/history.json',hist)
    subprocess.run([str(PYTHON),str(MAIN_WORK/'scripts/check_checkpoint.py'),'--path',str(last),'--step','1','--independent','--receipt',str(dest/'reports/INDEPENDENT_RELOAD.json')],env=env,check=True)
    events=read_table(dest/'results/optimizer_calls.csv');counts={n:sum(r['event']=='success' and r['optimizer']==n for r in events) for n in ['optimizer1','optimizer2']};assert counts==dict(optimizer1=1,optimizer2=1)
    c1=torch.load(dest/'runs/step1.ckpt',map_location='cpu',weights_only=False);c2=torch.load(last,map_location='cpu',weights_only=False);pre=torch.load(cfg['pretrained_model'],map_location='cpu',weights_only=False)
    changes={}
    for b in ['model_1','model_2','ema_model']:
        changes[b]=dict(first_update_changed=any(not torch.equal(c1[b][k],v) for k,v in pre['net'].items()),zero_update_resume_preserved=all(torch.equal(c1[b][k],v) for k,v in c2[b].items()))
        assert all(changes[b].values())
    val=read_table(dest/'results/validation_trace.csv');assert len(val)==1 and int(val[0]['step'])==1
    trace=read_table(dest/'results/C1_control_trace.csv');assert [int(r['step']) for r in trace]==[1]
    assert all(float(r['used_per_image_spatial_range_max'])==0 and float(r['max_per_image_mean_error'])<=1e-6 for r in trace)
    write_csv(MAIN_WORK/'results/short_all_optimizer_calls.csv',prior+events)
    result=dict(status='PASS',time=now(),code_identity=read_json(MAIN_WORK/'run_manifest.json')['code_identity'],physical_joint_updates=2,optimizer_calls_total=dict(optimizer1=2,optimizer2=2),successful_check_optimizer_calls=counts,failed_attempt_optimizer_calls=priorcounts,failed_attempt=str(failedroot),failed_attempt_history=read_json(failedroot/'state/history.json'),model_teacher_changes=changes,actual_EMA_algebra_checked=True,validation_entry_full_20_frames=True,stop_step=1,resume_probe_step=1,resume_probe_optimizer_calls=0,independent_reload=True,segments=hist,wall_seconds_this_short_driver=time.monotonic()-start,worker_resume_exact=False,formal_reuses_short_state=False)
    atomic_json(receipt,result)
    (MAIN_WORK/'PREFLIGHT_C1_CN.md').write_text('C1预检PASS。协议、数据/预训练/父运行身份、控制器、真实函数绑定、合成权重/梯度、隔离真实短程检查均通过。\n\n短程共2个物理联合更新：首个工程检查在EMA整数状态断言发生类型错误，已保留日志和1步物理调用；修复检查后仅用剩余1步完成两学生/教师变化、EMA代数、原20帧验证、安全完整保存及独立加载。随后通过真实恢复入口零更新探测完整加载两学生/教师/优化器/RNG/采样器，模型状态逐tensor一致；未再做优化器调用。isolated_control_test另验证控制器stop标志、独占锁和旧PID身份拒绝。正式训练另建模型/优化器/seed42 RNG/采样器，从原预训练第0步开始。16 worker随机流/prefetch不能位级恢复。\n\n唯一科学干预CTW逐图自身整图同均值均匀权重。概率L1、student2 stop-gradient、lambda0.05、TRC与TCSMix v2_B均保持。新增统计不读取GT、不增加前向/更新。旧Full仍NEEDS_MAPPING；历史预训练生成命令/训练H5逐文件历史哈希NA。当前资产已完整冻结。\n\n正式第一次优化器调用前另存INITIALIZATION_ACTUAL.json；当前不宣称已发生。\n')
    print('Short check PASS: 1 failed + 1 successful physical step; zero-update real resume probe; formal uses none of this state')
if __name__=='__main__':main()
