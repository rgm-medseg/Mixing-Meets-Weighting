import argparse
from common import *
def check(path,expected,task,independent=False):
    import torch
    c=torch.load(path,map_location='cpu',weights_only=False);cfgpath,cfg=config_for(task)
    needed={'model_1','model_2','optimizer1','optimizer2','random_state','sampler_state','amp_state','lr_state','scheduler_state','best_metrics','epoch','global_step','args'}
    needed|={'ema_model','controller_metadata','lr_state_optimizer2'} if task=='E4' else {'adamix_age','metadata','teacher_state','pretraining_stage'}
    assert needed<=set(c) and c['global_step']==expected and c['amp_state'] is c['scheduler_state'] is None
    meta=c['controller_metadata' if task=='E4' else 'metadata'];assert meta['cfg_sha256']==sha(cfgpath) and meta['code_identity']==task_identity(task)
    assert set(c['random_state'])=={'python_random','numpy_random','torch_random','torch_cuda_random'}
    for k in ['model_1','model_2']+(['ema_model'] if task=='E4' else []):assert all(not v.is_floating_point() or torch.isfinite(v).all() for v in c[k].values())
    if task=='E4':assert c['lr_state']==c['lr_state_optimizer2']==[.01,.01]
    else:assert c['lr_state']==[[1e-4],[1e-4]] and meta['curriculum_total_steps']==7500 and c['teacher_state'] is None
    if independent:
        if task=='E4':
            from resources import bind_gpu_pid
            bind_gpu_pid();bootstrap();import revision_runtime as r
            from v14_mask_adamix_components import make_optimizer
            r.configure(cfg)
            for branch in ['model_1','model_2','ema_model']:
                m=r.BCP_net(1,4,ema=branch=='ema_model');m.load_state_dict(c[branch],strict=True);assert sum(p.numel() for p in m.parameters())==1827475
                if branch!='ema_model':o=make_optimizer(m,.01,True);o.load_state_dict(c['optimizer1' if branch=='model_1' else 'optimizer2']);del o
                del m
        else:
            from official_adapter import load_official
            n=load_official()
            for branch in ['model_1','model_2']:
                m=n['networks'].UNet(in_channels=1,out_channels=4);m.load_state_dict(c[branch],strict=True);o=torch.optim.AdamW(m.parameters(),lr=1e-4);o.load_state_dict(c['optimizer1' if branch=='model_1' else 'optimizer2']);del m,o
    return dict(status='PASS',time=now(),task=task,path=str(path),sha256=sha(path),step=expected,keys=sorted(c),optimizer_physical_calls=meta['optimizer_success_calls'],independent_model_optimizer_reload=independent,worker_resume_exact=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--path',required=True);p.add_argument('--step',required=True,type=int);p.add_argument('--task',required=True,choices=['E4','E5']);p.add_argument('--independent',action='store_true');p.add_argument('--receipt',required=True);a=p.parse_args()
    atomic_json(a.receipt,check(Path(a.path),a.step,a.task,a.independent));print('checkpoint PASS',a.task,a.step)
if __name__=='__main__':main()
