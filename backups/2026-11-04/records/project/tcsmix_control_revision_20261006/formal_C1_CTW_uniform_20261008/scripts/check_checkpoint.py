import argparse
from common import *
def check(path,expected,independent=False):
    import torch
    ck=torch.load(path,map_location='cpu',weights_only=False)
    needed={'model_1','model_2','ema_model','optimizer1','optimizer2','random_state','sampler_state','lr_state','lr_state_optimizer2','scheduler_state','amp_state','controller_metadata','best_metrics','epoch','global_step','args'}
    assert needed<=set(ck) and ck['global_step']==expected and ck['amp_state'] is None and ck['scheduler_state'] is None
    cfg=read_json(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml')
    assert ck['controller_metadata']['cfg_sha256']==sha(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml')
    assert ck['controller_metadata']['code_identity']==read_json(MAIN_WORK/'run_manifest.json')['code_identity'] and ck['controller_metadata']['weight_mode']=='uniform'
    assert set(ck['random_state'])=={'python_random','numpy_random','torch_random','torch_cuda_random'}
    assert len(ck['sampler_state']['batches'])>=ck['sampler_state']['consumed']
    assert ck['lr_state']==ck['lr_state_optimizer2']==[.01,.01]
    for branch in ['model_1','model_2','ema_model']:
        assert all(not v.is_floating_point() or torch.isfinite(v).all() for v in ck[branch].values())
    if independent:
        from resources import bind_gpu_pid
        bind_gpu_pid();bootstrap();import revision_runtime as r
        from v14_mask_adamix_components import make_optimizer
        r.configure(cfg)
        for branch in ['model_1','model_2','ema_model']:
            m=r.BCP_net(1,4,ema=branch=='ema_model');m.load_state_dict(ck[branch],strict=True)
            if branch!='ema_model':
                o=make_optimizer(m,.01,True);o.load_state_dict(ck['optimizer1' if branch=='model_1' else 'optimizer2'])
                assert all(not torch.is_tensor(v) or torch.isfinite(v).all() for st in o.state.values() for v in st.values())
                del o
            del m
        r.restore_rng(ck['random_state'])
    return dict(status='PASS',time=now(),path=str(path),sha256=sha(path),step=expected,keys=sorted(ck),optimizer_physical_calls=ck['controller_metadata']['optimizer_success_calls'],independent_model_optimizer_reload=independent,worker_resume_exact=False)
def main():
    p=argparse.ArgumentParser();p.add_argument('--path',required=True);p.add_argument('--step',type=int,required=True);p.add_argument('--independent',action='store_true');p.add_argument('--receipt',required=True);a=p.parse_args()
    atomic_json(a.receipt,check(Path(a.path),a.step,a.independent));print('checkpoint PASS',a.step)
if __name__=='__main__':main()
