"""Zero-update actual construction and covered-MAC accounting, one input."""
from common import *
def main():
    import torch
    from resources import bind_gpu_pid
    bind_gpu_pid();bootstrap();import revision_runtime as r
    e4=read_json(MAIN_WORK/'configs/E4_plain_residual_v2B.yaml');r.configure(e4)
    models=[('E4',r.BCP_net(1,4))]
    r.configure(dict(e4,trc='token'));models.append(('R1',r.BCP_net(1,4)))
    from official_adapter import load_official
    ns=load_official();models.append(('E5',ns['networks'].UNet(in_channels=1,out_channels=4).cuda()))
    rows=[];layers=[];details=[]
    for task,m in models:
        m.eval();total=[0];hooks=[]
        def hook(name,module,inputs,out):
            if isinstance(module,(torch.nn.Conv2d,torch.nn.ConvTranspose2d)):
                # Match previous conv/linear accounting: output spatial positions.
                mac=out.numel()*(module.in_channels//module.groups)*module.kernel_size[0]*module.kernel_size[1]
            else:mac=out.numel()*module.in_features
            total[0]+=mac;layers.append(dict(task=task,layer=name,type=type(module).__name__,parameters=sum(p.numel() for p in module.parameters(recurse=False)),covered_MACs=int(mac)))
        for n,mod in m.named_modules():
            if isinstance(mod,(torch.nn.Conv2d,torch.nn.ConvTranspose2d,torch.nn.Linear)):hooks.append(mod.register_forward_hook(lambda mod,inp,out,n=n:hook(n,mod,inp,out)))
        with torch.inference_mode():m(torch.zeros(1,1,256,256,device='cuda'))
        for h in hooks:h.remove()
        extra=2*4*16*256*256 if task=='R1' else 0
        params=sum(p.numel() for p in m.parameters())
        if task=='E4':assert params==1827475 and sum(p.numel() for p in m.decoder.mask_token_router.parameters())==647
        if task=='R1':assert params==1827469 and sum(p.numel() for p in m.decoder.mask_token_router.parameters())==641
        for n,p in m.named_parameters():details.append(dict(task=task,name=n,shape=json.dumps(list(p.shape)),numel=p.numel(),requires_grad=p.requires_grad))
        rows.append(dict(task=task,parameters=params,adapter_parameters=647 if task=='E4' else 641 if task=='R1' else 0,covered_MACs=total[0]+extra,conv_linear_MACs=total[0],TRC_einsum_MACs=extra,input='1x1x256x256',precision='FP32',full_FLOPs='NA',coverage='conv/transpose-conv/linear output-grid convention + R1 two token einsums; excludes LN/BN/GELU/softmax/interpolate/pooling/additions',structure='16->18 Conv1x1bias/LN18/GELU/Conv1x1bias16 + gamma0 residual' if task=='E4' else 'token residual' if task=='R1' else 'native official UNet; no added adapters'))
    write_csv(MAIN_WORK/'MODEL_STRUCTURE_COMPARISON.csv',rows);write_csv(MAIN_WORK/'reports/MODEL_LAYER_MACS.csv',layers);write_csv(MAIN_WORK/'reports/MODEL_PARAMETER_SHAPES.csv',details)
    atomic_json(MAIN_WORK/'reports/MODEL_COUNT_RECEIPT.json',dict(status='PASS',time=now(),actual_instances=True,training_updates=0,models=rows))
if __name__=='__main__':main()
