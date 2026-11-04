"""Independent zero-update CPU architecture/import inventory."""
import inspect,subprocess
from common import *
def main():
    import torch
    # Local process only: architecture inventory requires no GPU or optimizer.
    original=torch.nn.Module.cuda;torch.nn.Module.cuda=lambda self,*a,**k:self
    try:
        bootstrap();import revision_runtime as r
        cfg=read_json(MAIN_WORK/'configs/E4_plain_residual_v2B.yaml');r.configure(cfg);e4=r.BCP_net(1,4)
        r.configure(dict(cfg,trc='token'));r1=r.BCP_net(1,4)
        from official_adapter import load_official
        ns=load_official();e5=ns['networks'].UNet(in_channels=1,out_channels=4)
        shapes=[]
        for name,model in [('E4',e4),('R1',r1),('E5',e5)]:
            (MAIN_WORK/'reports'/f'{name}_MODEL_ARCHITECTURE.txt').write_text(str(model)+'\n')
            for n,m in model.named_modules():
                shapes.append(dict(task=name,path=n,class_name=type(m).__module__+'.'+type(m).__name__,own_params=sum(p.numel() for p in m.parameters(recurse=False)),detail=str(m).splitlines()[0]))
        write_csv(MAIN_WORK/'reports/MODEL_ARCHITECTURE_MODULES.csv',shapes)
        old=PROJECT/'revision_runs_20261006/external/AdaMix';commit=subprocess.check_output(['git','-C',str(old),'rev-parse','HEAD'],text=True).strip()
        assert commit=='85361fb91cf43a69a158d41dfa9e7ee9fce36db9'
        atomic_json(MAIN_WORK/'reports/E5_IMPORT_BINDING.json',dict(status='PASS',time=now(),actual_commit=commit,official_repository='https://github.com/Senyh/AdaMix',native_model_class=type(e5).__module__+'.'+type(e5).__name__,native_model_file=inspect.getfile(type(e5)),AdaMix_file=inspect.getfile(ns['AdaMix']),loss_file=inspect.getfile(ns['DSCLossH']),dataset_file=inspect.getfile(ns['ACDCDataset']),literal_step_source=str(MAIN_WORK/'runs/E5_isolated_short/runs/short/official_step_source.py'),literal_step_sha256=sha(MAIN_WORK/'runs/E5_isolated_short/runs/short/official_step_source.py'),models_parameters=[sum(p.numel() for p in m.parameters()) for m in [e4,r1,e5]],scope='independent import/CPU architecture inventory; actual runtime shape counts and source hashes additionally in initialization receipts',training_updates=0))
    finally:torch.nn.Module.cuda=original
if __name__=='__main__':main()
