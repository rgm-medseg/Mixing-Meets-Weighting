"""Same-protocol orchestration of official AdaMix-CT's literal training step.

Method code, native U-Net, AdaMix curriculum, CE/Dice, confidence threshold,
four loaders and AdamW are from pinned upstream. Only split, total updates,
validation selection and durability are adapted. No beta-FFT loss is used.
"""
import ast
import copy
import math
import sys
import textwrap
import time
from pathlib import Path
import numpy as np
import torch
from torch.utils.data import ConcatDataset,DataLoader,Subset,Sampler
from common import *
import revision_runtime as r

OFFICIAL=WORK/'external/AdaMix'
OFFICIAL_CACHE=None

def load_official():
    global OFFICIAL_CACHE
    if OFFICIAL_CACHE is not None: return OFFICIAL_CACHE
    sys.path.insert(0,str(WORK/'external/python_deps'))
    sys.path.insert(0,str(OFFICIAL))
    source=(OFFICIAL/'exp_acdc/train_adamix_ct.py').read_text()
    source=source.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "2"','os.environ["CUDA_VISIBLE_DEVICES"] = "0"')
    source=source.replace('from utils import ensure_dir','def ensure_dir(path):\n    os.makedirs(path, exist_ok=True)')
    source=source.replace('import pandas as pd','')  # unused upstream report dependency
    source=source.replace('from torch.utils.tensorboard import SummaryWriter','')  # orchestration writes local CSV
    prefix=source[:source.index('def main(is_debug=False):')]
    ns={'__file__':str(OFFICIAL/'exp_acdc/train_adamix_ct.py'),'__name__':'revision_official_definitions'}
    exec(compile(prefix,str(OFFICIAL/'exp_acdc/train_adamix_ct.py'),'exec'),ns)
    OFFICIAL_CACHE=ns
    return ns

class RecordedRandomBatches(Sampler):
    def __init__(self,dataset,batch=16):
        self.n=len(dataset); self.batch=batch; self.plan=[]; self.replay=None
    def __len__(self): return math.ceil(self.n/self.batch)
    def __iter__(self):
        if self.replay is not None:
            s=self.replay; self.replay=None; self.plan=s['plan']
            yield from self.plan[s['consumed']:]
        else:
            # Match PyTorch RandomSampler's default local-generator seed draw.
            seed=int(torch.empty((),dtype=torch.int64).random_().item())
            gen=torch.Generator(); gen.manual_seed(seed)
            ids=torch.randperm(self.n,generator=gen).tolist()
            self.plan=[ids[i:i+self.batch] for i in range(0,self.n,self.batch)]
            yield from self.plan

def make_loaders(ns,cfg):
    OfficialDataset=ns['ACDCDataset']
    class Dataset(OfficialDataset):
        def __init__(self,labeled):
            self.image_size=256; self.sep='/'; self.stage='train'; self.is_augmentation=True
            self.image_path=str(DATA)
            all_ids=(DATA/'train_slices.list').read_text().splitlines()
            # Exact 136 labeled slices, same seven patients; no official shuffle list needed.
            self.sample_list=all_ids[:136] if labeled else all_ids[136:]
            self.augmentation=self.augmentation_transform()
            self.post_transform=OfficialDataset.post_transform(self)
            self.label_transform=OfficialDataset.label_transform(self)
            self.pre_transform=OfficialDataset.pre_transform(self)
    labeled=Dataset(True); unlabeled=Dataset(False)
    train=ConcatDataset([labeled,unlabeled])
    repeated=ConcatDataset([labeled for _ in range(len(train)//len(labeled))])
    repeated=ConcatDataset([repeated,Subset(repeated,range(len(train)-len(repeated)))])
    datasets=[repeated,train,repeated,train]
    loaders=[]; samplers=[]
    for index,ds in enumerate(datasets):
        sampler=RecordedRandomBatches(ds,cfg['batch_size']); samplers.append(sampler)
        kw={}
        if index>=2:
            generator=ns['BoxMaskGenerator'](prop_range=(.25,.5),n_boxes=3,random_aspect_ratio=True,prop_by_area=True,within_bounds=True,invert=True)
            kw['collate_fn']=ns['SegCollate'](batch_aug_fn=ns['AddMaskParamsToBatch'](generator))
        loaders.append(DataLoader(ds,batch_sampler=sampler,num_workers=cfg['num_workers'],pin_memory=True,**kw))
    return loaders,samplers

def run_official(cfg,resume=False):
    from evaluate import evaluate_model
    ns=load_official(); ns['seed_torch'](cfg['seed'])
    run=Path(cfg['run_root']); (run/'checkpoints').mkdir(parents=True,exist_ok=True)
    source=(OFFICIAL/'exp_acdc/train_adamix_ct.py').read_text()
    begin=source.index('            # labeled data\n')
    end=source.index("            writer.add_scalar('train_s_loss'",begin)
    step_source=textwrap.dedent(source[begin:end])
    (run/'official_step_source.py').write_text(step_source)
    step_code=compile(step_source,str(run/'official_step_source.py'),'exec')
    device=torch.device('cuda:0')
    # Upstream initializes each classifier after constructing its student.
    models=[]
    for _ in range(2):
        model=ns['networks'].UNet(in_channels=1,out_channels=4).to(device)
        ns['init_weight'](model.net.classifier,torch.nn.init.kaiming_normal_,torch.nn.BatchNorm2d,1e-5,.1,mode='fan_in',nonlinearity='relu')
        models.append(model)
    loaders,samplers=make_loaders(ns,cfg)
    iters=len(loaders[0]); optimizers=[torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999)) for m in models]
    criterion_s=ns['DSCLossH'](num_classes=4,device=device)
    criterion_u=ns['DSCLossH'](num_classes=4,device=device)
    # Curriculum uses the locked formal total, including during smoke checks.
    mixers=[ns['AdaMix'](total_steps=30000//4,num_classes=4,image_size=256,patch_size=8,topk=16,mode='hard',p=.5) for _ in range(4)]
    step=0; best=-1.; history=[]; resume_rng=None
    if resume:
        ck=torch.load(run/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        for m,key in zip(models,['model_1','model_2']): m.load_state_dict(ck[key])
        for o,key in zip(optimizers,['optimizer1','optimizer2']): o.load_state_dict(ck[key])
        step=ck['global_step']; best=ck['best_metrics']['model1_best_val_dice']; resume_rng=ck['random_state']
        history=read_table(run/'validation.csv') if (run/'validation.csv').exists() else []
        history=[row for row in history if int(row['step'])<=step]
        for mixer,age in zip(mixers,ck['adamix_age']): mixer.age=age
        if step%iters:
            for sampler,s in zip(samplers,ck['sampler_state']): sampler.replay=s
        r.restore_rng(resume_rng)
    def save(path):
        consumed=step%iters
        ck=dict(model_1=models[0].state_dict(),model_2=models[1].state_dict(),optimizer1=optimizers[0].state_dict(),optimizer2=optimizers[1].state_dict(),
                global_step=step,epoch=step//iters,random_state=r.rng_state(),best_metrics={'model1_best_val_dice':best},
                sampler_state=[dict(plan=s.plan,consumed=consumed) for s in samplers],adamix_age=[m.age for m in mixers],
                args=cfg,amp_state=None,lr_state=[o.param_groups[0]['lr'] for o in optimizers],resume_exact=False)
        r.atomic_torch(ck,path)
    ctx=dict(student1=models[0],student2=models[1],optimizer1=optimizers[0],optimizer2=optimizers[1],criterion_s=criterion_s,criterion_u=criterion_u,
             s1_adamix_l=mixers[0],s1_adamix_u=mixers[1],s2_adamix_l=mixers[2],s2_adamix_u=mixers[3],device=device,
             conf_threshold=.95,pbar=range(iters),torch=torch)
    val_cases=(DATA/'val.list').read_text().splitlines()
    epoch=step//iters+1
    while step<cfg['max_iterations']:
        for m in models: m.train()
        names=['iter_train_labeled_dataloader','iter_train_unlabeled_dataloader','iter_train_labeled_aux_dataloader','iter_train_unlabeled_aux_dataloader']
        for name,loader in zip(names,loaders): ctx[name]=iter(loader)
        if resume_rng:
            r.restore_rng(resume_rng); resume_rng=None
        offset=step%iters
        for idx in range(offset,iters):
            ctx.update(idx=idx,epoch=epoch)
            exec(step_code,ctx)
            loss=ctx['loss']
            if not torch.isfinite(loss): raise RuntimeError('Non-finite official AdaMix-CT loss')
            step+=1
            r.LATEST.clear(); r.LATEST.update(supervised_loss=ctx['loss_s'].item(),unsupervised_loss=ctx['loss_u'].item(),included_ratio=ctx['included_ratio'].item())
            r.record_step(run,step,optimizers[0],loss)
            print(f"iteration {step}: official AdaMix-CT loss={loss.item():.8f}",flush=True)
            if step%cfg['validation_interval']==0:
                models[0].eval()
                summary,_=evaluate_model(models[0],val_cases,'E5',str(step))
                score=summary['dice']
                history.append(dict(step=step,dice=score))
                if score>best:
                    best=score; save(run/'checkpoints/best.ckpt')
                write_csv(run/'validation.csv',history,['step','dice'])
                models[0].train()
            if step%cfg['save_interval']==0: save(run/'checkpoints/last.ckpt')
            if r.STOP:
                save(run/'checkpoints/last.ckpt'); r.completion(run,step,True); raise SystemExit(75)
            if step>=cfg['max_iterations']: break
        epoch+=1
    save(run/'checkpoints/last.ckpt')
    if cfg['max_iterations']<=2: save(run/'checkpoints/best.ckpt')
    if not (run/'checkpoints/best.ckpt').exists(): raise RuntimeError('No validation-selected official checkpoint')
    r.completion(run,step)
