"""Pinned official AdaMix CT computation; explicit common-protocol orchestration."""
import copy,math,sys,textwrap,time,signal
import numpy as np
import torch
import h5py
from PIL import Image
from torch.utils.data import ConcatDataset,DataLoader,Subset,Sampler
from common import *
OFFICIAL=MAIN_WORK/'external/AdaMix'
CACHE=None

def load_official():
    global CACHE
    if CACHE is not None:return CACHE
    sys.path.insert(0,str(MAIN_WORK/'external/python_deps'));sys.path.insert(0,str(OFFICIAL))
    src=(OFFICIAL/'exp_acdc/train_adamix_ct.py').read_text()
    src=src.replace('os.environ["CUDA_VISIBLE_DEVICES"] = "2"','os.environ["CUDA_VISIBLE_DEVICES"] = "0"')
    src=src.replace('from utils import ensure_dir','def ensure_dir(path):\n    os.makedirs(path, exist_ok=True)')
    src=src.replace('import pandas as pd','').replace('from torch.utils.tensorboard import SummaryWriter','')
    ns={'__file__':str(OFFICIAL/'exp_acdc/train_adamix_ct.py'),'__name__':'pinned_official_CT'}
    exec(compile(src[:src.index('def main(is_debug=False):')],ns['__file__'],'exec'),ns)
    CACHE=ns;return ns

class RecordedRandomBatches(Sampler):
    def __init__(self,dataset,batch=6):self.n=len(dataset);self.batch=batch;self.plan=[];self.replay=None
    def __len__(self):return math.ceil(self.n/self.batch)
    def __iter__(self):
        if self.replay is not None:
            s=self.replay;self.replay=None;self.plan=s['plan'];yield from self.plan[s['consumed']:]
        else:
            seed=int(torch.empty((),dtype=torch.int64).random_().item());g=torch.Generator().manual_seed(seed)
            ids=torch.randperm(self.n,generator=g).tolist();self.plan=[ids[i:i+self.batch] for i in range(0,self.n,self.batch)];yield from self.plan

def make_loaders(ns,cfg):
    Original=ns['ACDCDataset']
    class CommonDataset(Original):
        def __init__(self,role):
            self.image_size=256;self.sep='/';self.stage='val' if role=='val' else 'train';self.role=role
            self.is_augmentation=self.stage=='train';self.image_path=str(DATA)
            ids=(MAIN_WORK/'configs/train_slices.list').read_text().splitlines()
            if role=='labeled':self.sample_list=ids[:136]
            elif role=='unlabeled':self.sample_list=ids[136:]
            else:
                self.sample_list=[]
                for case in (MAIN_WORK/'configs/val.list').read_text().splitlines():
                    with h5py.File(DATA/'data'/(case+'.h5'),'r') as f:nz=f['image'].shape[0]
                    self.sample_list.extend((case,z) for z in range(nz))
            if self.is_augmentation:self.augmentation=Original.augmentation_transform()
            self.post_transform=Original.post_transform(self);self.label_transform=Original.label_transform(self);self.pre_transform=Original.pre_transform(self)
        def __getitem__(self,item):
            if self.role=='val':case,z=self.sample_list[item];source=DATA/'data'/(case+'.h5')
            else:case=self.sample_list[item];source=DATA/'data/slices'/(case+'.h5')
            with h5py.File(source,'r') as f:
                image=(f['image'][z] if self.role=='val' else f['image'][:])*255.
                # Never read or expose GT for any pseudo-label-stream case.
                label=np.zeros(image.shape,dtype=np.uint8) if self.role=='unlabeled' else f['label'][z] if self.role=='val' else f['label'][:]
            image=Image.fromarray(image).convert('L');label=Image.fromarray(label).convert('L')
            if self.stage=='train':
                image,label=self.pre_transform(image,label);imageA1,imageA2=copy.deepcopy(image),copy.deepcopy(image)
                imageA1,_=self.augmentation(imageA1,label);imageA2,_=self.augmentation(imageA1,label)
                image,label=self.post_transform(image),self.label_transform(label)
                return image,torch.from_numpy(np.array(label)).unsqueeze(0),self.post_transform(imageA1),self.post_transform(imageA2)
            return self.post_transform(image),torch.from_numpy(np.array(self.label_transform(label))).unsqueeze(0)
    labeled=CommonDataset('labeled');unlabeled=CommonDataset('unlabeled');val=CommonDataset('val')
    assert len(labeled)==136 and len(unlabeled)==1176
    repeated=ConcatDataset([labeled for _ in range(len(unlabeled)//136)])
    repeated=ConcatDataset([repeated,Subset(repeated,range(len(unlabeled)-len(repeated)))])
    loaders=[];samplers=[]
    for i,ds in enumerate([repeated,unlabeled,repeated,unlabeled]):
        sampler=RecordedRandomBatches(ds,6);samplers.append(sampler);kw={}
        if i>=2:
            gen=ns['BoxMaskGenerator'](prop_range=(.25,.5),n_boxes=3,random_aspect_ratio=True,prop_by_area=True,within_bounds=True,invert=True)
            kw['collate_fn']=ns['SegCollate'](batch_aug_fn=ns['AddMaskParamsToBatch'](gen))
        loaders.append(DataLoader(ds,batch_sampler=sampler,num_workers=cfg['num_workers'],pin_memory=True,**kw))
    val_loader=DataLoader(val,batch_size=6,num_workers=cfg['num_workers'],shuffle=False,pin_memory=True)
    assert all(len(x)==196 for x in loaders)
    return loaders,samplers,val_loader

def append(path,row):
    path=Path(path)
    with path.open('a',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(row))
        if f.tell()==0:w.writeheader()
        w.writerow(row);f.flush()

def run_official(cfg,resume=False,short_target=None,resume_probe=False):
    from unified_evaluation import evaluate_model
    ns=load_official();ns['seed_torch'](42)
    torch.backends.cudnn.allow_tf32=True;torch.backends.cuda.matmul.allow_tf32=False
    run=Path(cfg['run_root']);(run/'checkpoints').mkdir(parents=True,exist_ok=True)
    source=(OFFICIAL/'exp_acdc/train_adamix_ct.py').read_text();begin=source.index('            # labeled data\n');end=source.index("            writer.add_scalar('train_s_loss'",begin)
    literal=textwrap.dedent(source[begin:end]);(run/'official_step_source.py').write_text(literal);code=compile(literal,str(run/'official_step_source.py'),'exec')
    loaders,samplers,val_loader=make_loaders(ns,cfg);device=torch.device('cuda:0');models=[]
    for _ in range(2):
        model=ns['networks'].UNet(in_channels=1,out_channels=4).to(device)
        ns['init_weight'](model.net.classifier,torch.nn.init.kaiming_normal_,torch.nn.BatchNorm2d,1e-5,.1,mode='fan_in',nonlinearity='relu');models.append(model)
    opts=[torch.optim.AdamW(m.parameters(),lr=1e-4,betas=(.9,.999)) for m in models]
    criterion_s=ns['DSCLossH'](num_classes=4,device=device);criterion_u=ns['DSCLossH'](num_classes=4,device=device)
    mixers=[ns['AdaMix'](total_steps=7500,num_classes=4,image_size=256,patch_size=8,topk=16,mode='hard',p=.5) for _ in range(4)]
    step=0;best=-1.;best_official=100.;resume_rng=None;start=time.monotonic();interrupted=False;counts={n:0 for n in ['optimizer1','optimizer2']}
    ledger=WORK/'results/optimizer_calls.csv'
    if ledger.exists():
        for row in read_table(ledger):
            if row['event']=='success':counts[row['optimizer']]+=1
    def guard(o,name):
        original=o.step
        def call(*a,**kw):
            try:result=original(*a,**kw)
            except BaseException as e:
                append(ledger,dict(time=now(),segment=cfg['segment_id'],logical_step=step+1,optimizer=name,event='failed',cumulative_calls=counts[name],reason=type(e).__name__));raise
            counts[name]+=1;append(ledger,dict(time=now(),segment=cfg['segment_id'],logical_step=step+1,optimizer=name,event='success',cumulative_calls=counts[name],reason=''));return result
        o.step=call
    for o,n in zip(opts,counts):guard(o,n)
    if resume:
        ck=torch.load(run/'checkpoints/last.ckpt',map_location='cpu',weights_only=False)
        assert ck['metadata']['cfg_sha256']==cfg['config_sha256'] and ck['metadata']['code_identity']==task_identity('E5')
        for m,k in zip(models,['model_1','model_2']):m.load_state_dict(ck[k],strict=True)
        for o,k in zip(opts,['optimizer1','optimizer2']):o.load_state_dict(ck[k])
        step=ck['global_step'];best=ck['best_metrics']['model1_best_val_dice'];best_official=ck['best_metrics']['official_val_loss'];resume_rng=ck['random_state']
        for mix,age in zip(mixers,ck['adamix_age']):mix.age=age
        if step%196:
            for s,plan in zip(samplers,ck['sampler_state']):s.replay=plan
        restore_rng(resume_rng)
        if resume_probe:
            assert all(torch.equal(v.cpu(),ck[k][n].cpu()) for m,k in zip(models,['model_1','model_2']) for n,v in m.state_dict().items())
            atomic_json(WORK/'reports/RESUME_ZERO_UPDATE.json',dict(status='PASS',step=step,optimizer_calls=0,models_exact=True,optimizers_loaded=True,main_RNG_loaded=True,sampler_remaining_plans_loaded=True,worker_resume_exact=False));return
    else:
        assert all(not o.state for o in opts)
        atomic_json(WORK/'reports/INITIALIZATION_ACTUAL.json',dict(time=now(),joint_step=0,fresh=True,random_seed=42,pretrain='none: official CT random initialization',pretrain_budget=0,teacher='none',smoke_state_reused=False,code_identity=task_identity('E5'),official_commit=cfg['official_commit'],official_entry=str(OFFICIAL/'exp_acdc/train_adamix_ct.py'),official_entry_sha256=sha(OFFICIAL/'exp_acdc/train_adamix_ct.py'),literal_step_sha256=sha(run/'official_step_source.py'),rng_sha256=rng_digest(rng_state()),data_manifest_sha256=sha(MAIN_WORK/'manifests/data_assets.csv'),models=[dict(branch=k,parameters=sum(p.numel() for p in m.parameters()),shapes={n:list(p.shape) for n,p in m.named_parameters()}) for m,k in zip(models,['model_1','model_2'])],optimizer_groups=[[{k:v for k,v in g.items() if k!='params'} for g in o.param_groups] for o in opts],backend=dict(cudnn_deterministic=torch.backends.cudnn.deterministic,cudnn_benchmark=torch.backends.cudnn.benchmark,cudnn_allow_tf32=True,matmul_allow_tf32=False,AMP=False),batch_definition='4 streams x6 =24 images; main6+aux6 GT=12; other12 strict unlabeled',curriculum_total_steps=7500,test_access=False))
    def flag(signum,frame):
        nonlocal interrupted
        interrupted=True
    signal.signal(signal.SIGTERM,flag);signal.signal(signal.SIGINT,flag)
    def save(path):
        atomic_torch(dict(model_1=models[0].state_dict(),model_2=models[1].state_dict(),optimizer1=opts[0].state_dict(),optimizer2=opts[1].state_dict(),global_step=step,epoch=step//196,random_state=rng_state(),best_metrics=dict(model1_best_val_dice=best,official_val_loss=best_official),sampler_state=[dict(plan=s.plan,consumed=step%196) for s in samplers],adamix_age=[m.age for m in mixers],lr_state=[[g['lr'] for g in o.param_groups] for o in opts],scheduler_state=None,amp_state=None,pretraining_stage='not required',teacher_state=None,args=cfg,resume_exact=False,metadata=dict(task='E5',cfg_sha256=cfg['config_sha256'],code_identity=task_identity('E5'),official_commit=cfg['official_commit'],optimizer_success_calls=counts,curriculum_total_steps=7500)),path)
    ctx=dict(student1=models[0],student2=models[1],optimizer1=opts[0],optimizer2=opts[1],criterion_s=criterion_s,criterion_u=criterion_u,s1_adamix_l=mixers[0],s1_adamix_u=mixers[1],s2_adamix_l=mixers[2],s2_adamix_u=mixers[3],device=device,conf_threshold=.95,pbar=range(196),torch=torch)
    names=['iter_train_labeled_dataloader','iter_train_unlabeled_dataloader','iter_train_labeled_aux_dataloader','iter_train_unlabeled_aux_dataloader']
    def official_val():
        nonlocal best_official
        t=time.monotonic();models[0].eval();losses=[]
        with torch.no_grad():
            for image,label in val_loader:losses.append(criterion_s(models[0](image.to(device))['out'],label.to(device).squeeze(1).long()).item())
        value=float(np.mean(losses));changed=value<=best_official
        if changed:best_official=value;save(run/'checkpoints/official_val_loss_best.ckpt')
        append(WORK/'results/official_selection_trace.csv',dict(time=now(),segment=cfg['segment_id'],step=step,epoch=step//196,val_loss=value,best_val_loss=best_official,replaced=changed,criterion='official <= supervised val_loss; common patients; official PIL/bilinear input; per_stream batch6',seconds=time.monotonic()-t));models[0].train()
    while step<30000:
        for m in models:m.train()
        for name,loader in zip(names,loaders):ctx[name]=iter(loader)
        if resume_rng:restore_rng(resume_rng);resume_rng=None
        for idx in range(step%196,196):
            t=time.monotonic();ctx.update(idx=idx,epoch=step//196+1);exec(code,ctx);step+=1
            assert torch.isfinite(ctx['loss']) and all(g['lr']==1e-4 for o in opts for g in o.param_groups)
            # IDs come from recorded official RandomSampler plans; no additional draws.
            ids=[s.plan[idx] for s in samplers];lids=(MAIN_WORK/'configs/train_slices.list').read_text().splitlines()
            labeled_ids=[[lids[j%136] for j in ids[i]] for i in [0,2]];unlabeled_ids=[[lids[136+j] for j in ids[i]] for i in [1,3]]
            assert all(x.split('_')[0] in {z.split('_')[0] for z in lids[:136]} for group in labeled_ids for x in group)
            assert all(x.split('_')[0] not in {z.split('_')[0] for z in lids[:136]} for group in unlabeled_ids for x in group)
            torch.cuda.synchronize()
            row=dict(step=step,time=now(),segment=cfg['segment_id'],loss=ctx['loss'].item(),supervised_loss=ctx['loss_s'].item(),unsupervised_loss=ctx['loss_u'].item(),included_ratio=ctx['included_ratio'].item(),lr=1e-4,cur_step=step-1,curriculum_total_steps=7500,ages=json.dumps([m.age for m in mixers]),optimizer1_calls=counts['optimizer1'],optimizer2_calls=counts['optimizer2'],step_seconds=time.monotonic()-t,peak_cuda_bytes=torch.cuda.max_memory_allocated(),peak_cuda_reserved_bytes=torch.cuda.max_memory_reserved(),labeled_ids=json.dumps(labeled_ids),unlabeled_ids=json.dumps(unlabeled_ids),total_images=24,visible_GT_images=12)
            append(WORK/'results/E5_control_trace.csv',row)
            atomic_json(WORK/'state/control_progress.json',dict(pid=os.getpid(),time=now(),joint_step=step,optimizer_calls=counts,step_seconds=row['step_seconds'],peak_cuda_bytes=row['peak_cuda_bytes']))
            print(f'iteration {step}: official AdaMix-CT loss={row["loss"]:.8f}',flush=True)
            if step%196==0 or (short_target and step==1):official_val()
            if step%200==0 or (short_target and step==1):
                tv=time.monotonic();models[0].eval();score,_=evaluate_model(models[0],(MAIN_WORK/'configs/val.list').read_text().splitlines(),'E5',str(step));value=score['dice']
                if value>best:best=value;save(run/'checkpoints/best.ckpt')
                append(WORK/'results/validation_trace.csv',dict(time=now(),segment=cfg['segment_id'],step=step,student1_macro_dice=value,student2_macro_dice='NA: official model1 primary only',best_student1_macro_dice=best,seconds=time.monotonic()-tv,selection='strict student1 common val macro Dice; test excluded'));models[0].train()
            if step%200==0:save(run/'checkpoints/last.ckpt')
            if stop_requested() or interrupted or (short_target and step>=short_target):
                save(run/'checkpoints/last.ckpt');atomic_json(run/'TRAIN_RECEIPT.json',dict(status='INTERRUPTED',task='E5',global_step=step,optimizer_success_calls=counts,time=now(),segment_seconds=time.monotonic()-start,resume_exact=False,test_access_during_training=0));raise SystemExit(75)
            if step==30000:break
    save(run/'checkpoints/last.ckpt')
    atomic_json(run/'TRAIN_RECEIPT.json',dict(status='TRAINED',task='E5',global_step=step,optimizer_success_calls=counts,time=now(),segment_seconds=time.monotonic()-start,resume_exact=False,test_access_during_training=0))

if __name__=='__main__':raise RuntimeError('Use train_E5_entry.py')
