"""ACDC unified frame-volume metrics; patient aggregation is separate in A4."""
import argparse
from pathlib import Path
import h5py
import numpy as np
import torch
from medpy import metric
from scipy.ndimage import zoom
from common import *
bootstrap()

def metrics(pred,gt):
    pred=pred.astype(bool); gt=gt.astype(bool)
    if pred.any() and gt.any():
        inter=int((pred&gt).sum())
        return dict(dice=2*inter/(int(pred.sum())+int(gt.sum())),iou=inter/int((pred|gt).sum()),
                    hd95=float(metric.binary.hd95(pred,gt)),asd=float(metric.binary.asd(pred,gt)))
    if not pred.any() and not gt.any():
        return dict(dice=0.,iou=0.,hd95=0.,asd=0.)
    return dict(dice=0.,iou=0.,hd95=float(np.linalg.norm(pred.shape)),asd=float(np.linalg.norm(pred.shape)))

def load_model(cfg,checkpoint=None,branch='model_1'):
    import revision_runtime as r
    r.configure(cfg)
    if cfg['model_family']=='official_adamix_ct':
        from official_adapter import load_official
        official=load_official()
        model=official['networks'].UNet(in_channels=1,out_channels=4).cuda()
    else:
        model=r.BCP_net(in_chns=1,class_num=4,ema=False,mask_token_enabled=cfg['trc']!='off')
    ck=torch.load(checkpoint or cfg['checkpoint'],map_location='cpu',weights_only=False)
    model.load_state_dict(ck[branch],strict=True)
    model.eval()
    return model,ck

@torch.inference_mode()
def infer_volume(model,image):
    result=np.zeros(image.shape,dtype=np.uint8)
    for z,source in enumerate(image):
        h,w=source.shape
        x=zoom(source,(256/h,256/w),order=0)
        out=model(torch.from_numpy(x).float()[None,None].cuda())
        if isinstance(out,dict): out=out['out']
        if isinstance(out,(tuple,list)): out=out[0]
        p=out.softmax(1).argmax(1)[0].cpu().numpy()
        result[z]=zoom(p,(h/256,w/256),order=0)
    return result

def evaluate_model(model,cases,run,checkpoint,output=None,save_predictions=False):
    rows=[]
    for case in cases:
        with h5py.File(DATA/'data'/f'{case}.h5','r') as f:
            image=f['image'][:]; gt=f['label'][:]
        pred=infer_volume(model,image)
        if save_predictions:
            dst=Path(output)/'predictions'; dst.mkdir(parents=True,exist_ok=True)
            # No source image or GT in prediction archive.
            np.savez_compressed(dst/f'{case}.npz',prediction=pred,case=case,run=run,checkpoint=checkpoint)
        for cls,name in [(1,'RV'),(2,'MYO'),(3,'LV')]:
            rows.append(dict(run=run,checkpoint=checkpoint,case=case,patient=case.split('_')[0],class_id=cls,class_name=name,
                             unit='grid',**metrics(pred==cls,gt==cls)))
    summary={k:float(np.mean([r[k] for r in rows])) for k in ('dice','iou','hd95','asd')}
    return summary,rows

def main():
    p=argparse.ArgumentParser()
    p.add_argument('--task',required=True)
    p.add_argument('--split',choices=['test','val'],default='test')
    p.add_argument('--smoke',action='store_true')
    a=p.parse_args()
    if a.task.startswith('Historical'):
        cfg=read_json(WORK/f'configs/{a.task}.json'); checkpoint=Path(cfg['checkpoint'])
    else:
        _,cfg=config_for(a.task)
        checkpoint=Path(cfg['run_root'])/'checkpoints/best.ckpt'
    if a.smoke:
        checkpoint=WORK/f'runs/smoke/{a.task}/checkpoints/last.ckpt'
    output=WORK/('runs/smoke' if a.smoke else 'runs')/a.task/'evaluation'
    output.mkdir(parents=True,exist_ok=True)
    receipt=output/'EVALUATION_RECEIPT.json'
    ch=sha(checkpoint)
    if receipt.exists():
        prior=read_json(receipt)
        if prior['checkpoint_sha256']==ch and prior['split']==a.split and not a.smoke:
            print('Existing identical evaluation reused:',receipt); return
        if not a.smoke: raise RuntimeError('Refusing to overwrite different evaluation identity')
    model,ck=load_model(cfg,checkpoint)
    cases=[c.strip() for c in (DATA/f'{a.split}.list').read_text().splitlines() if c.strip()]
    if a.smoke: cases=cases[:1]
    summary,rows=evaluate_model(model,cases,a.task,str(checkpoint),output,True)
    write_csv(output/'case_class_metrics.csv',rows)
    class_rows=[dict(run=a.task,class_name=name,unit='grid',**{k:float(np.mean([r[k] for r in rows if r['class_name']==name])) for k in summary}) for name in ('RV','MYO','LV')]
    write_csv(output/'class_metrics.csv',class_rows)
    atomic_json(receipt,dict(status='EVALUATED',run=a.task,split=a.split,checkpoint=str(checkpoint),checkpoint_sha256=ch,
                            checkpoint_step=ck.get('global_step'),manifest=str(DATA/f'{a.split}.list'),manifest_sha256=sha(DATA/f'{a.split}.list'),
                            protocol='ACDC_v22_grid_directed_ASD',smoke=a.smoke,cases=len(cases),time=now(),metrics=summary))
    print(json.dumps(summary))

if __name__=='__main__':
    main()
