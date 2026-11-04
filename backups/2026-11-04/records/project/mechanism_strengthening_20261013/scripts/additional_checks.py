"""CPU-only evidence checks; never attaches to or mutates the training process."""
import json,h5py,numpy as np,torch,platform,importlib
from collections import defaultdict
from common import *
from augmentation import seed
def main():
    from dataloaders.dataset import WeakStrongAugment
    plan=json.loads((W/'configs/joint_input_plan.json').read_text());cache={};checks=[]
    for bp in plan['batches']:
        for s in bp['slots']:
            if s['frame'] not in cache:
                with h5py.File(s['source']) as f:cache[s['frame']]=f['image'][:]
            image=cache[s['frame']][s['z']];seed(s['seed'])
            z=WeakStrongAugment([256,256])(dict(image=image.copy(),label=np.zeros(image.shape,np.uint8)))
            with np.load(s['path']) as a:
                assert np.array_equal(z['image'].numpy(),a['weak']) and np.array_equal(z['image_strong'].numpy(),a['strong'])
            checks.append(dict(batch=s['batch'],slot=s['slot'],sample=s['sample'],image_outputs_identical_with_zero_GT=True))
    csvout(W/'results/GT_independence_checks.csv',checks);js(W/'receipts/GT_independence.json',dict(status='PASS',comparisons=len(checks),scope='replacing validation GT with zeros does not change either augmented image; GT only defines diagnostic statistics'))
    groups=defaultdict(list)
    for frame in CFG['val_frames']:
        files=sorted((W/'arrays/trc').glob(frame+'_z*.npz'))
        for p in files:
            with np.load(p) as a:
                groups[frame.split('_')[0]].append(dict(pixels=a['gt'].size,residual_sum=float(a['residual_norm'].sum(dtype=np.float64)),feature_sum=float(a['feature_norm'].sum(dtype=np.float64)),logit_sum=float(a['logit_change'].sum(dtype=np.float64)),residual_spatial_sd=float(a['residual_norm'].std(dtype=np.float64)),residual_min=float(a['residual_norm'].min()),residual_max=float(a['residual_norm'].max())))
    rows=[]
    for patient,v in sorted(groups.items()):
        n=sum(r['pixels'] for r in v)
        rows.append(dict(patient=patient,slices=len(v),pixels=n,mean_applied_residual_norm=sum(r['residual_sum'] for r in v)/n,mean_feature_norm=sum(r['feature_sum'] for r in v)/n,mean_logit_response_norm=sum(r['logit_sum'] for r in v)/n,mean_slice_residual_spatial_sd=float(np.mean([r['residual_spatial_sd'] for r in v])),min_applied_residual_norm=min(r['residual_min'] for r in v),max_applied_residual_norm=max(r['residual_max'] for r in v)))
    csvout(W/'results/trc_patient_feature_response.csv',rows)
    env=dict(python=platform.python_version(),executable=sys.executable,torch_cuda=torch.version.cuda,cudnn=torch.backends.cudnn.version(),CPU_threads_training=4)
    for name in ['torch','numpy','scipy','skimage','h5py','torchvision','matplotlib','PIL']:
        env[name]=importlib.import_module(name).__version__
    js(W/'configs/environment.json',env)
    print(json.dumps(dict(GT_independence=len(checks),residual_patient_count=len(rows),environment=env)),flush=True)
if __name__=='__main__':main()
