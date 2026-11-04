"""Freeze inputs before prediction; instrumentation preserves all original RNG draws."""
import random,pickle
import numpy as np,torch,h5py
from scipy.ndimage import zoom,rotate
from measurement import imagehash
from common import *
import adapter
from dataloaders import dataset as ds
def rng():return random.getstate(),np.random.get_state(),torch.get_rng_state()
def restore(s):random.setstate(s[0]);np.random.set_state(s[1]);torch.set_rng_state(s[2])
def seed(n):random.seed(n);np.random.seed(n);torch.manual_seed(n)
def boundary(g):
 b=np.zeros(g.shape,bool);a=g[1:]!=g[:-1];b[1:]|=a;b[:-1]|=a;a=g[:,1:]!=g[:,:-1];b[:,1:]|=a;b[:,:-1]|=a;return b
def recorded_aug(image,gt):
 meta={};geom={'kind':'identity'};original_rot=ds.random_rot_flip;original_angle=ds.random_rotate;original_cut=ds.cutout_gray;original_params=ds.transforms.ColorJitter.get_params
 def rot(a,b=None):
  r=np.random.RandomState();r.set_state(np.random.get_state());k=int(r.randint(0,4));axis=int(r.randint(0,2));geom.update(kind='rot_flip',k=k,axis=axis);return original_rot(a,b)
 def angle(a,b):
  r=np.random.RandomState();r.set_state(np.random.get_state());deg=int(r.randint(-20,20));geom.update(kind='rotate',angle=deg);return original_angle(a,b)
 def cut(a,b,**kw):
  before=rng();out=original_cut(a,b,**kw);after=rng();restore(before)
  _,ones=original_cut(a,np.ones(b.shape,np.uint8),**kw);restore(after);meta['occlusion']=np.array(ones)==0
  pos=np.argwhere(meta['occlusion']);meta['cutout_rectangle']=[int(pos[:,0].min()),int(pos[:,1].min()),int(pos[:,0].max()+1),int(pos[:,1].max()+1)] if len(pos) else None
  return out
 def params(*args):
  ret=original_params(*args);meta['color_jitter']=[ret[0].tolist(),*[float(x) if x is not None else None for x in ret[1:]]];return ret
 ds.random_rot_flip=rot;ds.random_rotate=angle;ds.cutout_gray=cut;ds.transforms.ColorJitter.get_params=staticmethod(params)
 try:out=ds.WeakStrongAugment([256,256])(dict(image=image.copy(),label=gt.copy()))
 finally:ds.random_rot_flip=original_rot;ds.random_rotate=original_angle;ds.cutout_gray=original_cut;ds.transforms.ColorJitter.get_params=staticmethod(original_params)
 def transform(a):
  if geom['kind']=='rot_flip':a=np.flip(np.rot90(a,geom['k']),axis=geom['axis']).copy()
  elif geom['kind']=='rotate':a=rotate(a,geom['angle'],order=0,reshape=False)
  return zoom(a,(256/a.shape[0],256/a.shape[1]),order=0)
 weak_true=transform(gt);anat=transform(boundary(gt).astype(np.uint8)).astype(bool);valid=transform(np.ones(gt.shape,np.uint8)).astype(bool)
 assert np.array_equal(weak_true.astype(np.uint8),out['label'].numpy())
 occ=meta['occlusion'];strong_label=weak_true.copy();strong_label[occ]=0
 assert np.array_equal(strong_label.astype(np.uint8),out['label_strong'].numpy())
 return out,dict(geometry=geom,cutout_rectangle=meta['cutout_rectangle'],color_jitter=meta.get('color_jitter'),weak_true=weak_true,anat=anat,geom_valid=valid,occlusion=occ)
def main():
 cfg=config();ud=unitdir('INPUTS');complete=ud/'complete.json'
 if complete.exists():return
 samples=[r for r in rows(W/'manifests/fixed_samples.csv') if r['cohort']=='validation'];samples=sorted(samples,key=lambda r:hashlib.sha256(('fixed-pair-v1:'+r['sample']).encode()).hexdigest());assert len(samples)==210
 manifest=[];valid=[];cache={};exposure={r['sample']:0 for r in samples}
 for b in range(9):
  indices=[(24*b+i)%210 for i in range(24)];selected=[samples[i] for i in indices];batch=[]
  for slot,r in enumerate(selected):
   exposure[r['sample']]+=1;n=cfg['a3_seed']+b*24+slot;frame=r['frame'];z=int(r['slice_index'])
   if frame not in cache:
    with h5py.File(r['source']) as f:cache[frame]=(f['image'][:],f['label'][:])
   image,gt=cache[frame][0][z],cache[frame][1][z]
   seed(n);before=rng();literal=ds.WeakStrongAugment([256,256])(dict(image=image.copy(),label=gt.copy()));after=rng();restore(before);out,meta=recorded_aug(image,gt);instrumented_after=rng()
   exact=all(torch.equal(out[k],literal[k]) for k in ['image','image_strong','label','label_strong'])
   rngexact=after[0]==instrumented_after[0] and after[1][0]==instrumented_after[1][0] and np.array_equal(after[1][1],instrumented_after[1][1]) and after[1][2:]==instrumented_after[1][2:] and torch.equal(after[2],instrumented_after[2])
   assert exact and rngexact
   arr=dict(weak=out['image'].numpy().astype(np.float32),strong=out['image_strong'].numpy().astype(np.float32),gt=meta['weak_true'].astype(np.uint8),strong_artificial_gt=out['label_strong'].numpy(),anatomy=meta['anat'],geometry_valid=meta['geom_valid'],occlusion=meta['occlusion'])
   # Strong shape is [1,H,W]; literal weak is [1,H,W]. Keep exact original views.
   dst=W/'state/input_arrays'/identity()/f'b{b}_slot{slot}.npz';dst.parent.mkdir(parents=True,exist_ok=True);tmp=dst.with_suffix('.tmp.npz');np.savez_compressed(tmp,**arr);os.replace(tmp,dst)
   role=['img_a','img_b','uimg_a','uimg_b'][slot//6];entry=dict(batch=b,slot=slot,role=role,role_index=slot%6,sample=r['sample'],patient=r['patient'],frame=frame,slice_index=z,source=r['source'],source_sha256=r['source_sha256'],analysis_seed=n,input_array_path=str(dst),input_array_sha256=sha(dst),weak_sha256=imagehash(arr['weak']),strong_sha256=imagehash(arr['strong']),gt_sha256=imagehash(arr['gt']),geometry=json.dumps(meta['geometry']),cutout_rectangle=json.dumps(meta['cutout_rectangle']),color_jitter=json.dumps(meta['color_jitter']),literal_augmentation_exact=exact,RNG_after_exact=rngexact,simulated_role=True)
   batch.append(entry);valid.append(entry)
  js(ud/f'batch_{b}.json',dict(identity=identity(),batch=b,slots=batch,position_uniform_y=((b*37+19)%997)/997,position_uniform_x=((b*71+43)%991)/991,matched_rectangle=92))
 for r in valid:r['sample_total_exposures']=exposure[r['sample']]
 csvout(W/'results/A3_input_pair_manifest.csv',valid)
 js(W/'results/A3_alignment_receipt.json',dict(status='PASS',literal_aug_comparisons=216,outputs_exact=True,RNG_after_exact=True,unique_slices=210,unique_frames=20,unique_patients=10,exposures=216,repeat_exposures=6,batches=9,batch_size=24,roles='img_a6 img_b6 uimg_a6 uimg_b6; val simulated roles',pairing='unl: uimg_a outside/img_a inside; labeled: img_b outside/uimg_b inside; student1 weak and student2 strong share source composition',geometry_shared=True,cutout_tracked=True,cutout_artificial_labels_zero=True,GT_use='only post-hoc anatomy/error; controller and w receive no GT',GT_true='anatomical weak transformed GT retained; strong artificial GT recorded separately; errors exclude occlusion and rotation padding',seed=cfg['a3_seed'],identity=identity()))
 js(complete,dict(status='DONE',identity=identity(),input_units=216,source=sha(W/'scripts/frozen/train_ACDC/dataloaders/dataset.py')))
 print('A3 INPUTS FROZEN AND EXACT',flush=True)
if __name__=='__main__':main()
