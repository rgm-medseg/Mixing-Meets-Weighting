"""Replay legacy metrics and separately correct demonstrated COVID geometry.

No training. Historical evaluator reconstructs XYZ with stack(axis=2): axis
order is correct, but in-plane resize spacing was not updated. Native-grid reevaluation upsamples saved categorical predictions
with nearest SciPy mapping and uses canonical GT and spacing (Z,X,Y).
Legacy values are never overwritten or silently relabeled.
"""
import json
import shutil
import sys
from pathlib import Path
import nibabel as nib
import numpy as np
import torch
from scipy.ndimage import zoom
from common import *
bootstrap()
from train_COVID19.covid19_volume_evaluator import binary_volume_metrics

def load_volumes():
    manifest=PROJECT/'data/COVID19_OFFICIAL20/manifests/TEST_ALL_SLICES.tsv'
    grouped={}
    for row in read_table(manifest): grouped.setdefault(row['patient_id'],[]).append(row)
    volumes={}
    for patient,rows in grouped.items():
        rows=sorted(rows,key=lambda r:int(r['slice_idx']))
        assert [int(r['slice_idx']) for r in rows]==list(range(len(rows)))
        images=[]; targets=[]; legacy_spacing=None
        for row in rows:
            path=Path(row['derived_path']); path=path if path.is_absolute() else PROJECT/path
            payload=np.load(path)
            images.append(payload['image']); targets.append(payload['mask'])
            current=tuple(float(v) for v in payload['spacing'])
            if legacy_spacing is None: legacy_spacing=current
            assert current==legacy_spacing
        native_path=PROJECT/rows[0]['mask_path']
        nii=nib.as_closest_canonical(nib.load(native_path))
        native_xyz=np.asarray(nii.dataobj)>0
        native_gt=native_xyz.transpose(2,0,1)
        sx,sy,sz=(float(v) for v in nii.header.get_zooms()[:3])
        derived=np.stack(targets)
        native_spacing=(sz,sx,sy)
        # scipy zoom(grid_mode=False) maps first/last pixel centers exactly.
        corrected_spacing=(sz,sx*(native_xyz.shape[0]-1)/(derived.shape[1]-1),sy*(native_xyz.shape[1]-1)/(derived.shape[2]-1))
        volumes[patient]=dict(image=np.stack(images),derived_gt=derived,native_gt=native_gt,legacy_spacing=legacy_spacing,
                              native_spacing=native_spacing,derived_corrected_spacing=corrected_spacing,
                              native_gt_path=str(native_path),native_gt_sha256=sha(native_path),canonical_shape=list(native_xyz.shape))
    return manifest,volumes

def load_full_model(receipt):
    dst=WORK/'source/promise_runtime/networks'
    if not dst.exists():
        shutil.copytree(PROJECT/'train_promise12/networks',dst,ignore=shutil.ignore_patterns('__pycache__','*.pth'))
    sys.path.insert(0,str(dst.parent))
    from networks.net_factory import net_factory
    cp=PROJECT/receipt['CheckpointPath']
    assert sha(cp)==receipt['CheckpointSHA']
    model=net_factory(net_type='unet',in_chns=1,class_num=2,use_mask_token_prehead_router=True,mask_token_num=4,mask_token_dim=16)
    model.load_state_dict(torch.load(cp,map_location='cpu',weights_only=False),strict=True)
    model.eval()
    return model,cp

@torch.inference_mode()
def predict(model,images):
    output=np.zeros(images.shape,np.uint8)
    for z in range(len(images)):
        x=torch.from_numpy(images[z]).float()[None,None].cuda()
        output[z]=model(x).softmax(1).argmax(1)[0].cpu().numpy()
    return output

def main():
    receipt_path=WORK/'reports/covid_reevaluation_receipt.json'
    if receipt_path.exists():
        old=read_json(receipt_path)
        if old.get('geometry_policy_version')=='resize_spacing_v2_XYZ_reconstruction_verified' and old['status']=='DONE':
            print('Completed identical COVID three-policy reevaluation reused',flush=True); return
    manifest,volumes=load_volumes()
    records=[]; geometry=[]
    for patient,v in volumes.items():
        geometry.append(dict(patient=patient,derived_shape=json.dumps(list(v['derived_gt'].shape)),native_shape=json.dumps(list(v['native_gt'].shape)),
                        legacy_spacing_XYZ_unscaled=json.dumps(v['legacy_spacing']),derived_corrected_spacing_ZXY=json.dumps(v['derived_corrected_spacing']),
                        native_spacing_ZXY=json.dumps(v['native_spacing']),native_gt_path=v['native_gt_path'],native_gt_sha256=v['native_gt_sha256']))
    write_csv(WORK/'reports/covid_geometry_audit.csv',geometry)
    for percent in (5,10):
        folder=PROJECT/f'formal_test/covid19_seed1337/COVID19_{percent}P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE'
        receipt=read_json(folder/'FORMAL_TEST_RECEIPT.json')
        predictions={}
        for entry in receipt['prediction_paths']:
            path=Path(entry['path']); assert sha(path)==entry['output_sha256']
            predictions[entry['patient_id']]=dict(pred=np.load(path)['prediction'],path=path,checkpoint=receipt['checkpoint_path'],checkpoint_sha256=receipt['checkpoint_sha256'],
                                                   original=receipt['metric_result']['per_patient'][entry['patient_id']])
        full_receipt=read_json(PROJECT/f'final_report/three_innovation_postfull_test_v22_1/raw_test/COVID{percent}_FINAL3/TEST_RESULT_FULL_PRECISION.json')
        original_full=read_json(PROJECT/f'final_report/three_innovation_postfull_test_v22_1/raw_test/COVID{percent}_FINAL3/patient_metrics.json')['per_patient']
        destination=WORK/f'runs/COVID{percent}_Full/evaluation/predictions'
        destination.mkdir(parents=True,exist_ok=True)
        needed=[p for p in volumes if not (destination/f'{p}.npz').exists()]
        if needed:
            model,checkpoint=load_full_model(full_receipt)
            for patient in needed:
                pred=predict(model,volumes[patient]['image'])
                np.savez_compressed(destination/f'{patient}.npz',prediction=pred,patient_id=patient,checkpoint=str(checkpoint),checkpoint_sha256=full_receipt['CheckpointSHA'])
            del model; torch.cuda.empty_cache()
        for method in ('baseline','Full'):
            for patient,v in volumes.items():
                if method=='baseline': p=predictions[patient]
                else:
                    path=destination/f'{patient}.npz'
                    p=dict(pred=np.load(path)['prediction'],path=path,checkpoint=str(PROJECT/full_receipt['CheckpointPath']),checkpoint_sha256=full_receipt['CheckpointSHA'],original=original_full[patient])
                assert p['pred'].shape==v['derived_gt'].shape
                native_pred=np.stack([zoom(s,(v['native_gt'].shape[1]/s.shape[0],v['native_gt'].shape[2]/s.shape[1]),order=0) for s in p['pred']])
                policies=[('legacy_replay',p['pred'].transpose(1,2,0),v['derived_gt'].transpose(1,2,0),v['legacy_spacing']),
                          ('corrected_derived_center_grid',p['pred'],v['derived_gt'],v['derived_corrected_spacing']),
                          ('corrected_native_grid',native_pred,v['native_gt'],v['native_spacing'])]
                for policy,pred,gt,spacing in policies:
                    m=binary_volume_metrics(pred>0,gt>0,spacing)
                    records.append(dict(run=f'COVID{percent}_{method}',patient=patient,policy=policy,prediction=str(p['path']),prediction_sha256=sha(p['path']),
                        checkpoint=p['checkpoint'],checkpoint_sha256=p['checkpoint_sha256'],spacing=json.dumps(spacing),unit='legacy_mm_label_geometrically_inconsistent' if policy=='legacy_replay' else 'mm',
                        **{k:float(vv) if np.isfinite(vv) else '' for k,vv in m.items()},
                        maximum_difference_from_original=max(abs(m[k]-p['original'][k]) for k in m) if policy=='legacy_replay' else 'NA'))
                print(percent,method,patient,'evaluated three frozen policies',flush=True)
    write_csv(WORK/'reports/covid_existing_prediction_reevaluation.csv',records)
    macro=[]
    for run in sorted({r['run'] for r in records}):
        for policy in sorted({r['policy'] for r in records}):
            data=[r for r in records if r['run']==run and r['policy']==policy]
            macro.append(dict(run=run,policy=policy,patients=len(data),**{k:float(np.mean([r[k] for r in data])) if all(isinstance(r[k],float) for r in data) else 'NA' for k in ('dice','iou','hd95_mm','asd_mm')}))
    write_csv(WORK/'reports/covid_reevaluation_macro.csv',macro)
    atomic_json(WORK/'reports/covid_reevaluation_receipt.json',dict(status='DONE',case_records=len(records),manifest=str(manifest),manifest_sha256=sha(manifest),
                  evaluator=str(WORK/'source/train_COVID19/covid19_volume_evaluator.py'),evaluator_sha256=sha(WORK/'source/train_COVID19/covid19_volume_evaluator.py'),
                  training_started=False,legacy_maximum_metric_difference=max(float(r['maximum_difference_from_original']) for r in records if r['policy']=='legacy_replay'),
                  actual_geometry_problem='in-plane resize512to256 retained original spacing; axis2 reconstruction correctly restores XYZ',
                  geometry_policy_version='resize_spacing_v2_XYZ_reconstruction_verified',script_sha256=sha(__file__),time=now()))

if __name__=='__main__': main()
