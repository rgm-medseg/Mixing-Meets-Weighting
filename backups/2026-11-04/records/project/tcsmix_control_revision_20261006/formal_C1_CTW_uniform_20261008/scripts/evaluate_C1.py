"""One final test evaluation of frozen validation-selected student1 only."""
import h5py
import numpy as np
from common import *
from unified_evaluation import load_model,infer_volume,metrics

def main():
    selection=read_json(WORK/'reports/SELECTED_CHECKPOINT.json');cfg=read_json(WORK/'configs/C1_CTW_mean_uniform_v2B.yaml')
    cp=selection['checkpoint'];ch=selection['sha256'];assert sha(cp)==ch
    receipt=WORK/'results/EVALUATION_RECEIPT.json'
    if receipt.exists():
        old=read_json(receipt);assert old['checkpoint_sha256']==ch
        if old['status']=='EVALUATED':print('Reused identical completed final evaluation');return
    from resources import bind_gpu_pid
    bind_gpu_pid()
    model,ck=load_model(cfg,cp,'model_1');rows=[];index=[]
    cases=(DATA/'test.list').read_text().splitlines();assert len(cases)==40
    for case in cases:
        source=DATA/'data'/f'{case}.h5'
        with h5py.File(source,'r') as f:image=f['image'][:];gt=f['label'][:]
        dst=WORK/'results/predictions'/f'{case}.npz';dst.parent.mkdir(exist_ok=True)
        if dst.exists():
            with np.load(dst) as saved:
                assert str(saved['checkpoint_sha256'])==ch and str(saved['case'])==case
                pred=saved['prediction']
        else:
            pred=infer_volume(model,image);tmp=dst.with_suffix('.npz.tmp')
            with tmp.open('wb') as f:np.savez_compressed(f,prediction=pred,case=case,run='C1_CTW_mean_uniform_v2B',checkpoint_sha256=ch)
            os.replace(tmp,dst)
        assert pred.shape==gt.shape
        index.append(dict(case=case,patient=case.split('_')[0],prediction=str(dst),prediction_sha256=sha(dst),GT_source=str(source),GT_source_sha256=sha(source),
            array_axes='z,y,x',original_shape='x'.join(map(str,gt.shape)),evaluation_shape='x'.join(map(str,pred.shape)),
            evaluation_spacing='1,1,1',spacing_meaning='reconstruction grid, NOT physical mm',postprocessing='none',checkpoint_sha256=ch))
        for c,name in [(1,'RV'),(2,'MYO'),(3,'LV')]:
            rows.append(dict(run='C1_CTW_mean_uniform_v2B',case=case,patient=case.split('_')[0],class_id=c,class_name=name,unit='grid',**metrics(pred==c,gt==c)))
        print('Final selected-checkpoint test case',case,flush=True)
    keys=['dice','iou','hd95','asd']
    write_csv(WORK/'results/case_class_metrics.csv',rows)
    classrows=[dict(run='C1_CTW_mean_uniform_v2B',class_name=name,unit='grid',**{k:float(np.mean([r[k] for r in rows if r['class_name']==name])) for k in keys}) for name in ['RV','MYO','LV']]
    write_csv(WORK/'results/C1_class_metrics.csv',classrows)
    caserows=[dict(run='C1_CTW_mean_uniform_v2B',case=c,patient=c.split('_')[0],unit='grid',**{k:float(np.mean([r[k] for r in rows if r['case']==c])) for k in keys}) for c in cases]
    write_csv(WORK/'results/C1_case_metrics.csv',caserows)
    patientrows=[dict(run='C1_CTW_mean_uniform_v2B',patient=p,frames=len([r for r in caserows if r['patient']==p]),unit='grid',
        **{k:float(np.mean([r[k] for r in caserows if r['patient']==p])) for k in keys}) for p in sorted({r['patient'] for r in caserows})]
    write_csv(WORK/'results/C1_patient_metrics.csv',patientrows);write_csv(WORK/'results/prediction_index.csv',index)
    summary={k:float(np.mean([r[k] for r in rows])) for k in keys}
    write_csv(WORK/'results/C1_results.csv',[dict(run='C1_CTW_mean_uniform_v2B',status='EVALUATED',checkpoint=cp,checkpoint_sha256=ch,selected_step=ck['global_step'],
        cases=40,patients=20,unit='grid',**summary)])
    atomic_json(receipt,dict(status='EVALUATED',time=now(),checkpoint=cp,checkpoint_sha256=ch,selected_step=ck['global_step'],
        test_manifest=str(DATA/'test.list'),test_manifest_sha256=sha(DATA/'test.list'),metrics=summary,protocol='ACDC_v22_grid_directed_ASD',
        average='mean RV/MYO/LV and 40 reconstructed ED/ES frame volumes; 20-patient paired results separately',
        unit='HD95 and directed pred-to-GT ASD are reconstruction grid distances, not mm',postprocessing='none',
        empty_both='all four0',empty_one='Dice/IoU0,HD95/ASD sqrt(Nz²+Ny²+Nx²)',resize='nearest order0 to256² and back to native in-plane grid',
        case_axis='z untouched',test_selection=False))
    print(json.dumps(summary))

if __name__=='__main__':main()
