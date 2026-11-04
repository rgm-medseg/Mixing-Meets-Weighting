"""Evaluate only the new run, after validation-best is frozen, using original test metric functions."""
import ast,h5py,numpy as np,torch
from scipy.ndimage import zoom
from medpy import metric
from common import *
def main():
    torch.set_num_threads(4)
    receipt=json.loads((W/'receipts/fixed_training_complete.json').read_text());assert receipt['updates_per_student']==30000
    cfg=json.loads((W/'configs/fixed_training_args.json').read_text());cp=Path(cfg['run_root'])/'checkpoints/best.ckpt';assert sha(cp)==receipt['best_sha256']
    m,load_receipt=load('model_1',str(cp));before=digest(m)
    evaluator=ROOT/'tools/v21_5_acdc10_test_frozen_checkpoint.py'
    fn=[n for n in ast.parse(evaluator.read_text()).body if isinstance(n,ast.FunctionDef) and n.name=='safe_metrics'];ns=dict(np=np,metric=metric);exec(compile(ast.Module(body=fn,type_ignores=[]),str(evaluator),'exec'),ns)
    cases=sorted(DATA.joinpath('test.list').read_text().split());js(W/'receipts/fixed_test_selection_lock.json',dict(checkpoint=str(cp),sha256=sha(cp),step=receipt['best_step'],selection='validation best; frozen before new test access',test_manifest_sha256=sha(DATA/'test.list')))
    rows=[]
    for case in cases:
        with h5py.File(DATA/'data'/f'{case}.h5') as f:image=f['image'][:];gt=f['label'][:]
        pred=np.zeros_like(gt)
        with torch.no_grad():
            for z,a in enumerate(image):
                h,w=a.shape;x=zoom(a,(256/h,256/w),order=0);out=m(torch.from_numpy(x).float()[None,None].cuda()).softmax(1).argmax(1)[0].cpu().numpy();pred[z]=zoom(out,(h/256,w/256),order=0)
        dst=W/'arrays/fixed_test';dst.mkdir(parents=True,exist_ok=True);np.savez_compressed(dst/f'{case}.npz',prediction=pred)
        for cls,name in [(1,'RV'),(2,'MYO'),(3,'LV')]:
            vals=ns['safe_metrics'](pred==cls,gt==cls);rows.append(dict(case=case,patient=case.split('_')[0],class_name=name,unit='grid',**dict(zip(['dice','iou','hd95','asd'],vals))))
        print('fixed test',case,flush=True)
    assert digest(m)==before
    csvout(W/'results/fixed_test_case_class.csv',rows)
    cr=[dict(class_name=c,**{k:float(np.mean([r[k] for r in rows if r['class_name']==c])) for k in ['dice','iou','hd95','asd']}) for c in ['RV','MYO','LV']];csvout(W/'results/fixed_test_class.csv',cr)
    summary={k:float(np.mean([r[k] for r in rows])) for k in ['dice','iou','hd95','asd']}
    js(W/'receipts/fixed_test_complete.json',dict(status='DONE',load=load_receipt,cases=len(cases),evaluator=str(evaluator),evaluator_sha256=sha(evaluator),selection_lock='fixed_test_selection_lock.json',metrics=summary,postprocessing='literal original argmax + nearest native-grid resize; no student LCC'))
    print(json.dumps(summary),flush=True)
if __name__=='__main__':main()
