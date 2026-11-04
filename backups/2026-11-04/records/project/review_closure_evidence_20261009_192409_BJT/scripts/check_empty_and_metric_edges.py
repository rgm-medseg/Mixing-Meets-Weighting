"""CPU checks of saved mask counts and isolated scalar metric functions only."""
import ast
import csv
import hashlib
import json
import warnings
from pathlib import Path

import h5py
import numpy as np
from medpy import metric
from scipy.ndimage import binary_erosion, distance_transform_edt

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

def writecsv(p,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with p.open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

rows=list(csv.DictReader((OUT/'TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv').open()))
lookup={r['row_id']:r for r in rows}
zero=list(csv.DictReader((OUT/'ZERO_OVERLAP_CASES.csv').open()))
counts=[];mask_cache={}
for z in zero:
    row=lookup[z['row_id']]
    if row['dataset']!='ACDC':
        counts.append({**z,'confirmed_status':'NOT_CHECKED_OUTSIDE_ACDC'})
        continue
    job=row['historical_run']
    pred=ROOT/'external_comparators/2026_reproduction/predictions'/job/(z['case']+'.npz')
    config=json.loads(Path(row['config_path']).read_text())
    truth=Path(config['data']['root'])/'data'/(z['case']+'.h5')
    key=(str(pred),str(truth))
    if not pred.exists() or not truth.exists():
        counts.append({**z,'confirmed_status':'NOT_FOUND_SAVED_MASK','prediction_source':str(pred),'GT_source':str(truth)})
        continue
    if key not in mask_cache:
        with np.load(pred) as d:p=d['prediction']
        with h5py.File(truth,'r') as h:t=h['label'][:]
        mask_cache[key]=(p,t,sha(pred),sha(truth))
    p,t,ps,ts=mask_cache[key]
    cls={'RV':1,'MYO':2,'LV':3}[z['class']]
    pn=int(np.count_nonzero(p==cls));tn=int(np.count_nonzero(t==cls))
    status='BOTH_EMPTY' if pn==tn==0 else ('PRED_EMPTY_GT_NONEMPTY' if pn==0 else ('GT_EMPTY_PRED_NONEMPTY' if tn==0 else 'BOTH_NONEMPTY_ZERO_OVERLAP'))
    penalty=float(np.linalg.norm(p.shape))
    counts.append({**z,'confirmed_status':status,'pred_class_pixels':pn,'GT_class_pixels':tn,
                   'prediction_shape':str(p.shape),'GT_shape':str(t.shape),'shape_matches':p.shape==t.shape,
                   'grid_diagonal_penalty':repr(penalty),'single_empty_rule_matches_record':(float(z['hd95'])==penalty and float(z['asd'])==penalty) if status in ['PRED_EMPTY_GT_NONEMPTY','GT_EMPTY_PRED_NONEMPTY'] else '',
                   'prediction_source':str(pred),'prediction_sha256':ps,'GT_source':str(truth),'GT_sha256':ts,
                   'handling':'Retain original finite shape-norm penalty and include in ordinary macro; no exclusion.'})
writecsv(OUT/'EMPTY_CASE_COUNTS_FROM_SAVED_MASKS.csv',counts)

# Compile only pure metric functions from their AST. Do not import a training/evaluation entry.
functions=[
    ('ACDC','tools/v22_0_acdc_test_frozen_checkpoint.py',['safe_metrics'],'safe_metrics'),
    ('PROMISE12','tools/v22_1_postfull_test_launcher.py',['promise_binary_metrics'],'promise_binary_metrics'),
    ('COVID-19','train_COVID19/covid19_volume_evaluator.py',['_surface','surface_distances_mm','binary_volume_metrics'],'binary_volume_metrics')]
edge=[]
empty=np.zeros((2,3,4),dtype=bool);one=empty.copy();one[0,1,2]=True
for ds,rel,names,fn in functions:
    p=ROOT/rel;tree=ast.parse(p.read_text());nodes=[n for n in tree.body if isinstance(n,(ast.FunctionDef,ast.AsyncFunctionDef)) and n.name in names]
    namespace={'np':np,'metric':metric,'binary_erosion':binary_erosion,'distance_transform_edt':distance_transform_edt}
    exec(compile(ast.Module(body=nodes,type_ignores=[]),str(p),'exec'),namespace)
    for case,a,b in [('both_empty',empty,empty),('prediction_empty',empty,one),('GT_empty',one,empty),('perfect_nonempty',one,one)]:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter('always')
            result=namespace[fn](a,b,(1.,1.,1.)) if ds=='COVID-19' else namespace[fn](a,b)
        if isinstance(result,tuple):result=dict(zip(['dice','iou','hd95','asd'],result))
        edge.append({'dataset':ds,'case':case,'source':str(p),'source_sha256':sha(p),'function':fn,
                     'numpy_version':np.__version__,'result':json.dumps(result,allow_nan=True),
                     'warning_messages':' | '.join(str(w.message) for w in caught),
                     'scope':'Current isolated metric-function CPU check; not a historical patient result or historical NumPy version proof.'})
writecsv(OUT/'METRIC_EMPTY_EDGE_CPU_CHECK.csv',edge)
summary={'numpy_version':np.__version__,'zero_overlap_records':len(zero),
         'verified_saved_mask_cases':len(mask_cache),'single_empty_class_rows':sum(r.get('confirmed_status') in ['PRED_EMPTY_GT_NONEMPTY','GT_EMPTY_PRED_NONEMPTY'] for r in counts),
         'both_nonempty_zero_overlap_rows':sum(r.get('confirmed_status')=='BOTH_NONEMPTY_ZERO_OVERLAP' for r in counts),
         'all_single_empty_penalties_match':all(r['single_empty_rule_matches_record'] for r in counts if r.get('confirmed_status') in ['PRED_EMPTY_GT_NONEMPTY','GT_EMPTY_PRED_NONEMPTY']) if any(r.get('confirmed_status') in ['PRED_EMPTY_GT_NONEMPTY','GT_EMPTY_PRED_NONEMPTY'] for r in counts) else None,
         'model_created':False,'forward_called':False,'predictions_changed':False,'images_copied':False}
(OUT/'EMPTY_CASE_CHECK_RECEIPT.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(summary,ensure_ascii=False))
print('COVID edges:',[r['result'] for r in edge if r['dataset']=='COVID-19'])
