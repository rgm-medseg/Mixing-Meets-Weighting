"""Recompute ten saved binary masks on CPU; no inference or source mutation."""
import ast
import csv
import hashlib
import json
import time
from pathlib import Path

import numpy as np
from medpy import metric

ROOT=Path(__file__).resolve().parents[2]
OUT=Path(__file__).resolve().parents[1]
SOURCE=ROOT/'audits/run_v19_5_promise20_and_mscmr_final_r3.py'
DATA=ROOT/'data_staging/promise12/20260624_141558/data_promise12'
PRED=ROOT/'formal_test/promise20_v19_5/PROMISE20_BASELINE_BETA_FFT_R1/test/predictions_npz'

def sha(p):
    h=hashlib.sha256()
    with p.open('rb') as f:
        for b in iter(lambda:f.read(1024*1024),b''):h.update(b)
    return h.hexdigest()

tree=ast.parse(SOURCE.read_text())
functions=[n for n in tree.body if isinstance(n,ast.FunctionDef) and n.name=='binary_metrics']
ns={'np':np,'metric':metric}
exec(compile(ast.Module(body=functions,type_ignores=[]),str(SOURCE),'exec'),ns)
start=time.monotonic();rows=[]
for case in DATA.joinpath('test.list').read_text().splitlines():
    case=case.strip().split('.')[0]
    if not case:continue
    pp=PRED/(case+'_prediction.npz');gt=DATA/'npy_image'/(case+'_segmentation.npy')
    with np.load(pp) as d:
        pred=d['prediction'];label_shape=tuple(d['label_shape'].tolist())
    truth=np.load(gt,mmap_mode='r')
    assert tuple(pred.shape)==tuple(truth.shape)==label_shape
    vals=ns['binary_metrics'](pred,truth)
    rows.append({'case':case,'unit':'grid','prediction_shape':str(pred.shape),'stored_label_shape':str(label_shape),
                 'prediction_positive_pixels':int(np.count_nonzero(pred)),'GT_positive_pixels':int(np.count_nonzero(truth)),
                 **vals,'prediction_source':str(pp),'prediction_sha256':sha(pp),'GT_source':str(gt),'GT_sha256':sha(gt),
                 'metric_source':str(SOURCE),'metric_source_sha256':sha(SOURCE),
                 'scope':'New CPU recomputation of existing saved masks using isolated located historical metric function; no checkpoint selection or inference.'})
    print(case,vals,flush=True)
with (OUT/'PROMISE_BASELINE_SAVED_PRED_CPU_RECALC.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
macro={k:float(np.mean([r[k] for r in rows])) for k in ['dice','iou','hd95','asd']}
prior=json.loads((ROOT/'audits/PROMISE20_FROZEN_BEST_TEST_IDENTITY_AUDIT.json').read_text())['baseline_test']
receipt={'cases':len(rows),'seconds':time.monotonic()-start,'macro':macro,
         'historical_precision_record':{k:prior[k] for k in macro},
         'absolute_errors':{k:abs(macro[k]-prior[k]) for k in macro},
         'all_masks_nonempty':all(r['prediction_positive_pixels']>0 and r['GT_positive_pixels']>0 for r in rows),
         'all_metrics_finite':all(np.isfinite(r[k]) for r in rows for k in macro),
         'checkpoint':prior['checkpoint_path'],'checkpoint_sha256_recorded':prior['checkpoint_sha256'],
         'checkpoint_sha256_current':sha(ROOT/prior['checkpoint_path']),
         'prediction_grid':'existing staged (Z,224,224); no resampling','postprocessing':'none added',
         'empty_rule':'historical binary_metrics: both_empty Dice/IoU1, distance0; one_empty Dice/IoU0, distancesinf',
         'historical_import_binding':'Located historical script/current bytes; old receipt does not freeze imported dependencies. CPU metric compatibility is not full runtime provenance.',
         'model_created':False,'forward_called':False,'checkpoint_reselection':False,'predictions_modified':False,'source_modified':False}
(OUT/'PROMISE_BASELINE_CPU_RECALC_RECEIPT.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps(receipt,ensure_ascii=False),flush=True)
