"""Recompute existing R2 class summary with CPU CSV parsing only."""
import csv
import json
from pathlib import Path
from statistics import mean

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
R2=ROOT/'tcsmix_control_revision_20261006/formal_R2_fixed92_20261007/results'
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))

cases=read(R2/'case_class_metrics.csv')
original=read(R2/'R2_class_metrics.csv')
rows=[]
for r in original:
    group=[x for x in cases if x['class_name']==r['class_name']]
    assert len(group)==40
    out={'class_name':r['class_name'],'case_class_rows':len(group),'source':str(R2/'R2_class_metrics.csv')}
    for metric in ['dice','iou','hd95','asd']:
        a=mean(float(x[metric]) for x in group);b=float(r[metric])
        out[metric+'_recomputed']=a;out[metric+'_original']=b;out[metric+'_absolute_error']=abs(a-b)
        assert abs(a-b)<1e-12
    rows.append(out)
with (OUT/'R2_CLASS_RECOMPUTATION.csv').open('w',newline='') as f:
    w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
receipt=json.loads((OUT/'R2_RECORD_VERIFICATION.json').read_text())
receipt['class_summary_rows']=len(rows)
receipt['class_table_max_error']=max(r[m+'_absolute_error'] for r in rows for m in ['dice','iou','hd95','asd'])
receipt['class_summary_source']=str(R2/'R2_class_metrics.csv')
(OUT/'R2_RECORD_VERIFICATION.json').write_text(json.dumps(receipt,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'class_rows':len(rows),'max_error':receipt['class_table_max_error']}))
