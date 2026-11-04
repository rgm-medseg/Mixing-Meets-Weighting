"""Verify Oct-6 legacy replays against existing COVID baseline receipts, CSV only."""
import csv
import hashlib
import json
import math
from pathlib import Path
from statistics import mean

OUT=Path(__file__).resolve().parents[1]
ROOT=OUT.parent
source=ROOT/'revision_runs_20261006/covid_metrics_before_after.csv'
def read(p):
    with p.open(newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    fields=list(dict.fromkeys(k for r in rows for k in r))
    with (OUT/name).open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)

mapped=read(OUT/'TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv')
cases=read(source)
agg=[r for r in read(OUT/'AGGREGATE_RECOMPUTATION.csv') if r['row_id'] not in ['ROW07','ROW09']]
carried=[]
source_sha=hashlib.sha256(source.read_bytes()).hexdigest()
for row in mapped:
    if row['row_id'] not in ['ROW07','ROW09']:continue
    run=row['historical_run']+'_CONFIRMATORY_TEST_ONCE'
    group=[r for r in cases if r['run']==run and r['policy']=='spacing_only_derived_center_grid']
    assert len(group)==4 and len({r['patient'] for r in group})==4
    for r in group:
        assert r['checkpoint_sha256']==row['checkpoint_sha256_current']
        assert all(r['changed_'+k]=='False' for k in ['prediction','GT','resampling','postprocess'])
        carried.append({'row_id':row['row_id'],'historical_run':row['historical_run'],**r,
                        'case_source_kind':'Existing Oct-6 legacy replay, not a newly found original training-time case table',
                        'source':str(source),'source_sha256':source_sha})
    for metric in ['dice','iou','hd95','asd']:
        key='before_'+metric+('_mm' if metric in ['hd95','asd'] else '')
        values=[float(r[key]) for r in group]
        assert all(math.isfinite(x) for x in values)
        expected=float(row[metric+'_raw']);actual=mean(values)
        assert abs(actual-expected)<1e-12
        agg.append({'row_id':row['row_id'],'source':str(source),'rows':len(group),'metric':metric,
                    'mean_recomputed':actual,'receipt_mean':expected,'absolute_error':abs(actual-expected),
                    'finite':True,'within_1e_12':True})
    assert all(float(r['before_dice'])>0 for r in group)
    row['case_records']=str(source)
    row['case_records_filter']='run='+run+'; policy=spacing_only_derived_center_grid; metrics=before_*'
    row['case_records_origin']='Oct-6 saved-prediction legacy replay, separately identified from original receipt'
write('COVID_BASELINE_CASE_REPLAY_VERIFICATION.csv',carried)
write('TABLE2_4_RUN_CHECKPOINT_SPLIT_EVALUATOR.csv',mapped)
write('AGGREGATE_RECOMPUTATION.csv',agg)
s=json.loads((OUT/'COLLECTION_SUMMARY.json').read_text())
s['aggregate_tests']=len(agg)
s['aggregate_failures']=[r for r in agg if str(r['within_1e_12'])!='True']
s['covid_baseline_case_replay_rows']=len(carried)
(OUT/'COLLECTION_SUMMARY.json').write_text(json.dumps(s,ensure_ascii=False,indent=2)+'\n')
print(json.dumps({'baseline_case_replay_rows':len(carried),'aggregate_tests':len(agg),'replay_origin':'existing Oct-6 records'}))
