"""Post-training CPU-only aggregation; preserve the existing Full result and checkpoint selection."""
from collections import defaultdict
import csv,json
import numpy as np
from common import *
def readcsv(p,delimiter=','):
    with Path(p).open() as f:return list(csv.DictReader(f,delimiter=delimiter))
def main():
    complete=json.loads((W/'receipts/fixed_test_complete.json').read_text());assert complete['status']=='DONE'
    trace=W/'runs/F97_main_signal_20261013/rectangle_trace.csv';rr=readcsv(trace)
    assert len(rr)==30000 and [int(r['execution_index']) for r in rr]==list(range(1,30001))
    assert all(int(r['fixed_h'])==int(r['fixed_w'])==97 for r in rr)
    counts=defaultdict(int);rare=[]
    for r in rr:
        q=float(r['q']);hyp=max(1,min(255,round(256*(.18+.20*q))));counts[hyp]+=1
        if hyp!=97:rare.append(dict(execution_index=int(r['execution_index']),q=q,hypothetical_original_side=hyp,actual_fixed_side=97))
    q=np.array([float(r['q']) for r in rr])
    distribution=[dict(actual_fixed_side=97,actual_area_ratio=97*97/65536,hypothetical_original_side=s,updates=n,fraction=n/30000,signal_owner='new fixed run EMA, not original Full historical trajectory') for s,n in sorted(counts.items())]
    csvout(W/'results/fixed_run_signal_geometry_distribution.csv',distribution);csvout(W/'results/fixed_run_signal_geometry_rare.csv',rare)
    js(W/'receipts/fixed_trace_summary.json',dict(status='PASS',records=30000,actual_fixed_side=97,q_min=float(q.min()),q_max=float(q.max()),q_mean=float(q.mean()),hypothetical_original_side_counts={str(k):v for k,v in counts.items()},scope='recorded new fixed-run teacher signals; cannot reconstruct the original Full closed-loop trajectory'))
    source=ROOT/'final_report/acdc10_top6_all_test_v21_5/raw_test/X03/case_class_metrics.tsv';old=readcsv(source,'\t');new=readcsv(W/'results/fixed_test_case_class.csv')
    om={(r['case'],r['class_name']):r for r in old};nm={(r['case'],r['class_name']):r for r in new};assert om.keys()==nm.keys() and len(nm)==120
    frame=[];groups=defaultdict(list)
    for case,cls in sorted(nm):
        row=dict(patient=case.split('_')[0],case=case,class_name=cls,main_dice=float(om[case,cls]['dice']),fixed_dice=float(nm[case,cls]['dice']))
        row['fixed_minus_main_dice_pp']=100*(row['fixed_dice']-row['main_dice']);frame.append(row);groups[row['patient'],cls].append(row)
    csvout(W/'results/fixed_vs_main_frame_class.csv',frame)
    patients=[]
    for (pat,cls),v in sorted(groups.items()):
        assert len(v)==2
        patients.append(dict(patient=pat,class_name=cls,frames=2,main_dice=float(np.mean([r['main_dice'] for r in v])),fixed_dice=float(np.mean([r['fixed_dice'] for r in v])),fixed_minus_main_dice_pp=float(np.mean([r['fixed_minus_main_dice_pp'] for r in v]))))
    csvout(W/'results/fixed_vs_main_patient_class.csv',patients)
    macro=[]
    for pat in sorted({r['patient'] for r in patients}):
        v=[r for r in patients if r['patient']==pat];assert len(v)==3
        macro.append(dict(patient=pat,frames=2,classes=3,main_dice=float(np.mean([r['main_dice'] for r in v])),fixed_dice=float(np.mean([r['fixed_dice'] for r in v])),fixed_minus_main_dice_pp=float(np.mean([r['fixed_minus_main_dice_pp'] for r in v]))))
    assert len(macro)==20
    csvout(W/'results/fixed_vs_main_patient_dice.csv',macro)
    assert abs(np.mean([r['fixed_dice'] for r in macro])-complete['metrics']['dice'])<1e-12
    assert abs(np.mean([r['main_dice'] for r in macro])-.9034965765204998)<1e-12
    js(W/'receipts/fixed_patient_aggregation.json',dict(status='PASS',test_frames=40,patients=20,class_records=120,frames_per_patient=2,aggregation='native frame-volume class Dice; average two frames and three classes per patient, then equal-patient mean',main_source=str(source),main_source_sha256=sha(source),main_not_reevaluated=True,mean_main_dice=float(np.mean([r['main_dice'] for r in macro])),mean_fixed_dice=float(np.mean([r['fixed_dice'] for r in macro])),mean_fixed_minus_main_pp=float(np.mean([r['fixed_minus_main_dice_pp'] for r in macro])),patients_fixed_higher=sum(r['fixed_dice']>r['main_dice'] for r in macro),scope='descriptive single-run comparison; no new training, reselection, or test evaluation'))
    print(json.dumps(dict(side_counts=dict(counts),mean_fixed_minus_main_pp=float(np.mean([r['fixed_minus_main_dice_pp'] for r in macro])))))
if __name__=='__main__':main()
