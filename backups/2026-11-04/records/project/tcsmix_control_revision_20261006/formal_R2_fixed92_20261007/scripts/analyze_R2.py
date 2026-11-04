"""CPU-only fixed-area audit and descriptive four-method comparisons, all cases retained."""
import collections
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
KEYS=['dice','iou','hd95','asd']

def save(fig,name):
    for ext in ['png','svg']:fig.savefig(WORK/f'analysis/{name}.{ext}',dpi=160,bbox_inches='tight')
    plt.close(fig)

def trajectory():
    summaries=[];final=[];series={}
    for method,path in [('R1_v2_B',PARENT_R1/'results/R1_control_trace.csv'),('R2_fixed92',WORK/'results/R2_control_trace.csv')]:
        rows=read_table(path) if path.exists() else [];latest={int(r['step']):r for r in rows};ordered=[latest[k] for k in sorted(latest)]
        series[method]=ordered
        if method=='R2_fixed92':final=ordered;raw_n=len(rows)
        for lo,hi in [(1,5000),(5001,10000),(10001,15000),(15001,20000),(20001,25000),(25001,30000),(1,30000)]:
            rr=[r for r in ordered if lo<=int(r['step'])<=hi];ds=[int(r['rectangle_h']) for r in rr];counts=collections.Counter(ds)
            summaries.append(dict(method=method,segment_start=lo,segment_end=hi,n_recorded=len(rr),expected=hi-lo+1,coverage='COMPLETE' if len(rr)==hi-lo+1 else 'PARTIAL',sizes=json.dumps(dict(sorted(counts.items()))),n_sizes=len(counts),modal_fraction=max(counts.values())/len(rr) if rr else 'NA',q_B_min=min(float(r['q_B']) for r in rr) if rr else 'NA',q_B_max=max(float(r['q_B']) for r in rr) if rr else 'NA',q_B_mean=float(np.mean([float(r['q_B']) for r in rr])) if rr else 'NA',actual_area_mean=float(np.mean([float(r['actual_area']) for r in rr])) if rr else 'NA',actual_area_min=min(float(r['actual_area']) for r in rr) if rr else 'NA',actual_area_max=max(float(r['actual_area']) for r in rr) if rr else 'NA',size_changes=sum(x!=y for x,y in zip(ds,ds[1:])),fallback=sum(r['fallback']=='True' for r in rr),scope='actual logged iterations; q_B is diagnostic only in R2'))
    write_csv(WORK/'analysis/scale_distribution.csv',summaries)
    errors=[]
    for r1 in summaries:
        if r1['method']!='R1_v2_B':continue
        r2=next(r for r in summaries if r['method']=='R2_fixed92' and r['segment_start']==r1['segment_start'] and r['segment_end']==r1['segment_end'])
        errors.append(dict(segment_start=r1['segment_start'],segment_end=r1['segment_end'],R1_mean_actual_area=r1['actual_area_mean'],R2_recorded_mean_area=r2['actual_area_mean'],R2_locked_area=8464/65536,signed_area_error=8464/65536-r1['actual_area_mean'],error_percentage_points=100*(8464/65536-r1['actual_area_mean']),relative_error=(8464/65536-r1['actual_area_mean'])/r1['actual_area_mean'],interpretation='full-run mean match does not match each stage or individual step'))
    write_csv(WORK/'analysis/area_matching_by_stage.csv',errors)
    fig,axes=plt.subplots(2,2,figsize=(11,7),layout='constrained')
    for method,rr in series.items():
        if not rr:continue
        steps=[int(r['step']) for r in rr]
        axes[0,0].plot(steps,[int(r['rectangle_h']) for r in rr],label=method,linewidth=.5)
        axes[0,1].plot(steps,[100*float(r['actual_area']) for r in rr],label=method,linewidth=.5)
        axes[1,0].plot(steps,[float(r['q_B']) for r in rr],label=method,linewidth=.5)
    for r in summaries:
        if r['method']=='R1_v2_B' and r['segment_end']-r['segment_start']==4999:
            counts=json.loads(r['sizes']);axes[1,1].plot([int(x) for x in counts],[n/r['n_recorded'] for n in counts.values()],marker='o',label=f"{r['segment_start']}-{r['segment_end']}")
    axes[1,1].axvline(92,color='black',linestyle='--',label='R2 fixed92')
    for ax,title in zip(axes.ravel(),['Actual side (pixels)','Actual area (%)','Foreground balanced margin q_B','R1 predefined stage side distribution']):ax.set_title(title);ax.legend(fontsize=7)
    for ax in axes.ravel()[:3]:ax.set_xlabel('Joint iteration')
    axes[1,1].set(xlabel='Side (pixels)',ylabel='Stage fraction')
    fig.suptitle('R1 dynamic vs R2 fixed area | independent training trajectories');save(fig,'dynamic_fixed_trajectory')
    stable=[float(r['step_seconds']) for r in final if int(r['step'])>50 and int(r['step'])%200 not in range(0,11)]
    segments=read_json(WORK/'state/train_segments.json') if (WORK/'state/train_segments.json').exists() else []
    atomic_json(WORK/'analysis/runtime_cost.json',dict(recorded_logical_steps=len(final),raw_trace_rows=raw_n,training_active_segment_seconds=sum(r['elapsed_seconds'] for r in segments),stable_timer_median_seconds=float(np.median(stable)) if stable else 'NA',timer_scope='step_seconds excludes batch waiting, following validation/EMA/checkpoint; end-to-end active segments include all of them',peak_allocated_bytes=max(int(r['peak_cuda_bytes']) for r in final) if final else 'NA',median_controller_wall_ms=float(np.median([float(r['controller_ms']) for r in final])) if final else 'NA',host='n1',gpu='GPU0 A10040GB',precision='FP32',batch24=24,R1_runtime=read_json(PARENT_R1/'analysis/runtime_cost.json'),log_overhead='new trace/event/progress costs in reports/logging_cost_*.json; no paired claim about inherited logging overhead',stage_history=str(WORK/'state/stage_history.json')))
    return final

def compare():
    definitions={'R2_fixed92':(WORK/'results/case_class_metrics.csv',WORK/'results/EVALUATION_RECEIPT.json'),
                 'R1_v2_B':(PARENT_R1/'results/case_class_metrics.csv',PARENT_R1/'results/EVALUATION_RECEIPT.json'),
                 'Old_Full':(WORK/'reports/Historical_Full_case_class_metrics.csv',WORK/'reports/Historical_Full_EVALUATION_RECEIPT.json'),
                 'beta_FFT_baseline':(WORK/'reports/Historical_baseline_case_class_metrics.csv',WORK/'reports/Historical_baseline_EVALUATION_RECEIPT.json')}
    methods={};provenance=[];overall=[];missing=[]
    for name,(path,receipt) in definitions.items():
        if not path.exists() or not receipt.exists():
            missing.append(dict(method=name,reason='unified CSV or evaluation receipt missing'));overall.append(dict(method=name,**{k:'NA' for k in KEYS}));continue
        r=read_json(receipt);assert r['protocol']=='ACDC_v22_grid_directed_ASD' and r.get('test_manifest_sha256',r.get('manifest_sha256'))==sha(DATA/'test.list')
        rows=read_table(path);assert len(rows)==120
        methods[name]={(r['case'],r['class_name']):r for r in rows}
        values={k:float(np.mean([float(x[k]) for x in rows])) for k in KEYS}
        assert all(abs(values[k]-r['metrics'][k])<1e-10 for k in KEYS)
        overall.append(dict(method=name,unit='grid',cases=40,patients=20,checkpoint=r['checkpoint'],checkpoint_sha256=r['checkpoint_sha256'],selected_step=r.get('selected_step',r.get('checkpoint_step')),**values))
        provenance.append(dict(method=name,CSV=str(path),CSV_sha256=sha(path),receipt=str(receipt),receipt_sha256=sha(receipt),protocol=r['protocol'],test_manifest_sha256=sha(DATA/'test.list')))
    differences=[];aggregates=[]
    for method,ref in [('R2_fixed92','R1_v2_B'),('R2_fixed92','Old_Full'),('R2_fixed92','beta_FFT_baseline'),('R1_v2_B','Old_Full')]:
        if method not in methods or ref not in methods:continue
        assert set(methods[method])==set(methods[ref])
        paired=[]
        for key,r in methods[method].items():
            old=methods[ref][key]
            paired.append(dict(method=method,reference=ref,case=r['case'],patient=r['patient'],class_name=r['class_name'],unit='grid',**{k+'_delta':float(r[k])-float(old[k]) for k in KEYS},**{k+'_method':float(r[k]) for k in KEYS},**{k+'_reference':float(old[k]) for k in KEYS}))
        differences.extend(paired)
        for level,field in [('class','class_name'),('frame','case'),('patient','patient'),('overall',None)]:
            values=sorted({r[field] for r in paired}) if field else ['ALL']
            for value in values:
                rr=[r for r in paired if not field or r[field]==value]
                aggregates.append(dict(method=method,reference=ref,level=level,id=value,n_rows=len(rr),**{k+'_delta':float(np.mean([r[k+'_delta'] for r in rr])) for k in KEYS}))
    write_csv(WORK/'analysis/method_comparison.csv',overall)
    write_csv(WORK/'analysis/comparison_provenance.csv',provenance)
    write_csv(WORK/'analysis/paired_case_class_differences.csv',differences);write_csv(WORK/'analysis/paired_aggregated_differences.csv',aggregates)
    if missing:write_csv(WORK/'analysis/missing_references.csv',missing)
    pp=[r for r in aggregates if r['method']=='R2_fixed92' and r['reference']=='R1_v2_B' and r['level']=='patient']
    if pp:
        pp.sort(key=lambda r:r['dice_delta']);fig,axes=plt.subplots(2,2,figsize=(12,8),layout='constrained')
        for ax,key in zip(axes.ravel(),KEYS):
            factor=100 if key in ['dice','iou'] else 1;ax.bar(range(len(pp)),[factor*r[key+'_delta'] for r in pp]);ax.axhline(0,color='black',linewidth=.8);ax.set_xticks(range(len(pp)),[r['id'] for r in pp],rotation=70,fontsize=7);ax.set_ylabel('R2 − R1 '+key+(' (pp)' if factor==100 else ' (grid)'))
        save(fig,'patient_differences_R2_R1')
    return dict(status='DONE' if len(methods)==4 else 'PARTIAL',methods=len(methods),paired_rows=len(differences),missing=missing)

def main():
    plt.rcParams.update({'svg.fonttype':'none','font.size':9})
    final=trajectory();comparison=compare();error=WORK/'analysis/error_R1_R2/ERROR_ANALYSIS_RECEIPT.json'
    diagnostic=dict(source=str(PARENT_R1/'analysis/confidence_quality.csv'),sha256=sha(PARENT_R1/'analysis/confidence_quality.csv'),frozen_samples=str(PARENT_R1/'analysis/diagnostic_samples.csv'),GT_use='fixed labeled/validation offline diagnostics already completed, NOT this post-test error analysis',new_inference=0)
    atomic_json(WORK/'analysis/ANALYSIS_RECEIPT.json',dict(status='ANALYZED' if len(final)==30000 and comparison['status']=='DONE' else 'PARTIAL',time=now(),recorded_logical_steps=len(final),fixed_h=92,fixed_w=92,predefined_segments='six5000-step segments plus whole; average match only globally',historical_comparison=comparison,R1_confidence_quality=diagnostic,error_analysis=read_json(error) if error.exists() else dict(status='PARTIAL',reason='error geometry stage not completed'),optimizer_updates=0,GPU_inference=0,interpretation='single run; dynamic versus near-mean-area fixed strategy, no separation of information/variation/curriculum effects or CTW interaction'))
    print('R2 analysis finished, all favorable/equal/unfavorable paired rows retained.',flush=True)

if __name__=='__main__':main()
