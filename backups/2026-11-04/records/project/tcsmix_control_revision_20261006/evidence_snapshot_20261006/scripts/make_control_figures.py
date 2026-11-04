"""Plots and corrected timing summaries from completed, immutable short records."""
import collections
from datetime import datetime
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *

def main():
    rows=read_table(WORK/'short_trace.csv');stat=read_json(WORK/'reports/short_statistics.json');segments=read_json(WORK/'reports/short_segments.json')
    timing=[]
    for item in segments:
        command=read_json(WORK/f"reports/short_command_{item['segment']}.json")
        elapsed=(datetime.fromisoformat(item['time'])-datetime.fromisoformat(command['time'])).total_seconds()
        timing.append(dict(segment=item['segment'],start=command['time'],end=item['time'],seconds=elapsed,
            convention='validated/configured entry command timestamp to subprocess completion; early imports/admission before timestamp excluded'))
    first=read_json(WORK/'reports/short_first_segment_TRAIN_RECEIPT.json')
    overhead=first['training_seconds']-float(rows[199]['cumulative_active_seconds'])
    total=sum(r['seconds'] for r in timing)
    measurement=dict(time=now(),actual_measured_segment_seconds=total,segments=timing,raw_compatibility_receipt_seconds=stat['active_segment_seconds'],
        legacy_receipt_limitation='resume offset comes from per-step progress saved before step200 validation; raw cumulative receipt omits that prior validation tail; do not use as total observed cost',
        first_validation_checkpoint_stop_tail_seconds=overhead,
        raw_receipts_preserved=True,stable_step_median_seconds=stat['stable_step_median_seconds'],
        compute_only_30000_hours=stat['full30000_compute_hours_estimate'],
        projected_with_observed_200step_tail_hours=stat['full30000_compute_hours_estimate']+overhead*150/3600,
        planning_range_per_full_run_hours=[6.7,7.5],planning_range_R1_R2_hours=[13.4,15.0],
        limits='240-step early segment; includes artificial save/stop in tail; no middle/late v2 speed and no guarantee of long-run rates; final inference/analysis and resource waiting additional')
    atomic_json(WORK/'reports/measurement_summary.json',measurement)
    plt.rcParams.update({'svg.fonttype':'none','font.size':9,'axes.spines.top':False,'axes.spines.right':False})
    fig,axes=plt.subplots(3,2,figsize=(10,8),layout='constrained');steps=[int(r['step']) for r in rows]
    axes[0,0].plot(steps,[float(r['old_q']) for r in rows],label='Old q replay on SAME v2 teacher',lw=.8,c='#0072B2')
    axes[0,0].plot(steps,[float(r['q_A']) for r in rows],label='A signal replay (not A training)',lw=.8,c='#E69F00')
    axes[0,0].plot(steps,[float(r['q']) for r in rows],label='Selected B, actual short run',lw=.8,c='#009E73');axes[0,0].legend(fontsize=7)
    axes[0,0].set(ylabel='Detached batch control',xlabel='Joint SSL update')
    axes[0,1].step(steps,[int(r['rectangle_h']) for r in rows],where='post',c='#009E73',label='Actual v2_B rectangle')
    axes[0,1].axhline(97,c='#0072B2',ls='--',lw=.8,label='Old controller replay (all240)');axes[0,1].legend(fontsize=7)
    axes[0,1].set(ylabel='Side length (pixels)',xlabel='Joint SSL update')
    for c,name,color in [(1,'RV','#56B4E9'),(2,'MYO','#E69F00'),(3,'LV','#CC79A7')]:axes[1,0].plot(steps,[int(r[f'n_{c}']) for r in rows],label=name,lw=.7,c=color)
    axes[1,0].set(ylabel='Original LCC foreground support (pixels)',xlabel='Joint SSL update');axes[1,0].legend(fontsize=7)
    axes[1,1].plot(steps,[100*float(r['actual_area']) for r in rows],c='#009E73',lw=.8);axes[1,1].set(ylabel='Actual exchanged area (%)',xlabel='Joint SSL update')
    axes[2,0].plot(steps,[float(r['loss']) for r in rows],lw=.8,c='#D55E00');axes[2,0].set(ylabel='Total loss, not efficacy',xlabel='Joint SSL update')
    count=collections.Counter(int(r['rectangle_h']) for r in rows);axes[2,1].bar(sorted(count),[count[k] for k in sorted(count)],color='#009E73');axes[2,1].set(xlabel='Actual rectangle side (pixels)',ylabel='Count of executed short updates')
    for ax in axes.flat:
        if ax is not axes[2,1]:ax.axvline(200,c='gray',ls=':',lw=.8)
    fig.suptitle('TCSMix v2_B short check | 240 joint updates / 480 optimizer calls | Full spatial CTW\nDotted line: same-run save/resume boundary; no formal/test metrics')
    for ext in ['png','svg']:fig.savefig(WORK/f'analysis/short_control_trace.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    controlled=read_table(WORK/'analysis/controlled_probability_checks.csv')
    rr=[r for r in controlled if r['test']=='background multiplicity']
    fig,axs=plt.subplots(1,2,figsize=(8,3),layout='constrained')
    for field,label,color in [('old_q','Old all-pixel','#0072B2'),('q_fg','A foreground class-balanced','#E69F00'),('q_margin','B foreground margin','#009E73')]:
        axs[0].plot([0,100],[float(r[field]) for r in rr],marker='o',label=label,c=color)
    axs[0].set(xlabel='Added background pixels; same foreground P/support',ylabel='Control value',title='Controlled tensors, not patient efficacy');axs[0].legend(fontsize=7)
    rr=[r for r in controlled if r['test']=='foreground monotonicity']
    for field,label,color in [('q_fg','A','#E69F00'),('q_margin','B','#009E73')]:axs[1].plot([float(r['condition']) for r in rr],[float(r[field]) for r in rr],marker='o',label=label,c=color)
    axs[1].set(xlabel='Foreground class probability, competitors equal',ylabel='Control value',title='Monotonic signal response');axs[1].legend()
    for ext in ['png','svg']:fig.savefig(WORK/f'analysis/controlled_signal_response.{ext}',dpi=170,bbox_inches='tight')
    plt.close(fig)
    atomic_json(WORK/'analysis/figure_provenance.json',dict(time=now(),figures={'short_control_trace':'short_trace.csv','control_replay':'control_diagnostics.csv','controlled_signal_response':'analysis/controlled_probability_checks.csv'},
        version='v2_B',official_metrics='NA',medical_images_exported=False,plotting='Matplotlib real statistics, editable SVG text',checkpoint_inventory='reports/offline_checkpoint_inventory.json',
        teacher_for_short='new evolving v2_B teacher, no old-teacher trajectory substitution'))

if __name__=='__main__':main()
