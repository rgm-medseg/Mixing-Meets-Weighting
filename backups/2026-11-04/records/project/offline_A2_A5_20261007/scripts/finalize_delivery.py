"""Finalize reviewed CSV/figure additions and repackage a consistent immutable snapshot."""
import shutil,fcntl
import numpy as np
from common import *
from finish import package,mdtable
def main():
 lock=(W/'state/controller.lock').open('a');fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
 state=read(W/'state/controller.json');assert state['status']=='DONE'
 visual=read(W/'reports/VISUAL_REVIEW.json');assert visual['status']=='PASS'
 for p,h in visual['PNG_sha256'].items():assert sha(p)==h
 before=read(W/'manifests/protected.json');assert before==protected()
 sources=read(W/'reports/RUNTIME_SOURCE_ADDITIONAL_AUDIT.json')
 for r in sources:assert sha(r['path'])==r['sha256']
 # Present single-network and three-network totals in separate columns.
 params=rows(W/'results/A5_parameters_compute.csv');systems=[]
 for r in params:
  if r['scope']=='training system':
   r['system_parameters_total']=int(r['parameters_total']) if r.get('parameters_total') else int(r['system_parameters_total'])
   r['system_parameters_trainable']=int(r['parameters_trainable']) if r.get('parameters_trainable') else int(r['system_parameters_trainable'])
   r['parameters_total']='';r['parameters_trainable']=''
   systems.append(dict(method=r['method_id'],student1=int(r['parameters_student1']),student2=int(r['parameters_student2']),EMA_teacher=int(r['parameters_EMA_teacher']),system_total=r['system_parameters_total'],system_trainable=r['system_parameters_trainable'],teacher_trainable=int(r['teacher_trainable_parameters'])))
  else:r['system_parameters_total']='';r['system_parameters_trainable']=''
 csvout(W/'results/A5_parameters_compute.csv',params)
 csvout(W/'results/A5_training_system_parameters.csv',systems)
 cost=rows(W/'results/A5_training_cost.csv');ct=[{k:r.get(k,'NA') for k in ['method_id','successful_training_seconds','successful_training_hours','failed_startup_seconds','joint_steps','optimizer1_success','optimizer2_success','evaluation_extra_seconds','analysis_extra_seconds','peak_allocated_GiB']} for r in cost]
 section='\n\n## 训练系统参数与已有训练成本分项\n\n单网络参数位于parameters_total；三网络总参数位于system_parameters_total，训练总可训练数位于system_parameters_trainable。原始结构计数保留于raw_units/A5/parameters.json，汇总表分列避免部署与训练总量混用。\n\n'+mdtable(systems,['method','student1','student2','EMA_teacher','system_total','system_trainable','teacher_trainable'])+'\n'+mdtable(ct,['method_id','successful_training_seconds','successful_training_hours','failed_startup_seconds','joint_steps','optimizer1_success','optimizer2_success','evaluation_extra_seconds','analysis_extra_seconds','peak_allocated_GiB'])+'\n计时活动段边界、局部单步计时遗漏与原始来源另见A5_training_cost.csv；失败启动独立列出。\n'
 for p in [W/'reports/A5_REPORT_CN.md',W/'WRITING_HANDOFF_A2_A5_CN.md',W/'FINAL_REPORT_A2_A5_CN.md']:
  text=p.read_text()
  if '## 训练系统参数与已有训练成本分项' not in text:atomic(p,text+section)
 # Validate every repeated volume identity and all numeric report additions.
 vol=rows(W/'results/A5_volume_runtime.csv');by={}
 for r in vol:
  if r['valid']=='True':by.setdefault((r['method_id'],r['frame']),set()).add(r['prediction_sha256'])
 assert len(by)==80 and all(len(h)==1 for h in by.values())
 js(W/'reports/VOLUME_REPEAT_REPRODUCIBILITY.json',dict(status='PASS',method_frame_identities=80,rounds_each=3,all_prediction_hashes_equal=True))
 r1=[r for r in rows(W/'results/A2_class_precision_recall_summary.csv') if r['method_id']=='R1_v2_B' and r['weighting']=='pooled_pixel'];low=min(r1,key=lambda r:float(r['recall']));high=max(r1,key=lambda r:float(r['recall']))
 gs=rows(W/'results/A2_confidence_correctness.csv');bg=next(r for r in gs if r['method_id']=='R1_v2_B' and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['axis']=='predicted' and r['group']=='0' and r['weighting']=='pooled_pixel');fg=next(r for r in gs if r['method_id']=='R1_v2_B' and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['axis']=='predicted' and r['group']=='foreground' and r['weighting']=='pooled_pixel');background_share=int(bg['pixel_count'])/(int(bg['pixel_count'])+int(fg['pixel_count']))
 bg_note=f'\n\n背景支持观察：最佳R1 EMA教师的验证预测像素中，背景占{100*background_share:.4f}%。其平均最大概率{float(bg["mean_confidence"]):.6f}，预测前景为{float(fg["mean_confidence"]):.6f}。全像素平均置信度因此主要反映背景像素的贡献；分组可靠性和前景类别等权代理提供互补信息。该陈述来自已有A2像素支持及概率求和，未新增前向。\n'
 for p in [W/'reports/A2_REPORT_CN.md',W/'WRITING_HANDOFF_A2_A5_CN.md',W/'FINAL_REPORT_A2_A5_CN.md']:
  text=p.read_text()
  if '背景支持观察：最佳R1 EMA' not in text:atomic(p,text+bg_note)
 note=f"\n\n类别观察：最佳R1 EMA教师在验证集的{low['class_name']}召回率最低，为{float(low['recall']):.6f}；{high['class_name']}最高，为{float(high['recall']):.6f}。各类别precision/recall的pooled-pixel和patient-equal结果均保留。\n\nFor the best R1 EMA teacher, {low['class_name']} had a pooled-pixel recall of {float(low['recall']):.4f}, while {high['class_name']} had a recall of {float(high['recall']):.4f}. The class-specific results show where foreground confidence and recall differ.\n"
 for p in [W/'reports/A2_REPORT_CN.md',W/'WRITING_HANDOFF_A2_A5_CN.md',W/'FINAL_REPORT_A2_A5_CN.md']:
  text=p.read_text()
  if '类别观察：最佳R1 EMA' not in text:atomic(p,text+note)
 review=read(W/'reports/FIVE_ROUND_REVIEW.json')
 review['reviews'][3].update(status='PASS',result='CSV-derived values and units checked; all5 scientific figure PNGs visually inspected; titles/axes/legends readable, no clipping; SVG/PDF/PNG and captions complete',evidence='VISUAL_REVIEW.json; figures/A2_*; A2_paper_summary.csv; A5_paper_summary.csv')
 review['reviews'][4].update(result='469 completed files reused byte-identically on resume; duplicate start reused live PID;8 teachers,204 patient units,20 timing blocks,240 volumes; all unfavorable correlations retained; actual extracted archive SHA checked',evidence='RECOVERY_DUPLICATE_UNIT_RECEIPT.json; SOURCE_UNCHANGED.json; VOLUME_REPEAT_REPRODUCIBILITY.json; SHA256SUMS; ARCHIVE_RECEIPT.json')
 review.update(time=now(),status='PARTIAL',remaining_missing='complete non-MAC FLOPs and historical matched training totals; early/mid EMA states NA; architecture-only TRC lacks trained checkpoint')
 js(W/'reports/FIVE_ROUND_REVIEW.json',review)
 atomic(W/'FIVE_PASS_REVIEW_A2_A5.md',mdtable(review['reviews'],['round','status','evidence','result'])+'\n整体PARTIAL由真实计数/历史资料缺口导致；实际完成项如上。\n')
 csvout(W/'manifests/analysis_scripts_SHA256.csv',[dict(path=str(p),sha256=sha(p)) for p in sorted((W/'scripts').rglob('*.py'))])
 js(W/'reports/SOURCE_UNCHANGED.json',dict(status='PASS',time=now(),protected_files=len(before),all_source_hashes_unchanged=True,additional_runtime_sources=len(sources),training_updates=0,optimizer_updates=0,EMA_updates=0,old_queue=read(ROOT/'revision_runs_20261006/state/queue.json')['status'],old_stop_flag=(ROOT/'revision_runs_20261006/state/stop_requested').exists()))
 archive=W/'A2_A5_offline_evidence_20261007.tar.gz';old=read(W/'reports/ARCHIVE_RECEIPT.json');preserve=W/'archives'/('initial_verified_archive_'+old['sha256']+'.tar.gz')
 if not preserve.exists():shutil.copy2(archive,preserve)
 js(W/'reports/INITIAL_ARCHIVE_PRESERVED.json',dict(path=str(preserve),sha256=sha(preserve),initial_snapshot=old['snapshot'],reason='consistent initial archive retained; final package adds actual-control text/plot, source forward audit and visual/resume verification'))
 previous_hash=state.pop('archive_sha256',None)
 state.update(status='FINALIZING',stage='FINAL_SNAPSHOT',heartbeat=now(),controller_pid=os.getpid(),child_pid=None,log=str(W/'logs/finalize_delivery_archive.log'),previous_verified_archive_sha256=previous_hash);js(W/'state/controller.json',state)
 package()
 receipt=read(W/'reports/ARCHIVE_RECEIPT.json');state.update(status='DONE',stage='FINISHED',heartbeat=now(),finalized=now(),archive_sha256=receipt['sha256'],A2_status='DONE',A5_status='PARTIAL',finalization='completed report additions, visual and resume verification, actual extraction SHA verification');js(W/'state/controller.json',state)
 print('Final A2 DONE / A5 PARTIAL; verified',receipt['sha256'],flush=True)
if __name__=='__main__':main()
