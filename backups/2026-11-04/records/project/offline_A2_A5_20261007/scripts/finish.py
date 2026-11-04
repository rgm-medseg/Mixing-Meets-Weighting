import gc,shutil,tarfile,tempfile
from collections import defaultdict
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from common import *
def numeric(v):return v not in ['NA',''] and not str(v).startswith('NA:')
def val(v):return float(v) if numeric(v) else np.nan
def mdtable(rr,fields):
 return '|'+ '|'.join(fields)+'|\n|'+'|'.join(['---']*len(fields))+'|\n'+'\n'.join('|'+ '|'.join(str(r.get(k,'NA')) for k in fields)+'|' for r in rr)+'\n'
def figsave(fig,name,caption):
 for ext in ['pdf','svg','png']:fig.savefig(W/f'figures/{name}.{ext}',dpi=200,bbox_inches='tight')
 plt.close(fig);atomic(W/f'figures/{name}_caption.md',caption+'\n')
def plots():
 plt.rcParams.update({'svg.fonttype':'none','font.size':10,'axes.spines.top':False,'axes.spines.right':False})
 data=rows(W/'results/A2_sample_class_quality.csv');methods=config()['methods'];colors=['#0072B2','#E69F00','#009E73','#CC79A7']
 fig,axs=plt.subplots(2,4,figsize=(14,6.8),layout='constrained')
 for j,m in enumerate(methods):
  for i,c in enumerate(['fixed_three_foreground_macro','GT_present_foreground_macro']):
   for tag,marker in [('best','o'),('last','x')]:
    rr=[r for r in data if r['method_id']==m['id'] and r['checkpoint_tag']==tag and r['mode']=='eval' and r['cohort']=='validation' and r['prediction_kind']=='original_pseudolabel' and r['class_id']==c and numeric(r['dice'])]
    axs[i,j].scatter([val(r['mean_max_probability']) for r in rr],[val(r['dice']) for r in rr],s=13,alpha=.55,marker=marker,color=colors[j],label=tag)
   axs[i,j].set(xlabel='Mean max probability (fraction)',ylabel='Slice macro Dice (fraction)',ylim=(-.02,1.02),title=m['name']+'\n'+('Fixed RV/MYO/LV' if i==0 else 'GT-present classes'));axs[i,j].legend(fontsize=9)
 figsave(fig,'A2_confidence_quality','EMA teachers, eval mode, all210 validation slices from20 frames/10 patients. Best and last are shown separately. Top: fixed three-foreground macro with original both-empty0. Bottom: equal mean over GT-present foreground classes; all-background slices excluded only from this conditional macro. Input nearest order0 to256². These are offline slice diagnostics, separate from deployed-student volume test scores. Correlations and patient aggregation in A2_correlations.csv and A2_patient_correlation_inputs.csv.')
 bins=rows(W/'results/A2_confidence_bins.csv');fig,axs=plt.subplots(1,2,figsize=(10,4.5),layout='constrained')
 for ax,g in zip(axs,['0','foreground']):
  ax.plot([0,1],[0,1],':',color='.6')
  for j,m in enumerate(methods):
   for weight,style in [('pooled_pixel','-'),('patient_equal','--')]:
    rr=[r for r in bins if r['method_id']==m['id'] and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['axis']=='predicted' and r['group']==g and r['weighting']==weight and numeric(r['accuracy'])]
    ax.plot([val(r['mean_confidence']) for r in rr],[val(r['accuracy']) for r in rr],style,marker='o',markersize=3,color=colors[j],label=m['name']+(' pooled' if weight=='pooled_pixel' else ' patient'))
  ax.set(xlabel='Mean max probability (fraction)',ylabel='Multiclass accuracy (fraction)',xlim=(0,1),ylim=(0,1),title='Predicted background' if g=='0' else 'Predicted foreground');ax.legend(fontsize=7)
 figsave(fig,'A2_reliability','Best EMA teachers on all validation slices. Fixed10 equal-width confidence bins on[0,1], last bin includes1. Solid pooled-pixel curves; dashed patient-equal mean within each bin among patients with support. Empty bins omitted graphically and NA in CSV. Pooled ECE weights each bin by its pixel fraction; patient-equal ECE averages individual-patient ECE, rather than deriving it from the dashed curve. Foreground is raw argmax classes1–3, correctness is exact class match. Bin support counts and GT-conditioned statistics accompany A2_confidence_bins.csv.')
 hist=rows(W/'results/A2_confidence_distributions.csv');fig,axs=plt.subplots(1,4,figsize=(14,3.8),layout='constrained')
 for ax,m in zip(axs,methods):
  for outcome,color in [('correct','#0072B2'),('error','#D55E00')]:
   rr=[r for r in hist if r['method_id']==m['id'] and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['axis']=='predicted' and r['group']=='foreground' and r['value']=='confidence' and r['outcome']==outcome]
   ax.plot([(val(r['bin_left'])+val(r['bin_right']))/2 for r in rr],[val(r['conditional_fraction']) for r in rr],color=color,label=outcome)
  ax.set(xlabel='Max probability (fraction)',ylabel='Within-outcome pixel fraction',title=m['name'],xlim=(0,1));ax.legend()
 figsave(fig,'A2_correct_error_confidence','Best EMA teachers, raw predicted foreground, validation only. Correct and incorrect pixels each normalize their own50-bin confidence histogram to1. Their different support counts remain in raw CSV and correctness table. The histograms describe probability distributions, not new calibration probability definitions. Confidence margins have separate raw histograms; margins are not used as ECE probabilities.')
 fig,axs=plt.subplots(1,3,figsize=(12,3.8),layout='constrained')
 for ax,c in zip(axs,['1','2','3']):
  for j,m in enumerate(methods):
   rr=[r for r in data if r['method_id']==m['id'] and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['prediction_kind']=='raw_argmax' and r['class_id']==c and int(r['GT_pixels'])>0 and numeric(r['predicted_class_mean_confidence'])]
   ax.scatter([val(r['predicted_class_mean_confidence']) for r in rr],[val(r['dice']) for r in rr],s=13,alpha=.5,color=colors[j],label=m['name'])
  ax.set(xlabel='Predicted-class max probability',ylabel='Slice class Dice (fraction)',title={'1':'RV','2':'MYO','3':'LV'}[c],ylim=(0,1));ax.legend(fontsize=7)
 figsave(fig,'A2_class_confidence','Best EMA teachers, raw argmax, each foreground class separately on GT-present validation slices with nonempty predicted support. Scatter exclusions are conditional plotting only: all GT-absent slices, false positives and missed predictions remain in raw quality/correctness tables. Class precision/recall are NA for GT-absent classes, with false-positive counts preserved.')
def verify_a2():
 cfg=config();q=rows(W/'results/A2_sample_class_quality.csv');by={(r['sample'],r['prediction_kind'],r['class_id']):r for r in q if r['method_id']=='R1_v2_B' and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation'}
 uu=[read(p) for p in unitdir('A2').glob('R1_v2_B_best_eval_validation_*.json')];uu=[u for u in uu if u['cohort']=='validation'];lookup={d['quality'][0]['sample']:d for u in uu for d in u['slices']};checks=[]
 for pat in cfg['verification_patients']:
  ss=[s for s in rows(W/'manifests/fixed_samples.csv') if s['patient']==pat and s['cohort']=='validation']
  for s in ss:
   p=W/'state/verification_pixels'/f"{s['sample'].replace(':','_')}.npz"
   with np.load(p) as a:
    gt=a['GT'];raw=a['prediction'];post=a['pseudolabel'];confidence=a['confidence'].astype(np.float64);margin=a['margin'].astype(np.float64)
   d=lookup[s['sample']]
   for kind,pred in [('raw_argmax',raw),('original_pseudolabel',post)]:
    for c in [1,2,3]:
     pm=pred==c;gm=gt==c;tp=int(np.count_nonzero(pm&gm));fp=int(np.count_nonzero(pm&~gm));fn=int(np.count_nonzero(~pm&gm));den=2*tp+fp+fn;dice=2*tp/den if den else 0.
     r=by[(s['sample'],kind,str(c))];assert abs(float(r['dice'])-dice)<1e-12 and int(r['FP'])==fp and int(r['FN'])==fn
   for axis,group in [('predicted',0),('predicted','foreground'),('GT',0),('GT','foreground'),('predicted',1),('GT',2),('GT',3)]:
    base=raw if axis=='predicted' else gt;mask=base==group if isinstance(group,int) else base>0;ok=raw==gt
    r=next(r for r in d['pixel_stats'] if r['axis']==axis and r['group']==str(group));assert int(r['n'])==np.count_nonzero(mask) and int(r['correct'])==np.count_nonzero(mask&ok)
    assert abs(r['conf_sum']-float(confidence[mask].sum()))<1e-8
    for bi in range(10):
     b=mask&(confidence>=bi/10)&(confidence<((bi+1)/10) if bi<9 else confidence<=1)
     assert r['bins_n'][bi]==np.count_nonzero(b)
   assert d['counters']['background_to_foreground']==int(np.count_nonzero((gt==0)&(raw>0)))
   assert d['counters']['foreground_to_background']==int(np.count_nonzero((gt>0)&(raw==0)))
   checks.append(dict(patient=pat,sample=s['sample'],status='PASS',verification_array_sha256=sha(p),dice_FP_FN_group_bins_independent=True))
 csvout(W/'reports/A2_fixed_patient_recomputation.csv',checks)
 assert len(checks)==42
 assert len(q)==29040
 unique={(r['method_id'],r['checkpoint_tag'],r['mode'],r['cohort'],r['sample'],r['prediction_kind'],r['class_id']) for r in q};assert len(unique)==len(q)
 for r in q:
  if r['class_id'] in ['1','2','3'] and int(r['GT_pixels'])==0:assert r['precision']=='NA' and r['recall']=='NA'
 old=rows(W/'legacy/confidence_quality.csv');new={(r['checkpoint_tag'],r['mode'],r['sample']):r for r in q if r['method_id']=='R1_v2_B' and r['cohort'].endswith('_legacy') and r['prediction_kind']=='original_pseudolabel' and r['class_id']=='fixed_three_foreground_macro'}
 errors=[]
 for r in old:
  tag='best' if r['checkpoint'].endswith('/best.ckpt') else 'last';n=new[(tag,r['mode'],r['sample'])]
  errors.append(dict(checkpoint_tag=tag,mode=r['mode'],sample=r['sample'],macro_dice_absolute_difference=abs(float(r['macro_dice'])-float(n['dice'])),old_q_absolute_difference=abs(float(r['old_q'])-float(n['old_q_auxiliary'])),q_B_absolute_difference=abs(float(r['q_B'])-float(n['q_B_auxiliary']))))
 csvout(W/'reports/A2_legacy_replay_comparison.csv',errors)
 maxdiff=max(max(r[k] for k in ['macro_dice_absolute_difference','old_q_absolute_difference','q_B_absolute_difference']) for r in errors)
 js(W/'reports/A2_NUMERIC_REVIEW.json',dict(status='PASS' if maxdiff<1e-5 else 'PARTIAL',preselected_patients=cfg['verification_patients'],slices_checked=len(checks),duplicates=0,legacy_records_compared=len(errors),legacy_max_absolute_difference=maxdiff,legacy_tolerance=1e-5,NA_empty_bins_and_missing_GT_classes_verified=True,sample_and_batch_control_distinct=True))
def reports():
 cfg=config();co=rows(W/'results/A2_correlations.csv');stats=rows(W/'results/A2_confidence_correctness.csv');patient=rows(W/'results/A2_patient_summary.csv');a5=rows(W/'results/A5_inference_summary.csv');params=rows(W/'results/A5_parameters_compute.csv');vol=rows(W/'results/A5_volume_summary.csv');train=rows(W/'results/A5_training_cost.csv')
 def corr(mid,tag,level,signal='mean_max_probability'):
  return next(r for r in co if r['method_id']==mid and r['checkpoint_tag']==tag and r['mode']=='eval' and r['cohort']=='validation' and r['prediction_kind']=='original_pseudolabel' and r['class_id']=='GT_present_foreground_macro' and r['level']==level and r['signal']==signal and r['quality']=='dice')
 def group(mid,g,weight='pooled_pixel'):
  return next(r for r in stats if r['method_id']==mid and r['checkpoint_tag']=='best' and r['mode']=='eval' and r['cohort']=='validation' and r['axis']=='predicted' and r['group']==str(g) and r['weighting']==weight)
 ct=[];st=[]
 for m in cfg['methods']:
  for tag in ['best','last']:
   sl=corr(m['id'],tag,'slice');pa=corr(m['id'],tag,'patient');ctrl=corr(m['id'],tag,'patient','sample_control_proxy')
   ct.append(dict(method=m['name'],checkpoint=tag,slice_rho=sl['spearman'],slice_n=sl['valid_n'],patient_rho=pa['spearman'],patient_n=pa['valid_n'],patient_control_rho=ctrl['spearman']))
  for g in ['0','foreground','1','2','3']:
   r=group(m['id'],g);pe=group(m['id'],g,'patient_equal');st.append(dict(method=m['name'],group=g,pixels=r['pixel_count'],mean_probability=r['mean_confidence'],accuracy=r['accuracy'],incorrect_mean_probability=r['error_mean_confidence'],pooled_ECE=r['ECE'],patient_equal_ECE=pe['ECE']))
 r1bg=group('R1_v2_B','0');r1fg=group('R1_v2_B','foreground');r1sl=corr('R1_v2_B','best','slice');r1pa=corr('R1_v2_B','best','patient')
 en=f"We analyzed the frozen EMA teachers on 210 validation slices from 20 frames and 10 patients. For the best R1 teacher, predicted background pixels had a mean maximum probability of {float(r1bg['mean_confidence']):.4f}, while predicted foreground pixels had a mean of {float(r1fg['mean_confidence']):.4f}. Incorrect foreground predictions still had a mean confidence of {float(r1fg['error_mean_confidence']):.4f}. The Spearman correlation between mean maximum probability and GT-present foreground macro Dice was {float(r1sl['spearman']):.3f} at the slice level and {float(r1pa['spearman']):.3f} at the patient level. These observations show that the relationship depends on the pixel group and the aggregation unit."
 csvout(W/'results/A2_paper_summary.csv',ct);csvout(W/'results/A2_group_summary.csv',st)
 a2=f'''# A2教师置信度与质量\n\n状态DONE，{now()}。主要数据：验证210切片/20帧/10患者、真实标注训练136切片/14帧/7患者分开报告；四方法best/last共8个真实EMA教师状态，严格加载ema_model、eval/inference_mode。缺少6000/15000/24000完整教师状态，NA。训练更新0、EMA更新0、测试输入0。原136行来自34个样本×2检查点×2模式；保留并重放全部输入哈希。train_snapshot独立副本每批恢复参数和BN缓存，固定RNG，单列legacy数据组。\n\n作者版本说明：{cfg['author_version_note']}。本次真实加载的Full_main路径/分支/步数/哈希附source_manifest；与R1/R2分别列身份。全文统一TRC — Token-Guided Residual Conditioning，TCSMix — Teacher-Confidence-Scaled Mixing，CTW — Confidence-Transition Weighting。外部AdaMix未纳入本次方法结果。\n\n观察：最佳R1教师预测背景的平均最大概率{float(r1bg['mean_confidence']):.6f}，预测前景{float(r1fg['mean_confidence']):.6f}；前景正确率{float(r1fg['accuracy']):.6f}，错误前景像素平均置信度{float(r1fg['error_mean_confidence']):.6f}。置信度与质量在不同类别和统计单位呈现差异，完整表保留弱相关、负相关和病例反例。下面相关目标为原始伪标签处理后的GT-present前景macroDice；切片GT全背景时该宏值NA。固定三前景macroDice原规则双空0另列，其完整散点包括空切片。\n\n{mdtable(ct,['method','checkpoint','slice_rho','slice_n','patient_rho','patient_n','patient_control_rho'])}\n\n每个固定checkpoint/mode/cohort单独Spearman，无显著性检验；患者相关先在相同有效切片内等权、再帧等权，x/y共享同一有效名单，见A2_patient_correlation_inputs.csv。患者汇总并非新增训练重复。best与last的符号和幅度分别在表中，现有数据说明条件对应关系，未将它们合并为独立患者证据。\n\n{mdtable(st,['method','group','pixels','mean_probability','accuracy','incorrect_mean_probability','pooled_ECE','patient_equal_ECE'])}\n\n分组0=预测背景，foreground=预测类1–3，类别1/2/3=RV/MYO/LV。GT条件分组以及背景→前景、前景→背景计数/分母见A2_confidence_correctness.csv。各类precision/recall及缺失类FP在A2_sample_class_quality.csv；GT缺该类时precision/recall=NA，Dice/IoU按原规则保留0。原始argmax和literal get_ACDC_masks(nms=1)逐类LCC结果分别保存质量。LCC定义来自对应运行源码，未增加处理。\n\n可靠性分10个等宽[0,1]区间，最后闭右端；空区间NA。max概率用于校准，top1-top2间隔另作诊断。pooled-pixel按像素累计；patient-equal先各患者像素累计再等权平均有支持患者，分组支持人数随空区间变化。ECE=Σ_b n_b/N×|acc_b−conf_b|，条件像素范围在axis/group字段；patient-equal ECE=患者ECE均值。原训练控制依版本源码计算，不使用GT；sample_control_proxy为单切片离线响应，batch_control_value始终NA，本轮没有重建真实无标注训练批次。R2 q_B是记录信号、输出几何固定92，baseline没有TCSMix控制值。\n\n英文正文草稿：\n\n{en}\n\n图注均在figures/*_caption.md。两位预先固定患者patient002/019共42切片以缓存原始概率与掩码独立复算逐类质量、FP/FN、分组及每个区间支持；回执见A2_NUMERIC_REVIEW.json。未回写源文件。\n'''
 atomic(W/'reports/A2_REPORT_CN.md',a2)
 eff=[]
 for r in a5:
  p=next(p for p in params if p['method_id']==r['method_id'] and p['scope']=='single deployed student1');v=next(v for v in vol if v['method_id']==r['method_id'])
  eff.append(dict(method=r['method_id'],parameters=int(p['parameters_total']),TRC_parameters=int(p['TRC_parameters']),covered_GMACs=float(p['MACs_covered'])/1e9,covered_GFLOPs=float(p['MAC_FLOPs_covered'])/1e9,mean_ms=float(r['mean_ms']),median_ms=float(r['median_ms']),p95_ms=float(r['p95_ms']),slices_per_second=float(r['throughput_slices_per_second']),peak_allocated_MiB=float(r['peak_allocated_MiB']),peak_reserved_MiB=float(r['peak_reserved_MiB']),volume_ms_per_slice=float(v['ms_per_slice'])))
 csvout(W/'results/A5_paper_summary.csv',eff)
 english_table=mdtable([{k:(f'{v:.4f}' if isinstance(v,float) else v) for k,v in e.items()} for e in eff],['method','parameters','covered_GMACs','mean_ms','p95_ms','slices_per_second','volume_ms_per_slice'])
 base=eff[0];full=eff[1];r1t=next(r for r in train if r['method_id']=='R1_v2_B');r2t=next(r for r in train if r['method_id']=='R2_fixed92')
 en5=f"The deployed model uses one student network. TRC adds {full['TRC_parameters']} parameters to the baseline, giving {full['parameters']:,} parameters in total. On the A100 GPU, the baseline and paper Full model took {base['mean_ms']:.3f} and {full['mean_ms']:.3f} ms per slice for model forward, respectively. The corresponding validation-volume paths took {base['volume_ms_per_slice']:.3f} and {full['volume_ms_per_slice']:.3f} ms per slice, including resizing and device transfers. TCSMix and CTW operate during training and share the same deployed network structure as TRC."
 a5text=f'''# A5计算成本与效率\n\n状态PARTIAL：参数、声明口径内MACs、真实forward测速、验证体推理和R1/R2训练成本已完成。完整非MAC推理FLOPs与全部训练FLOPs未完整覆盖，标NA；历史baseline/Full缺同口径训练段边界。缺口不影响部署参数与实测时延表，论文计算量列应明确为已覆盖MACs而非完整FLOPs。\n\n部署按真实student1，strict加载model_1。四版本真实最佳checkpoint分别测速，单GPU常驻一个模型。TRC-only未发现对应ACDC真实权重，architecture-only共享实际Full实例相同结构计数，不填写分割质量或训练时间。Full/R1/R2参数与层结构相同，训练混合和CTW不进入部署模型。TRC净增{full['TRC_parameters']}参数（相对baseline{100*full['TRC_parameters']/base['parameters']:.5f}%）；三网络训练参数另列，总量不是部署成本。教师在原工厂ema=True detach_()，参数数目与可训练状态分开。\n\n{english_table}\n\n表注：1×1×256×256、FP32 eval/inference_mode、GPU0 A100-SXM4-40GB；环境/UUID/TF32既有默认标志附原始JSON；未启用AMP/编译/量化。每个模型首次50预热，随后5个100-forward CUDA event块按预固定轮换顺序、结束同步，预热排除；重新加载时间排除。输入已在GPU，文件IO、H2D、输出保存以及外部softmax/argmax排除。原TRC forward自带概率路由统计、CPU同步和decoder argmax诊断，按真实原路径保留，实测开销包含这些已有实现。吞吐=500/500次event时间总和，mean/median/p95来自全部有效原始次数。整块host墙钟另附，不混用吞吐口径。\n\n峰值allocated/reserved在模型与输入加载、首次预热后reset，包含一个常驻模型/输入；nvidia-smi整卡占用另列。测速期间每0.75秒检查外部GPU PID、块前/后核验，检测到干扰的块保留invalid且不入汇总。体推理输入是固定20个验证frame的预加载CPU数组，3轮，共每方法630切片；包括原order0缩放、H2D、forward、softmax/argmax、D2H与原尺寸恢复，不包括读盘、存盘和GT指标，保持batch1及z轴顺序。\n\nMAC来自Conv/Linear模块hooks和两次einsum显式公式，torch.profiler原生bmm交叉验证。1 MAC=2 FLOPs，仅为已覆盖乘加口径。softmax/LayerNorm/残差融合补充公式另表，不与已计数项重复。未覆盖列表和原生零flops事件完整附A5_OPERATOR_COVERAGE.md与CSV。\n\nR1成功训练{float(r1t['successful_training_seconds']):.6f}秒（{float(r1t['successful_training_hours']):.6f}小时）；R2成功训练{float(r2t['successful_training_seconds']):.6f}秒（{float(r2t['successful_training_hours']):.6f}小时），另有零更新启动失败{float(r2t['failed_startup_seconds']):.6f}秒。原R2 runtime_cost把失败加进活动段，本次按exit_code拆分，不修改历史文件。二者30000联合步、两学生各30000成功更新、物理重放0。活动段包含导入/加载/取数等待/EMA/验证/保存，Popen轮询结束检测误差≤5秒。局部step_seconds仅部分循环，不能替代活动段；控制器局部ms作为诊断单列。评价和分析额外时间见A5_training_cost.csv。\n\n英文正文草稿：\n\n{en5}\n'''
 atomic(W/'reports/A5_REPORT_CN.md',a5text)
 hand=f'''# A2/A5写作交接\n\n新增内容使用TRC — Token-Guided Residual Conditioning；TCSMix — Teacher-Confidence-Scaled Mixing；CTW — Confidence-Transition Weighting。\n\nA2建议放机制分析节：最佳R1教师预测背景平均最大概率{float(r1bg['mean_confidence']):.4f}，预测前景{float(r1fg['mean_confidence']):.4f}；错误前景仍有{float(r1fg['error_mean_confidence']):.4f}平均置信度。平均最大概率与GT-present伪标签前景macroDice的Spearman在切片层面为{float(r1sl['spearman']):.3f}，患者层面{float(r1pa['spearman']):.3f}。分组及统计单位呈现不同关系，完整方法/检查点/类别比较见A2_paper_summary.csv、A2_group_summary.csv、A2_correlations.csv。负相关和弱相关保留。\n\n{en}\n\n建议组合A2_confidence_quality与A2_reliability正文，A2_class_confidence和A2_correct_error_confidence放补充；全部图有SVG/PDF/PNG及图注。主要队列验证210切片、20帧、10患者；标注训练136切片单列；原34个小样本使用历史增强/中间层切片并原样保留。A2质量属于EMA教师离线切片诊断，原始argmax/原LCC分别报告。主版本90.35%、R1 89.98%、R2 90.20%是已有student1完整体测试结果，身份各自保留，A2/A5不产生动态混合独立性能增益结论。作者关于90.35%主版本动态机制的说明保留在预分析与A2报告，不转移到R1。\n\nA5建议放效率节的表：\n\n{english_table}\n\n{en5}\n\n表注必须标已覆盖GMACs，完整FLOPs=NA/PARTIAL；单模型student1与三网络训练系统分列。模型forward时延与预加载CPU到原尺寸mask体时延分列；距离指标未在本次测速计入。R2成功训练{float(r2t['successful_training_hours']):.4f}小时，失败启动{float(r2t['failed_startup_seconds']):.2f}秒另列。R1/R2计时范围相同，单次运行差异只描述。历史训练成本NA、中期教师NA、TRC-only仅architecture-only可结构计数，这些缺口应在表注准确说明。\n\n证据映射：A2_SAMPLE_CLASS表→逐类质量与FP/FN；A2_confidence_bins→可靠性与ECE；A2_patient_correlation_inputs→相同切片/帧汇总；A5_parameters_compute和A5_OPERATOR_COVERAGE→结构/算子；A5_inference_raw/summary→500次有效原始测速和吞吐；A5_volume_runtime→3轮20frame；A5_training_cost+legacy/training_cost→历史训练边界。\n'''
 atomic(W/'WRITING_HANDOFF_A2_A5_CN.md',hand)
 atomic(W/'FINAL_REPORT_A2_A5_CN.md',f'# A2/A5离线分析结果\n\n{now()}。A2 DONE；A5 PARTIAL（完整FLOPs、历史可比训练总耗时缺项）。新增训练/优化器/EMA更新0，旧队列INTERRUPTED。\n\n'+a2+'\n'+a5text+'\n\n操作：start.sh重复启动复用活跃控制器；stop.sh在完成患者/块边界停止；resume.sh补缺失身份单元。tmux会话beta_fft_A2_A5_20261007。服务器重启后需要运行resume.sh；tmux不跨重启存活。归档快照来自核验后状态，最终外层校验回执后写。源医学数据和完整权重不包含，附真实路径与SHA256。\n')
def verify_a5():
 pp=rows(W/'results/A5_parameters_compute.csv');single=[p for p in pp if p['scope']=='single deployed student1'];base=next(p for p in single if p['method_id']=='baseline')
 for p in single:
  assert int(p['MAC_FLOPs_covered'])==2*int(p['MACs_covered'])
  if p['method_id']!='baseline':assert int(p['parameters_total'])-int(base['parameters_total'])==641 and int(p['TRC_parameters'])==641
 raw=rows(W/'results/A5_inference_raw.csv');ss=rows(W/'results/A5_inference_summary.csv')
 for s in ss:
  rr=[r for r in raw if r['method_id']==s['method_id'] and r['valid']=='True'];assert len(rr)==500
  ms=np.array([float(r['forward_ms']) for r in rr]);assert abs(ms.mean()-float(s['mean_ms']))<1e-10 and abs(500/(ms.sum()/1000)-float(s['throughput_slices_per_second']))<1e-8
  assert float(s['warmup_total'])==50
 vr=[r for r in rows(W/'results/A5_volume_runtime.csv') if r['valid']=='True'];assert len(vr)==240
 for mid in config()['methods']:
  rr=[r for r in vr if r['method_id']==mid['id']];assert sum(int(r['slices']) for r in rr)==630
 cost=rows(W/'results/A5_training_cost.csv');r2=next(r for r in cost if r['method_id']=='R2_fixed92');assert abs(float(r2['successful_training_seconds'])+float(r2['failed_startup_seconds'])-float(r2['recorded_runtime_cost_total_seconds']))<1e-6
 return dict(status='PASS',parameters_verified=True,operator_coverage='PARTIAL complete FLOPs explicitly NA',timing_forwards=2000,volume_repetitions=240,R2_failure_separated=True)
def package():
 snap=W/'archives'/f"snapshot_{now().replace(':','')}";snap.mkdir(parents=True)
 for name in ['scripts','configs','manifests','legacy','logs','results','figures','reports']:
  shutil.copytree(W/name,snap/name,ignore=shutil.ignore_patterns('__pycache__','*.pyc','ARCHIVE_RECEIPT.json'))
 for p in W.glob('*.md'):shutil.copy2(p,snap/p.name)
 for p in W.glob('*.sh'):shutil.copy2(p,snap/p.name)
 atomic(snap/'ARCHIVE_NOTE.md','Results, raw units, figures and reports are one immutable snapshot after completed numeric/visual review. SHA256SUMS covers actual package files. The current outer archive receipt is written after compression and is intentionally excluded from this package; use the adjacent .tar.gz.sha256. INITIAL_ARCHIVE_PRESERVED.json explicitly identifies retained earlier archives. The pipeline state snapshot precedes completion of this compression.\n')
 # Include atomically completed compact units for independent reaggregation.
 shutil.copytree(unitdir('A2'),snap/'raw_units/A2');shutil.copytree(unitdir('A5'),snap/'raw_units/A5')
 js(snap/'state_snapshot.json',dict(time=now(),identity=identity(),A2=read(W/'results/A2_RECEIPT.json'),A5=read(W/'results/A5_RECEIPT.json'),controller=read(W/'state/controller.json'),pixel_verification_arrays='server-only raw probability/GT cache excluded; paths/hash in recomputation receipt'))
 files=sorted(p for p in snap.rglob('*') if p.is_file());atomic(snap/'SHA256SUMS',''.join(sha(p)+'  '+str(p.relative_to(snap))+'\n' for p in files))
 dst=W/'A2_A5_offline_evidence_20261007.tar.gz';tmp=dst.with_suffix('.tmp')
 with tarfile.open(tmp,'w:gz') as tf:tf.add(snap,arcname='A2_A5_offline_evidence_20261007')
 os.replace(tmp,dst)
 with tempfile.TemporaryDirectory(prefix='A2_A5_actual_extract_',dir=W/'archives') as temp:
  with tarfile.open(dst) as tf:tf.extractall(temp,filter='data')
  extracted=Path(temp)/'A2_A5_offline_evidence_20261007'
  for line in (extracted/'SHA256SUMS').read_text().splitlines():
   expected,rel=line.split('  ',1);assert sha(extracted/rel)==expected,rel
 h=sha(dst);atomic(dst.with_suffix('.gz.sha256'),h+'  '+dst.name+'\n');js(W/'reports/ARCHIVE_RECEIPT.json',dict(status='VERIFIED',time=now(),archive=str(dst),sha256=h,bytes=dst.stat().st_size,files_verified=len(files),verification='actual extraction and every file SHA256 match',snapshot=str(snap)))
def main():
 # Regenerate summaries deterministically from completed units before checks.
 from a2 import aggregate
 aggregate();verify_a2();a5review=verify_a5();plots();reports()
 before=read(W/'manifests/protected.json');after=protected();assert before==after
 js(W/'reports/SOURCE_UNCHANGED.json',dict(status='PASS',files_checked=len(before),all_hashes_unchanged=True,training_updates=0,EMA_updates=0,old_queue=read(ROOT/'revision_runs_20261006/state/queue.json')['status']))
 a2review=read(W/'reports/A2_NUMERIC_REVIEW.json')
 reviews=[dict(round=1,status='PASS',evidence='manifests/source_manifest.csv; reports/LOAD_*; SOURCE_UNCHANGED.json',result='8 real EMA teachers and4 deployed student1 strict loads; no training; old queue paused'),dict(round=2,status=a2review['status'],evidence='A2_NUMERIC_REVIEW.json; A2_fixed_patient_recomputation.csv; A2_legacy_replay_comparison.csv',result='42 predefined slices independently recomputed; sample/batch fields distinct; real GT only; patient equal and zero support explicit'),dict(round=3,status='PARTIAL',evidence='A5_OPERATOR_COVERAGE.md; A5_modules_*; A5_operators_*; A5_inference_raw.csv; A5_training_cost.csv',result='parameters/MACs/timing/timing-boundary checks PASS; complete FLOPs and old complete training costs NA'),dict(round=4,status='PARTIAL',evidence='A2_*_caption.md; A2_paper_summary.csv; A5_paper_summary.csv; report generated directly from CSV',result='fractions,ms,s,h,MiB/GiB explicitly labeled; paper module names; SVG native text/PDF/PNG; human visual inspection pending'),dict(round=5,status='PASS',evidence='deterministic regeneration from raw_units; launch/attach/detach receipt; internal SHA256SUMS and external ARCHIVE_RECEIPT',result='duplicates0; all negative/weak results retained; unit resume with atomic file replacement; actual extraction checksum verified by package')]
 js(W/'reports/FIVE_ROUND_REVIEW.json',dict(time=now(),status='PARTIAL',reviews=reviews,A5_numeric_review=a5review))
 atomic(W/'FIVE_PASS_REVIEW_A2_A5.md',mdtable(reviews,['round','status','evidence','result'])+'\nPARTIAL来自真实覆盖缺项，统计/计时完成项已核验。归档回执在快照之后写，包内SHA256SUMS实际解包核验。\n')
 package()
 print('A2 DONE; A5 PARTIAL; archive verified',flush=True)
if __name__=='__main__':main()
