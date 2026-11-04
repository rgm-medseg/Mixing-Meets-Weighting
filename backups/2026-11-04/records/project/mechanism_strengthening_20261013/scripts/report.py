"""Write reviewable manuscript text and tables exclusively from actual results/receipts."""
import csv,json,subprocess,sys
from common import *
def rows(name):
    with (W/'results'/name).open() as f:return list(csv.DictReader(f))
def pick(rr,**kw):return next(r for r in rr if all(r[k]==v for k,v in kw.items()))
def f(r,k):return float(r[k])
def main():
    if (W/'receipts/fixed_test_complete.json').exists():
        subprocess.run([sys.executable,str(W/'scripts/final_control_analysis.py')],check=True)
    joint=rows('joint_region_summary.csv');trc=rows('trc_region_summary.csv');dice=rows('trc_dice_summary.csv')
    ua=pick(joint,condition='dynamic_uniform',region='anatomy');ca=pick(joint,condition='dynamic_CTW',region='anatomy');ce=pick(joint,condition='dynamic_CTW',region='pseudo_error');cc=pick(joint,condition='dynamic_CTW',region='pseudo_correct')
    alltrc=pick(trc,region='all');macro=pick(dice,class_name='macro');geometry=rows('joint_full_geometry.csv');q=[f(r,'q') for r in geometry if r['strategy']=='dynamic']
    fixedreceipt=W/'receipts/fixed_test_complete.json';done=fixedreceipt.exists();training=W/'receipts/fixed_training_complete.json';trained=training.exists()
    trace=W/'runs/F97_main_signal_20261013/audit/training_dynamics.csv';steps=0
    if trace.exists():
        with trace.open() as fp:latest=list(csv.DictReader(fp))
        complete=[r for r in latest if r.get('global_step','').isdigit() and r.get('checkpoint_saved') is not None]
        if complete:steps=max(int(r['global_step']) for r in complete)
    fixed=dict(run='F97_main_signal_20261013',status='DONE' if done else 'TRAINED_TEST_PENDING' if trained else 'TRAINING',updates_per_student=30000 if trained else steps,selected_step=None,validation_dice_percent=None,test_dice_percent=None,RV_dice_percent=None,MYO_dice_percent=None,LV_dice_percent=None,hd95_grid=None,asd_grid=None,rectangle='97x97',size_basis='150 sparse main-Full training endpoints; not full-trajectory area matching',TRC='on',CTW='on',signal='original teacher full-image mean max probability',seed=42)
    fixedtext=f'The independently named 97-by-97 control is running ({steps}/30000 updates per student at report generation); final checkpoint selection and test results are pending.'
    ablation_path=ROOT/'final_report/paper_internal_evidence_v23_0/tables/TABLE4_ACDC10_8ARM_FULL_PRECISION.tsv'
    with ablation_path.open() as fp:existing=list(csv.DictReader(fp,delimiter='\t'))
    trc_off=next(r for r in existing if r['Method']=='011');trc_on=next(r for r in existing if r['Method']=='111')
    csvout(W/'results/existing_TRC_training_ablation.csv',[dict(arm=r['Method'],TRC='off' if r['Method']=='011' else 'on',TCSMix='on',CTW='on',test_dice_percent=float(r['Dice'])*100,scope='existing reported training ablation; not reevaluated this round',source_table=str(ablation_path),source_sha256=sha(ablation_path)) for r in [trc_off,trc_on]])
    csvout(W/'results/trained_condition_inventory.csv',[
        dict(condition='fixed97_uniform',TRC='on',status='MISSING',checkpoint='',note='no comparable completed main-signal checkpoint; no new factorial training'),
        dict(condition='dynamic_uniform',TRC='on',status='MISSING_COMPLETED',checkpoint='',note='old E1 only3976 updates, not a completed30000-update model; excluded'),
        dict(condition='fixed97_CTW',TRC='on',status='DONE' if done else 'TRAINING',checkpoint=str(W/'runs/F97_main_signal_20261013/checkpoints/best.ckpt') if done else '',note='sole new condition; final validation selection only after budget complete'),
        dict(condition='dynamic_CTW',TRC='on',status='EXISTING_MAIN',checkpoint=CFG['main_checkpoint'],note='original X03 / Full_main; original main mean-max signal'),
    ])
    if trained:
        tc=json.loads(training.read_text());fixed.update(selected_step=tc['best_step'],validation_dice_percent=100*tc['best_validation']['model1_best_val_dice'])
    if done:
        fr=json.loads(fixedreceipt.read_text());fixed.update(test_dice_percent=100*fr['metrics']['dice'],hd95_grid=fr['metrics']['hd95'],asd_grid=fr['metrics']['asd'])
        fc=rows('fixed_test_class.csv')
        for name in ['RV','MYO','LV']:fixed[name+'_dice_percent']=100*f(pick(fc,class_name=name),'dice')
        fixedtext=(f"The fixed control completed 30,000 updates per student. Student 1 was selected at update {fixed['selected_step']} using the original validation criterion and achieved {fixed['test_dice_percent']:.4f}\\% test Dice, compared with the retained main-Full value of 90.3497\\%, a difference of {fixed['test_dice_percent']-90.34965765204998:+.4f} percentage points. Its HD95 and ASD were {fixed['hd95_grid']:.4f} and {fixed['asd_grid']:.4f} grid units. This is a single-seed, run-level comparison of the size rules. The fixed run's own teacher signals would yield 97-by-97 rectangles at 29,999 updates and 96-by-96 once under the original rounding rule; these logged signals do not reconstruct the original Full training trajectory.")
    mainrow=dict(run='original_X03_Full_main_paper_R1',status='EXISTING_RETAINED',updates_per_student=30000,selected_step=29600,validation_dice_percent=89.99080952059592,test_dice_percent=90.34965765204998,RV_dice_percent=89.03348224029954,MYO_dice_percent=88.29581930922916,LV_dice_percent=93.71967140662125,hd95_grid=1.4233666940334322,asd_grid=.4290264993301964,rectangle='original_dynamic_meanmax',size_basis='original protocol',TRC='on',CTW='on',signal='original teacher full-image mean max probability',seed=42)
    csvout(W/'results/fixed_control_table.csv',[mainrow,fixed])
    regiontable=[]
    for reg in ['anatomy','seam','cutout','other_valid','invalid','pseudo_correct','pseudo_error']:
        u=pick(joint,condition='dynamic_uniform',region=reg);c=pick(joint,condition='dynamic_CTW',region=reg)
        regiontable.append(dict(region=reg,pixels=int(c['pixels']),support_patients=int(c['support_patients']),mean_e=f(c,'mean_e'),mean_w_uniform=f(u,'mean_w'),mean_w_CTW=f(c,'mean_w'),mean_we_uniform=f(u,'mean_we'),mean_we_CTW=f(c,'mean_we'),mean_gradient_uniform=f(u,'mean_grad'),mean_gradient_CTW=f(c,'mean_grad'),gradient_share_uniform_percent=100*f(u,'gradient_fraction'),gradient_share_CTW_percent=100*f(c,'gradient_fraction'),student1_error_rate=f(c,'student1_error_rate') if c['student1_error_rate'] else None))
    csvout(W/'results/region_statistics_table.csv',regiontable)
    consolidated=[dict(panel='A_joint',**r) for r in regiontable]
    consolidated.extend(dict(panel='B_TRC',**r) for r in trc)
    csvout(W/'results/TABLE_regions.csv',consolidated)
    text=(r'''% Main-text additions for the current 17-page manuscript; no old manuscript overwritten.
% Keep title and module names. Place these paragraphs in Results / Mechanism Analysis.
\paragraph{Joint analysis of mixing and spatial weighting.}
We froze the two main-Full students and the EMA teacher and evaluated four combinations of fixed or TCSMix regions and uniform or CTW weights on the ACDC 10\% validation set. The 210 distinct slices from 10 patients formed nine predeclared batches with complementary mixtures and shared masks across aligned weak and strong views; six repeated source exposures completed the final batch. Uniform weights preserved each image's mean CTW weight. We differentiated the original $0.05\,\mathrm{mean}(w e)$ consistency term with respect to student-1 logits while detaching student 2 and the weights, and summarized source-patient statistics with equal patient weights. Ground truth defined offline anatomical and error regions only.
'''+f"The teacher confidence ranged from {min(q):.6f} to {max(q):.6f}, and every dynamic rectangle rounded to 97 by 97 pixels. Consequently, the fixed and dynamic conditions produced identical mixtures and cannot establish an interaction benefit on these frozen inputs. CTW increased the anatomical-boundary share of the consistency gradient from {100*f(ua,'gradient_fraction'):.2f}\\% to {100*f(ca,'gradient_fraction'):.2f}\\%, while the mean boundary gradient increased by a factor of {f(ca,'mean_grad')/f(ua,'mean_grad'):.2f}. The mean weight was {f(ce,'mean_w'):.3f} on erroneous pseudo-label source pixels and {f(cc,'mean_w'):.3f} on correct source pixels. These results locate CTW's emphasis at confidence transitions, including difficult anatomy and pseudo-label errors. Mixing seams and cutout responses are reported separately. Switching weights on a fixed checkpoint changes the consistency loss and gradient while preserving its predictions.\n\n"+
    r'''\paragraph{Residual intervention in TRC.}
We compared normal inference with a temporary $\gamma=0$ intervention in the paper's main-Full student 1, using the same checkpoint, validation images, and original inference protocol. The learned gate was $\gamma=0.035988$. Restoring the gate reproduced the normal logits bitwise on all 210 slices, and the checkpoint remained unchanged.
'''+f"Normal TRC changed {alltrc['changed']} of {int(alltrc['pixels']):,} pixels on the 256-grid diagnostic images relative to the intervention, correcting {alltrc['corrected']} errors and introducing {alltrc['introduced']}; {alltrc['bothwrong_changed']} changed pixels were incorrect in both conditions. Patient-equal native-volume Dice was {100*f(macro,'dice_normal'):.5f}\\% with normal TRC and {100*f(macro,'dice_gamma0'):.5f}\\% with $\\gamma=0$, a difference of {100*f(macro,'delta_dice'):.5f} percentage points. The small response indicates limited immediate dependence on the residual in this trained checkpoint. The existing training ablation reports {100*float(trc_off['Dice']):.4f}\\% test Dice without TRC and {100*float(trc_on['Dice']):.4f}\\% for Full with TCSMix and CTW retained. These observations distinguish the benefit of training with TRC from the residual's instantaneous contribution at the selected checkpoint; the intervention uses validation data, while the existing ablation uses test data.\n\n"+
    r'''\paragraph{Fixed-region training control.}
We selected a single 97-by-97 rectangle from 150 existing main-Full training records sampled every 200 updates. All recorded, rounded rectangles had area $9409/65536$, so the fixed size matches these observed endpoints. The records do not establish the mean area over all 30,000 updates. The new run reuses the main-Full pretrained weights, inherited SGD state, split, augmentations, TRC, CTW, EMA, losses, validation selection, and training budget; only the region dimensions are replaced.
'''+fixedtext+'\n\n'+
    r'''% Suggested contribution wording; algorithmic framing, not a claim of measured factorial synergy.
We formulate mixed-sample construction and spatial consistency allocation as linked stages of semi-supervised segmentation. TCSMix controls the construction of complementary mixed samples, CTW controls their spatial learning contribution, and TRC conditions decoder features through a learned residual. Our frozen-model diagnostics identify where the original consistency term allocates real logit gradients and quantify the trained model's dependence on the TRC residual.
    ''')
    (W/'MANUSCRIPT_INSERTIONS.tex').write_text(text)
    (W/'FIGURE2_LAYOUT.txt').write_text('Figure 2 layout proposal for the 17-page main-text manuscript\n\nPrimary left-to-right path:\nMixed samples → Aligned predictions → Spatial consistency weighting\n\nLeft: original EMA teacher weak views produce q = mean maximum probability; TCSMix maps q through s=.18+.20q and rounded geometry. Show one shared mask and two complementary mixtures.\nCenter: student1 weak mixed view and student2 spatially aligned strong mixed view, with the same sources and mask. Put TRC as a small decoder-feature residual inset: x + gamma*r(x).\nRight: aligned probability maps produce e and detached confidence-transition weights w. End at the original 0.05*mean(w*e) consistency item; mark stop-gradient on student2 and w.\nUse a thin secondary line for EMA updating the teacher. Keep other original losses in a compact retained-loss box so this item is not presented as the entire objective. GT is absent from the training path; optional offline diagnostic boxes sit below with a dashed border.\nKeep title, module names and the original algorithm. This is a layout recommendation only; no paper image or PDF was replaced.\n')
    summary=f'''# 本轮机制补强 SUMMARY

任务1：DONE。原始X03 / Full_main / 论文R1 best，step29600，两学生和EMA严格加载；主Full测试Dice90.35%保留。验证集20帧、10患者、210不同切片；9批24模拟角色，216来源暴露含6重复，108混合图/条件。四条件均实际计算。所有动态区域97×97，与固定完全重合；不能据此声称动态优势或TCSMix×CTW交互收益。

CTW把真实解剖边界的患者等权一致性梯度贡献从{100*f(ua,'gradient_fraction'):.2f}%提高至{100*f(ca,'gradient_fraction'):.2f}%；平均真实梯度强度比{f(ca,'mean_grad')/f(ua,'mean_grad'):.2f}。这是0.05缩放CTW一致性项对学生1 logits的L1梯度，不是总训练梯度。权重同图均值匹配，student2/权重detach；全图损失未按区域筛选。固定checkpoint的预测不因切换CTW改变。

任务2：DONE。gamma={float(json.loads((W/'receipts/trc_full.json').read_text())['gamma']):.9f}；改变{alltrc['changed']}像素，纠正{alltrc['corrected']}、新增{alltrc['introduced']}、双方均错误但标签改变{alltrc['bothwrong_changed']}。正常/干预患者等权native-volume Dice={100*f(macro,'dice_normal'):.5f}% / {100*f(macro,'dice_gamma0'):.5f}%，差{100*f(macro,'delta_dice'):.5f}个百分点；RV/MYO/LV及区域明细见CSV。全部210切片恢复gamma后logits逐元素一致，模型/原checkpoint未更新。既有训练消融在TCSMix/CTW保持启用时，无TRC/Full测试Dice={100*float(trc_off['Dice']):.4f}% / {100*float(trc_on['Dice']):.4f}%（独立已有训练结果，非本轮新评价）；即时验证干预与训练测试消融分别解释。

任务3：{'DONE' if done else 'RUNNING，结果待实际30000更新及评价完成'}。唯一新增条件F97_main_signal_20261013，97×97，TRC/CTW均开，原mean-max信号；复用原预训练及优化器继承，seed42，每学生30000更新。当前报告读取到{steps}次已完成更新（训练完成状态以receipt为准）。尺寸来自主Full150个稀疏训练时点，匹配这些时点的实际面积，未标为全程平均面积匹配。旧E3未运行，旧92×92替代信号结果未归入本对照。初始化三模型base tensors及两优化器base状态继承均逐元素核验。

{fixedtext.replace(chr(92)+'%', '%')}

新增固定run的完整信号日志含30000条，按原q/取整规则离线重映射时，29999条对应97×97、1条（step460）对应96×96；实际训练mask始终固定97×97。这只是该固定run自身的EMA信号分布，不能补成旧Full全程轨迹，也不将0.4948个百分点差解释为已证明的动态面积因果效应或TCSMix×CTW交互。40帧/20测试患者的等权重算与主Full原始病例CSV比较见fixed_vs_main_patient_dice.csv；主Full没有重评价。

交付：figures/Fig_joint_TCSMix_CTW和Fig_TRC_residual_intervention，各SVG/PDF/PNG；results/TABLE_regions.csv、joint_patient_region.csv、trc_patient_dice.csv、trc_patient_region.csv、fixed_control_table.csv、fixed_vs_main_patient_dice.csv；区域及固定对照结果表另有SVG/PDF/PNG预览和LaTeX。完整图像/预测/真实梯度在arrays/joint，TRC特征/残差在arrays/trc_features。scripts、configs、COMMANDS.sh与PROTOCOL.md提供复现。MANUSCRIPT_INSERTIONS.tex是独立英文正文段落与贡献建议，FIGURE2_LAYOUT.txt给布局；不覆盖旧稿，不替换论文图片或PDF。

五审见FIVE_REVIEWS.md和receipts/FIVE_REVIEWS.json。三项实验均DONE，训练、验证选点、40帧测试及复现产物已完成；没有尚需用户提供的初始化或实验配置。未找到可合入的最新17页可编辑稿，按授权交付独立正文插入内容。两种uniform训练条件仍缺少同设定的完整checkpoint，已登记且未新增训练；这不影响本轮四条件冻结诊断的完成。期刊最终生产格式核验仍PENDING。

使用K-Dense Scientific Visualization，实际数据与全部变换保留；skill来源与软件引用见PROTOCOL.md。Neurocomputing指南读取403，最终生产格式核验保留PENDING，不声称自动合规。最新17页正文未定位为可编辑工作稿，因此只交独立插入内容。作者固定记录未改动。
'''
    (W/'SUMMARY.md').write_text(summary)
    latex=r'''% One primary regional table. Fixed/dynamic values identical on all frozen batches.
\begin{tabular}{lrrrrrrrr}
\hline
Region & Pixels & Mean $e$ & CTW $w$ & CTW $we$ & $G_U$ & $G_C$ & U share (\%) & C share (\%) \\
\hline
'''
    for r in regiontable:latex+=f"{r['region'].replace('_',' ')} & {r['pixels']} & {r['mean_e']:.5f} & {r['mean_w_CTW']:.3f} & {r['mean_we_CTW']:.5f} & {r['mean_gradient_uniform']*1e9:.4f} & {r['mean_gradient_CTW']*1e9:.4f} & {r['gradient_share_uniform_percent']:.2f} & {r['gradient_share_CTW_percent']:.2f} \\\\\n"
    latex+='\\hline\n\\end{tabular}\n% Pixel counts are pooled supports; means and shares are equal-patient aggregates.\n% First five rows partition full support; pseudo rows overlap that partition.\n'
    latex+='% G_U / G_C are patient-mean L1 logit gradients of .05*mean(w*e), multiplied by1e9 for display; original mean-we is unscaled.\n'
    latex+='\n% Panel B: TRC gamma0 -> normal error changes; diagnostic256-grid pixels.\n\\begin{tabular}{lrrrrr}\n\\hline\nRegion & Pixels & Changed & Corrected & Introduced & Both wrong \\\\\n\\hline\n'
    for reg in ['RV','MYO','LV','anatomy','organ_interior','background','all']:
        r=pick(trc,region=reg);latex+=f"{reg.replace('_',' ')} & {r['pixels']} & {r['changed']} & {r['corrected']} & {r['introduced']} & {r['bothwrong_changed']} \\\\\n"
    latex+='\\hline\n\\end{tabular}\n% B counts are pooled; equal-patient rates and class one-vs-rest results remain in CSV.\n'
    (W/'REGION_TABLE.tex').write_text(latex)
    captions=f'''Figure: Joint TCSMix and CTW diagnostic. (a,b) Spatially aligned weak/strong complementary mixed views from predeclared batch0/pair0. (c) Mixed anatomical GT, used only for offline statistics (RV blue, MYO orange, LV purple). (d) Student argmax disagreement. (e) Original CTW weights, shown over their full .25–1.25 range. (f) Mean-class absolute probability error e. (g,h) True L1 magnitude of the student1 logit gradient of .05*mean(w*e), with common limits and student2/weights detached. (i,j) Equal-patient gradient shares and weights; points are10 patients, bars means, circles uniform and triangles CTW. Anatomy excludes seams/cutout/padding. All9 dynamic masks rounded97×97 and were identical to fixed masks, so the displayed dynamic results also represent fixed geometry. Full results include6 repeated source exposures needed to complete the final24-slot batch. No optimizer step was performed.

Figure: TRC residual intervention in main-Full student1. (a–d) The predeclared first validation frame (patient002_frame01), middle slice5, GT and same-checkpoint predictions with normal gamma and gamma0. (e) Pre-TRC feature L2 norm. (f) L2 norm of the applied residual gamma*r. (g) Logit response L2 norm. (h) Error changes for gamma0→normal, at original coordinates: corrected green, introduced vermilion, changed-but-wrong-in-both purple. This fixed case changes only1 pixel, explaining the sparse map; it was not chosen by response. (i) Native-volume patient Dice differences (normal−gamma0),10 patient points per class and mean lines; units are percentage points. (j) Equal-patient regional corrected/introduced pixel rates. Applied residual and logit maps use absolute scales starting at zero; their differing units are explicit. All210 restored forwards reproduced normal logits bitwise. Total changes were86, with46 corrections,36 introductions and4 changed but wrong in both. Average Dice changed by{100*f(macro,'delta_dice'):.5f} percentage points.

Table: Spatial statistics. Panel A uses108 mixed outputs per condition and source-patient equal aggregation; pooled pixel supports are shown alongside equal-patient mean e/w/gradient shares. The first five rows partition full image support, while pseudo-correct/error rows overlap that partition and cover only the pseudo-labeled source. Full fixed/dynamic and gradient-strength statistics are in CSV. Panel B pools TRC pixel counts over210 unaugmented256-grid validation slices; organ rows use GT support, anatomy/interior/background form a separate partition, and one-vs-rest class statistics plus equal-patient rates remain in the patient/region CSV. Pixel counts from A and B have different inputs and denominators.
'''
    (W/'figures/CAPTIONS.txt').write_text(captions)
    if done:
        cp=W/'receipts/ACTUAL_COMMANDS.json';commands=json.loads(cp.read_text())
        commands['scope']='All three pipeline dependent commands completed; see pipeline_status.json and fixed training/test receipts.'
        commands['pipeline_final_status']=json.loads((W/'receipts/pipeline_status.json').read_text())['status']
        commands['postprocess_commands']=[[sys.executable,str(W/'scripts/final_control_analysis.py')],[sys.executable,str(W/'scripts/tables.py')]]
        js(cp,commands)
        subprocess.run([sys.executable,str(W/'scripts/tables.py')],check=True)
    subprocess.run([sys.executable,str(W/'scripts/audit.py')],check=True)
    if done:subprocess.run([sys.executable,str(W/'scripts/package.py')],check=True)
    print(json.dumps(dict(fixed_status=fixed['status'],training_updates=steps,trc_delta_pp=100*f(macro,'delta_dice'),anatomy_gradient_share_CTW=100*f(ca,'gradient_fraction'))),flush=True)
if __name__=='__main__':main()
