"""Read-only evidence audit with per-fact paths, fields, lines and source hashes."""
import difflib
import json
import re
import subprocess
from common import *

def line(path,needle):
    for index,text in enumerate(Path(path).read_text().splitlines(),1):
        if needle in text: return index
    return 'NA'

def fact(rows,name,value,path,field,needle=None,run='NA',checkpoint='NA'):
    rows.append(dict(fact=name,value=json.dumps(value,ensure_ascii=False) if not isinstance(value,str) else value,
                     source=str(path),field=field,line=line(path,needle) if needle and Path(path).exists() else 'NA',
                     run_id=run,checkpoint=checkpoint,source_sha256=sha(path) if Path(path).exists() else 'NA'))

def main():
    protocol=read_json(WORK/'protocol.json'); rows=[]
    full_log=FULL/'train.log'; baseline_log=BASE/'train.log'
    for name in ('seed','batch_size','labeled_bs','base_lr','max_iterations','consistency','consistency_rampup','u_weight'):
        fact(rows,'Full '+name,protocol['original_namespace'][name],full_log,'Namespace.'+name,needle='Namespace(',run=str(FULL),checkpoint=str(FULL/'checkpoints/best.ckpt'))
    fact(rows,'baseline actual namespace',baseline_log.read_text().splitlines()[0],baseline_log,'Namespace',needle='Namespace(',run=str(BASE),checkpoint=str(BASE/'checkpoints/best.ckpt'))
    fact(rows,'seven labeled patients',protocol['labeled_patients'],DATA/'train_slices.list','first136 patient IDs; historical patients_to_slices(7)=136',run=str(FULL))
    fact(rows,'TRC zero gate/shared tokens','4x16 tokens, 1x1+LN retrieval residual, gamma0',PROJECT/'train_ACDC/v14_mask_adamix_components.py','SpatialMaskTokenRouter',needle='class SpatialMaskTokenRouter',run=str(FULL))
    fact(rows,'TCSMix mapping','s=.18+.2*q; Python round+clamp; shared batch mask',PROJECT/'train_ACDC/v14_mask_adamix_components.py','generate_adamix_bcp_lite_v1_mask',needle='def generate_adamix',run=str(FULL))
    fact(rows,'CTW detach/weight/loss','backwardabs q-difference, perimage minmax floor1e-6; .25+r; mean-classL1, student2 detached',PROJECT/'train_ACDC/v21_3_acdc10_third_innovation_components.py','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',needle='        contour =',run=str(FULL))
    fact(rows,'CTW cutout exclusion','none: full mixed grid participates',FULL/'effective_source.py','compute_v21_3_innovation_x(out_1,out_2...)',needle='                innovation_x_loss, innovation_x_endpoint',run=str(FULL))
    fact(rows,'ACDC strong cutout','strong artificial label set0; strong teacher predicts cutout input+foreground LCC, CPS and CTW have no separate cutout exclusion mask',PROJECT/'train_ACDC/dataloaders/dataset.py','WeakStrongAugment/cutout_gray',needle='        mask[y:y + erase_h',run=str(FULL))
    fact(rows,'EMA all state entries','alpha=.99, student1 state_dict including router',FULL/'effective_source.py','update_model_ema',needle='def update_model_ema',run=str(FULL))
    fact(rows,'Validation selection','strict > student1 foreground Dice, every200; db_test aliases val',FULL/'effective_source.py','performance1 > best_performance1; db_test split=val',needle='    db_test =',run=str(FULL))
    fact(rows,'ACDC final surface units/empty/directed ASD','grid, no spacing; one empty diagonal; both empty0; MedPy directed ASD',PROJECT/'tools/v22_0_acdc_test_frozen_checkpoint.py','safe_metrics',needle='def safe_metrics')
    fact(rows,'COVID two views','collate clones same input and masks, no view perturbation in collate',PROJECT/'train_COVID19/covid19_dataset_adapter.py','covid19_two_stream_collate.image_strong',needle='        "image_strong": image_batch.clone(),')
    covid_source=PROJECT/'experiments/three_innovation_ablation_transfer_v22_0/COVID10_FINAL3/self_train/train_promise12_bcp_fft.py'
    fact(rows,'COVID FFT actual inputs','net_input_1/net_input_2 same-coordinate mixed duplicate views; low-frequency spatial components exchanged, threshold30',covid_source,'process_image_batches_for_model(net_input_1,net_input_2,30)',needle='            image_patch_last = process_image_batches_for_model')
    fact(rows,'COVID geometry generation error','512→256 in-plane resize retains XYZ original spacing; no resize spacing update',PROJECT/'train_COVID19/covid19_official20_v2.py','build_master_manifest spacing / resize_slice',needle='                spacing=np.array(image.header.get_zooms()[:3]')
    fact(rows,'COVID reconstruction correct axis order','dataset stackaxis0 gives ZXY for inference, evaluator stackaxis2 reconstructs XYZ and uses XYZ spacing; axis order itself correct',PROJECT/'train_COVID19/covid19_volume_evaluator.py','reconstruct_patient_volume',needle='    pred_vol = np.stack')
    fact(rows,'pretraining creation provenance','NA: no generation run log/command proves file budget/seed; exact file identity and net/opt available',PROJECT/'train_ACDC/Pre_model/label7/unet_best_model.pth','net/opt',checkpoint=protocol['pretrain'])
    fact(rows,'lambda scan switches','TRC ON,TCSMix ON,seed42,30000updates,valbest,no test access',PROJECT/'configs/paper_internal_evidence_v23_0/ACDC10_X03_WEIGHT_001.json','mask_token_enabled/adamix_enabled/seed/max_iterations/checkpoint_selection')
    fact(rows,'default lambda provenance','.05 was V21.3 shared candidate-hook default before sensitivity; no record proving optimized default rationale',PROJECT/'train_ACDC/v21_3_acdc10_third_innovation_transfer.py','parser.v21_3_innovation_x_weight',needle='default=0.05')
    fact(rows,'E5 official entry/commit',subprocess.check_output(['git','-C',str(WORK/'external/AdaMix'),'rev-parse','HEAD'],text=True).strip(),WORK/'external/AdaMix/exp_acdc/train_adamix_ct.py','main / two students / four AdaMix objects')
    write_csv(WORK/'reports/fact_provenance.csv',rows)
    # Estimate wall time only from actual historical time-stamped training log.
    times=[]
    for text in full_log.read_text().splitlines():
        m=re.match(r'\[(\d+):(\d+):(\d+)\.(\d+)\]',text)
        if m: times.append(sum(int(v)*k for v,k in zip(m.groups(),[3600,60,1,.001])))
    elapsed=times[-1]-times[0]+86400*sum(b<a for a,b in zip(times,times[1:]))
    estimate=dict(historical_full_log=str(full_log),sha256=sha(full_log),first_seconds_of_day=times[0],last_seconds_of_day=times[-1],
                  elapsed_seconds=elapsed,hours=elapsed/3600,updates=30000,interpretation='historical observed duration, not guarantee on current server; E5 duration unknown')
    atomic_json(WORK/'reports/historical_budget.json',estimate)
    units=[]
    for percent in (5,10):
        path=PROJECT/f'formal_test/covid19_seed1337/COVID19_{percent}P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json'
        payload=read_json(path)
        units.append(dict(method='beta-FFT baseline',ratio=f'{percent}%',unit='mm',aggregation='patient volume macro, n=4',
                          source=str(path),evaluator=payload['evaluator_path'],evaluator_sha256=payload['evaluator_sha256'],
                          checkpoint=payload['checkpoint_path'],checkpoint_sha256=payload['checkpoint_sha256'],status='LEGACY_MM_LABEL_GEOMETRY_ERROR_CONFIRMED'))
        path=PROJECT/f'final_report/three_innovation_postfull_test_v22_1/raw_test/COVID{percent}_FINAL3/TEST_RESULT_FULL_PRECISION.json'
        payload=read_json(path)
        units.append(dict(method='Full',ratio=f'{percent}%',unit=payload['BoundaryUnit'],aggregation='patient volume macro, n=4',source=str(path),
                          evaluator=payload['EvaluatorPath'],evaluator_sha256=payload['EvaluatorSHA'],checkpoint=payload['CheckpointPath'],checkpoint_sha256=payload['CheckpointSHA'],status='LEGACY_MM_LABEL_GEOMETRY_ERROR_CONFIRMED'))
        dest=WORK/'reports/evidence_sources'/path.relative_to(PROJECT); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(path.read_text())
    # External-row identity cannot be invented from an unavailable manuscript.
    units.append(dict(method='external rows in manuscript Table4',ratio='5%/10%',unit='NA',aggregation='NA',source='NA: no local frozen Table4 external-row/evaluator mapping located',
                      evaluator='NA',evaluator_sha256='NA',checkpoint='NA',checkpoint_sha256='NA',status='BLOCKED_UNIT_AUDIT_ONLY'))
    for percent in (5,10):
        for method in ('baseline','Full'):
            units.append(dict(method=method+' corrected-native-grid reevaluation',ratio=f'{percent}%',unit='mm',aggregation='patient volume macro, n=4',
                              source=str(WORK/'reports/covid_existing_prediction_reevaluation.csv'),evaluator=str(WORK/'scripts/reevaluate_existing_covid.py'),
                              evaluator_sha256=sha(WORK/'scripts/reevaluate_existing_covid.py'),checkpoint='see per-case CSV',checkpoint_sha256='see per-case CSV',status='CORRECTED_SEPARATE_POLICY'))
    write_csv(WORK/'reports/covid_metric_unit_audit.csv',units)
    for percent in (5,10):
        path=PROJECT/f'formal_test/covid19_seed1337/COVID19_{percent}P_BASELINE_CANON1337_R1_CONFIRMATORY_TEST_ONCE/FORMAL_TEST_RECEIPT.json'
        dest=WORK/'reports/evidence_sources'/path.relative_to(PROJECT); dest.parent.mkdir(parents=True,exist_ok=True); dest.write_text(path.read_text())
    (WORK/'protocol_audit.md').write_text(f'''# 协议核查

服务器n1，持久NFS目录{WORK}，实际A100-SXM4-40GB，Python {PYTHON}（torch2.4.1+cu121）。无历史原稿/权重覆盖，无数据大规模复制。本轮只新增ACDC10%E1–E5各单次，未自动重跑其他配置。

Full来源{FULL}/effective_source.py与train.log，baseline来源{BASE}/source_snapshot.py与train.log。每项事实和代码行见reports/fact_provenance.csv，每个原始文件/检查点SHA见reports/source_index.csv。当前拷贝源码和产生论文数值的历史运行分列，不用当前代码推测失落历史日志。

病例划分70/10/20患者，对应1312训练切片、20验证frame体积、40测试frame体积。标注病例：{', '.join(protocol['labeled_patients'])}，历史规则首136切片。seed42，2D256²，B24含12标注，30000updates，SGD .01/momentum.9/wd1e-4，预训练optimizer状态恢复；循环无LR调度。验证每200update，student1前景macroDice严格提高才选择best。历史名为test的重复验证loader实际split=val；ACDC测试不参与方法选择，遵循作者更正。

原始预训练文件net/opt及SHA已核验。生成该文件的训练日志、实际命令/seed/预算未定位，保持NA，不能用当前官方pre_train默认10000来补写历史事实。E1–E4仍能定义为完全复用同一已知文件初始化的对照。

最终统一评价逐切片nearest resize、回原H5网格重建frame体积，无预测LCC，RV/MYO/LV与frame等权平均。HD95为MedPy双向95百分位，ASD为prediction→GT单向距离，无spacing参数，单位grid。空预测/非空GT给体积网格对角线惩罚；都空时Dice/IoU/HD95/ASD=0。这与验证辅助函数空预测HD=0的口径不同；锁定原验证Dice选择，最终评价使用统一完整程序。A4统计配对差值时先把两个frame聚合到patient，独立单位为20患者。

COVID本地baseline和Full的5%/10%行历史均标mm，patient-volume macro、n=4；baseline确认receipt和Full调用都指向covid19_volume_evaluator.py，但512→256缩放后仍保留原spacing，距离未体现正确像素间距。最终沿axis=2重建XYZ的轴序正确。历史口径逐病例精确复现，另存两种修正spacing/原网格口径，不覆盖旧值。该程序ASD是合并双向表面距离均值，与ACDC单向ASD不同。外部文献Table4的逐行身份/程序/spacing映射未在本地定位，仅该单位审计项BLOCKED，继续五项训练和其余分析，不索取论文/审查报告。见reports/covid_metric_unit_audit.csv。

历史Full实际日志耗时{elapsed/3600:.3f}小时，当前机器耗时不能据此保证。E1–E4合计可参考约{4*elapsed/3600:.2f}历史小时，E5未知；全部预算150000新增SSL/co-trainingupdates，没有新增预训练。计时smoke不作为正式初始化或科学结果。
''')
    (WORK/'implementation_audit.md').write_text('''# 实现核查

旧Mask Token对应TRC（4个16维共享学习参考，没有预设解剖语义），旧AdaMix BCP-lite对应本文TCSMix；外部Senyh/AdaMix单独命名official AdaMix-CT。旧ContourReliabilityWeight/CRW对应CTW。

ACDC两视图共用空间rot/flip/rotate及resize，strong再cutout和ColorJitter。cutout人工标签为0；teacher目标由strong输入重新预测并每前景LCC，未强制cutout区域teacher目标为背景；CPS在全mixed输出上计算，CTW全部mixed像素参与，没有独立cutout遮挡排除。本轮保持历史行为，不凭正文疑问宣称实现错误。ICT强视图teacher概率同样从扰动输入取得。

CTW q来自两学生detached最大类别概率的均值。上下/左右后向绝对差相加，边界方向首项为0；逐图minmax分母下限1e-6；w=.25+r；类别均值L1(p1,detach p2)，全图均值，lambda.05，仅student1受此项梯度。E1每图mean(w)广播、均值和w detached，实际参与区域全mixed网格。E2/E3同一97²矩形、同位置随机规则/两视图共享掩码、互补交换和目标处理；诊断统计不修改训练损失。EMA系数.99包含router参数和state buffers。

E4只选宽18的普通1x1(16→18)/LN/GELU/1x1(18→16)零gamma残差，无token/原型/注意力。647参数vs641（+0.936%），接口与原特征16×256²一致；计算MAC/pixel投影576vs512，TRC另有两次tokeneinsum128，因此所计MAC合计576vs640（−10%），不含归一化/激活等。学生2与EMA教师都替换。宽度按最近参数数目决定，没有多宽度训练或测试选择。

COVID克隆输入后collate不再增强，两条mixed输入用同一mask，实际FFT交换net_input_1/2的低频空间分量（ifft取abs），不是随意声称幅度谱交换。纯复制情况下两输入的低频交换无额外内容扰动（可能仅浮点舍入）；模型dropout和两学生参数仍可产生不同预测。Full与baseline均由同一covid19数据适配读取manifest及collate；需用历史配置核对其他default-off扩展，不能把代码中未启用的25D功能算入COVID机制。

保存学生1/2、teacher、优化器、step、epoch、主进程Python/NumPy/torch/CUDA RNG、采样batch IDs和LR状态。FP32无AMP，amp_state=None。断点原子写盘，last为完整续训状态，best仅按验证提高选择；16worker增强RNG/预取队列无法完整恢复，故resume_exact=false。记录这一限制，不声称原历史恢复实现已位级精确。工程修改见source补丁及每run的revision_effective_source.py；旧代码未修改。
''')
    (WORK/'hyperparameter_provenance.md').write_text('''# 系数与调度来源

图5扫描0/.01/.03/.05/.07/.10为TRC ON、TCSMix ON，seed42，30000updates，best validation Dice，保持现有原始扫描，无重新扫描。原始表与配置见reports/evidence_sources；数值90.1655/90.0328/90.0488/89.9908/90.0887/90.1823，不是多种子方差。

.05在V21.3候选hook中是共用默认值，并用于原Full；未发现证明该值通过预先数值优化确定的历史理由，保持NA，不因.10验证值更高改主系数。

TCSMix端点.18/.38不是训练过程观测范围。实际q批均高；固定尺度源自历史150个每200步记录，实际面积均97²/256²=14.3569946%。记录是稀疏时间覆盖，不能冒充全步轨迹。E2/E3固定s=.37890625，按round/clamp实际面积误差0；RMS边长仅辅助，未用test反推。A1新增逐step记录真实q/s/面积/LR/损失，旧轨迹不补写。

AdaMix-CT pinned commit85361fb91cf43a69a158d41dfa9e7ee9fce36db9，官方exp_acdc/train_adamix_ct.py。原生U-Net[16,32,64,128,256]，AdamW1e-4默认wd.01，无teacher/β-FFT/TRC/CTW；保留四加载器、DSCLossH、.95阈值、patch_size8、topk16、hard模式p.5、自步课程总步的四分之一。锁定30000updates以和Full比较，改变官方200epoch及val-loss选择为同协议每200step的student1 val-macroDice。每加载器B16、标注集重复至1312与原训练全集辅助池；7病例标签固定，官方图像PIL量化/增强照旧。初始化从头、无预训练；总预算不隐藏阶段。
''')
    (WORK/'dependency_report.md').write_text('''# 依赖准备

绝对解释器/home/renguanming/.conda/envs/beta_fft/bin/python，torch2.4.1+cu121，torchvision.19.1+cu121，h5py/MedPy/scikit-image/tensorboardX/matplotlib/SimpleITK已实测导入；CUDA可用。使用A10040GB，驱动550.54.14。

官方AdaMix固定版本已完整git clone在external/AdaMix，所有代码本地可用；新增easydict1.13/unfoldNd.2.3/pandas2.3.3/thop.1.1.post2209072238下载到独立external/python_deps，不修改原环境。pandas仅上游报告用，运行入口去除这一未使用import，不需要其pytz。官方开发环境Python3.8/torch1.11，实际采用当前2.4.1并用真实batch smoke检查兼容性，差异明示。

后台禁用在线跟踪登录；数据、初始化、源码、依赖在服务器持久磁盘，无临时授权链接、客户端挂载、代理隧道依赖。GPU资源不足时WAITING_RESOURCE每30秒检查，不停止其他任务，不OOM后静默改B/loss/budget，不自动重试代码错误。
''')
    (WORK/'README_RUN.md').write_text(f'''# β-FFT ACDC10% 修订实验

工作目录{WORK}。普通Python队列scripts/run_queue.py负责E1→E2→E3→E4→E5、最终评价、A1–A6、汇总和打包；脱离agent/SSH继续。正式训练由通过的state/preflight.json及源码/配置SHA gate控制；不重复完成项、不读取smoke作初始化。

配置configs/E*.yaml采用JSON（YAML1.2合法子集），protocol.json是固定原协议，fixed_scale.json是锁定面积来源。scripts/train_E*.sh显式cwd和绝对环境。run_manifest.json记录命令/配置SHA/预算/依赖/状态，state/queue.json为原子心跳，logs留traceback和退出码。锁state/controller.lock防双控制器。state/stop_requested为安全停止请求；下个训练step完成后保存last并退出。

`{WORK}/scripts/start_tmux.sh`启动；`{WORK}/scripts/status.sh`查看；`{WORK}/scripts/stop.sh`安全停止；`{WORK}/scripts/resume.sh`同run恢复。失败配置保持FAILED且无无限自动重试，其他独立配置继续。已完成DONE不重训。中断续训主状态和batch IDs保留，workerRNG边界见implementation_audit.md。

会话beta_fft_revision_20261006，`tmux attach -t beta_fft_revision_20261006`进入，Ctrl-b随后d detach。tmux能跨SSH断连，不能跨服务器/容器重启；重启后运行resume.sh，不修改系统启动服务。当前执行namespace中n1有GPU与持久NFS盘；容器/作业被系统终止同样属于需手动恢复的中断，不保证tmux存活。

最后自动生成experiment_results.csv、case_metrics.csv、class_metrics.csv、analysis真实统计/PNG/SVG、FINAL_REPORT_CN.md、EDITOR_HANDOFF_CN.md、SHA256SUMS.txt和revision_evidence_20261006.tar.gz。内部CSV分数为比例，ACDC表面距离grid；COVID历史mm标签存在缩放spacing问题，修正距离另列mm。归档排除原始影像/GT数组/权重/凭据，仅checkpoint路径+hash。负结果保留。
''')
    # Concrete code patch lets reviewers inspect the frozen-loop transformations.
    from train_entry import build_beta_source
    for task in ('E1','E2','E3','E4'):
        _,cfg=config_for(task)
        before=(WORK/'source/historical_full.py').read_text(); after=build_beta_source(cfg)
        patch=''.join(difflib.unified_diff(before.splitlines(True),after.splitlines(True),fromfile='historical_full.py',tofile=task+'_revision_effective_source.py'))
        (WORK/f'reports/{task}_source.patch').write_text(patch)
    geometry_addendum='\n\n## 已证实的COVID缩放spacing问题\n\n代码与原NIfTI头共同证实：512→256的in-plane resize未更新spacing；最终evaluator沿axis=2正确重建XYZ，所以轴序本身正确。此前仅根据dataset ZXY堆叠作出的轴序判断已撤回，以完整重建链路为准。历史距离调用的mm标签没有反映resize后的正确像素间距；前述单位说明只描述标签/调用一致性。没有修改旧值，另存corrected-derived-center-grid及corrected-native-grid评价。具体患者shape/spacing/原GT文件SHA见reports/covid_geometry_audit.csv，全部三口径原始统计见covid_existing_prediction_reevaluation.csv。新native口径对预测nearest回原canonical网格，GT不缩放，spacing(Z,X,Y)。该问题不改变ACDC grid距离对照，也不新增COVID训练。\n'
    for path in ('protocol_audit.md','implementation_audit.md','hyperparameter_provenance.md'):
        with (WORK/path).open('a') as f: f.write(geometry_addendum)
    sourcefiles={str(p.relative_to(WORK)):sha(p) for p in (WORK/'source').rglob('*.py') if '__pycache__' not in p.parts}
    atomic_json(WORK/'reports/frozen_source_versions.json',dict(git_repository=False,source_sha256=sourcefiles,official_commit=read_json(WORK/'configs/E5_adamix_ct.yaml')['official_commit']))
    print(f'Audited {len(rows)} facts; historical Full duration={elapsed/3600:.3f} hours')

if __name__=='__main__': main()
