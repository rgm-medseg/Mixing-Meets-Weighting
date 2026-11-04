"""Write factual phase handoff. No training, no tmux, no approval creation."""
import collections
import difflib
import json
import shutil
import subprocess
from common import *

def main():
    assert read_json(WORK/'reports/postcheck_PASS.json')['status']=='PASS'
    cfg=read_json(WORK/'configs/short_selected.yaml');stat=read_json(WORK/'reports/short_statistics.json');measurement=read_json(WORK/'reports/measurement_summary.json')
    selection=read_json(WORK/'reports/controller_selection.json');cost=read_table(WORK/'analysis/controller_cost.csv')
    checkpoints=read_json(WORK/'reports/short_checkpoint_checks.json')['checkpoints'];diagnostics=read_table(WORK/'control_diagnostics.csv')
    cp_table='\n'.join(f"|{x['step']}|`{x['path']}`|`{x['sha256']}`|" for x in checkpoints)
    scope=[
        ('教师控制统计','预期改变','旧全图平均最大概率→原LCC支持上类别平衡概率间隔；背景不参与；没有前景时0','scripts/controller.py;control_diagnostics.csv;short_trace.csv'),
        ('尺度映射与取整','保持一致','.18+.20q；Python round ties-even；尺寸clamp[1,H−1]；边长比例，面积约s²','controller.py:geometry;逐step反算PASS'),
        ('单矩形与位置采样','保持一致','一个批次共享矩形；np.random.randint(0,max(1,H−ph))原exclusive约定；不改选区/方向','controller.py:generate_mask;control_loop.patch'),
        ('混合图与目标/双视图','保持一致','同一个image mask/loss mask；互补混合；weak/strong空间对应；图/目标同步','preflight loss块逐字一致;short_trace mask检查'),
        ('教师伪标签后处理','保持一致','复用本来已生成的plab_a/plab_b：argmax+每前景类LCC；不新增阈值或LCC调用','control_loop.patch;source/historical_full.py:get_ACDC_masks'),
        ('有效像素','保持一致','weak参考视图的全部像素；前景支持按原pseudo；不读无标注GT、不新设过滤阈值','controller.py:signals;offline样本清单'),
        ('TRC','保持一致','4token、dim16、原decoder接口、gamma0；641参数/router','source/train_ACDC/v14_mask_adamix_components.py;checkpoint strictload'),
        ('模型和参数','保持一致','原两个UNet学生和EMA教师；student1参数1827469；新增可学习参数0','source/train_ACDC/networks/unet.py;short_checkpoint_checks.json'),
        ('CTW','保持一致','空间权重.25+r；原差分/minmax/floor1e−6；概率类别均L1、student2 detach、只student1、lambda.05','control_preflight.json真实张量与原CTW逐值一致;short_trace逐step系数'),
        ('全部其他loss','保持一致','mixed CE/Dice、CPS、FFT、student1 supervised/ICT；ICT最大系数.5、u_weight.5、原ramp','preflight从consistency_weight到zero_grad块逐字一致'),
        ('EMA','保持一致','.99从student1整个state_dict，包括TRC与buffers；每步教师原4次前向，无新增','source historical loop;preflight forward count'),
        ('split和标注病例','保持一致','ACDC70/10/20；原7标注patient021/038/049/050/058/099/100，136切片','protocol.json;data_manifest_hashes.csv'),
        ('初始化','保持一致','复用精确相同pretrain net与opt；正式R1/R2重新从该源开始；短程不继承E1','short初始log;protocol pretrain hash477a666…'),
        ('batch/增强','保持一致','B24/12 labeled；6+6 unlabeled weak两组；256²；16workers；原WeakStrongAugment cutout/ColorJitter','copied dataset source hash;actualNamespace'),
        ('随机性和优化器','保持一致','seed42；SGD.01/momentum.9/wd1e−4；恒定LR；移植原预训练opt','full有效循环/short Namespace/LR每step'),
        ('验证与选择','保持一致','每200步原student1/2 val程序；严格提高student1前景macroDice选best；历史名testloader仍val','source loop;short200日志'),
        ('最终评价','保持一致/本阶段未执行','保持原ACDC nearest回原体积、grid单位、无预测LCC、原空例/类frame均值；正式指标NA','scripts/evaluate.py与旧程序同SHA；本阶段未调用'),
        ('运行预算','短程检查差异','正式配置仍30000联合步/单次；本阶段240联合步=480 optimizer调用，500调用硬上限','update_budget.json;short_segments.json'),
        ('耐久状态/恢复','工程差异','复用已审计原子checkpoint/sampler IDs/main RNG；加入无状态controller版本/config SHA；worker增强流非精确','checkpoint metadata;short_checkpoint_checks.json'),
        ('必要日志/安全','工程差异','逐step支持数量/margin/尺寸/面积/fallback；调用计数和正式phase拒绝；NaN/Inf控制输入失败退出','control_runtime.py;formal_refusal_checks.json'),
        ('正式R2','预期改变/尚未执行','同B信号仍记录，仅固定几何；尺度NA，必须由R1全轨迹面积另锁定','R2_planned.yaml;lock_R2_from_R1.py'),
        ('历史完整源码版本','无法核验','旧run有效循环可核验，但当时每个import/预训练生成预算未完整保留；不补造','原mechanism audit、protocol.json'),
    ]
    write_csv(WORK/'reports/scope_diff.csv',[dict(item=a,status=b,behavior=c,evidence=d) for a,b,c,d in scope])
    manifests=[]
    for split in ['train_slices','val','test']:
        path=DATA/f'{split}.list';manifests.append(dict(split=split,path=str(path),sha256=sha(path),entries=len(path.read_text().splitlines())))
        shutil.copyfile(path,WORK/f'reports/{split}.list.txt')
    write_csv(WORK/'reports/data_manifest_hashes.csv',manifests)
    original=PROJECT/'revision_runs_20261006'
    for f in ['TCSMIX_MECHANISM_AUDIT.md','scale_distribution.csv','EXPERIMENT_DECISION_TABLE.md']:
        shutil.copyfile(original/'mechanism_audit_20261006_v2'/f,WORK/f'reports/prior_{f}')
    for name in ['historical_scale_log_excerpt.txt','Full_namespace_and_initialization.txt']:
        shutil.copyfile(original/'mechanism_audit_20261006_v2/sources'/name,WORK/f'reports/prior_{name}')
    (WORK/'reports/new_controller.patch').write_text(''.join(difflib.unified_diff(
        [],
        (WORK/'scripts/controller.py').read_text().splitlines(True),fromfile='frozen_parent_components.py',tofile='new_stateless_controller.py')))
    (WORK/'REFERENCES_CN.md').write_text('''# 针对性方法来源

只核查以下两项，不开展综述，也不把它们的结果归给本项目设计。

- Adaptive Mix：使用预测与伪标签的代理分割损失和age控制规则/patch数量，并按confidence排序patch；本项目不移植其选择方向或多patch结构。[作者全文v2](https://arxiv.org/html/2407.21586v2)、[作者仓库](https://github.com/Senyh/AdaMix)。仓库及Crossref正式记录为Medical Image Analysis 108，103857，2026年2月；[正式DOI](https://doi.org/10.1016/j.media.2025.103857)。arXiv v2是2025-04-20，不能称作2026新预印本。正式元数据保存于reports/AdaMix_publication_metadata.json；本地原官方commit85361fb…仅作代码参照，未启动AdaMix训练。
- UCAD：用teacher entropy估计区域不确定性，在superpixel上做选择/位移，并有不确定性加权loss；本轮借鉴“控制对象需明确”的比较，不引入superpixel、位置选择或新loss。[作者全文v1](https://arxiv.org/html/2601.17366v1)、[作者代码](https://github.com/dcb937/UCAD)。作者[arXiv记录](https://arxiv.org/abs/2601.17366)声明Accepted by ISBI2026；本次实际核验的是2026-01-24预印本v1和作者代码，未核对最终会议出版页/DOI。代码树SHA及entropy/softmax位置存于reports/UCAD_source_tree_metadata.json、UCAD_code_inspection.json。

前景类别平衡概率/间隔及empty fallback0是本项目本轮候选设计。引用上述方法不构成该候选有效性证明。
''')
    (WORK/'TCSMIX_CONTROL_DESIGN_CN.md').write_text(f'''# TCSMix 控制设计：v2_B

本轮锁定**前景类别平衡概率间隔控制器v2_B**；映射端点、取整、随机位置及混合结构不变。状态等待正式训练确认，正式新版Dice/IoU/HD95/ASD均NA。旧Full90.35%仍属于old版本。时间{now()}。

## 原路径与改动位置

```mermaid
flowchart LR
  X[无标注weak参考视图 两组6+6] --> T[原EMA教师 logits BxCxHxW]
  T --> P[softmax 类别轴dim1 detach]
  P --> O[old 全像素最大概率均值]
  P --> L[原argmax及每前景类LCC 已有伪标签]
  P --> M[本类概率 减最大其他类概率]
  L --> S[按预测前景类别支持 汇总两组像素]
  M --> S
  S --> Q[v2_B 有支持类等权 无前景取0]
  O --> F[s等于0.18加0.20乘q]
  Q --> F
  F --> D[H与W乘s Python round并原限幅]
  D --> R[原随机位置 单矩形 batch及双视图共享]
  R --> Y[互补混合 图与目标同mask]
```

原信号包含大量背景，连续s变化在97格内被取整吸收；未发现softmax维度/传值错误。旧Full150已记录步和旧E1全部3976已执行步都是97²，缺失Full迭代仍未知。新控制信号在原plab_a/plab_b已生成位置读取，不新增教师前向或伪标签后处理。

令P=detach(softmax(z,dim=1))，支持Ω_c为两组参考视图**原LCC处理后**预测为前景类c的全部有效像素并集，n_c=|Ω_c|。类别映射来自configs/ACDC_class_map.json（background0、RV1/MYO2/LV3）；核心函数按实际C与foreground_ids工作，已检查2/5类例子。

```text
m_c = (1/n_c) × Σ[v∈Ω_c] (P_c(v) − max[k≠c] P_k(v))
A = {{c : n_c > 0}}
q_B = (1/|A|) × Σ[c∈A] m_c   when |A| > 0
q_B = 0                       when |A| = 0
```

原支持是argmax类（LCC只移除区域），所以每个间隔数学上在[0,1]；代码只对浮点负误差作0下限，并拒绝与argmax不符的负间隔支持。背景不参与q；未出现类别跳过，不除零、不填1；无前景时q=0、s=.18、256²取46²。缺失类别不自动说明分割失败。没有额外阈值、温度、可学习分支、状态平滑或GT筛选。

仍s=.18+.20q，h=clamp(round(Hs),1,H−1)、w同理。s是边长比例，实际面积h*w/(H*W)，并非s。位置仍原np.random.randint上界exclusive约定，mask内0外1；一个mask共享批次/双视图，互补图像和目标同步。q与mask全程no_grad；CTW等损失和EMA维持原Full。

伪代码：

```python
P = cat([softmax(logits_a.detach(), dim=1),
         softmax(logits_b.detach(), dim=1)], dim=0)  # N,C,H,W
y = cat([existing_plab_a, existing_plab_b], dim=0)   # original LCC
class_margins = []
for c in configured_foreground_ids:
    support = (y == c)       # across BOTH equal-size unlabeled groups
    n_c = support.sum()
    if n_c > 0:
        others = [k for k in range(P.shape[1]) if k != c]
        margin = P[:, c] - P[:, others].amax(dim=1)
        class_margins.append(mean(margin[support]))
q = mean(class_margins) if class_margins else 0
s = 0.18 + 0.20*q
h, w = original_round_and_clamp(H*s, W*s)
mask = original_shared_random_rectangle(h, w)
```

## 候选取舍与真实诊断

先评估A=前景类别均衡mean(P_c)。70个固定离线batch（5来源×2模式×7批）中，old全部97；A总体91–95，但E1/3200与3976的train_snapshot每个7批都94，Full/29600每批都95。因这一阶段内剩余取整退化，仅离线比较一次B，不训练A、不调端点。B令两个E1快照各出现91/92、Full/29600出现92/93；Full/30000的train_snapshot仍全部93。取舍据前景/背景不变性、整数响应及简洁性，不读取测试分数或按Dice选方案。

所有70批：B尺寸86–94；train_snapshot35批最多93，最常见93占13/35=37.14%，old97占35/35=100%；eval35批最常见93占12/35=34.29%。这些混合了不同来源/时刻，不能误写为单一训练轨迹。实际240步新B短程才是新机制局部轨迹，85–90、88最多占38.75%、变化176次、fallback0；同一新版teacher回放old均97。变化不是效益证明。

来源：相同pretrain重建的SSL0教师（非留存step0快照）、E1均匀CTW教师3200/3976、旧Full教师29600/30000。Full中期检查点未保留，不能声称早/中/末完整覆盖。固定70样本：24索引等距无标注切片+6个无标注患者两端切片12张+原7标注患者14张+10验证患者20个frame中间切片；参考weak seed4261006，纯背景/缺类通过预测或已允许GT诊断标明，不用未标注GT挑片。eval和train_snapshot（每批恢复独立副本buffers、固定dropout）分别记录；静态回放不等于v2教师全程。

控制概率张量检查显示增加100个背景像素或改变类别像素数100:1，A/B保持同前景控制值，old改变；前景概率/间隔单调提高时q和整数尺寸单调非降。真实数据跨批背景占比与预测状态一起变化，不能仅凭散点称其完全背景无关。单样本回放只用于响应诊断，正式训练仍12张汇总；其中eval有10次fallback、train_snapshot2次，真实batch均无fallback，极小预测类唯一支持train_snapshot可只有1像素、eval可只有2像素，类均衡可放大其影响。原单切片回放调用signals(l,l)以复用双组信号，曾重复记录像素数；已将单样本计数除2，保留原表及修正receipt，概率/尺寸/质量未改。

GT只用于原标注/验证诊断，预测quality按相同LCC后的标签统计；fixed三前景macro双空类0，同时给GT-present-class Dice、GT类别数量和像素比例。不把切片/重复checkpoint当独立病例。例E1/3976 eval patient030_frame02:z5：q_B=.86061007，但三类macroDice=.40042189，真实前景类仅2/3、present-class Dice=.60063283；说明高间隔仍可能错误，不能叫校准正确率或轮廓可靠性。相关性表均为描述性，不能说明因果或用于候选挑选。

实现和范围见scripts/controller.py、reports/control_loop.patch、SCOPE_DIFF_AUDIT.md；针对性两方法比较与正式/预印本区别见REFERENCES_CN.md。控制器无状态，无新增模型参数或教师前向。正式收益目前未知，后续R1/R2单次正式实验尚待用户明确确认。
''')
    (WORK/'SCOPE_DIFF_AUDIT.md').write_text('# 改动范围逐项核查\n\n'+
        '|项目|分类|实际行为|证据|\n|---|---|---|---|\n'+'\n'.join(f'|{a}|{b}|{c}|{d}|' for a,b,c,d in scope)+
        '\n\n旧工程耐久层被独立复制到新目录；历史入口和全部旧锁定文件仍原SHA。相对历史Full的工程差异包括原子存盘、保存采样ID/主RNG、16worker非位级恢复和日志；科学loss块、EMA、混合方向保持一致。控制前NaN/Inf拒绝是本轮数值检查，不能在异常输入上制造高置信fallback。\n\n短程真实执行的controller原文件保存在reports/controller_executed_short.py。完成后仅修正尚未运行的R2固定分支日志，使其继续记录B信号；B动态执行路径逐字未改，补充CPU检查已通过。R2_logging_alignment.json记录两个源hash，不能把未运行R2写成已验完整训练。\n\n短程只是max_iterations从30000缩为240并在200处保存/恢复；一轮联合训练调用两个optimizer.step，实际480次调用已累计计数。所有Tensor代码验证、离线前向及成本重复计时为0次优化器调用。\n')
    (WORK/'SHORT_RUN_REPORT_CN.md').write_text(f'''# 短程检查报告

唯一SHORT_TCSMIX_V2运行已完成，240个联合SSL步、**480次真实optimizer.step调用**，本阶段500调用硬上限未触发。为满足500次optimizer.step硬上限，主体取240个联合步；每联合步调用两个optimizer.step，不能同时采用300–500联合步的建议。没有给A单独训练、没有其他smoke优化器更新、没有修复后训练重试。new方法正式指标NA。

fresh段1–200安全退出75，最后完整checkpoint200；resume段201–240正常退出0。两段同一锁定配置与run，从原Full相同pretrain初始化，非E1续训；controller B、原空间CTW及TRC。实际命令、解释器、时间及完整stdout/train.log均入包。停止后未接管旧队列或正式R1/R2。

|实际尺寸|步数|占比|
|---|---:|---:|
{chr(10).join(f'|{d}×{d}|{n}|{n/240:.4%}|' for d,n in stat['sizes'].items())}

q_B范围{stat['q_min']:.9f}–{stat['q_max']:.9f}；实际面积{100*stat['area_min']:.8f}%–{100*stat['area_max']:.8f}%；最常见88²占38.75%，相邻step尺寸变化176次，fallback0。old_q回放这同一新教师输出时240步全97；不是另训old Full的对照，也不是旧Full全程轨迹。short_trace.csv覆盖连续1–240，逐step公式、每类margin/数量、面积、mask、lambda.05与LR.01核查PASS。

200/240完整checkpoint都实际严格载入三模型和两套optimizer，controller元数据hash相符、stateless=true，保留RNG/采样计划。resume日志与201–240真实轨迹证明续训；16worker增强流/预取不保存，不能声称位级无中断重现。短程checkpoint仅诊断，不可作为正式初始化。

|step|服务器checkpoint|SHA256|
|---|---|---|
{cp_table}

已通过：范围/损失块一致、原空间CTW与更新方向、无新增参数/教师前向、数据与预训练来源、图/目标共享mask、逐步控制公式、数值稳定、原子完整状态、同run恢复、自动预算退出、正式入口拒绝。未知/未证明：v2完整训练指标、全程尺度/分割收益、完整阶段稳定性、校准或因果解释；NaN/Inf测试拒绝策略属于代码检查。

耗时：校验与配置完成的入口timestamp到各段子进程结束，合计{measurement['actual_measured_segment_seconds']:.3f}s（约{measurement['actual_measured_segment_seconds']/60:.2f}min）。该口径排除timestamp之前早期imports/admission，resume之间阶段分别计数。原兼容receipt给194.075s，因其恢复偏移读取200步验证前的progress，遗漏上一段验证尾部，**不作为实际总耗时**；原receipt保留并在measurement_summary.json解释。正式估时采用179个稳定步median {stat['stable_step_median_seconds']:.6f}s，加单个200步周期观测的验证/保存/安全退出尾部{measurement['first_validation_checkpoint_stop_tail_seconds']:.3f}s，非使用有偏receipt。

训练峰值CUDA allocated为{stat['peak_training_cuda_bytes']}bytes={stat['peak_training_cuda_bytes']/2**30:.4f}GiB（FP32、B24/12标签、256²、双学生+EMA，同A100）。该值是本轮运行实测，不声称与old配对训练差值。相同Full29600 teacher预计算logits+LCC支持上的控制器/生成mask微基准：old median{float(cost[0]['median_ms']):.4f}ms，B {float(cost[1]['median_ms']):.4f}ms，增量{float(cost[1]['median_ms'])-float(cost[0]['median_ms']):.4f}ms；临时allocated增量差{(int(cost[1]['peak_incremental_allocated_bytes'])-int(cost[0]['peak_incremental_allocated_bytes']))/2**20:.3f}MiB。warmup10/同步计时30，无训练更新，排除teacher及LCC前向处理；不是端到端训练成本增幅。参数增量0，教师前向增量0。
''')
    (WORK/'FULL_READINESS_CN.md').write_text(f'''# 正式训练准备状态

**WAITING_FOR_FULL_APPROVAL**。控制器实现、离线诊断、240步短程、同run保存恢复、范围与成本检查均完成，可以提交R1正式实验方案；本阶段不创建tmux、不启动正式Full、不恢复旧E1。R1/R2实际入口当前都在模型初始化前拒绝formal配置，这是用户明确阶段边界，不是运行故障。

R1：原Full仅换TCSMix v2_B，空间CTW/TRC/EMA及其余协议维持；configs/R1_planned.yaml；同一pretrain net+opt（SHA477a666c3c603c44a87247d3c3cbc76d89bcaa7aeb03db234ca9ac3a36827bf3）、seed42、相同7标注/136切片、B24/12、SGD.01、FP32、30000联合更新，单次，从原pretrain重新初始化。不得使用short200/240或E1/3976权重。

R2：configs/R2_planned.yaml同协议，只固定几何，B控制仍用于可比日志；fixed_side=NA待R1全程真实面积。lock_R2_from_R1.py要求TRAINED/30000及完整control_trace1..30000，按原方形比选择最接近mean实际像素面积的整数平方，另写R2_fixed_locked.yaml及绝对/相对/百分点误差，保留原template；不读测试指标。恢复产生重复记录原表保留，均值用最后一次对应最终逻辑step的记录并披露重复数。当前offline/short面积只能估计，不能锁作R1全程均值。R2运行准备依赖R1完成与尺度锁定，当前不能启动。

稳定训练计时median {stat['stable_step_median_seconds']:.6f}s/联合步，纯计算约{stat['full30000_compute_hours_estimate']:.2f}h/30000；加现有每200步验证/存盘尾部估计约{measurement['projected_with_observed_200step_tail_hours']:.2f}h。规划**每项约6.7–7.5 GPU小时，两项约13.4–15.0 GPU小时**，另加评价/分析及资源等待；来自本机179个稳定短程步，不保证完整阶段速度，单个尾部还包含人为安全停止、误差需披露。新版R1完整指标全部NA。

仍未解决：B也可能在某一时刻所有固定batch同尺寸（Full30000 train_snapshot全部93）；类平衡放大极小/错误前景类，概率间隔不是校准准确率；仅有初始化/早期E1/末期Full静态来源，无Full中期teacher；短程不能证明完整训练收益；worker恢复非位级精确。选择B只据控制响应，不据测试或诊断Dice排名。

后续只有用户明确确认R1/R2正式训练后，才进入独立正式phase、构建新的tmux与普通服务器队列，R1训练→冻结val-best→确定固定面积→R2及统一评价/分析。当前短程入口上限250联合步且正式配置拒绝；不通过修改其max步数或拿short checkpoint冒充正式启动。正式阶段需单独批准的入口/队列并保留本证据快照。原E1–E5队列保持INTERRUPTED，原配置、checkpoint与历史结果均不变。
''')
    (WORK/'EDITOR_HANDOFF_CN.md').write_text('''# 给写作助手的交接

本轮版本TCSMix v2_B，保留模块TRC/TCSMix/CTW名称及论文标题。准确方法：teacher weak logits→类别轴softmax、detach→原argmax/每类LCC前景支持→各前景类mean(P_c−max其他类P)→有支持类别等权均值；全无前景取0。原s=.18+.20q，整数round/clamp、随机位置、单共享mask/双视图互补混合保持原定义。背景与像素面积不直接进入控制平均；不是概率校准、GT可靠率或轮廓度量。公式与伪代码见TCSMIX_CONTROL_DESIGN_CN.md。

可写事实：受控背景/类别面积变化时A/B均值保持、前景概率提高时控制单调；70个固定离线batch old全97而B86–94，部分stage内A仍固定所以选择B；Full末期一个snapshot仍单尺度。真实short240步B85–90、88最多38.75%、176次变化、实际面积11.0244751%–12.3596191%，fallback0。同一新teacher old回放全97，明确是反事实控制回放，不能称另训baseline收益。仅诊断响应/实现，不是完整版训练提升。

数据与GT说明：原7标注患者14张与10验证患者20个中间frame切片用于quality；另外36张无标注训练/边缘切片不读取GT。每个来源各自保留patient/slice/checkpoint/hash，早期E1为uniform CTW教师，SSL0为原pretrain构建的零门控teacher，不是存下的历史step0；中期Full缺失。切片/重复checkpoint不作独立病例，Spearman只描述；all3宏Dice双空类0，另给GT-present宏Dice/GT类别与像素数。高控制仍可能错误；保留不利例与极小预测区域。

修改论文公式/图注时区分old与v2。旧Full Dice90.35%（统一旧复评精确90.349657652%）仍只属于old。v2正式Dice/IoU/HD95/ASD=NA，R1/R2未启动。原八组合/原E1控制不能自动充当v2消融或交互证据；本阶段没有方法收益、跨seed显著性或新版CTW交互结论。后续R1/R2仅单次、固定面积从R1完整实际轨迹锁定，不据测试指标挑值，也不能证明teacher条件化优于同分布随机尺度。

图：analysis/control_replay.png/SVG为静态同输入teacher回放；short_control_trace为新B真实局部训练、200标记恢复边界，old/A线只是同teacher信号回放；controlled_signal_response是人工概率验证。各图原始CSV与manifest均入包。CSV概率/Dice/面积为fraction0–1，尺寸pixel、step联合更新、计数pixel；short预算严格按两个optimizer调用计，240联合步=480次调用。成本计时repeat不是训练重复。

原CTW空间差分/.25+r/L1/stop-gradient/.05/方向、TRC4×16、EMA.99、loss/数据/增强/优化器保持原Full，逐项证据见SCOPE_DIFF_AUDIT.md。正式阶段仍需用户明确确认，当前WAITING_FOR_FULL_APPROVAL。短程权重不可作为正式初始化。
''')
    env=subprocess.run([str(PYTHON),'-m','pip','freeze'],capture_output=True,text=True,check=True).stdout
    (WORK/'reports/environment_freeze.txt').write_text(env)
    tracked=read_json(WORK/'state/old_inputs_locked.json');assert all(sha(p)==h for p,h in tracked.items())
    atomic_json(WORK/'reports/code_identity.json',dict(time=now(),source_files={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'source').rglob('*.py'))},
        script_files={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'scripts').glob('*.py'))},
        config_files={str(p.relative_to(WORK)):sha(p) for p in sorted((WORK/'configs').glob('*')) if p.is_file()},
        actual_short_controller_sha256=sha(WORK/'reports/controller_executed_short.py'),
        old_inputs_unchanged=len(tracked),host='n1',gpu='GPU0 A10040GB',python=str(PYTHON),
        source_control='project has no Git repository; exact frozen sources, content SHA and actual unified patches supplied'))
    (WORK/'README_CN.md').write_text('''# TCSMix 控制修订证据

先读TCSMIX_CONTROL_DESIGN_CN.md、SHORT_RUN_REPORT_CN.md、FULL_READINESS_CN.md、SCOPE_DIFF_AUDIT.md、EDITOR_HANDOFF_CN.md。正式实验未授权，当前WAITING_FOR_FULL_APPROVAL，原队列保持暂停。

control_diagnostics.csv：同输入/同checkpoint的old/A/B控制与原LCC支持，scope=batch与single_sample_response分开；row cohort不同勿当训练总体分布。short_trace.csv：唯一B短程真实1–240步、两optimizer调用总480；old_q/q_A是同v2 teacher的回放，不是独立old/A训练。
analysis/包含尺寸分布、相关性、GT类别存在性、概率受控检查、微基准成本、PNG/SVG图与来源。reports/offline_A_initial保留先A诊断及原脚本；report/control_diagnostics_before_GT_metadata保留追加GT语义字段前的同分数版本，不能计为独立样本/训练。
checkpoint路径与hash见reports/short_checkpoint_checks.json、offline_checkpoint_inventory.json，weights只在服务器，不入包。无原始影像/GT数组/凭据。

可审查入口：scripts/offline_diagnostics.py --compare-b（已有结果不能当额外独立实验），check_controller.py、preflight_control.py、run_short_check.py、postcheck_control.py。短程现已完成，run_short_check/入口的复跑会拒绝现有run或预算；不要修改配置绕过。正式R1/R2 planned配置会在初始化前被当前入口拒绝，待用户正式确认后另立阶段和服务器队列。lock_R2_from_R1.py只在R1完整真实轨迹可用后产出新锁定配置。
源代码与变更patch、实际命令/日志、环境、input SHA、五轮检查和静态SHA256清单均入包。原结果和被暂停E1不会被本阶段脚本恢复。
''')
    print('Five reports and editor handoff written. No training or tmux command executed.')

if __name__=='__main__':main()
