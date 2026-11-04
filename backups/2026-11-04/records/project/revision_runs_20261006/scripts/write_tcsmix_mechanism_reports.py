"""Assemble mechanism reports and a frozen small archive; never starts a queue."""
import argparse
import ast
import csv
import hashlib
import json
import math
import shutil
import subprocess
import tarfile
from pathlib import Path
from common import *

def text_new(path,text):
    with Path(path).open('x') as f: f.write(text)

def main():
    p=argparse.ArgumentParser(); p.add_argument('--output',required=True); a=p.parse_args(); out=Path(a.output).resolve()
    assert read_json(out/'DIAGNOSTIC_RECEIPT.json')['status']=='DONE'
    assert read_json(WORK/'state/queue.json')['status']=='INTERRUPTED' and (WORK/'state/stop_requested').exists()
    rows=read_table(out/'confidence_pooled_statistics.csv'); distributions=read_table(out/'scale_distribution.csv')
    cfgs={t:read_json(next((WORK/'configs').glob(t+'_*.yaml'))) for t in ('E1','E2','E3','E4','E5')}
    receipt=read_json(out/'DIAGNOSTIC_RECEIPT.json'); resume=read_json(out/'E1_checkpoint_restoration_check.json')
    def pooled(run,step,cohort,region):
        return next(r for r in rows if r['run']==run and int(r['checkpoint_step'])==step and r['mode']=='train_snapshot' and r['cohort']==cohort and r['region']==region)
    sources=[(FULL/'effective_source.py','historical_full_effective_source.py'),
        (FULL/'source_snapshot.py','historical_full_source_snapshot.py'),
        (PROJECT/'configs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT.json','historical_full_config.json'),
        (WORK/'source/train_ACDC/v14_mask_adamix_components.py','v14_mask_adamix_components.py'),
        (WORK/'source/train_ACDC/v21_3_acdc10_third_innovation_components.py','v21_3_acdc10_third_innovation_components.py'),
        (WORK/'source/train_ACDC/dataloaders/dataset.py','dataset.py'),
        (WORK/'source/train_ACDC/networks/unet.py','unet.py'),
        (WORK/'scripts/revision_runtime.py','revision_runtime.py'),
        (WORK/'scripts/train_entry.py','train_entry.py'),
        (WORK/'scripts/common.py','common.py'),
        (WORK/'scripts/evaluate.py','evaluate.py'),
        (WORK/'scripts/official_adapter.py','official_adapter.py'),
        (WORK/'scripts/tcsmix_mechanism_diagnostics.py','tcsmix_mechanism_diagnostics.py'),
        (WORK/'scripts/probe_tcsmix_quantization.py','probe_tcsmix_quantization.py'),
        (WORK/'external/AdaMix/exp_acdc/train_adamix_ct.py','official_AdaMix_CT_recipe.py'),
        (Path(__file__),'write_tcsmix_mechanism_reports.py'),
        (WORK/'runs/E1/revision_effective_source.py','E1_revision_effective_source.py'),
        (WORK/'runs/E1/audit/revision_trace.csv','E1_revision_trace.csv'),
        (WORK/'runs/E1/TRAIN_RECEIPT.json','E1_TRAIN_RECEIPT.json'),
        (WORK/'runs/E1/progress.json','E1_progress.json'),
        (WORK/'state/queue.json','paused_queue_snapshot.json'),
        (WORK/'state/preflight.json','preflight_PASS.json'),
        (WORK/'protocol.json','protocol.json'),
        (WORK/'run_manifest.json','run_manifest.json'),
        (WORK/'configs/diagnostic_cases.json','prior_locked_diagnostic_cases.json'),
        (WORK/'reports/E5_OFFICIAL_METHOD_AUDIT.json','E5_OFFICIAL_METHOD_AUDIT.json'),
        (WORK/'COMPARABILITY_AUDIT.md','prior_COMPARABILITY_AUDIT.md'),
        (WORK/'protocol_comparison.csv','prior_protocol_comparison.csv'),
        (WORK/'runs/Historical_Full/evaluation/EVALUATION_RECEIPT.json','Historical_Full_EVALUATION_RECEIPT.json')]
    sources += [(next((WORK/'configs').glob(t+'_*.yaml')),t+'_locked.yaml') for t in cfgs]
    for t in ('E4','E5'):
        sources += [(WORK/f'runs/smoke/{t}/TRAIN_RECEIPT.json',f'{t}_prior_smoke_TRAIN_RECEIPT.json'),
                    (WORK/f'runs/smoke/{t}/progress.json',f'{t}_prior_smoke_progress.json')]
    source_rows=[]
    for src,name in sources:
        dst=out/'sources'/name; shutil.copyfile(src,dst)
        assert sha(dst)==sha(src)
        source_rows.append(dict(original_path=str(src),package_path='sources/'+name,sha256=sha(src),bytes=src.stat().st_size))
    oldcopy=PROJECT/'final_report/paper_evidence_for_mep_draft8_20260907T100152Z/sources/train_ACDC/v14_mask_adamix_components.py'
    assert sha(oldcopy)==sha(WORK/'source/train_ACDC/v14_mask_adamix_components.py')
    source_rows.append(dict(original_path=str(oldcopy),package_path='sources/v14_mask_adamix_components.py (same bytes)',sha256=sha(oldcopy),bytes=oldcopy.stat().st_size))
    write_csv(out/'source_manifest.csv',source_rows)
    # Actual original lines; no reconstructed shell arguments presented as historical command.
    text_new(out/'sources/Full_namespace_and_initialization.txt','\n'.join((FULL/'train.log').read_text().splitlines()[:3])+'\n')
    lines=(WORK/'runs/E1/train.log').read_text().splitlines()
    text_new(out/'sources/E1_log_initialization_and_pause.txt','\n'.join(lines[:4]+lines[-12:])+'\n')
    snippets=[]
    for filename,names in [('v14_mask_adamix_components.py',['generate_adamix_bcp_lite_v1_mask']),
                           ('revision_runtime.py',['generate_adamix_bcp_lite_v1_mask','ctw_maps','sampler_state','set_sampler_resume','resumed_iterator']),
                           ('train_entry.py',['build_beta_source','main'])]:
        path=out/'sources'/filename; body=path.read_text(); ss=body.splitlines()
        for node in ast.parse(body).body:
            if isinstance(node,(ast.FunctionDef,ast.ClassDef)) and node.name in names:
                snippets.append(f'### {filename}:{node.lineno} `{node.name}`\n\n```python\n'+'\n'.join(ss[node.lineno-1:node.end_lineno])+'\n```\n')
    for filename,start,end in [('historical_full_effective_source.py',333,374),('historical_full_effective_source.py',441,462),
                               ('historical_full_effective_source.py',34,70),('historical_full_effective_source.py',105,115),
                               ('dataset.py',436,461),('unet.py',376,394)]:
        ss=(out/'sources'/filename).read_text().splitlines()
        snippets.append(f'### {filename}:{start}–{end}\n\n```python\n'+'\n'.join(ss[start-1:end])+'\n```\n')
    text_new(out/'sources/CODE_CHAIN_EXCERPTS.md','# 实际代码摘录\n\n行号针对包内相同字节的来源文件，完整源文件同时保留。\n\n'+'\n'.join(snippets))
    # Numeric rounding boundaries: exact ties go to the even integer; float32 mean may perturb an exact tie.
    thresholds=[]
    for d in range(59,98):
        lo=max(.25,((d-.5)/256-.18)/.2); hi=min(1.,((d+.5)/256-.18)/.2)
        thresholds.append(dict(side_pixels=d,q_lower=lo,q_upper=hi,area_fraction=d*d/65536,
            exact_tie_rule='Python round ties to even; even d includes half-integer endpoints, odd d excludes them',
            numerical_boundary='analytical real-valued q; actual q float32, scale Python float'))
    write_csv(out/'quantization_thresholds.csv',thresholds)
    checks=[]
    import sys
    sys.path.insert(0,str(WORK/'scripts'))
    from train_entry import build_beta_source
    base=(WORK/'runs/E1/revision_effective_source.py').read_text()
    for task,cfg in cfgs.items():
        if task=='E5': continue
        result=build_beta_source(cfg); assert result==base
        checks.append(dict(task=task,generated_loop_sha256=hashlib.sha256(result.encode()).hexdigest(),
            equals_actual_E1_loop=True,mix_mode=cfg['mix_mode'],weight_mode=cfg['weight_mode'],trc=cfg['trc'],
            fixed_side=cfg['fixed_side'],fixed_used=cfg['mix_mode']=='fixed',cfg_sha256=sha(next((WORK/'configs').glob(task+'_*.yaml')))))
    write_csv(out/'effective_entry_comparison.csv',checks)
    e1=pooled('E1',3976,'unlabeled_train','all'); bg=pooled('E1',3976,'unlabeled_train','pred_bg'); fg=pooled('E1',3976,'unlabeled_train','pred_fg')
    full=pooled('Historical_Full',29600,'unlabeled_train','all'); fbg=pooled('Historical_Full',29600,'unlabeled_train','pred_bg'); ffg=pooled('Historical_Full',29600,'unlabeled_train','pred_fg')
    diagtable=[]
    for run,step in [('Historical_Full',29600),('Historical_Full',30000),('E1',3200),('E1',3976)]:
        all_=pooled(run,step,'unlabeled_train','all'); bg_=pooled(run,step,'unlabeled_train','pred_bg'); fg_=pooled(run,step,'unlabeled_train','pred_fg')
        diagtable.append(f"|{run} / {step}|{float(all_['q']):.9f}|{100*float(bg_['pixel_fraction']):.4f}%|{float(bg_['confidence_mean']):.9f}|{float(fg_['confidence_mean']):.9f}|")
    gtrows=[]
    for run,step in [('Historical_Full',29600),('E1',3976)]:
        for cohort in ('labeled_train','validation'):
            a_=pooled(run,step,cohort,'all'); f_=pooled(run,step,cohort,'gt_fg'); b_=pooled(run,step,cohort,'gt_bg')
            gtrows.append(f"|{run}/{step}|{cohort}|{100*float(f_['pixel_fraction']):.4f}%|{float(f_['confidence_mean']):.8f}|{float(b_['confidence_mean']):.8f}|{float(a_['fg_macro_dice']):.8f}|")
    text_new(out/'TCSMIX_MECHANISM_AUDIT.md',f'''# TCSMix 机制核查

本轮结论：**B 在已覆盖迭代中成立，完整历史轨迹仍有 D 的证据缺口；未发现 A 型实现错误，也没有发现 C 型额外几何变化。** 公式中的连续 s 有变化，实际整数矩形在旧 Full 的 150 个已记录迭代及 E1 的全部 3976 个已执行迭代中均为 97×97。两者不能混称为“全部 Full 训练恒定”。本轮未恢复 E1、未训练、未更新参数、未访问测试数据，所有历史与锁定配置保持原字节。时间：{now()}。

## 1. 教师输入到掩码的计算链

以下路径均为包内实际源文件；original_path 与 SHA256 见 source_manifest.csv。完整摘录见 sources/CODE_CHAIN_EXCERPTS.md。

|阶段|实际行为|代码与配置证据|
|---|---|---|
|教师输入|B24 中无标注12张，拆成6+6；共享几何后的 weak image；strong另一次前向产生目标，但不用于q_mix|historical_full_effective_source.py:333–357；dataset.py:436–461|
|原始输出|pre_a=ema_model(uimg_a)、pre_b=ema_model(uimg_b)，形状[6,4,256,256]；UNet最终out_conv输出logits，不预先softmax|historical_full_effective_source.py:345；unet.py:390；v14_mask_adamix_components.py:_install_router|
|概率|softmax(logits.detach(),dim=1)，类别维正确；max(dim=1)后为[6,256,256]|v14_mask_adamix_components.py:128–130；离线probability_sum误差≤2.3842e-7|
|batch平均|每组对样本和所有空间位置平均，然后两组平均；组等大小时等价全12张平均|v14_mask_adamix_components.py:128–130|
|s映射|q clamp[0,1]；s=.18+(.38−.18)q；min/max参数的保护范围不改变当前端点|v14_mask_adamix_components.py:125–131；Full Namespace及config .18/.38|
|取整/裁剪|int(round(H*s))、int(round(W*s))；Python round 为最近整数、.5向偶数；clamp[1,H−1]/[1,W−1]|v14_mask_adamix_components.py:132–133|
|最终掩码|矩形内0、外1；同一位置/大小扩展到6张，loss_mask一致；两组与双视图复用它|v14_mask_adamix_components.py:134–155；historical_full_effective_source.py:364–373|
|图/目标交换|weak/strong均以同一mask配对互补；目标与loss_mask对应；无其他s输入损失|historical_full_effective_source.py:364–397|

硬标签 get_ACDC_masks(...,nms=1)用于混合目标；生成矩形传入的是 **pre_a/pre_b 原始 logits**，没有传plab_a/plab_b。argmax仅另作patch class_histogram诊断，不参与尺度。未发现张量/维度误用。教师对象保持默认 train 模式；no_grad禁梯度，但不关闭dropout或BN行为。教师EMA在本步优化后更新整个state_dict，包括buffers。因此离线eval、train_snapshot和训练中某一步不是同一统计条件。额外synthetic_probability_mask_probe.csv用人工常数softmax分布验证实际动态函数：q约.25/.50/.90/.98/.995时取59/72/92/96/97，而相同输入下E2/E3一直97；表明动态s确实影响函数，不是被固定覆盖。该探针不是真实教师或历史证据。

日志中的side_fraction是计算得到的连续s；patch_area_ratio是由最终取整后patch_x*patch_y计算的实际交换面积。两者不可互换。原始日志没有逐步保存mask数组/坐标/宽高；在已确认256²、同一s控制两轴、无裁剪触发的实现下，可由面积与计算链唯一核验97×97。离线诊断还直接检查了mask零区域的宽、高及像素数。

E1实际入口 train_entry.py → build_beta_source → revision_effective_source.py，从原Full冻结循环引入revision_runtime；动态分支直接调用上述parent函数。E2/E3仅改runtime的mix_mode为fixed，s=97/256=.37890625；该值也存在于E1配置，但E1的dynamic分支**不读取它**。effective_entry_comparison.csv证明E1–E4生成循环与实际E1循环相同，差异在明确runtime配置。fixed分支仍计算q用于记录，但不据其改变边长。

旧Full每个run保存了有效循环和Namespace；未保存当时所有import模块的逐文件哈希。当前冻结组件与2026-09-07旧证据包副本及项目组件均为SHA256 75754fa75e6fd6fc01c97682262895998799c7ce3c0bc2dcbe058ef4c63dcaee；旧endpoint数值与其规则一致。它加强实现证据，但不能补造2026-08训练时未保存的模块哈希或完整shell argv。

## 2. 轨迹、来源和覆盖范围

|运行|独立尺度记录|step覆盖|实际尺寸种类/占比|实际面积范围|记录间变化次数|未覆盖|
|---|---:|---|---|---|---:|---|
|旧Full|150|200,400,…,30000|97×97：150/150=100%|14.3569946289%–14.3569946289%|0|其余29850步，包括1–199；完整轨迹变化次数NA|
|新增E1|3976|连续1–3976，无重复/缺行|97×97：3976/3976=100%|14.3569946289%–14.3569946289%|0|尚未执行3977–30000|

旧Full q范围0.988724470139–0.997913122177，s范围0.377744894028–0.379582624435，均值0.379189255555。E1 q范围0.985085248947–0.996754169464，s范围0.377017049789–0.379350833893，均值0.378668454599。完整原始逐步E1 trace和两份原始Full endpoint摘录均入包；scale_records.csv标明每条来源行号，scale_distribution.csv给出占比/面积/变化及证据性质。

额外找到controller stdout/stderr的同样150条endpoint，逐值与Full train.log一致，**没有增加独立step**。Full training_dynamics有30000行但没有q/s/mask字段；TFRecord只有total_loss/model1_loss/model2_loss/consistency_weight各30000条，没有尺度。正式run只找到best29600、last30000，未找到中间教师快照或保存掩码；另有X03工程smoke检查点，属于不同run，不能填补Full。history_source_inventory.csv及history_search.json记录搜索范围和文件SHA；不是宣称整个服务器不存在任何未索引备份。

97²匹配旧Full **已记录150个迭代**平均实际面积，取整后面积匹配误差0；不等于证明全程平均面积误差0。旧Full高置信度连续s发生变化而实际尺度未变化；E1直到3976步实际几何确已恒定。

## 3. 独立离线置信度诊断

固定输入在前向前锁定：24张无标注训练切片按列表索引等距选取（不读其GT作评价）；14张既有标注切片来自已锁定7例；全部20个验证frame各取固定中间z切片（10患者）。弱增强seed4261006；分12张批次为两等大小组，尾批明确较小。样本路径/hash和实际输入hash见diagnostic_batch_manifest.csv；本轮不访问测试。

4个现有教师checkpoint：Full best29600/last30000，E1 best3200/last3976。每个都做eval和train_snapshot两种模式，总48批；train_snapshot在每批前后恢复独立模型副本的参数和BN buffers、固定dropout RNG，模型state_digest前后完全相同。无优化器step。它近似当前训练前向的模式，**不是恢复历史batch/worker随机流/连续强弱前向序列**。checkpoint是post-step EMA状态，而本步训练q是pre-update teacher。所有48离线批次的动态mask与重置同一随机状态后的固定97mask完全一致，全部97²，不能据此补造历史轨迹。

|教师/step|无标注整体q（像素汇总）|预测背景比例|预测背景平均置信度|预测前景平均置信度|
|---|---:|---:|---:|---:|
{chr(10).join(diagtable)}

|教师/step|有GT诊断域|真实前景比例|真实前景平均最大概率|真实背景平均最大概率|汇总前景macro Dice|
|---|---|---:|---:|---:|---:|
{chr(10).join(gtrows)}

这里Dice是固定2D诊断样本的逐类汇总，并非模型正式测试/验证整卷得分；all3类double-empty记0，另给GT-present-class Dice防止遗漏类影响被隐藏。例：patient030_frame02:z5，Full29600的q=0.997173667而all3诊断Dice=0.45405、present-class Dice=0.68108；E1/3976 q=0.995913267，对应0.41840/0.62760。高全图q不能保证前景质量。E1/3976 train_snapshot描述性Spearman：14标注切片−0.10330、20验证中间切片0.45865；不同域/模式不稳定，不作独立样本检验/因果推断。

确认事实：背景在这个固定批次中占约95.5%，其高概率在全图平均中占主要权重；前景置信度也较高，但明显低于背景。不能据预测背景多断言前景全部塌缩：有GT样本也约94–95%为真实背景且存在较好前景分割。可能原因：预训练及学习后教师饱和、全像素聚合稀释前景不确定性、batch平均进一步压缩变化；本轮没有初始教师或历史每batch概率分布，不能把这些原因分别定量归因到Full全程。

## 4. 取整阈值与“动态”失效位置

H=W=256时d=round(46.08+51.2q)。每减一像素通常需q变化0.01953125。97区间为 **q>0.984765625** 到1；精确q=.984765625得到96.5，round取偶数96。96内部区间约0.965234375–0.984765625，两个精确半整数端点取96。float32 softmax/均值与Python浮点会影响恰好临界的数值，报告的是解析阈值。裁剪在当前范围不触发。

E1最小q只比97阈值高0.000319623947，但全部已执行step仍落在97区间；不能保证未来始终如此。四类softmax最大概率≥1/4，所以合法q实际范围[.25,1]，对应s约[.23,.38]和59–97像素；46是把形式端点q=0代入的值，不是4类概率可达到的下限。quantization_thresholds.csv列全部可达整数区间。

问题发生在**高且变化窄的全像素批均q，经线性映射后全部落入同一个整数取整格**，而非已查出的参数覆盖或硬标签错误。真实s变化与有效几何变化必须分开描述。随机位置仍每次变化；它不会成为“动态尺度”的证据。当前np.random.randint上界为H−d（exclusive），最后一个合法起点未被采到，是原有/固定控制共有的采样约定，不能将其当成尺度恒定的原因或在本轮静默修正。

## 5. 对实验与论文的影响

E1保留独立CTW空间分配对照价值。CTW_actual_difference.csv基于E1/3976两学生真实输出、12固定标注诊断样本、同一弱/强混合输入，原权重空间SD范围0.03474959–0.06213243，替代权重空间SD为0；空间损失减均匀损失0.000273042–0.001868169。均值一致不保证损失/梯度相同；本轮只前向，未生成参数梯度或更新。lambda .05、概率类别平均L1、student2 detach、student1方向仍由实际函数和既有独立梯度gate支持。

E2相对Full：已记录迭代几何一致，未记录Full迭代可能不同，只有独立控制实际未来全程轨迹才能排除。但在目前证据下，它不能直接支撑“置信度动态尺度优于固定尺度”。E3相对E1：1–3976已执行轨迹在同RNG条件下几何一致，尾程未知，因而存在很高冗余风险。原2×2的尺度因素在已观测范围没有实际干预；即使最终数值不同，也不能仅由开关名称解释为动态调度或协同收益。

E4仍可比较TRC与普通匹配残差适配层；E5仍可比较官方AdaMix-CT核心方法的同协议适配。TCSMix这项发现不否定它们或CTW。具体准备状态、资源和建议见EXPERIMENT_DECISION_TABLE.md与NEXT_STEP_PLAN_CN.md。建议属于方案，队列/config均未改。
''')
    remaining_old=26024*25434.563/30000/3600
    remaining_actual=26024*read_json(WORK/'runs/E1/TRAIN_RECEIPT.json')['training_seconds']/3976/3600
    text_new(out/'EXPERIMENT_DECISION_TABLE.md',f'''# E1–E5 实验决策表

截至{now()}，全队列INTERRUPTED；E1完整checkpoint第3976步，E2–E5正式run均PENDING/0。queue.json的latest_step=3975是控制器最后一次采样值，最终receipt/progress/checkpoint均为3976；本轮保留原状态文件。下表是建议，不修改队列执行顺序或配置。

|实验|实际干预与证据|目前能回答的问题|建议|依赖/准备状态|资源估计，GPU0 A100|
|---|---|---|---|---|---|
|E1|用每图原CTW mean广播空间；原L1/detach/.05/方向不变；原权重有空间SD，均匀后0，同输出损失改变|CTW空间分配相对均匀分配的单次控制；不证明跨seed显著性|保留；建议用户确认方案后按原配置续训；本轮不恢复|第3976步full-state可加载，剩余26024步；预训练、配置、source已锁定；worker随机流恢复不精确|剩余约{remaining_old:.2f}–{remaining_actual:.2f}小时；已用0.9683h；历史/当前速度估算|
|E2|固定97；Full已记录150步97，相同位置RNG则mask同；Full缺29850步尺度日志|最多检验固定控制在完整新run上的行为；当前不能归因动态收益|暂缓；无须为维持原表格而训练近似空干预；尚不能断言全程严格等价|配置锁定、既有3步工程smoke PASS；正式未启动；缺真实Full全轨迹无法补造|若保留约7.07h，潜在冗余|
|E3|同E1均匀函数、同E2固定97；E1全部3976步97|与E1尾程几何是否分叉；目前尺度因子退化|暂缓；冗余风险高，等待方案决定是否取消原定义|同函数/固定值核验、旧smoke PASS；未来26024步q未知|若保留约7.07h，潜在冗余|
|E4|TRC641参数换1×1/LN/GELU/1×1残差647，gamma0，两学生/EMA均替换|共享token检索相对普通残差适配，控制其余实际算法；仅单次对照|保留价值，待统一方案后再启动|同预训练/split/source；既有3步fresh/restore smoke及EMA PASS，无新增依赖；必须继续记录实际尺度|约7.07h粗估；+6参数/student，残差MAC−10%，非总网变化；不能据此精确推算耗时|
|E5|官方核心双学生、四loader、原生AdaMix/DSCLossH，无EMA/TRC/CTW/FFT；wrapper适配协议|与最近外部方法同split/评价的受控比较，不是无改动官方CLI原样运行|保留价值，待方案后再启动|commit85361fb…已锁，依赖本地、四loader+3步保存/恢复smoke PASS；训练入口train_entry→official_adapter；预算/选择/PIL等差异有完整清单|30000 updates，四loader各B16；可靠总耗时NA，3步smoke含启动不足外推；旧smoke显存峰值约8.00GiB仅工程参考|

E1恢复核查实际加载了三个state_dict、两套SGD、主Python/NumPy/Torch/CUDA RNG，重放本epoch剩余6个batch ID且逐项一致；未调用训练入口、DataLoader训练或optimizer.step。epoch361、采样计划11批、已消耗5批；主LR两组均.01、FP32无AMP（amp_state=None），val-best metadata保留。150个参数/学生中90个已有momentum state；其余是未在当前loss图产生梯度的projection/prediction/contrastive heads，SGD按需创建state，完整载入原保存状态而未补造。checkpoint含全部模型参数与buffers，不是best-only权重。

恢复边界：16worker RNG、预取增强张量未保存；重启worker会改变其后数据增强随机实现，resume_exact=False；不宣称与不间断训练位级一致。diagnostic process不改变checkpoint或原进程RNG。本轮恢复可行性由实际加载与采样重放验证，尚未实际执行第3977步。

E2/E3差别的必要条件是动态分支在某执行step取到非97边长；连续s或日志名称不同不是干预。固定分支q本身仍来自softmax，动态分支额外argmax/class_hist只作统计，不新增随机抽样；同97时两者np.random.randint调用范围/次数相同。48个离线固定批次在同RNG下mask逐元素相同，证据限于这些批次，不能推广成历史30000步或未来训练等价证明。

E5完整命令与官方repo/source/adapter哈希在sources/E5_OFFICIAL_METHOD_AUDIT.json；所有E1–E5原配置入包。既有smoke是此前工程检查，不能当正式实验或把其checkpoint用于正式初始化。E4/E5不因TCSMix尺度退化被判为无价值。
''')
    text_new(out/'NEXT_STEP_PLAN_CN.md',f'''# 下一步方案（只提出，不执行）

当前判断：**B：定义已落实，但有效尺度在已观测区间退化；D：Full未记录迭代和后续E1不可核验。** A型张量/维度/覆盖/取整传值错误未发现。C型实际尺寸变化未在额外日志或现有离线诊断出现。保持队列暂停、E1第3976步原checkpoint、全部锁定代码/配置/历史数值。不得将本轮分析当作恢复或更改实验预算的授权。

## 分类对应动作

|情况|判据|E1建议|E2/E3建议|机制说明|
|---|---|---|---|---|
|A 实现错误|发现错误教师张量/维度、s被覆盖、尺寸与mask不符|若改变共同混合定义则不能混用3976步旧机制继续成新Full；原run保留，修订另立版；本轮未发现该情况|先暂缓受影响控制，按错误范围重定义，不能静默改尺度|披露旧实现与新定义差异；旧90.35%仍属于旧版|
|B 定义符合但几何退化|当前所有E1已执行步、150个Full记录都97|可按原配置续训回答CTW空间分配问题，明确非位级精确恢复|建议暂缓，最终可取消原尺度对照以节省计算；不能仅为出分保留空干预|旧机制为教师置信度参数化的混合规则，现有数据不支持有效动态调度的独立收益|
|C 实际变化、采样漏掉|新增原始历史mask/尺度或新run全step确有非97|仍可保留；后续诊断按全step统计|仅当完整实际干预存在且占比足够解释时保留；列真实面积/频次，不凭标签判断|报告变化位置/幅度/覆盖，动态vs固定仍不能证明教师条件化优于等分布随机尺度|
|D 证据不足|不能定位历史某段/q分布/未来分叉|可保留独立CTW价值，不把缺记录当反证|暂缓比立即花14h更合理；不能自动补跑Full|标未知，不补造轨迹，也不宣称整个算法理论上永远固定|

## 推荐的保守方案

保留E1、E4、E5；暂缓原E2/E3。统一方案经用户后续指令后，再恢复E1剩余26024步并执行独立E4/E5。暂缓不是删除原定义：五份cfg、smoke、E1已用计算与结果全部留存，正式队列仍未改变。继续E1时保留当前全部协议，并逐步记录q/s/实际面积；未来若出现96等尺寸，应如实补充，不能以本轮结论盖掉新事实。

E1剩余估计{remaining_old:.2f}–{remaining_actual:.2f}h，E4约7.07h，合计约13.19–13.40h + E5（可靠耗时NA）+ 已计划评价/分析。暂缓E2/E3最多避免约14.13h潜在冗余训练。均为单A100历史/当前单次速度线性估算，含当前协议周期验证，调度等待/后续变化未纳入；E5三步工程smoke无法提供可靠全程估算。不得额外运行计时训练来伪造精度。本轮两个只读诊断版本合计约33秒前向/统计，0 optimizer updates；首版补齐四类softmax统计后保存为v2，v1数据/脚本未覆盖。

## 若要研究TCSMix新机制：两个明确候选，均尚未实现

1. **前景条件置信度**：q_fg=mean(max softmax probability on teacher-predicted foreground pixels)，先合并两组有效像素再平均，避免组前景量不同造成额外权重；若全batch无预测前景，明确fallback为原q_all。保留原s映射、配对/共享几何/目标定义。依据：固定批次背景约95.5%、前景置信度明显较低，原全图平均掩盖这部分变化。风险：预测前景选择偏差、漏检前景不被计入；q_fg高仍不等于校准质量，也不保证经过取整后变化。不能先根据测试分数决定fallback或更改端点。
2. **按预测类别平衡的归一化熵代理**：q_entropy=1−mean_c(mean_pixels(pred=c)[H(p)/log4])，对本batch存在的预测类别等权，空类别不计。仍只来自教师概率，不读训练未标注GT。保留原几何与混合形式，和候选1互斥、只选择一个版本；依据是全像素占比与最大概率饱和，可对概率分布的非最大类别质量更敏感。风险：class-balanced校准/预测类别选择偏差、仍可能饱和；先用预先锁定训练/验证诊断证明它是否产生可分辨整数面积及前景质量对应，不选最高验证/测试Dice的公式。

这些是有待核验的新定义，**不是本轮确认的修复**。不要仅放大映射范围/换取整规则来制造“动起来”的表象。随机取整即使带来几何变化，也可能只是噪声，不能当可靠置信度调度；本轮不把它列为优先方案。若两个候选的固定批次几何仍恒定，应报告失败并选择保守论文修订，不自动继续搜参数。

## 新版与剩余预算的边界

若用户决定研究一个新版，独立命名TCSMix-v2并新建源文件/commit或内容哈希、配置和run ID；原Full/E1以及未启动原E2/E3仍保留。改变q定义会改变整个混合训练过程，**不能从3976步旧E1接续后称作新版Full**。新版从原已核实初始化按完整30000步单次训练，其分数、A1/A2/A3/A5及模型指标重新计算；旧Full90.349657652%归旧定义，不能转给新版。

可供之后批准的资源替换方案：使用原E2/E3的两个尚未花费完整训练预算，另命名R2-v2-dynamic、R3-v2-fixed（不修改原E2/E3文件）。先预注册一个q定义、所有共同协议、分析和选择规则；完成v2 dynamic后，仅用其训练全step面积轨迹、在读取测试分数前锁定v2固定矩形，报告可表示面积的匹配误差，再单次训练v2 fixed。这样至多两项新完整训练，约14.13h旧协议粗估，不能声称这是已授权自动重新分配。若新版无可辨几何变化，则不启动第二项冗余训练。

这个两项对照能评估**新版实际变量几何相对面积匹配固定几何**，仍不能单独证明置信度条件化优于独立随机但同分布的尺度；后者需要预注册的等分布控制，当前预算不能同时补齐所有控制，应收窄结论，不自动追加。v1的E1/E4仍支持v1 CTW/TRC比较，不能直接充当v2对应消融；如要声称新版三组件交互，需另批准相应重新评估，不能沿用退化的旧2×2差值。

## 论文可以保留与必须调整的说法

可以给出实际q、映射式、取整/随机位置/共享mask及旧Full90.35%的原始来源；明确连续s自适应但观察到的整数面积不变。应删除或降格“实际尺度随置信度在训练中扩大/动态调度带来独立性能收益/2×2证明尺度与权重协同”的现有证据性表述。改为“教师置信度参数化混合；在所核查ACDC设置中有效交换面积接近固定97²，动态几何贡献未被现有记录建立”。E1/E4的CTW/TRC机制及E5比较独立陈述，持平/下降结果也保留。其他数据集未做本轮尺度全轨迹核查，不把ACDC发现自动推广，也不替其宣称已验证动态收益。
''')
    # Keep the first diagnostic pass in the small archive, including its actual script and receipt.
    prior=WORK/'mechanism_audit_20261006_v1'
    if prior.exists():
        shutil.copytree(prior,out/'prior_diagnostic_v1')
    readme='''# 小型机制证据包

先读TCSMIX_MECHANISM_AUDIT.md、EXPERIMENT_DECISION_TABLE.md和NEXT_STEP_PLAN_CN.md。
scale_distribution.csv是按run/尺寸聚合的真实记录；scale_records.csv为逐step的原始来源映射。
confidence_diagnostics.csv按offline batch、sample、region分别记录；scope=batch时无样本Dice，未标注域没有GT指标。
confidence_pooled_statistics.csv按固定诊断域汇总；confidence_probability_histograms.csv和softmax_probability_statistics.csv给出原始4类概率与最大概率分布。
batch_confidence_distribution.csv描述固定批次间分布，不是训练时间序列；所有概率/Dice为0–1，面积fraction亦0–1，尺寸单位pixel。
gt_present_macro_dice仅平均GT存在的前景类；fg_macro_dice固定3类且双空记0。切片诊断分数不能当正式整卷测试结果。
offline_mask_comparison.csv固定同一checkpoint/输入/随机状态比较真实输出mask；不能替代历史mask记录。
E1_checkpoint_restoration_check.json只有独立加载/状态重放检查，不执行续训；CTW_actual_difference.csv只有同输出前向损失比较。
sources/保存实际代码、5份锁定配置、Full原始endpoint摘录、E1完整trace、队列暂停快照、既有smoke/官方方法来源与哈希。
prior_diagnostic_v1/保留本轮首版诊断和原脚本；v2补齐四类softmax原始统计，未改算法、样本或原状态。两版数据不得当独立运行/种子。
诊断图为Matplotlib实测数据图（PNG/SVG），图例run/step对应teacher_checkpoint_inventory.json。无生成影像、原始患者数组或大型权重入包。
SHA256SUMS只校验这个静态快照里的文件，不要求继续刷新的服务器其他报告一致。归档外部另有sha256与receipt。
'''
    text_new(out/'README_CN.md',readme)
    tracked=read_json(out/'input_hashes_before.json'); actual={p:sha(p) for p in tracked}; assert tracked==actual
    atomic_json(out/'FINAL_VALIDATION.json',dict(time=now(),status='PASS',all_tracked_inputs_unchanged=True,tracked_files=len(tracked),
        queue_status='INTERRUPTED',E1_checkpoint_step=3976,stop_requested=True,configs_changed=False,
        new_full_training_runs=0,optimizer_updates=0,diagnostics_only=True,required_files_present=all((out/f).exists() for f in
        ['TCSMIX_MECHANISM_AUDIT.md','scale_distribution.csv','confidence_diagnostics.csv','EXPERIMENT_DECISION_TABLE.md','NEXT_STEP_PLAN_CN.md'])))
    files=sorted(p for p in out.rglob('*') if p.is_file())
    text_new(out/'SHA256SUMS',''.join(f'{sha(p)}  {p.relative_to(out)}\n' for p in files))
    # The output directory is immutable during archiving; validate archive member bytes against its manifest.
    dst=WORK/'tcsmix_mechanism_evidence_20261006_v2.tar.gz'; assert not dst.exists()
    with tarfile.open(dst,'w:gz') as t: t.add(out,arcname=out.name,recursive=True)
    with tarfile.open(dst,'r:gz') as t:
        manifest=t.extractfile(f'{out.name}/SHA256SUMS').read().decode()
        for line in manifest.splitlines():
            digest,name=line.split('  ',1)
            actual=hashlib.sha256(t.extractfile(f'{out.name}/{name}').read()).hexdigest()
            assert actual==digest, name
    outer=sha(dst)
    text_new(Path(str(dst)+'.sha256'),f'{outer}  {dst.name}\n')
    atomic_json(WORK/'reports/tcsmix_mechanism_archive_receipt_v2.json',dict(time=now(),status='VERIFIED',archive=str(dst),
        bytes=dst.stat().st_size,sha256=outer,manifest_records=len(files),all_archive_member_hashes_verified=True,
        snapshot=str(out),queue_status='INTERRUPTED',E1_step=3976))
    print(json.dumps(dict(archive=str(dst),bytes=dst.stat().st_size,sha256=outer,files=len(files)),ensure_ascii=False))

if __name__=='__main__': main()
