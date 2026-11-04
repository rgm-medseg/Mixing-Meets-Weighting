"""CPU-only reporting and paired-state comparisons; no network forward."""
import platform
import sys
sys.dont_write_bytecode = True
from analyze_direction import *


def read(name):
    return list(csv.DictReader((OUT / 'results' / name).open()))


def pick(rows, **kw):
    return next(r for r in rows if all(r[k] == v for k, v in kw.items()))


def val(r, k):
    return float(r[k])


def reaggregate():
    # Reconstruct optional raw directional means after inference, with no new
    # gradient computation. This also makes report regeneration independent of GPU.
    for prefix in ['main', 'factorial']:
        raw = read(prefix + '_image_region.csv')
        for r in raw:
            for k in SUMS:
                r[k] = float(r[k])
            for k in ['pixels', 'error_support', 'source_patient_pixels', 'aligned_pixels',
                      'opposing_pixels', 'neutral_pixels', 'zero_gradient_pixels',
                      'student1_errors', 'student2_errors', 'disagreement_count']:
                r[k] = int(r[k])
        aggregate(raw, prefix)


def state_comparisons():
    patient = read('factorial_patient_region.csv')
    keys = ['mean_e', 'mean_grad', 'disagreement_rate', 'student1_error_rate', 'student2_error_rate',
            'gradient_fraction', 'aligned_gradient_fraction', 'opposing_gradient_fraction',
            'mean_a_signed', 'a_net_fraction']
    contrasts = [('010', '000'), ('001', '000'), ('011', '010'), ('011', '001')]
    rows, summary = [], []
    for child, parent in contrasts:
        for region in sorted({r['region'] for r in patient}):
            for condition in ['uniform', 'CTW']:
                vals = []
                for pid in json.loads((OLD / 'configs/joint_input_plan.json').read_text())['patients']:
                    c = pick(patient, model='arm_'+child, patient=pid, region=region, condition=condition)
                    u = pick(patient, model='arm_'+parent, patient=pid, region=region, condition=condition)
                    row = dict(child_arm=child, parent_arm=parent, region=region, condition=condition, patient=pid)
                    for k in keys:
                        row['delta_'+k] = val(c,k)-val(u,k) if c[k] and u[k] else None
                    rows.append(row); vals.append(row)
                for k in keys:
                    vv = [r['delta_'+k] for r in vals if r['delta_'+k] is not None]
                    summary.append(dict(child_arm=child, parent_arm=parent, region=region, condition=condition,
                                        metric=k, paired_patients=len(vv), patient_mean_difference=float(np.mean(vv)) if vv else None,
                                        positive_patients=sum(x>0 for x in vv), negative_patients=sum(x<0 for x in vv)))
    write_csv(OUT / 'results/factorial_state_patient_contrasts.csv', rows)
    write_csv(OUT / 'results/factorial_state_contrasts.csv', summary)


def make_table(main, factorial):
    labels = [('anatomy', 'Clean anatomy, all sources'),
              ('pseudo_error', 'Teacher-error source, all eligible'),
              ('anatomy__pseudo_correct', 'Anatomy × teacher correct'),
              ('anatomy__pseudo_error', 'Anatomy × teacher error'),
              ('other_valid__pseudo_correct', 'Other valid × teacher correct'),
              ('other_valid__pseudo_error', 'Other valid × teacher error'),
              ('seam__pseudo_error', 'Seam × teacher error'),
              ('cutout__pseudo_error', 'Visible cutout edge × teacher error')]
    table = []
    for region, label in labels:
        u = pick(main, region=region, condition='uniform')
        c = pick(main, region=region, condition='CTW')
        r = dict(panel='A: original main Full', label=label, region=region,
                 pixels=int(c['pixels']), support_patients=int(c['support_patients']))
        for k in ['mean_w', 'mean_grad', 'gradient_fraction', 'aligned_gradient_fraction', 'opposing_gradient_fraction', 'mean_a_signed']:
            r[k+'_uniform'] = val(u,k); r[k+'_CTW'] = val(c,k); r['delta_'+k] = val(c,k)-val(u,k)
        table.append(r)
    names = {'000':'Baseline (000)', '010':'TCSMix (010)', '001':'CTW (001)', '011':'TCSMix+CTW (011)'}
    for arm in ['000','010','001','011']:
        u = pick(factorial, model='arm_'+arm, region='pseudo_error', condition='uniform')
        c = pick(factorial, model='arm_'+arm, region='pseudo_error', condition='CTW')
        r = dict(panel='B: TRC-off trained states, common teacher-error support', label=names[arm], region='pseudo_error',
                 pixels=int(c['pixels']), support_patients=int(c['support_patients']))
        for k in ['mean_w', 'mean_grad', 'gradient_fraction', 'aligned_gradient_fraction', 'opposing_gradient_fraction', 'mean_a_signed']:
            r[k+'_uniform'] = val(u,k); r[k+'_CTW'] = val(c,k); r['delta_'+k] = val(c,k)-val(u,k)
        table.append(r)
    write_csv(OUT / 'results/SUMMARY_TABLE.csv', table)
    text = '''# Quantitative summary: uniform → CTW

One table; panel A is the original Full and panel B is the actual four-model TRC-off factorial. All numbers are source-patient-equal means. L1 shares (%) divide by each patient's full mixed-image consistency-gradient L1 sum. `mean a` uses the supported region's pixel mean, then equal patient mean, in units of 1e-10; positive a locally favours GT cross entropy. Pixel counts are exposures. Each row has 10 supported patients. Spatial rows and teacher rows overlap; do not sum them together. U→C denotes two loss probes at fixed predictions, not before/after training.

| Panel / region or model | Pixels | Total L1 share (%) | GT-aligned L1 share (%) | GT-opposed L1 share (%) | Mean a (×10⁻¹⁰) |
| --- | ---: | ---: | ---: | ---: | ---: |
'''
    for r in table:
        f = lambda k,scale: f"{r[k+'_uniform']*scale:.2f} → {r[k+'_CTW']*scale:.2f}"
        text += f"| {r['panel'][0]} / {r['label']} | {r['pixels']:,} | {f('gradient_fraction',100)} | {f('aligned_gradient_fraction',100)} | {f('opposing_gradient_fraction',100)} | {f('mean_a_signed',1e10)} |\n"
    text += '\nThe total shares in the first two rows reproduce existing observations; their directional columns and the spatial×correctness intersections are new. The baseline model was trained without CTW; panel B still applies the same diagnostic loss probe to every frozen state. Common teacher correctness comes from the archived main-Full EMA target, not each compared model’s own teacher. Visible cutout-edge support excludes occluded interiors. Neutral mass is retained in CSVs. Mean weights, raw gradients, disagreements, errors, signed magnitudes, paired differences and patient counts are in the complete CSVs.\n'
    (OUT / 'SUMMARY_TABLE.md').write_text(text)


def main():
    assert json.loads((OUT/'receipts/primary.json').read_text())['status']=='PASS'
    assert json.loads((OUT/'receipts/factorial.json').read_text())['status']=='PASS'
    reaggregate()
    state_comparisons()
    main = read('main_region_summary.csv'); fac = read('factorial_region_summary.csv')
    make_table(main, fac)
    checks = list(csv.DictReader((OUT/'receipts/main_analytic_checks.csv').open()))
    pp = read('main_patient_region.csv')
    rawmass = sum(val(r,'raw_flip_uniform_gradient_mass') for r in checks)
    fullmass = sum(val(r,'grad_sum') for r in pp if r['region']=='all' and r['condition']=='uniform')
    write_json(OUT/'receipts/environment.json', dict(python=sys.version, torch=torch.__version__, numpy=np.__version__,
               platform=platform.platform(), cpu_threads=4, cuda=torch.version.cuda,
               reporting='CPU aggregation only; no model instantiated by this script'))
    write_json(OUT/'receipts/precision_summary.json', dict(raw_signed_flips=sum(int(r['raw_signed_flips']) for r in checks),
               raw_flip_L1_mass_fraction=rawmass/fullmass,
               robust_direction_flips=sum(int(r['robust_direction_flips']) for r in checks),
               max_mean_weight_roundoff=max(val(r,'mean_weight_abs_difference') for r in checks),
               neutral_rule='relative 1e-6 plus 8 FP32 rounding units at original per-class coefficient'))
    english = '''# Independent manuscript insertion — 16-page 2026-10-17 revision

For Section 4.4:

We extended the frozen main-Full diagnostic by crossing clean anatomical regions with the correctness of the archived teacher hard targets on pseudo-label source pixels. The original 210 validation slices from 10 patients and the predeclared mixed inputs were retained; six repeated source exposures completed the batch plan. For the 0.05-scaled student-consistency term, we compared spatial CTW with each image’s mean CTW weight and computed a=〈g,p₁−onehot(GT)〉; positive a denotes a local logit direction that reduces GT cross entropy. Within teacher-error boundary regions, CTW increased the patient-equal share of the full consistency-gradient L1 mass from 7.88% to 10.18%. GT-aligned and GT-opposed contributions both increased, from 3.61% to 4.83% and from 4.26% to 5.35%, respectively, in all 10 patients. Across all eligible teacher-error pixels, these shares rose from 4.71% to 5.85% and from 6.37% to 6.97%; the patient-equal regional mean a remained negative (−9.38×10⁻¹¹ versus −1.12×10⁻¹⁰). Thus, boundary emphasis includes contributions in both GT directions.

The archived Baseline, TCSMix, CTW and TCSMix+CTW checkpoints with TRC disabled were all available and identity-verified. On identical mixed images, whole-image student disagreement was 1.296%, 1.210%, 1.273% and 1.177%, respectively. Their within-model weighting responses and common-reference teacher-error strata remained configuration dependent; this comparison does not reconstruct each model’s own input-generation rule.

For Discussion:

At fixed predictions, positive detached CTW weights rescale each pixel’s consistency gradient and cannot reverse its exact direction. The observed redistribution therefore does not establish selective correction of erroneous teacher targets: GT-opposed contributions also receive additional mass. Teacher correctness is an offline stratum, whereas CTW compares student probabilities. These first-order logit diagnostics and common-input checkpoint comparisons do not measure an actual parameter update, establish a training causal chain, or assess variability across training seeds. Prediction errors are unchanged when only the frozen loss weights are switched.
'''
    (OUT/'MANUSCRIPT_INSERTIONS.md').write_text(english)
    summary = f'''# 本轮机制方向分析 SUMMARY — 2026-10-17

已完成：复用全量归档、补齐主Full概率、空间×教师正确性分层、GT局部方向分析，以及通过盘点门槛后的TRC关闭四个已训练模型比较。结果是**混合观察**：CTW增加真实边界的即时梯度份额，同时增加GT同向和反向贡献，不能据此声称“不会放大错误”或“只在纠错”。

## 主Full：新增实质证据

- 教师错误来源区域的11.0772%→12.8183%是原有梯度份额，不是全图错误率；本轮逐项复现原表，未把它当作新实验。
- 新交叉分层：真实边界×教师错误支持11,030个暴露像素、10位患者，梯度份额7.8756%→10.1813%（+2.3057 pp，10/10增加）；其中GT同向3.6139%→4.8322%（+1.2183 pp，10/10增加），反向4.2617%→5.3491%（+1.0874 pp，10/10增加）。真实边界×教师正确为67,298像素、10患者；完整权重/强度/分歧/错误率均在CSV。
- 干净非边界×教师错误支持5,089像素、10患者，梯度份额2.3880%→1.7268%。这里同向和反向份额都下降，不能把总体教师错误区域的增加解释为所有空间区域一致增加。接缝与cutout边带独立列示，排除规则未改。
- 所有教师错误来源区域：GT同向全图份额4.7077%→5.8471%，反向6.3695%→6.9712%；份额分别10/10和9/10增加。两类原始L1贡献强度均在10/10患者增加。区域每像素mean a的患者等权均值为−9.3843e−11→−1.1166e−10，CTW−均匀=−1.7817e−11（6/10下降、4/10上升）。因此不能从归一化份额改善推断原始GT方向贡献改善。
- 全部真实边界原有46.60%→56.02%得到精确复现；其中同向24.41%→28.91%、反向22.19%→27.11%。两类均为10/10份额增加。误差率来自固定预测，切换权重必然不变，不能用来证明训练中不强化错误。

所有L1份额分母均为该来源患者全部混合图像的原一致性项梯度L1总和（包括原损失中的padding/cutout），之后患者等权。上述区域不是互斥全图分区。GT只作离线方向参照，没有进入CTW、混合控制、teacher处理或选点。

## 四个实际训练模型：盘点与条件性扩展

已核验000/010/001/011对应Baseline/TCSMix/CTW/TCSMix+CTW，TRC均关闭；best SHA与历史八组合表逐一一致，已完成预算均30,000，既定选点分别21200/22200/25200/26600，公共训练参数相同。TRC开启的另一组四配置也齐全，仅登记。本轮未补训、未按分数选择替代组。原包缺项登记针对固定/动态尺寸×权重组合，与此模块组合是不同清单，旧档未改。

四模型使用同一108个归档混合输入、同一来源患者和同一主Full EMA硬目标分层。全图学生argmax分歧依次1.2960%、1.2096%、1.2726%、1.1773%。联合模型分歧较低，但GT方向、教师错误区域和模型间差值并非所有指标一致改善；不能概括为完整协同因果证据。各模型内部均匀/CTW响应，以及配置间患者配对状态差和响应差均已交CSV。没有运行各自混合规则比较，更没有把冻结Full关模块当作四个训练模型。

## 数值、冻结与样本

主Full沿用原四通道梯度，仅一次冻结Student-1前向补齐缺失概率；概率缓存覆盖108个混合输入，后续只读缓存。216个原来源数组、432个原混合数组均在；新分析使用dynamic的均匀/CTW两组全量数组，不以示例代全体。原10患者/20帧/210独立切片与216来源暴露（6次重复）保持不变。每个动态矩形均为97×97，未挑输入制造尺寸差异。

逐通道缩放关系的最大峰值相对误差4.1212e−7；原始FP32有{sum(int(r['raw_signed_flips']) for r in checks)}个近零符号差异，对应均匀总L1质量的{rawmass/fullmass:.4e}。舍入感知容差外方向翻转为0。首次过严断言的日志保留，容差与中性/零梯度规则见PROTOCOL.md；原始带符号a未被清零。四模型另逐批通过原函数损失与解析导数校验。参数/BN状态摘要、checkpoint SHA前后一致，参数.grad均None，优化器步数0。

## 固定身份与范围

主Full仍为原始X03 / Full_main / 论文R1，Dice90.35%；89.98%的替代foreground-margin控制信号模型独立保留。既有TRC训练消融+0.26 pp；动态90.3497%与固定97×97 89.8548%的未舍入差0.4948 pp保留，均未重跑。多种子与E3取消；病例配对不是训练稳定性证据。PROMISE无新增训练/评价，历史loader未归档状态保留。作者确认的COVID HD95、DC-Net ACDC10% 2.28/0.68 grid记录未修改。

未找到20261017的16页PDF，按用户基准交独立英文与分析，不覆盖任何旧PDF。无Appendix、无新训练、无主表测试重评价、无checkpoint重选、无现有进程干预。

## 交付

- `SUMMARY_TABLE.md` / `results/SUMMARY_TABLE.csv`：唯一一张定量汇总表（两个panel），没有新增图。
- `results/main_*`：患者、区域、图像暴露、配对差值；`results/factorial_*`：四模型对应统计、模型状态配对差、CTW响应差。
- `MANUSCRIPT_INSERTIONS.md`：可用于4.4/Discussion的独立英文段落。
- `PROTOCOL.md`、`COMMANDS.sh`、`scripts/`、`receipts/`与`logs/`：计算与来源可复用。
- `FIVE_REVIEWS.md`：五项审查的实际证据。全量概率和四模型logits在本目录arrays中；紧凑交付包保留其路径/校验和，不重复打包约GB的原始数组。
'''
    (OUT/'SUMMARY.md').write_text(summary)
    print('CPU aggregation, one summary table and independent manuscript text written.')


if __name__=='__main__':
    main()
