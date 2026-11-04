"""Five evidence reviews, data alignment checks and final local artifact index."""
import sys
sys.dont_write_bytecode = True
from analyze_direction import *


def read(name):
    return list(csv.DictReader((OUT/'results'/name).open()))


def audit():
    primary = json.loads((OUT/'receipts/primary.json').read_text())
    factorial = json.loads((OUT/'receipts/factorial.json').read_text())
    gate = json.loads((OUT/'receipts/factorial_gate.json').read_text())
    assert primary['status']==factorial['status']==gate['status']=='PASS'
    assert primary['model_hash_before']==primary['model_hash_after']
    assert primary['load']['checkpoint_sha256']==primary['checkpoint_sha_after']
    for r in factorial['loads']:
        assert r['checkpoint_sha_before']==r['checkpoint_sha_after']
        assert r['model_hash_before']==r['model_hash_after']
    assert primary['optimizer_steps']==factorial['optimizer_steps']==0
    assert primary['parameter_gradients_none'] and factorial['parameter_gradients_none']
    inv = read('trained_condition_inventory.csv')
    assert all(r['status']=='IDENTITY_AND_BUDGET_PASS' for r in inv)
    assert all(r['selected_step']==r['expected_selected_step'] for r in inv)
    plan = json.loads((OLD/'configs/joint_input_plan.json').read_text())
    samples = [s['sample'] for bp in plan['batches'] for s in bp['slots']]
    assert len(set(samples))==210 and len(samples)==216
    assert sum(s['repeat_exposure'] for bp in plan['batches'] for s in bp['slots'])==6
    for bp in plan['batches']:
        for s in bp['slots']:
            assert sha(s['path'])==s['sha256']
    arrays = []
    for b in range(9):
        input_hashes = []
        for arm in ['000','010','001','011']:
            cache = OUT/'arrays/factorial_logits'/f'arm{arm}_b{b}.npz'
            with np.load(cache) as f:
                input_hashes.append(str(f['input_sha']))
        assert len(set(input_hashes))==1
        for j in range(12):
            a = read_array('dynamic_CTW',b,j)
            group,k = divmod(j,6)
            o,ins = (12+k,k) if group==0 else (6+k,18+k)
            slots = plan['batches'][b]['slots']
            outside, inside = (dict(np.load(slots[s]['path'])) for s in [o,ins])
            for key in ['gt','anatomy','geom_valid','occlusion']:
                assert np.array_equal(a[key],np.where(a['outside'],outside[key],inside[key]))
            assert np.array_equal(a['image1'],np.where(a['outside'],outside['weak'][0],inside['weak'][0]))
            assert np.array_equal(a['image2'],np.where(a['outside'],outside['strong'][0],inside['strong'][0]))
            pid = np.where(a['outside'],plan['patients'].index(slots[o]['patient']),plan['patients'].index(slots[ins]['patient']))
            assert np.array_equal(pid,a['patient_ids'])
            assert np.array_equal(a['pseudo_source'],a['outside'] if group==0 else ~a['outside'])
            rr = cross_regions(a)
            support = a['pseudo_source'] & a['geom_valid'] & ~a['occlusion']
            assert np.array_equal(rr['pseudo_correct'] | rr['pseudo_error'],support)
            assert not np.any(rr['pseudo_correct'] & rr['pseudo_error'])
            assert np.array_equal(sum(rr[k+'__pseudo_error'].astype(int) for k in ['anatomy','other_valid','seam','cutout','invalid']),rr['pseudo_error'].astype(int))
            assert not np.any(rr['anatomy'] & (rr['seam']|rr['cutout']|rr['invalid']))
            assert np.allclose(np.abs(a['grad_logits']).sum(0),a['grad'],rtol=1e-6,atol=1e-18)
    for path in sorted((OUT/'arrays').rglob('*.npz')):
        arrays.append(dict(path=str(path.relative_to(OUT)),bytes=path.stat().st_size,sha256=sha(path)))
    write_csv(OUT/'receipts/new_array_manifest.csv',arrays)
    for prefix in ['main','factorial']:
        p = read(prefix+'_patient_region.csv')
        for r in p:
            if float(r['grad_denominator']):
                assert abs(float(r['gradient_fraction'])-sum(float(r[k]) for k in ['aligned_gradient_fraction','opposing_gradient_fraction','neutral_gradient_fraction']))<1e-12
            if r['region']=='all':
                assert abs(float(r['gradient_fraction'])-1)<1e-12
        for model,patient,condition in {(r['model'],r['patient'],r['condition']) for r in p}:
            q = {r['region']:r for r in p if (r['model'],r['patient'],r['condition'])==(model,patient,condition)}
            for target in ['pseudo_correct','pseudo_error']:
                for key in ['pixels','grad_sum','aligned_grad_sum','opposing_grad_sum','a_signed_sum']:
                    total = sum(float(q[k+'__'+target][key]) for k in ['anatomy','other_valid','seam','cutout','invalid'])
                    assert np.isclose(total,float(q[target][key]),rtol=1e-12,atol=1e-18)
        pairs = read(prefix+'_patient_paired.csv')
        for r in pairs:
            for key in ['student1_error_rate','student2_error_rate','disagreement_rate','mean_e']:
                assert not r['delta_'+key] or float(r['delta_'+key])==0
    # Verify the GT-reference derivative independently on archived logits only.
    with np.load(OUT/'arrays/factorial_logits/arm000_b0.npz') as f:
        z = torch.from_numpy(f['logits1'][:1].copy()).requires_grad_(True)
    gt = torch.from_numpy(read_array('dynamic_CTW',0,0)['gt'].astype(np.int64))[None]
    h = z.detach().softmax(1)-torch.nn.functional.one_hot(gt,4).permute(0,3,1,2)
    actual = torch.autograd.grad(torch.nn.functional.cross_entropy(z,gt,reduction='sum'),z)[0]
    h_error = float((actual-h).abs().max())
    assert h_error<3e-7 and z.grad is None
    records = json.loads((OUT/'receipts/read_only_source_records.json').read_text())
    assert all(sha(r['path'])==r['sha256'] for r in records)
    checks = list(csv.DictReader((OUT/'receipts/main_analytic_checks.csv').open()))
    assert all(int(r['robust_direction_flips'])==0 for r in checks)
    fchecks = list(csv.DictReader((OUT/'receipts/factorial_analytic_checks.csv').open()))
    assert len(checks)==108 and len(fchecks)==432
    assert all(int(r['robust_direction_flips'])==0 for r in fchecks)
    table = read('SUMMARY_TABLE.csv')
    assert len(table)==12 and all(int(r['support_patients'])==10 for r in table)
    text=(OUT/'MANUSCRIPT_INSERTIONS.md').read_text()
    for required in ['7.88% to 10.18%', '3.61% to 4.83%', '4.26% to 5.35%',
                     '−9.38×10⁻¹¹ versus −1.12×10⁻¹⁰', 'GT-opposed', 'fixed predictions']:
        assert required in text
    assert not list(OUT.glob('*.pdf')) and not list(OUT.glob('*.svg')) and not list(OUT.glob('*.png'))
    reviews = [
        dict(review='1_identity_configuration',status='PASS',evidence='8 historical arm SHAs and selected steps match; all budgets 30000; actual TRC-off four used; main X03 retained; both student branches strict; frozen state/checkpoint hashes unchanged'),
        dict(review='2_alignment_and_groups',status='PASS',evidence='216 source hashes; all108 GT/image/geometry/occlusion/source-patient compositions exact; teacher strata exclude annotated sources; identical input hashes across four models; original exclusions preserved; 210 unique slices plus6 repeats'),
        dict(review='3_numerics_and_gradients',status='PASS',evidence=f'108 main and432 factorial scaling/direction checks; no robust flips; explicit FP32 cancellation handling; class/batch/lambda reductions checked; independent GT CE derivative max error {h_error:.3e}; L1/signed sums reconcile; weight-switch predictions/errors identical; optimizer steps0'),
        dict(review='4_claims_and_evidence',status='PASS_WITH_MIXED_FINDINGS',evidence='Both GT-aligned and GT-opposed mass rise in teacher-error boundary support; original existing shares identified; raw mean-a remains negative on teacher-error pixels; no selective-correction, no-error-amplification, training-stability or complete-causal-synergy claim; no PROMISE computation'),
        dict(review='5_table_and_english',status='PASS',evidence='One 12-row table in Markdown/CSV with full denominator/support/paired-probe caption; independent 4.4 and Discussion prose retains negative result and scope; no PDF replacement or Appendix')]
    write_json(OUT/'receipts/FIVE_REVIEWS.json',dict(status='PASS_WITH_MIXED_FINDINGS',reviews=reviews,
               method='five sequential evidence reviews by this execution; not five independent reviewers',
               GT_reference_autograd_max_error=h_error, original_source_record_hashes_unchanged=True))
    lines=['# 五项审查\n','这是本次执行的五项顺序审查，不冒充五位独立审稿人。原始过严断言失败与修正日志均保留。\n']
    for r in reviews:
        lines.append(f"## {r['review']}\n\n**{r['status']}** — {r['evidence']}\n")
    lines.append('## 结论界限\n\n审查通过指身份、实现和报告与证据一致；结果本身为混合观察，不能证明CTW只强化纠错或完整训练协同。该分析没有训练、重选checkpoint、重评价主表或PROMISE历史运行。\n')
    (OUT/'FIVE_REVIEWS.md').write_text('\n'.join(lines))
    print(json.dumps(dict(status='PASS_WITH_MIXED_FINDINGS',new_arrays=len(arrays),alignment_mixtures=108,
                          GT_CE_derivative_max_error=h_error,main_checks=len(checks),factorial_checks=len(fchecks))))


if __name__=='__main__':
    torch.set_num_threads(4)
    audit()
