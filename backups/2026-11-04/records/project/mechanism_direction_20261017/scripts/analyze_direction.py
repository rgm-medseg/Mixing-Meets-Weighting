"""Reuse full archived gradients; one cached frozen student-1 pass for missing p1.

Never calls the historical pipeline, training loop, test evaluator or optimizer.
All writes are confined to this new package. The original region implementation
is imported read-only; GT is used only after the predictions and gradients exist.
"""
import argparse
import csv
import hashlib
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.dont_write_bytecode = True
OUT = Path(__file__).resolve().parents[1]
ROOT = OUT.parent
OLD = ROOT / 'mechanism_strengthening_20261013'
sys.path.insert(0, str(OLD / 'scripts'))
import numpy as np
import torch
from common import load, digest, sha
from diagnose_joint import regions

TOL = 1e-6  # dimensionless |<g,h>| / ||g||_1


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def write_csv(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open('w', newline='') as f:
        w = csv.DictWriter(f, fields)
        w.writeheader()
        w.writerows(rows)


def read_array(condition, batch, pair):
    path = OLD / 'arrays/joint' / f'{condition}_b{batch}_p{pair}.npz'
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def cross_regions(a):
    rr = regions(a)
    # Existing cutout > seam > anatomy > other_valid > invalid priority is literal.
    for spatial in ['anatomy', 'other_valid', 'seam', 'cutout', 'invalid']:
        for target in ['pseudo_correct', 'pseudo_error']:
            rr[spatial + '__' + target] = rr[spatial] & rr[target]
    return {k: v for k, v in rr.items() if k not in ['RV', 'MYO', 'LV', 'background']}


def direction(g, p, gt):
    g = g.astype(np.float64)
    h = p.astype(np.float64).copy()
    np.put_along_axis(h, gt[None].astype(np.int64),
                      np.take_along_axis(h, gt[None].astype(np.int64), axis=0) - 1, axis=0)
    a = (g * h).sum(axis=0)
    l1 = np.abs(g).sum(axis=0)
    score = np.divide(a, l1, out=np.zeros_like(a), where=l1 > 0)
    labels = np.where(score > TOL, 1, np.where(score < -TOL, -1, 0))
    return a, l1, score, labels


def paired_direction(u, c, p):
    au, lu, su, du = direction(u['grad_logits'], p, u['gt'])
    ac, lc, sc, dc = direction(c['grad_logits'], p, c['gt'])
    ratio = c['w'].astype(np.float64) / u['w']
    h = p.astype(np.float64).copy()
    np.put_along_axis(h, u['gt'][None].astype(np.int64),
                      np.take_along_axis(h, u['gt'][None].astype(np.int64), axis=0) - 1, axis=0)
    # FP32 softmax backward subtracts nearly equal terms on saturated pixels.
    # Bound eight rounding units at the original per-class loss coefficient;
    # use the same uncertainty band in uniform-gradient units for both arms.
    coeff = .05 * u['w'].astype(np.float64) / (12 * 256 * 256 * 4)
    tau = TOL * np.maximum(lu, lc / ratio) + 8 * np.finfo(np.float32).eps * coeff * np.abs(h).sum(0)
    cu = np.where(au > tau, 1, np.where(au < -tau, -1, 0))
    cc = np.where(ac / ratio > tau, 1, np.where(ac / ratio < -tau, -1, 0))
    robust_flips = int(((cu * cc) < 0).sum())
    assert robust_flips == 0, 'direction flip exceeds FP32 uncertainty bound'
    common = np.where(cu == cc, cu, 0)
    raw_flips = (au * ac) < 0
    return common, dict(raw_signed_flips=int(raw_flips.sum()),
                       raw_flip_uniform_gradient_mass=float(u['grad'][raw_flips].sum(dtype=np.float64)),
                       robust_direction_flips=robust_flips,
                       tolerance_edge_disagreements=int((cu != cc).sum()),
                       max_normalized_direction_difference=float(np.abs(su-sc).max()))


SUMS = ['w_sum', 'e_sum', 'grad_sum', 'disagreement_count', 'student1_errors',
        'student2_errors', 'pixels', 'error_support', 'aligned_pixels',
        'opposing_pixels', 'neutral_pixels', 'zero_gradient_pixels',
        'aligned_grad_sum', 'opposing_grad_sum', 'neutral_grad_sum',
        'a_positive_sum', 'a_negative_magnitude_sum', 'a_signed_sum',
        'a_absolute_sum', 'grad_denominator', 'a_absolute_denominator',
        'source_patient_pixels']


def extract(rows, model_name, condition, batch, pair, a, p, patients, labels):
    adot, l1, _, own_labels = direction(a['grad_logits'], p, a['gt'])
    rr = cross_regions(a)
    for pid in np.unique(a['patient_ids']):
        pm = a['patient_ids'] == pid
        denom = float(a['grad'][pm].sum(dtype=np.float64))
        adenom = float(np.abs(adot[pm]).sum(dtype=np.float64))
        for region, mask in rr.items():
            m = pm & mask
            ev = m & a['geom_valid'] & ~a['occlusion']
            pos, neg, neutral = m & (labels == 1), m & (labels == -1), m & (labels == 0)
            r = dict(model=model_name, condition=condition, batch=batch, pair=pair,
                     patient=patients[int(pid)], region=region,
                     pixels=int(m.sum()), error_support=int(ev.sum()),
                     source_patient_pixels=int(pm.sum()), grad_denominator=denom,
                     a_absolute_denominator=adenom,
                     student1_errors=int(((a['pred1'] != a['gt']) & ev).sum()),
                     student2_errors=int(((a['pred2'] != a['gt']) & ev).sum()),
                     disagreement_count=int(a['disagree'][m].sum()),
                     w_sum=float(a['w'][m].sum(dtype=np.float64)),
                     e_sum=float(a['e'][m].sum(dtype=np.float64)),
                     grad_sum=float(a['grad'][m].sum(dtype=np.float64)),
                     aligned_pixels=int(pos.sum()), opposing_pixels=int(neg.sum()),
                     neutral_pixels=int(neutral.sum()), zero_gradient_pixels=int((m & (l1 == 0)).sum()),
                     aligned_grad_sum=float(a['grad'][pos].sum(dtype=np.float64)),
                     opposing_grad_sum=float(a['grad'][neg].sum(dtype=np.float64)),
                     neutral_grad_sum=float(a['grad'][neutral].sum(dtype=np.float64)),
                     a_positive_sum=float(np.maximum(adot[m], 0).sum()),
                     a_negative_magnitude_sum=float(np.maximum(-adot[m], 0).sum()),
                     a_signed_sum=float(adot[m].sum()), a_absolute_sum=float(np.abs(adot[m]).sum()))
            rows.append(r)


def aggregate(rows, prefix):
    groups = defaultdict(list)
    for r in rows:
        groups[r['model'], r['condition'], r['patient'], r['region']].append(r)
    patient_rows = []
    for (model, condition, patient, region), vals in sorted(groups.items()):
        r = dict(model=model, condition=condition, patient=patient, region=region,
                 mixed_image_exposures=len(vals))
        r.update({k: sum(v[k] for v in vals) for k in SUMS})
        n, ne, gd, ad = r['pixels'], r['error_support'], r['grad_denominator'], r['a_absolute_denominator']
        for name, key in [('mean_w', 'w_sum'), ('mean_e', 'e_sum'), ('mean_grad', 'grad_sum'),
                          ('mean_aligned_grad', 'aligned_grad_sum'), ('mean_opposing_grad', 'opposing_grad_sum'),
                          ('mean_neutral_grad', 'neutral_grad_sum'),
                          ('disagreement_rate', 'disagreement_count'), ('mean_a_signed', 'a_signed_sum'),
                          ('mean_a_positive', 'a_positive_sum'), ('mean_a_negative_magnitude', 'a_negative_magnitude_sum')]:
            r[name] = r[key] / n if n else None
        for name, key in [('student1_error_rate', 'student1_errors'), ('student2_error_rate', 'student2_errors')]:
            r[name] = r[key] / ne if ne else None
        for name, key in [('gradient_fraction', 'grad_sum'), ('aligned_gradient_fraction', 'aligned_grad_sum'),
                          ('opposing_gradient_fraction', 'opposing_grad_sum'), ('neutral_gradient_fraction', 'neutral_grad_sum')]:
            r[name] = r[key] / gd if gd else None
        for name, key in [('a_positive_fraction', 'a_positive_sum'), ('a_negative_fraction', 'a_negative_magnitude_sum'), ('a_net_fraction', 'a_signed_sum')]:
            r[name] = r[key] / ad if ad else None
        patient_rows.append(r)
    metrics = [k for k in patient_rows[0] if k.startswith('mean_') or k.endswith('_rate') or k.endswith('_fraction')]
    summary = []
    for model, condition, region in sorted({(r['model'], r['condition'], r['region']) for r in patient_rows}):
        vals = [r for r in patient_rows if (r['model'], r['condition'], r['region']) == (model, condition, region)]
        r = dict(model=model, condition=condition, region=region,
                 pixels=sum(v['pixels'] for v in vals), support_patients=sum(v['pixels'] > 0 for v in vals),
                 total_patients=len(vals), no_support_patients=';'.join(v['patient'] for v in vals if not v['pixels']),
                 aggregation='source_patient_equal')
        for k in metrics:
            vv = [v[k] for v in vals if v[k] is not None]
            r[k] = float(np.mean(vv)) if vv else None
        for k in ['aligned_pixels', 'opposing_pixels', 'neutral_pixels', 'zero_gradient_pixels']:
            r[k] = sum(v[k] for v in vals)
        summary.append(r)
    pairs = []
    for model, patient, region in sorted({(r['model'], r['patient'], r['region']) for r in patient_rows}):
        v = {r['condition']: r for r in patient_rows if (r['model'], r['patient'], r['region']) == (model, patient, region)}
        u, c = v['uniform'], v['CTW']
        r = dict(model=model, patient=patient, region=region, pixels=u['pixels'])
        for k in metrics:
            r['delta_' + k] = c[k] - u[k] if c[k] is not None and u[k] is not None else None
        pairs.append(r)
    paired_summary = []
    for model, region in sorted({(r['model'], r['region']) for r in pairs}):
        vals = [r for r in pairs if (r['model'], r['region']) == (model, region)]
        for k in metrics:
            vv = [r['delta_' + k] for r in vals if r['delta_' + k] is not None]
            mean = float(np.mean(vv)) if vv else None
            paired_summary.append(dict(model=model, region=region, metric=k, n_paired=len(vv),
                                       delta_patient_mean=mean, positive_patients=sum(x > 1e-15 for x in vv),
                                       negative_patients=sum(x < -1e-15 for x in vv),
                                       tied_patients=sum(abs(x) <= 1e-15 for x in vv),
                                       same_direction_as_mean=sum(x * mean > 0 for x in vv) if mean is not None else 0))
    for name, rr in [('image_region', rows), ('patient_region', patient_rows), ('region_summary', summary),
                     ('patient_paired', pairs), ('paired_summary', paired_summary)]:
        write_csv(OUT / 'results' / f'{prefix}_{name}.csv', rr)
    return patient_rows, summary


def primary():
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan = json.loads((OLD / 'configs/joint_input_plan.json').read_text())
    checkpoint = json.loads((OLD / 'configs/diagnostics.json').read_text())['main_checkpoint']
    original_receipt = json.loads((OLD / 'receipts/joint_full.json').read_text())
    assert sha(checkpoint) == original_receipt['loads'][0]['checkpoint_sha256']
    m, receipt = load('model_1')
    before = digest(m)
    assert before == original_receipt['model_hash_before'][0]
    rows, checks, sources = [], [], []
    for bp in plan['batches']:
        b = bp['batch']
        aa = [read_array('dynamic_CTW', b, j) for j in range(12)]
        cache = OUT / 'arrays/probabilities' / f'main_b{b}.npz'
        if cache.exists():
            with np.load(cache) as f:
                probs = f['p1']
                assert str(f['checkpoint_sha']) == receipt['checkpoint_sha256']
                assert str(f['image1_sha']) == hashlib.sha256(np.stack([a['image1'] for a in aa]).tobytes()).hexdigest()
            cache_used = True
        else:
            images = torch.from_numpy(np.stack([a['image1'] for a in aa]))[:, None].cuda()
            with torch.no_grad():
                # Match the historical two six-image forward calls and FP32 softmax.
                z = torch.cat([m(images[:6]), m(images[6:])])
                probs = z.softmax(1).cpu().numpy()
            np.savez_compressed(cache, p1=probs, checkpoint_sha=receipt['checkpoint_sha256'],
                                image1_sha=hashlib.sha256(np.stack([a['image1'] for a in aa]).tobytes()).hexdigest())
            cache_used = False
        for j, c in enumerate(aa):
            u = read_array('dynamic_uniform', b, j)
            for key in c:
                if key not in ['w', 'we', 'grad', 'grad_logits']:
                    assert np.array_equal(c[key], u[key]), (b, j, key)
            p = probs[j]
            assert np.array_equal(p.argmax(0), c['pred1'])
            assert np.isfinite(p).all() and np.allclose(p.sum(0), 1, atol=2e-7)
            assert np.all(c['w'] > 0)
            ratio = c['w'].astype(np.float64) / u['w']
            expected = u['grad_logits'].astype(np.float64) * ratio[None]
            gdiff = np.abs(c['grad_logits'] - expected)
            scale = max(float(np.abs(c['grad_logits']).max()), 1e-30)
            assert gdiff.max() / scale < 2e-6
            labels, precision = paired_direction(u, c, p)
            checks.append(dict(batch=b, pair=j, cache_used=cache_used,
                               max_gradient_scaling_abs_error=float(gdiff.max()),
                               max_gradient_scaling_peak_relative_error=float(gdiff.max()/scale),
                               **precision,
                               mean_weight_abs_difference=float(abs(float(c['w'].mean(dtype=np.float64)) - float(u['w'][0,0]))),
                               min_weight=float(c['w'].min()), max_weight=float(c['w'].max())))
            for condition, a in [('uniform', u), ('CTW', c)]:
                extract(rows, 'main_Full', condition, b, j, a, p, plan['patients'], labels)
                path = OLD / 'arrays/joint' / f'dynamic_{condition}_b{b}_p{j}.npz'
                sources.append(dict(path=str(path), sha256=sha(path)))
        assert digest(m) == before and all(q.grad is None and not q.requires_grad for q in m.parameters())
        print(f'primary batch {b}: 12 mixed images, cached={cache_used}, state unchanged', flush=True)
    pp, summary = aggregate(rows, 'main')
    # Reproduce old aggregates from the full arrays before accepting new intersections.
    old = list(csv.DictReader((OLD / 'results/joint_region_summary.csv').open()))
    reused = []
    for r in summary:
        if '__' in r['region']:
            continue
        ref = next(v for v in old if v['condition'] == 'dynamic_' + r['condition'] and v['region'] == r['region'])
        for k in ['mean_w', 'mean_e', 'mean_grad', 'gradient_fraction', 'disagreement_rate', 'student1_error_rate', 'student2_error_rate']:
            delta = abs(r[k] - float(ref[k])) if r[k] is not None else 0
            assert delta < 1e-12, (r['region'], k, delta)
        reused.append(dict(condition=r['condition'], region=r['region'], old_pixels=int(ref['pixels']),
                           reproduced_pixels=r['pixels'], reproduced_gradient_fraction=r['gradient_fraction']))
    write_csv(OUT / 'receipts/reproduced_existing.csv', reused)
    write_csv(OUT / 'receipts/main_analytic_checks.csv', checks)
    write_csv(OUT / 'receipts/main_source_arrays.csv', sources)
    write_json(OUT / 'receipts/primary.json', dict(status='PASS', load=receipt,
               model_hash_before=before, model_hash_after=digest(m), batch_norm_and_parameters_unchanged=True,
               checkpoint_sha_after=sha(checkpoint), parameter_gradients_none=True, optimizer_steps=0,
               unique_slices=plan['samples'], source_exposures=plan['exposures'], repeat_exposures=plan['repeat_exposures'],
               mixed_images=108, patients=plan['patients'], direction_tolerance=TOL,
               original_gradients_reused=True, new_model_forward='student1 only, one pass, cached by batch',
               max_scaling_peak_relative_error=max(c['max_gradient_scaling_peak_relative_error'] for c in checks),
               robust_direction_flips=sum(c['robust_direction_flips'] for c in checks),
               raw_signed_flips=sum(c['raw_signed_flips'] for c in checks),
               precision_note='raw FP32 sign disagreements at near-zero gradients are retained in analytic checks; paired labels use rounding-aware uncertainty bound'))


if __name__ == '__main__':
    p = argparse.ArgumentParser()
    p.parse_args()
    primary()
