"""Conditional four-trained-model diagnostic on archived common mixed inputs."""
import sys
sys.dont_write_bytecode = True
from analyze_direction import *
from v14_mask_adamix_components import v14_bcp_net
from v21_3_acdc10_third_innovation_components import _norm01, _spatial_gradient, compute_v21_3_innovation_x


def run():
    gate = json.loads((OUT / 'receipts/factorial_gate.json').read_text())
    if gate['status'] != 'PASS':
        print('No complete comparable TRC-off group; inventory only.')
        return
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    plan = json.loads((OLD / 'configs/joint_input_plan.json').read_text())
    inv = list(csv.DictReader((OUT / 'results/trained_condition_inventory.csv').open()))
    all_rows, checks, loads = [], [], []
    for arm in ['000', '010', '001', '011']:
        identity = next(r for r in inv if r['arm'] == arm)
        path = Path(identity['checkpoint'])
        assert sha(path) == identity['checkpoint_sha256']
        ck = torch.load(path, map_location='cpu', weights_only=False)
        models = []
        for branch in ['model_1', 'model_2']:
            m = v14_bcp_net(1, 4, ema=False, mask_token_enabled=False)
            m.load_state_dict(ck[branch], strict=True)
            m.eval()
            for q in m.parameters():
                q.requires_grad_(False)
            models.append(m)
        before = [digest(m) for m in models]
        for bp in plan['batches']:
            b = bp['batch']
            aa = [read_array('dynamic_CTW', b, j) for j in range(12)]
            cache = OUT / 'arrays/factorial_logits' / f'arm{arm}_b{b}.npz'
            cache.parent.mkdir(exist_ok=True, parents=True)
            images1 = np.stack([a['image1'] for a in aa])[:, None]
            images2 = np.stack([a['image2'] for a in aa])[:, None]
            image_sha = hashlib.sha256(images1.tobytes() + images2.tobytes()).hexdigest()
            if cache.exists():
                with np.load(cache) as f:
                    v1, v2 = torch.from_numpy(f['logits1']).cuda(), torch.from_numpy(f['logits2']).cuda()
                    assert str(f['checkpoint_sha']) == identity['checkpoint_sha256'] and str(f['input_sha']) == image_sha
                cache_used = True
            else:
                with torch.no_grad():
                    i1, i2 = torch.from_numpy(images1).cuda(), torch.from_numpy(images2).cuda()
                    v1 = torch.cat([models[0](i1[:6]), models[0](i1[6:])])
                    v2 = torch.cat([models[1](i2[:6]), models[1](i2[6:])])
                np.savez_compressed(cache, logits1=v1.cpu().numpy(), logits2=v2.cpu().numpy(),
                                    checkpoint_sha=identity['checkpoint_sha256'], input_sha=image_sha)
                cache_used = False
            z1 = v1.detach().requires_grad_(True)
            z2 = v2.detach().requires_grad_(True)
            p1, p2 = z1.softmax(1), z2.softmax(1)
            w = (.25 + _norm01(_spatial_gradient(.5 * (p1.detach().max(1).values + p2.detach().max(1).values)))).clamp(.25, 1.25).detach()
            e = (p1 - p2.detach()).abs().mean(1)
            literal, _ = compute_v21_3_innovation_x('V21_3_X03_CONTOUR_RELIABILITY_WEIGHT', z1, z2, z1, z2, z1, z2, 29600, 30000)
            assert torch.equal(literal, (w * e).mean())
            stats = {}
            sg = (p1.detach() - p2.detach()).sign()
            for condition in ['uniform', 'CTW']:
                ww = w.mean((1, 2), keepdim=True).expand_as(w) if condition == 'uniform' else w
                loss = .05 * (ww * e).mean()
                g, g2 = torch.autograd.grad(loss, (z1, z2), retain_graph=True, allow_unused=True)
                assert g2 is None and not ww.requires_grad
                expected = .05 * ww[:, None] * p1.detach() * (sg - (sg * p1.detach()).sum(1, keepdim=True)) / (e.numel() * 4)
                assert torch.allclose(g, expected, atol=2e-14, rtol=2e-5)
                stats[condition] = dict(w=ww.cpu().numpy(), grad_logits=g.cpu().numpy(),
                                        grad=g.abs().sum(1).cpu().numpy(), loss=float(loss.detach()))
            pn = p1.detach().cpu().numpy()
            pred1, pred2 = pn.argmax(1), p2.detach().cpu().numpy().argmax(1)
            for j, a in enumerate(aa):
                arr = {}
                for condition in ['uniform', 'CTW']:
                    v = a.copy()
                    v.update({k: stats[condition][k][j] for k in ['w', 'grad_logits', 'grad']})
                    v.update(e=e[j].detach().cpu().numpy(), pred1=pred1[j], pred2=pred2[j], disagree=pred1[j] != pred2[j])
                    arr[condition] = v
                u, c = arr['uniform'], arr['CTW']
                labels, precision = paired_direction(u, c, pn[j])
                ratio = c['w'].astype(np.float64) / u['w']
                diff = np.abs(c['grad_logits'] - u['grad_logits'].astype(np.float64) * ratio[None])
                peak_relative = float(diff.max() / max(np.abs(c['grad_logits']).max(), 1e-30))
                assert peak_relative < 2e-6
                checks.append(dict(arm=arm, batch=b, pair=j, cache_used=cache_used,
                                   scaling_peak_relative_error=peak_relative, **precision))
                for condition in ['uniform', 'CTW']:
                    extract(all_rows, 'arm_' + arm, condition, b, j, arr[condition], pn[j], plan['patients'], labels)
            assert [digest(m) for m in models] == before
            assert all(q.grad is None and not q.requires_grad for m in models for q in m.parameters())
            print(f'arm {arm}, batch {b}, cache={cache_used}, strict load, state unchanged', flush=True)
        loads.append(dict(arm=arm, checkpoint=str(path), checkpoint_sha_before=identity['checkpoint_sha256'],
                          checkpoint_sha_after=sha(path), strict_student_branches=['model_1', 'model_2'],
                          selected_step=ck['global_step'], model_hash_before=before, model_hash_after=[digest(m) for m in models]))
        del models, ck
        torch.cuda.empty_cache()
    patient, summary = aggregate(all_rows, 'factorial')
    # Compare the within-model CTW-minus-uniform response across paired patients.
    paired = list(csv.DictReader((OUT / 'results/factorial_patient_paired.csv').open()))
    comparisons = []
    contrasts = [('010', '000'), ('001', '000'), ('011', '010'), ('011', '001')]
    for child, parent in contrasts:
        for region in sorted({r['region'] for r in paired}):
            for metric in ['gradient_fraction', 'aligned_gradient_fraction', 'opposing_gradient_fraction', 'a_net_fraction']:
                vals = []
                for pid in plan['patients']:
                    c = next(r for r in paired if r['model'] == 'arm_' + child and r['patient'] == pid and r['region'] == region)
                    u = next(r for r in paired if r['model'] == 'arm_' + parent and r['patient'] == pid and r['region'] == region)
                    key = 'delta_' + metric
                    if c[key] and u[key]:
                        vals.append(float(c[key]) - float(u[key]))
                comparisons.append(dict(child_arm=child, parent_arm=parent, region=region, metric=metric,
                                        contrast='difference of within-model CTW-minus-uniform responses',
                                        paired_patients=len(vals), mean_difference=float(np.mean(vals)) if vals else None,
                                        positive_patients=sum(x > 0 for x in vals), negative_patients=sum(x < 0 for x in vals)))
    write_csv(OUT / 'results/factorial_response_contrasts.csv', comparisons)
    state_pairs = []
    for child, parent in contrasts:
        for region in sorted({r['region'] for r in patient}):
            for condition in ['uniform', 'CTW']:
                for pid in plan['patients']:
                    c = next(r for r in patient if r['model']=='arm_'+child and r['region']==region and r['condition']==condition and r['patient']==pid)
                    u = next(r for r in patient if r['model']=='arm_'+parent and r['region']==region and r['condition']==condition and r['patient']==pid)
                    row = dict(child_arm=child, parent_arm=parent, region=region, condition=condition, patient=pid)
                    for key in ['mean_e', 'mean_grad', 'disagreement_rate', 'student1_error_rate', 'student2_error_rate',
                                'gradient_fraction', 'aligned_gradient_fraction', 'opposing_gradient_fraction', 'a_net_fraction']:
                        row['delta_'+key] = c[key]-u[key] if c[key] is not None and u[key] is not None else None
                    state_pairs.append(row)
    write_csv(OUT / 'results/factorial_state_patient_contrasts.csv', state_pairs)
    state_summary = []
    for child, parent in contrasts:
        for region in sorted({r['region'] for r in patient}):
            for condition in ['uniform', 'CTW']:
                vals = [r for r in state_pairs if (r['child_arm'],r['parent_arm'],r['region'],r['condition'])==(child,parent,region,condition)]
                for key in [k for k in vals[0] if k.startswith('delta_')]:
                    vv = [r[key] for r in vals if r[key] is not None]
                    state_summary.append(dict(child_arm=child, parent_arm=parent, region=region, condition=condition,
                                              metric=key, paired_patients=len(vv), patient_mean_difference=float(np.mean(vv)) if vv else None,
                                              positive_patients=sum(x>0 for x in vv), negative_patients=sum(x<0 for x in vv)))
    write_csv(OUT / 'results/factorial_state_contrasts.csv', state_summary)
    write_csv(OUT / 'receipts/factorial_analytic_checks.csv', checks)
    write_json(OUT / 'receipts/factorial.json', dict(status='PASS', loads=loads, optimizer_steps=0,
               parameter_gradients_none=True, batch_norm_and_parameters_unchanged=True,
               common_input_images=108, common_source_patients=10, common_target='archived main-Full EMA LCC hard target',
               own_teacher_targets_used=False, own_mixing_rules_replayed=False,
               interpretation='common-input trained-state comparison and within-state loss probe; not a training causal chain',
               robust_direction_flips=sum(r['robust_direction_flips'] for r in checks)))


if __name__ == '__main__':
    run()
