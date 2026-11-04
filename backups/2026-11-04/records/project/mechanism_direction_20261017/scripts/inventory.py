"""Read-only identity and protocol gate for the actual 2x2 trained factorial."""
import sys
sys.dont_write_bytecode = True
from analyze_direction import *


def inventory():
    table_path = ROOT / 'final_report/three_innovation_postfull_test_v22_1/ACDC10_THREE_INNOVATION_8ARM_TEST_FULL_PRECISION.tsv'
    with table_path.open() as f:
        table = {r['Arm']: r for r in csv.DictReader(f, delimiter='\t')}
    paths = {
        '000': ROOT / 'runs/beta_fft_acdc_official_full_seed42_fresh_ckpt/checkpoints/best.ckpt',
        '110': ROOT / 'runs/beta_fft_acdc_10pct_mask_token_adamix_full_seed42/checkpoints/best.ckpt',
        '111': Path(json.loads((OLD / 'configs/diagnostics.json').read_text())['main_checkpoint']),
    }
    for arm in ['100', '010', '001', '101', '011']:
        c = json.loads((ROOT / f'configs/three_innovation_ablation_transfer_v22_0/ACDC10_ARM_{arm}.json').read_text())
        paths[arm] = Path(c['run_root']) / 'checkpoints/best.ckpt'
    # These are all eight pre-identified arms; no score-based alternative search.
    args_all, rows, provenance = {}, [], []
    keys = ['root_path', 'model', 'pre_iterations', 'max_iterations', 'batch_size', 'deterministic',
            'base_lr', 'image_size', 'seed', 'num_classes', 'labeled_bs', 'labelnum', 'u_weight',
            'consistency', 'consistency_rampup', 'magnitude', 's_param', 'patch_size', 'h_size', 'w_size', 'top_num']
    for arm in ['000', '010', '001', '011', '100', '110', '101', '111']:
        path = paths[arm]
        if not path.exists():
            rows.append(dict(arm=arm, TRC=arm[0], TCSMix=arm[1], CTW=arm[2], status='MISSING', checkpoint=str(path)))
            continue
        ck = torch.load(path, map_location='cpu', weights_only=False)
        args = ck['args']; args_all[arm] = args
        actual_sha = sha(path)
        last = path.with_name('last.ckpt')
        last_step = int(torch.load(last, map_location='cpu', weights_only=False)['global_step']) if last.exists() else None
        toggles = (bool(args.get('mask_token_enabled', False)), bool(args.get('adamix_enabled', False)), bool(args.get('enable_v21_3_innovation_x', False)))
        assert toggles == tuple(x == '1' for x in arm), (arm, toggles)
        identity = actual_sha == table[arm]['CheckpointSHA']
        branches = all(k in ck for k in ['model_1', 'model_2', 'ema_model'])
        status = 'IDENTITY_AND_BUDGET_PASS' if identity and branches and last_step == 30000 else 'INCOMPLETE_OR_MISMATCH'
        rows.append(dict(arm=arm, TRC=arm[0], TCSMix=arm[1], CTW=arm[2], status=status,
                         checkpoint=str(path), checkpoint_sha256=actual_sha, expected_sha256=table[arm]['CheckpointSHA'],
                         selected_step=ck['global_step'], expected_selected_step=table[arm]['BestValIter'], completed_updates=last_step,
                         branches_complete=branches, selection='existing validation-selected student1; no reselection',
                         main_signal='original EMA whole-image mean-max' if arm[1]=='1' else 'disabled; original deterministic top-left 128x128 mixing',
                         ctweight=args.get('v21_3_innovation_x_weight', .05), seed=args['seed']))
        for source in [path.parent.parent / 'effective_source.py', path.parent.parent / 'source_snapshot.py']:
            if source.exists():
                provenance.append(dict(path=str(source), sha256=sha(source)))
    ref = args_all.get('000', {})
    differences = {a: {k: [ref.get(k), args_all[a].get(k)] for k in keys if ref.get(k) != args_all[a].get(k)} for a in args_all}
    four = ['000', '010', '001', '011']
    complete = all(any(r['arm'] == a and r['status'] == 'IDENTITY_AND_BUDGET_PASS' for r in rows) for a in four)
    comparable = complete and all(not differences.get(a, {'MISSING': True}) for a in four)
    snapshots = OUT / 'receipts/arguments'
    for arm, args in args_all.items():
        write_json(snapshots / f'arm_{arm}.json', args)
    manifests = {n: dict(path=str(Path(ref['root_path']) / n), sha256=sha(Path(ref['root_path']) / n)) for n in ['train_slices.list', 'val.list']}
    write_csv(OUT / 'results/trained_condition_inventory.csv', rows)
    write_csv(OUT / 'receipts/training_sources.csv', provenance)
    write_json(OUT / 'receipts/factorial_gate.json', dict(status='PASS' if comparable else 'NOT_ELIGIBLE',
               reference_table=str(table_path), reference_table_sha=sha(table_path), TRC_off_arms=four,
               complete_same_group=comparable, protocol_argument_differences=differences, manifests=manifests,
               common_input='archived main-Full dynamic mixtures, unchanged',
               common_stratification='archived main-Full EMA LCC hard targets, NOT each model own teacher',
               comparison='frozen trained states under common input; within each state apply diagnostic uniform/CTW weights',
               own_mixing_rules_replayed=False, performance_based_substitution=False))
    print(json.dumps(dict(TRC_off_eligible=comparable, differences=differences, rows=rows), indent=2))


if __name__ == '__main__':
    inventory()
