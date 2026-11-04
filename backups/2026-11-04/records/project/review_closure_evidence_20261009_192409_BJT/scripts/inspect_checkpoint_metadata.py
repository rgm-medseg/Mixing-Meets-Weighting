"""Read existing checkpoint metadata on CPU; no model construction or forward."""
import gc
import hashlib
import json
import os
from pathlib import Path

os.environ['CUDA_VISIBLE_DEVICES'] = ''
import torch

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parents[1]
torch.set_num_threads(1)

def sha(path):
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()

def clean(v):
    if hasattr(v, '__dict__') and not isinstance(v, dict):
        v = vars(v)
    if isinstance(v, dict):
        return {str(k): clean(x) for k, x in v.items()}
    if isinstance(v, (list, tuple)):
        return [clean(x) for x in v]
    if isinstance(v, (str, int, float, bool)) or v is None:
        return v
    return str(v)

def optimizer_summary(state):
    groups = []
    for g in state.get('param_groups', []):
        row = {k: clean(v) for k, v in g.items() if k != 'params'}
        row['parameter_count'] = len(g['params'])
        groups.append(row)
    states = state.get('state', {})
    return {
        'groups': groups,
        'state_parameter_count': len(states),
        'momentum_buffer_count': sum('momentum_buffer' in v for v in states.values()),
        'explicit_step_values': sorted({str(v['step']) for v in states.values() if 'step' in v}),
        'note': 'SGD momentum has no step counter; update counts require training trace/source, not momentum contents.'
    }

paths = [
    'train_ACDC/Pre_model/label3/unet_best_model.pth',
    'train_ACDC/Pre_model/label7/unet_best_model.pth',
    'experiments/three_innovation_ablation_transfer_v22_0/ACDC5_FINAL3/checkpoints/last.ckpt',
    'runs/acdc10_third_innovation_codex_prep_top6_v21_3/V21_3_X03_CONTOUR_RELIABILITY_WEIGHT/checkpoints/last.ckpt',
]
rows = []
for rel in paths:
    p = ROOT / rel
    d = torch.load(p, map_location='cpu', weights_only=False)
    row = {'path': str(p), 'sha256': sha(p), 'bytes': p.stat().st_size,
           'keys': list(d), 'loaded_device': 'CPU', 'cuda_visible_devices': '',
           'model_created': False, 'forward_called': False, 'parameters_updated': False}
    for k in ['args', 'global_step', 'epoch', 'best_metrics', 'lr_state', 'schema', 'iteration', 'seed']:
        if k in d:
            row[k] = clean(d[k])
    row['state_dicts'] = {}
    for k, v in d.items():
        if isinstance(v, dict) and 'param_groups' in v:
            row[k] = optimizer_summary(v)
        elif isinstance(v, dict) and v and all(isinstance(x, torch.Tensor) for x in v.values()):
            row['state_dicts'][k] = {'tensor_keys': len(v), 'tensor_numel': sum(x.numel() for x in v.values()),
                                    'all_cpu': all(x.device.type == 'cpu' for x in v.values()),
                                    'TRC_tensor_keys': [n for n in v if 'mask_token_router' in n]}
    rows.append(row)
    print(json.dumps({'path': rel, 'keys': row['keys'], 'global_step': row.get('global_step'),
                      'optimizers': {k: v for k, v in row.items() if isinstance(v, dict) and 'groups' in v}}, ensure_ascii=False), flush=True)
    del d
    gc.collect()
(OUT / 'CHECKPOINT_CPU_METADATA.json').write_text(json.dumps(rows, ensure_ascii=False, indent=2) + '\n')
