"""Freeze only the inputs needed for this round, without changing old artifacts."""
import ast, hashlib, json, shutil
from pathlib import Path
W=Path(__file__).resolve().parents[1]; ROOT=W.parent
def sha(p): return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def main():
    dst=W/'scripts/frozen'; dst.mkdir(exist_ok=True)
    frozen=ROOT/'offline_A2_A5_20261007/scripts/frozen/train_ACDC'
    for name in ['networks','dataloaders']:
        shutil.copytree(frozen/name,dst/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    shutil.copy2(frozen/'v14_mask_adamix_components.py',dst)
    src=ROOT/'tcsmix_control_revision_20261006/source/train_ACDC'
    for name in ['utils']:
        shutil.copytree(src/name,dst/name,ignore=shutil.ignore_patterns('__pycache__'),dirs_exist_ok=True)
    for name in ['val_2D.py','v21_3_acdc10_third_innovation_components.py']:
        shutil.copy2(src/name,dst)
    cfg=json.loads((ROOT/'offline_A2_A5_20261007/configs/Full_main_checkpoint_args.json').read_text())
    run=Path(cfg['run_root']); source=run/'effective_source.py'
    shutil.copy2(source,W/'scripts/main_full_source.py')
    aug=ROOT/'paper_offline_followup_20261007/scripts/a3_inputs.py'
    nodes=[n for n in ast.parse(aug.read_text()).body if isinstance(n,ast.FunctionDef) and n.name in ['rng','restore','seed','boundary','recorded_aug']]
    out='import random\nimport numpy as np, torch\nfrom scipy.ndimage import zoom, rotate\nfrom dataloaders import dataset as ds\n\n'+ast.unparse(ast.Module(body=nodes,type_ignores=[]))+'\n'
    (W/'scripts/augmentation.py').write_text(out)
    records=ROOT/'revision_runs_20261006/reports/fixed_rectangle_historical_records.csv'
    shutil.copy2(records,W/'results/fixed_size_historical_records.csv')
    lock=json.loads((ROOT/'revision_runs_20261006/configs/fixed_scale.json').read_text())
    (W/'configs/fixed_size_basis.json').write_text(json.dumps(lock,indent=2)+'\n')
    cfg.update(run_root=str(W/'runs/F97_main_signal_20261013'))
    (W/'configs/fixed_training_args.json').write_text(json.dumps(cfg,indent=2)+'\n')
    data=Path(cfg['root_path'])
    diag=dict(main_checkpoint=str(run/'checkpoints/best.ckpt'),main_checkpoint_sha256=sha(run/'checkpoints/best.ckpt'),main_step=29600,
              main_identity='original X03 / Full_main / paper R1, test Dice 90.35%',branches=['model_1','model_2','ema_model'],trc=True,
              data_root=str(data),val_frames=sorted(data.joinpath('val.list').read_text().split()),seed=20261013,
              batch_size=24,roles='img_a6,img_b6,uimg_a6,uimg_b6; validation simulated roles',boundary_width=2,
              fixed_rectangle=[97,97],lambda_ctw=.05,mode='eval',parameter_updates=0,
              q='0.5*(mean(max(softmax(EMA(uimg_a))))+mean(max(softmax(EMA(uimg_b)))))',
              geometry='s=.18+.20q; Python round(256*s), clamp[1,255]',
              position='common predeclared u; floor(u*max(1,256-side)), original exclusive upper-bound support',
              loss='.05*mean_batch_pixels(w*mean_class(abs(p1-stopgrad(p2)))); w detached',
              uniform='each image whole-image mean of CTW weights; no regional renormalization',
              aggregation='sum pixel statistics within source patient, then equal mean over supported patients; fractions use all source-patient pixels',
              case_illustration='first sorted validation frame, middle slice by index; no outcome-dependent selection',
              cutout='occlusion interior plus width2 band at cutout edge; excluded from anatomy/error supports',
              region_partition='cutout > seam > anatomy > other_valid > invalid rotation padding; full image remains in loss',
              trc_error_direction='gamma0 -> normal: corrected=zero wrong/normal right; introduced=zero right/normal wrong')
    (W/'configs/diagnostics.json').write_text(json.dumps(diag,indent=2)+'\n')
    protected=[run/'checkpoints/best.ckpt',run/'checkpoints/last.ckpt',source,Path(cfg['pretrained_model'])/'unet_best_model.pth',data/'train_slices.list',data/'val.list',data/'test.list']
    receipt=dict(inputs={str(p):sha(p) for p in protected},source_copy=sha(W/'scripts/main_full_source.py'),dependencies={str(p.relative_to(W)):sha(p) for p in dst.rglob('*.py')},scope='actual round inputs only',fixed_checkpoint_missing='old E2 has no checkpoint; E3 stays cancelled',four_trained_combinations='not available as a complete comparable set with TRC on and main signal; partial E1 excluded')
    (W/'receipts/input_identity.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps(dict(main_checkpoint=diag['main_checkpoint'],sha256=diag['main_checkpoint_sha256'],fixed_rectangle=[97,97],frames=len(diag['val_frames']))))
if __name__=='__main__': main()
