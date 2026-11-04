import argparse

import json

import os

import sys

from pathlib import Path

from common import *

def replace(source,old,new):
    n=source.count(old)
    if n!=1: raise RuntimeError(f'Frozen source anchor expected once, found {n}: {old[:120]}')
    return source.replace(old,new,1)

def build_beta_source(cfg):
    s=(WORK/'source/historical_full.py').read_text()
    s=replace(s,'from utils import ramps, losses','import revision_runtime as revision\nfrom revision_runtime import BCP_net, generate_adamix_bcp_lite_v1_mask, compute_v21_3_innovation_x, mask_token_endpoint\nfrom revision_runtime import StatefulSampler as TwoStreamBatchSampler\nfrom utils import ramps, losses')
    s=replace(s,'        "args": vars(args),','        "args": vars(args),\n        "sampler_state": revision.sampler_state(global_step),\n        "amp_state": None,\n        "lr_state": [g["lr"] for g in optimizer1.param_groups],\n        "resume_exact": False,')
    s=replace(s,'    torch.save(ckpt, path)','    revision.atomic_torch(ckpt, path)')
    s=replace(s,'    return ckpt\n','    revision.set_sampler_resume(ckpt.get("sampler_state"))\n    return ckpt\n')
    s=replace(s,'        num_workers=16,',f'        num_workers={cfg["num_workers"]},')
    # Sampler itself replays the remaining IDs; do not discard batches twice.
    s=replace(s,'    start_batch_offset = iter_num % len(trainloader) if args.resume_ckpt else 0','    start_batch_offset = 0')
    s=replace(s,'        start_epoch = int(resume_state.get("epoch", 0))','        start_epoch = iter_num // len(trainloader)')
    s=replace(s,'        for batch_idx, sampled_batch in enumerate(trainloader):','        saved_rng = revision.rng_state()\n        batch_iterator = revision.resumed_iterator(trainloader, bool(args.resume_ckpt and epoch == start_epoch), saved_rng)\n        for batch_idx, sampled_batch in enumerate(batch_iterator):')
    s=replace(s,'            iter_num += 1','            iter_num += 1\n            revision.record_step(run_root, iter_num, optimizer1, loss)')
    s=replace(s,'            if iter_num >= args.max_iterations:\n                break\n        if iter_num >= args.max_iterations:',
              '            if revision.STOP:\n                best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}\n                save_full_ckpt(run_root / "checkpoints/last.ckpt", model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)\n                revision.completion(run_root, iter_num, interrupted=True)\n                writer.close()\n                raise SystemExit(75)\n            if iter_num >= args.max_iterations:\n                break\n        if iter_num >= args.max_iterations:')
    # Retain validation cadence and metric; save after validation so metadata are current.
    s=replace(s,'                model_2.train()','                model_2.train()\n                best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}\n                save_full_ckpt(run_root / "checkpoints/last.ckpt", model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)')
    s=replace(s,'    writer.close()\n\n\nif __name__', '    revision.completion(run_root, iter_num)\n    writer.close()\n\n\nif __name__')
    # No fallback to an unrelated initializer.
    s=replace(s,'    if not pre_model.exists():\n        pre_model = Path(args.root_path).parent / "best_results" / "ACDC_10" / "unet_best.pth"',
              '    if not pre_model.exists():\n        raise FileNotFoundError(pre_model)')
    return s
