# 实际代码摘录

行号针对包内相同字节的来源文件，完整源文件同时保留。

### v14_mask_adamix_components.py:122 `generate_adamix_bcp_lite_v1_mask`

```python
def generate_adamix_bcp_lite_v1_mask(img, logits_a, logits_b, min_side=0.18, max_side=0.38):
    """Frozen PROMISE12 AdaMix sampling/mixing rule; class-count independent."""
    batch_size, _, img_x, img_y = img.shape
    min_side = max(0.05, min(0.75, float(min_side)))
    max_side = max(min_side, min(0.90, float(max_side)))
    with torch.no_grad():
        conf_a = torch.softmax(logits_a.detach(), dim=1).max(dim=1)[0].mean()
        conf_b = torch.softmax(logits_b.detach(), dim=1).max(dim=1)[0].mean()
        teacher_conf = torch.clamp(0.5 * (conf_a + conf_b), 0.0, 1.0)
        side_fraction = min_side + (max_side - min_side) * teacher_conf.item()
    patch_x = max(1, min(img_x - 1, int(round(img_x * side_fraction))))
    patch_y = max(1, min(img_y - 1, int(round(img_y * side_fraction))))
    w = np.random.randint(0, max(1, img_x - patch_x))
    h = np.random.randint(0, max(1, img_y - patch_y))
    mask = torch.ones(img_x, img_y, device=img.device)
    loss_mask = torch.ones(batch_size, img_x, img_y, device=img.device)
    mask[w:w + patch_x, h:h + patch_y] = 0
    loss_mask[:, w:w + patch_x, h:h + patch_y] = 0
    with torch.no_grad():
        pseudo = logits_a.argmax(dim=1)
        patch_labels = pseudo[:, w:w + patch_x, h:h + patch_y]
        class_histogram = [float((patch_labels == c).float().mean().cpu()) for c in range(logits_a.shape[1])]
    stats = {
        "teacher_conf": float(teacher_conf.detach().cpu()),
        "side_fraction": float(side_fraction),
        "patch_area_ratio": float((patch_x * patch_y) / float(img_x * img_y)),
        "class_histogram": class_histogram,
        "foreground_coverage": float(1.0 - class_histogram[0]),
        "finite": bool(torch.isfinite(teacher_conf).item() and np.isfinite(side_fraction)),
    }
    # ACDC parent multiplies masks after an explicit channel unsqueeze and
    # therefore expects [B,H,W]; PROMISE used [H,W] broadcasting directly.
    # Expanding the same sampled rectangle is an interface-only adaptation.
    batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return batch_mask.long(), loss_mask.long(), stats
```

### revision_runtime.py:91 `sampler_state`

```python
def sampler_state(step):
    if SAMPLER is None:
        return None
    consumed = step % len(SAMPLER)
    return dict(batches=SAMPLER.batches, consumed=consumed,
                boundary='worker RNG/prefetch unavailable; sampled IDs/main RNG saved')
```

### revision_runtime.py:98 `set_sampler_resume`

```python
def set_sampler_resume(state):
    global SAMPLER_RESTORE
    SAMPLER_RESTORE = state
    if SAMPLER is not None and state and state['consumed']:
        if len(state['batches']) < len(SAMPLER):
            raise RuntimeError('Incomplete saved batch plan; cannot resume safely')
        SAMPLER.replay = state
```

### revision_runtime.py:106 `resumed_iterator`

```python
def resumed_iterator(loader, resumed, rng):
    iterator = iter(loader)
    # DataLoader consumes a base seed. Preserve saved main RNG on restart.
    if resumed:
        restore_rng(rng)
    return iterator
```

### revision_runtime.py:153 `ctw_maps`

```python
def ctw_maps(out1,out2,uniform=False):
    p1,p2=out1.softmax(1),out2.softmax(1)
    with torch.no_grad():
        q=(p1.detach().max(1).values+p2.detach().max(1).values)*.5
        dx=torch.zeros_like(q); dy=torch.zeros_like(q)
        dx[:,1:,:]=(q[:,1:,:]-q[:,:-1,:]).abs()
        dy[:,:,1:]=(q[:,:,1:]-q[:,:,:-1]).abs()
        raw=dx+dy
        lo=raw.flatten(1).min(1).values[:,None,None]
        hi=raw.flatten(1).max(1).values[:,None,None]
        normalized=((raw-lo)/(hi-lo).clamp_min(1e-6)).clamp(0,1)
        original=(.25+normalized).clamp(.25,1.25)
        weight=original.mean((1,2),keepdim=True).expand_as(original) if uniform else original
    error=(p1-p2.detach()).abs().mean(1)
    return raw,normalized,original,weight,error
```

### revision_runtime.py:179 `generate_adamix_bcp_lite_v1_mask`

```python
def generate_adamix_bcp_lite_v1_mask(img,la,lb,min_side=.18,max_side=.38):
    if CFG['mix_mode']=='dynamic':
        result=parent.generate_adamix_bcp_lite_v1_mask(img,la,lb,min_side,max_side)
    else:
        b,_,h,w=img.shape
        with torch.no_grad():
            q=.5*(la.detach().softmax(1).max(1).values.mean()+lb.detach().softmax(1).max(1).values.mean())
        s=CFG['fixed_side']; ph=max(1,min(h-1,round(h*s))); pw=max(1,min(w-1,round(w*s)))
        y=np.random.randint(0,max(1,h-ph)); x=np.random.randint(0,max(1,w-pw))
        mask=torch.ones((b,h,w),device=img.device,dtype=torch.long)
        mask[:,y:y+ph,x:x+pw]=0
        result=(mask,mask.clone(),dict(teacher_conf=q.item(),side_fraction=s,patch_area_ratio=ph*pw/(h*w)))
    LATEST.update(q=result[2]['teacher_conf'],s=result[2]['side_fraction'],actual_area=result[2]['patch_area_ratio'])
    return result
```

### train_entry.py:14 `build_beta_source`

```python
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
```

### train_entry.py:36 `main`

```python
def main():
    p=argparse.ArgumentParser()
    p.add_argument('--config',required=True)
    p.add_argument('--smoke',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    cfg=read_json(args.config)
    if args.smoke:
        cfg=dict(cfg,run_root=str(WORK/'runs/smoke'/cfg['task']),max_iterations=3 if args.resume else 2,save_interval=1)
    cfg=dict(cfg,_resume=args.resume)
    run=Path(cfg['run_root']); run.mkdir(parents=True,exist_ok=True)
    os.environ['CUDA_VISIBLE_DEVICES']='0'
    os.environ['WANDB_MODE']='disabled'
    os.environ['MLFLOW_TRACKING_URI']=''
    bootstrap()
    import revision_runtime as revision
    revision.configure(cfg)
    if cfg['task']=='E5':
        from official_adapter import run_official
        return run_official(cfg,args.resume)
    source=build_beta_source(cfg)
    (run/'revision_effective_source.py').write_text(source)
    argv=[str(WORK/'source/historical_full.py'),'--root_path',cfg['dataset_root'],'--run_root',str(run),
          '--pretrained_model',str(Path(cfg['pretrained_model']).parent),'--seed','42','--labelnum','7','--gpu','0',
          '--max_iterations',str(cfg['max_iterations']),'--save_interval',str(cfg['save_interval']),
          '--mask_token_enabled','true','--adamix_enabled','true','--innovation_x_id','V21_3_X03_CONTOUR_RELIABILITY_WEIGHT',
          '--enable_v21_3_innovation_x','true','--v21_3_innovation_x_weight','0.05','--mechanism_log_interval','200']
    if args.resume:
        ckpt=run/'checkpoints/last.ckpt'
        if not ckpt.exists(): raise FileNotFoundError('Resume requested without full state: '+str(ckpt))
        argv+=['--resume_ckpt',str(ckpt)]
    sys.argv=argv
    namespace={'__name__':'__main__','__file__':str(WORK/'source/historical_full.py'),'V21_3_EFFECTIVE_SOURCE':source}
    exec(compile(source,str(run/'revision_effective_source.py'),'exec'),namespace)
```

### historical_full_effective_source.py:333–374

```python
            volume_batch_strong = sampled_batch["image_strong"].cuda()
            label_batch_strong = sampled_batch["label_strong"].cuda()

            labeled_volume_batch = volume_batch[:args.labeled_bs]
            labeled_volume_batch_s = volume_batch_strong[:args.labeled_bs]
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            lab_a_s, lab_b_s = label_batch_strong[:labeled_sub_bs], label_batch_strong[labeled_sub_bs:args.labeled_bs]
            img_a_s, img_b_s = volume_batch_strong[:labeled_sub_bs], volume_batch_strong[labeled_sub_bs:args.labeled_bs]
            uimg_a_s, uimg_b_s = volume_batch_strong[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch_strong[args.labeled_bs + unlabeled_sub_bs:]

            with torch.no_grad():
                pre_a = ema_model(uimg_a)
                pre_b = ema_model(uimg_b)
                plab_a = get_ACDC_masks(pre_a, nms=1)
                plab_b = get_ACDC_masks(pre_b, nms=1)
                pre_a_s = ema_model(uimg_a_s)
                pre_b_s = ema_model(uimg_b_s)
                plab_a_s = get_ACDC_masks(pre_a_s, nms=1)
                plab_b_s = get_ACDC_masks(pre_b_s, nms=1)
                adamix_stats = None
                if args.adamix_enabled:
                    img_mask, loss_mask, adamix_stats = generate_adamix_bcp_lite_v1_mask(
                        img_a, pre_a, pre_b, args.adamix_min_side, args.adamix_max_side)
                else:
                    img_mask = torch.ones_like(img_a[:, 0])
                    img_mask[:, : img_mask.shape[1] // 2, : img_mask.shape[2] // 2] = 0
                    loss_mask = img_mask.clone()

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            net_input_unl_1 = uimg_a * img_mask.unsqueeze(1) + img_a * (1 - img_mask.unsqueeze(1))
            net_input_l_1 = img_b * img_mask.unsqueeze(1) + uimg_b * (1 - img_mask.unsqueeze(1))
            net_input_1 = torch.cat([net_input_unl_1, net_input_l_1], dim=0)
            net_input_unl_2 = uimg_a_s * img_mask.unsqueeze(1) + img_a_s * (1 - img_mask.unsqueeze(1))
            net_input_l_2 = img_b_s * img_mask.unsqueeze(1) + uimg_b_s * (1 - img_mask.unsqueeze(1))
            net_input_2 = torch.cat([net_input_unl_2, net_input_l_2], dim=0)

            out_unl_1 = model_1(net_input_unl_1)
            out_l_1 = model_1(net_input_l_1)
            out_1 = torch.cat([out_unl_1, out_l_1], dim=0)
            out_soft_1 = torch.softmax(out_1, dim=1)
```

### historical_full_effective_source.py:441–462

```python
            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            if args.mask_token_enabled and iter_num % args.mechanism_log_interval == 0:
                logging.info("iteration %d : mask_token_endpoint=%s", iter_num, json.dumps({"model1": mask_token_endpoint(model_1), "model2": mask_token_endpoint(model_2)}, sort_keys=True))
            if args.adamix_enabled and iter_num % args.mechanism_log_interval == 0:
                adamix_stats["mixed_loss_contribution"] = float((loss_dice_1 + loss_ce_1 + loss_dice_2 + loss_ce_2).detach().cpu())
                adamix_stats["model1_total_gradient_norm"] = float(torch.sqrt(sum((p.grad.detach() ** 2).sum() for p in model_1.parameters() if p.grad is not None)).cpu())
                logging.info("iteration %d : adamix_endpoint=%s", iter_num, json.dumps(adamix_stats, sort_keys=True))
            if args.enable_v21_3_innovation_x and iter_num % args.mechanism_log_interval == 0:
                payload = dict(innovation_x_endpoint or {})
                payload["candidate_id"] = args.innovation_x_id
                payload["innovation_x_loss"] = float(innovation_x_loss.detach().cpu())
                payload["weighted_innovation_x_loss"] = float((args.v21_3_innovation_x_weight * innovation_x_loss).detach().cpu())
                payload["finite"] = bool(torch.isfinite(innovation_x_loss).item())
                logging.info("iteration %d : v21_3_innovation_x_endpoint=%s", iter_num, json.dumps(payload, sort_keys=True))
            update_model_ema(model_1, ema_model, 0.99)

```

### historical_full_effective_source.py:34–70

```python
parser.add_argument("--root_path", type=str, default="../data/ACDC")
parser.add_argument("--exp", type=str, default="beta_fft_acdc_official_10pct_seed42")
parser.add_argument("--model", type=str, default="unet")
parser.add_argument("--pre_iterations", type=int, default=10000)
parser.add_argument("--max_iterations", type=int, default=30000)
parser.add_argument("--batch_size", type=int, default=24)
parser.add_argument("--deterministic", type=int, default=1)
parser.add_argument("--base_lr", type=float, default=0.01)
parser.add_argument("--image_size", type=int, nargs=2, default=[256, 256])
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--num_classes", type=int, default=4)
parser.add_argument("--labeled_bs", type=int, default=12)
parser.add_argument("--labelnum", type=int, default=7)
parser.add_argument("--u_weight", type=float, default=0.5)
parser.add_argument("--gpu", type=str, default="0")
parser.add_argument("--consistency", type=float, default=0.1)
parser.add_argument("--consistency_rampup", type=float, default=200.0)
parser.add_argument("--magnitude", type=float, default=6.0)
parser.add_argument("--s_param", type=int, default=6)
parser.add_argument("--patch_size", type=int, default=64)
parser.add_argument("--h_size", type=int, default=4)
parser.add_argument("--w_size", type=int, default=4)
parser.add_argument("--top_num", type=int, default=4)
parser.add_argument("--run_root", type=str, default="../runs/beta_fft_acdc_official_10pct_seed42")
parser.add_argument("--pretrained_model", type=str, default="")
parser.add_argument("--data_manifest_name", type=str, default="official")
parser.add_argument("--resume_ckpt", type=str, default="")
parser.add_argument("--save_interval", type=int, default=200)
parser.add_argument("--mask_token_enabled", type=str2bool, default=False)
parser.add_argument("--adamix_enabled", type=str2bool, default=False)
parser.add_argument("--mask_token_num", type=int, default=4)
parser.add_argument("--mask_token_dim", type=int, default=16)
parser.add_argument("--adamix_min_side", type=float, default=0.18)
parser.add_argument("--adamix_max_side", type=float, default=0.38)
parser.add_argument("--mechanism_log_interval", type=int, default=200)
parser.add_argument("--innovation_x_id", type=str, default="")
parser.add_argument("--enable_v21_3_innovation_x", type=str2bool, default=False)
```

### historical_full_effective_source.py:105–115

```python


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


```

### dataset.py:436–461

```python
            if validity_t is not None:
                sample["neighbor_validity"] = validity_t
                sample["neighbor_validity_strong"] = validity_t.clone()
        else:
            # Original 2D code path (unchanged)
            if random.random() > 0.5:
                image, label = random_rot_flip(image, label)
            elif random.random() > 0.5:
                image, label = random_rotate(image, label)
            x, y = image.shape
            image = zoom(image, (self.output_size[0] / x, self.output_size[1] / y), order=0)
            label = zoom(label, (self.output_size[0] / x, self.output_size[1] / y), order=0)

            image_strong, label_strong = cutout_gray(image, label, p=0.5)
            image_strong = color_jitter(image_strong).type("torch.FloatTensor")
            image = torch.from_numpy(image.astype(np.float32)).unsqueeze(0)

            label = torch.from_numpy(label.astype(np.uint8))
            label_strong = torch.from_numpy(label_strong.astype(np.uint8))
            sample = {
                "image": image,
                "image_strong": image_strong,
                "label": label,
                "label_strong": label_strong
            }
        return sample
```

### unet.py:376–394

```python
class UNet(nn.Module):
    def __init__(self, in_chns, class_num):
        super(UNet, self).__init__()

        params = {'in_chns': in_chns,
                  'feature_chns': [16, 32, 64, 128, 256],
                  'dropout': [0.05, 0.1, 0.2, 0.3, 0.5],
                  'class_num': class_num,
                  'bilinear': False,
                  'acti_func': 'relu'}

        self.encoder = Encoder(params)
        self.decoder = Decoder(params)

    def forward(self, x):
        feature = self.encoder(x)
        output, x_f = self.decoder(feature)
        return output

```
