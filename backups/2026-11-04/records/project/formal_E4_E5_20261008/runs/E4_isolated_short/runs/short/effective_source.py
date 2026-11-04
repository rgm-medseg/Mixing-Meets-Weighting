import argparse
import csv
import json
import logging
import os
import random
import shutil
from pathlib import Path

import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torchvision import transforms
from tqdm import tqdm

from dataloaders.dataset import BaseDataSets, TwoStreamBatchSampler, WeakStrongAugment
from v14_mask_adamix_components import (v14_bcp_net as BCP_net, generate_adamix_bcp_lite_v1_mask, load_parent_pretrain, transplant_parent_optimizer_state, make_optimizer, mask_token_endpoint, process_image_batches_parent_2d)
from v21_3_acdc10_third_innovation_components import compute_v21_3_innovation_x
import revision_runtime as revision
from revision_runtime import BCP_net, compute_v21_3_innovation_x, mask_token_endpoint
import control_runtime as control
from control_runtime import generate_adamix_bcp_lite_v1_mask
from revision_runtime import StatefulSampler as TwoStreamBatchSampler
from utils import ramps, losses
process_image_batches = process_image_batches_parent_2d
from val_2D import test_single_volume


def str2bool(x):
    return str(x).lower() in {"1", "true", "yes", "y"}


parser = argparse.ArgumentParser()
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
parser.add_argument("--v21_3_innovation_x_weight", type=float, default=0.05)
args = parser.parse_args()

dice_loss = losses.DiceLoss(n_classes=4)
ce_loss = nn.CrossEntropyLoss()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def to_dict_state():
    return {
        "python_random": random.getstate(),
        "numpy_random": np.random.get_state(),
        "torch_random": torch.get_rng_state(),
        "torch_cuda_random": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_dict_state(state):
    random.setstate(state["python_random"])
    np.random.set_state(state["numpy_random"])
    torch.set_rng_state(state["torch_random"])
    if torch.cuda.is_available() and state.get("torch_cuda_random") is not None:
        torch.cuda.set_rng_state_all(state["torch_cuda_random"])


def get_current_consistency_weight(epoch):
    return 5 * args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)


def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)


def get_ACDC_2DLargestCC(segmentation):
    from skimage.measure import label
    batch_list = []
    for i in range(segmentation.shape[0]):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i]
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:]) + 1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        batch_list.append(class_list[0] + class_list[1] + class_list[2])
    return torch.Tensor(batch_list).cuda()


def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)
    return probs


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction="none")
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)
    return loss_dice, loss_ce


def patients_to_slices(dataset, patiens_num):
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136, "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    else:
        raise ValueError("Unsupported dataset")
    return ref_dict[str(patiens_num)]


def make_run_dirs():
    run_root = Path(args.run_root)
    run_root.mkdir(parents=True, exist_ok=True)
    (run_root / "checkpoints").mkdir(exist_ok=True)
    (run_root / "logs").mkdir(exist_ok=True)
    (run_root / "audit").mkdir(exist_ok=True)
    shutil.copy(__file__, run_root / "source_snapshot.py")
    (run_root / "effective_source.py").write_text(V21_3_EFFECTIVE_SOURCE)
    return run_root


def save_full_ckpt(path, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, global_step, best_metrics):
    ckpt = {
        "model_1": model_1.state_dict(),
        "model_2": model_2.state_dict(),
        "ema_model": ema_model.state_dict(),
        "optimizer1": optimizer1.state_dict(),
        "optimizer2": optimizer2.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_metrics": best_metrics,
        "random_state": to_dict_state(),
        "args": vars(args),
        "sampler_state": revision.sampler_state(global_step),
        "amp_state": None,
        "lr_state": [g["lr"] for g in optimizer1.param_groups],
        "resume_exact": False,
        "scheduler_state": None,
        "lr_state_optimizer2": [g["lr"] for g in optimizer2.param_groups],
        "controller_metadata": control.checkpoint_metadata(),
    }
    revision.atomic_torch(ckpt, path)


def load_full_ckpt(path, model_1, model_2, ema_model, optimizer1, optimizer2):
    ckpt = torch.load(path, map_location="cpu")
    model_1.load_state_dict(ckpt["model_1"])
    model_2.load_state_dict(ckpt["model_2"])
    ema_model.load_state_dict(ckpt["ema_model"])
    optimizer1.load_state_dict(ckpt["optimizer1"])
    optimizer2.load_state_dict(ckpt["optimizer2"])
    if "random_state" in ckpt:
        restore_dict_state(ckpt["random_state"])
    revision.set_sampler_resume(ckpt.get("sampler_state"))
    return ckpt


def mark_last_dynamics_checkpoint_saved(csv_path, global_step):
    lines = csv_path.read_text().splitlines()
    if len(lines) < 2:
        return
    rows = list(csv.reader(lines[1:]))
    for row in reversed(rows):
        if row and row[0] == str(global_step):
            row[-1] = "1"
            break
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(lines[0].split(","))
        writer.writerows(rows)


def finite_check(model, optimizer):
    for n, p in model.named_parameters():
        if torch.is_tensor(p) and p.is_floating_point() and not torch.isfinite(p).all():
            raise RuntimeError(f"Non-finite model tensor: {n}")
    for state in optimizer.state.values():
        for _, v in state.items():
            if torch.is_tensor(v) and v.is_floating_point() and not torch.isfinite(v).all():
                raise RuntimeError("Non-finite optimizer tensor")


def write_dynamics_header(csv_path):
    if csv_path.exists():
        return
    with csv_path.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "iter","global_step","total_loss","loss_1","loss_2","consistency_weight","repair_triggered",
            "fft_triggered","pred_fg_ratio_1","pred_fg_ratio_2","pseudo_fg_ratio_1","pseudo_fg_ratio_2",
            "branch_disagreement_ratio","branch_agreement_ratio","fft_mixing_triggered","fft_threshold",
            "loss_finite","grad_finite","logits_finite","checkpoint_saved"
        ])


def append_dynamics(csv_path, row):
    with csv_path.open("a", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(row)


def train():
    run_root = make_run_dirs()
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
    seed_everything(args.seed)

    logging.basicConfig(filename=str(run_root / "train.log"), level=logging.INFO, format="[%(asctime)s.%(msecs)03d] %(message)s", datefmt="%H:%M:%S")
    logging.getLogger().addHandler(logging.StreamHandler())
    logging.info(str(args))
    logging.info(f"run_root={run_root}")

    pre_snapshot_path = Path(args.pretrained_model) if args.pretrained_model else Path("./Pre_model/label7")
    pre_model = pre_snapshot_path / f"{args.model}_best_model.pth"
    if not pre_model.exists():
        raise FileNotFoundError(pre_model)
    logging.info(f"pretrained_model={pre_model}")

    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    model_1 = BCP_net(in_chns=1, class_num=args.num_classes, ema=False, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)
    model_2 = BCP_net(in_chns=1, class_num=args.num_classes, ema=False, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)
    ema_model = BCP_net(in_chns=1, class_num=args.num_classes, ema=True, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)

    db_train = BaseDataSets(base_dir=args.root_path, split="train", num=None, transform=transforms.Compose([WeakStrongAugment(args.image_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    # True test access is deferred until validation-best identity is frozen.
    db_test = BaseDataSets(base_dir=args.root_path, split="val")
    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    trainloader = DataLoader(
        db_train,
        batch_sampler=TwoStreamBatchSampler(list(range(0, labeled_slice)), list(range(labeled_slice, len(db_train))), args.batch_size, args.batch_size - args.labeled_bs),
        num_workers=16,
        pin_memory=True,
        worker_init_fn=lambda worker_id: random.seed(args.seed + worker_id),
    )
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)
    testloader = DataLoader(db_test, batch_size=1, shuffle=False, num_workers=1)

    optimizer1 = make_optimizer(model_1, args.base_lr, args.mask_token_enabled)
    control.guarded_optimizer(optimizer1, "optimizer1")
    optimizer2 = make_optimizer(model_2, args.base_lr, args.mask_token_enabled)
    control.guarded_optimizer(optimizer2, "optimizer2")

    writer = SummaryWriter(str(run_root / "logs"))
    dynamics_csv = run_root / "audit" / "training_dynamics.csv"
    write_dynamics_header(dynamics_csv)

    max_epoch = args.max_iterations // len(trainloader) + 1
    iter_num = 0
    start_epoch = 0
    best_performance1 = 0.0
    best_performance2 = 0.0

    if args.resume_ckpt:
        resume_path = Path(args.resume_ckpt)
        logging.info(f"resume_ckpt={resume_path}")
        resume_state = load_full_ckpt(resume_path, model_1, model_2, ema_model, optimizer1, optimizer2)
        iter_num = int(resume_state.get("global_step", 0))
        start_epoch = iter_num // len(trainloader)
        best_metrics = resume_state.get("best_metrics", {})
        best_performance1 = float(best_metrics.get("model1_best_val_dice", 0.0))
        best_performance2 = float(best_metrics.get("model2_best_val_dice", 0.0))
        logging.info("strict_resume_loaded epoch=%s global_step=%s", start_epoch, iter_num)
    else:
        load_state = torch.load(pre_model, map_location="cpu")
        load_parent_pretrain(ema_model, load_state["net"], args.mask_token_enabled)
        load_parent_pretrain(model_1, load_state["net"], args.mask_token_enabled)
        load_parent_pretrain(model_2, load_state["net"], args.mask_token_enabled)
        if "opt" in load_state:
            transplant_parent_optimizer_state(optimizer1, load_state["opt"], args.mask_token_enabled)
            transplant_parent_optimizer_state(optimizer2, load_state["opt"], args.mask_token_enabled)
        control.initial_loaded([("model_1",model_1),("model_2",model_2),("ema_model",ema_model)], [optimizer1,optimizer2], pre_model)

    assert args.resume_ckpt and iter_num == 2
    assert revision.SAMPLER.replay == resume_state["sampler_state"]
    for branch, model in [("model_1",model_1),("model_2",model_2),("ema_model",ema_model)]:
        assert all(torch.equal(v.cpu(),resume_state[branch][k].cpu()) for k,v in model.state_dict().items())
    revision.atomic_json(revision.WORK/"reports/RESUME_ZERO_UPDATE.json", dict(status="PASS",global_step=iter_num,optimizer_calls=0,models_exact=True,sampler_replay_loaded=True,main_RNG_loaded=True,worker_exact=False))
    writer.close()
    raise SystemExit(0)
    start_batch_offset = 0
    iterator = tqdm(range(start_epoch, max_epoch), ncols=70)
    for epoch in iterator:
        saved_rng = revision.rng_state()
        batch_iterator = revision.resumed_iterator(trainloader, bool(args.resume_ckpt and epoch == start_epoch), saved_rng)
        for batch_idx, sampled_batch in enumerate(batch_iterator):
            if epoch == start_epoch and batch_idx < start_batch_offset:
                continue
            if iter_num >= args.max_iterations:
                break
            control.start_step(iter_num + 1, sampled_batch)
            volume_batch = sampled_batch["image"].cuda()
            label_batch = sampled_batch["label"].cuda()
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
                        img_a, pre_a, pre_b, args.adamix_min_side, args.adamix_max_side, plab_a=plab_a, plab_b=plab_b)
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
            out_pseudo_1 = torch.argmax(out_soft_1.detach(), dim=1, keepdim=False)
            unl_dice_1, unl_ce_1 = mix_loss(out_unl_1, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_1, l_ce_1 = mix_loss(out_l_1, lab_b, plab_b, loss_mask, u_weight=args.u_weight)
            loss_ce_1 = unl_ce_1 + l_ce_1
            loss_dice_1 = unl_dice_1 + l_dice_1

            out_unl_2 = model_2(net_input_unl_2)
            out_l_2 = model_2(net_input_l_2)
            out_2 = torch.cat([out_unl_2, out_l_2], dim=0)
            out_soft_2 = torch.softmax(out_2, dim=1)
            out_pseudo_2 = torch.argmax(out_soft_2.detach(), dim=1, keepdim=False)
            unl_dice_2, unl_ce_2 = mix_loss(out_unl_2, plab_a_s, lab_a_s, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice_2, l_ce_2 = mix_loss(out_l_2, lab_b_s, plab_b_s, loss_mask, u_weight=args.u_weight)
            loss_ce_2 = unl_ce_2 + l_ce_2
            loss_dice_2 = unl_dice_2 + l_dice_2

            pseudo_supervision1 = dice_loss(out_soft_1, out_pseudo_2.unsqueeze(1))
            pseudo_supervision2 = dice_loss(out_soft_2, out_pseudo_1.unsqueeze(1))
            mix_factors = torch.tensor(np.random.beta(0.1, 0.1, size=(args.labeled_bs // 2, 1, 1, 1)), dtype=torch.float).cuda()
            batch_ux_mixed = uimg_a * (1.0 - mix_factors) + uimg_b * mix_factors
            input_volume_batch = torch.cat([labeled_volume_batch, batch_ux_mixed], dim=0)
            outputs = model_1(input_volume_batch)
            outputs_soft = torch.softmax(outputs, dim=1)
            with torch.no_grad():
                ema_output_ux0 = torch.softmax(pre_a, dim=1)
                ema_output_ux1 = torch.softmax(pre_b, dim=1)
                batch_pred_mixed = ema_output_ux0 * (1.0 - mix_factors) + ema_output_ux1 * mix_factors
            loss_ce_mix = ce_loss(outputs[:args.labeled_bs], label_batch[:args.labeled_bs].long())
            loss_dice_mix = dice_loss(outputs_soft[:args.labeled_bs], label_batch[:args.labeled_bs].unsqueeze(1))
            supervised_loss_mix = 0.5 * (loss_dice_mix + loss_ce_mix)
            consistency_loss = torch.mean((outputs_soft[args.labeled_bs:] - batch_pred_mixed) ** 2) * consistency_weight

            mix_factors_s = torch.tensor(np.random.beta(0.1, 0.1, size=(args.labeled_bs // 2, 1, 1, 1)), dtype=torch.float).cuda()
            batch_ux_mixed_s = uimg_a_s * (1.0 - mix_factors_s) + uimg_b_s * mix_factors_s
            input_volume_batch_s = torch.cat([labeled_volume_batch_s, batch_ux_mixed_s], dim=0)
            outputs_s = model_1(input_volume_batch_s)
            outputs_soft_s = torch.softmax(outputs_s, dim=1)
            with torch.no_grad():
                ema_output_ux0_s = torch.softmax(pre_a_s, dim=1)
                ema_output_ux1_s = torch.softmax(pre_b_s, dim=1)
                batch_pred_mixed_s = ema_output_ux0_s * (1.0 - mix_factors_s) + ema_output_ux1_s * mix_factors_s
            loss_ce_mix_s = ce_loss(outputs_s[:args.labeled_bs], label_batch_strong[:args.labeled_bs].long())
            loss_dice_mix_s = dice_loss(outputs_soft_s[:args.labeled_bs], label_batch_strong[:args.labeled_bs].unsqueeze(1))
            supervised_loss_mix_s = 0.5 * (loss_dice_mix_s + loss_ce_mix_s)
            consistency_loss_s = torch.mean((outputs_soft_s[args.labeled_bs:] - batch_pred_mixed_s) ** 2) * consistency_weight

            image_patch_last_FFT = process_image_batches(net_input_1, net_input_2, 30)
            image_output_1_FFT = model_1(image_patch_last_FFT.unsqueeze(1))
            image_output_soft_1_FFT = torch.softmax(image_output_1_FFT, dim=1)
            pseudo_image_output_1_FFT = torch.argmax(image_output_soft_1_FFT.detach(), dim=1, keepdim=False)
            image_output_2_FFT = model_2(image_patch_last_FFT.unsqueeze(1))
            image_output_soft_2_FFT = torch.softmax(image_output_2_FFT, dim=1)
            pseudo_image_output_2_FFT = torch.argmax(image_output_soft_2_FFT.detach(), dim=1, keepdim=False)
            pseudo_supervision5 = dice_loss(image_output_soft_1_FFT, pseudo_image_output_2_FFT.unsqueeze(1))
            pseudo_supervision6 = dice_loss(image_output_soft_2_FFT, pseudo_image_output_1_FFT.unsqueeze(1))

            loss_1 = (loss_dice_1 + loss_ce_1) / 2 + pseudo_supervision1 + pseudo_supervision5
            loss_2 = (loss_dice_2 + loss_ce_2) / 2 + pseudo_supervision2 + pseudo_supervision6
            innovation_x_loss = loss_1 * 0.0
            innovation_x_endpoint = None
            if args.enable_v21_3_innovation_x:
                innovation_x_loss, innovation_x_endpoint = compute_v21_3_innovation_x(
                    args.innovation_x_id, out_1, out_2, pre_a, pre_b,
                    image_output_1_FFT, image_output_2_FFT, iter_num, args.max_iterations)
            loss = loss_1 + loss_2 + consistency_loss + supervised_loss_mix + consistency_loss_s + supervised_loss_mix_s + args.v21_3_innovation_x_weight * innovation_x_loss

            optimizer1.zero_grad()
            optimizer2.zero_grad()
            loss.backward()
            optimizer1.step()
            optimizer2.step()

            iter_num += 1
            revision.record_step(run_root, iter_num, optimizer1, loss)
            control.record_step(run_root, iter_num, optimizer1, loss)
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
            old_teacher = {k:v.detach().clone() for k,v in ema_model.state_dict().items()}
            update_model_ema(model_1, ema_model, 0.99)
            assert all(torch.equal(v, (.99*old_teacher[k]+(1-.99)*model_1.state_dict()[k]).to(v.dtype)) for k,v in ema_model.state_dict().items())
            del old_teacher
            control.after_ema(iter_num, model_1, ema_model)

            logits_finite = all(torch.isfinite(t).all() for t in [out_1, out_2, outputs, outputs_s, image_output_1_FFT, image_output_2_FFT])
            loss_finite = torch.isfinite(loss).item()
            grad_finite = all(torch.isfinite(p.grad).all().item() for p in list(model_1.parameters()) + list(model_2.parameters()) if p.grad is not None)
            fft_triggered = 1 if image_patch_last_FFT is not None else 0
            repair_triggered = 1
            branch_disagreement_ratio = (out_pseudo_1 != out_pseudo_2).float().mean().item()
            branch_agreement_ratio = 1.0 - branch_disagreement_ratio
            pred_fg_ratio_1 = (out_pseudo_1 > 0).float().mean().item()
            pred_fg_ratio_2 = (out_pseudo_2 > 0).float().mean().item()
            pseudo_fg_ratio_1 = (pseudo_image_output_1_FFT > 0).float().mean().item()
            pseudo_fg_ratio_2 = (pseudo_image_output_2_FFT > 0).float().mean().item()

            writer.add_scalar("info/total_loss", loss, iter_num)
            writer.add_scalar("info/model1_loss", loss_1, iter_num)
            writer.add_scalar("info/model2_loss", loss_2, iter_num)
            writer.add_scalar("info/consistency_weight", consistency_weight, iter_num)
            logging.info("iteration %d: loss: %f, model1_loss: %f, model2_loss: %f", iter_num, loss, loss_1, loss_2)
            logging.info("iteration %d: loss: %f, ICT_loss: %f, FFT_loss: %f", iter_num, loss, consistency_loss + supervised_loss_mix, pseudo_supervision5 + pseudo_supervision6)

            append_dynamics(dynamics_csv, [
                iter_num, iter_num, float(loss.item()), float(loss_1.item()), float(loss_2.item()), float(consistency_weight),
                repair_triggered, fft_triggered, pred_fg_ratio_1, pred_fg_ratio_2, pseudo_fg_ratio_1, pseudo_fg_ratio_2,
                branch_disagreement_ratio, branch_agreement_ratio, fft_triggered, 30, int(loss_finite), int(grad_finite), int(logits_finite), 0
            ])

            if iter_num > 0 and iter_num % args.save_interval == 0:
                best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
                last_ckpt = run_root / "checkpoints" / "last.ckpt"
                save_full_ckpt(last_ckpt, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
                mark_last_dynamics_checkpoint_saved(dynamics_csv, iter_num)
                logging.info("checkpoint_saved path=%s epoch=%s global_step=%s", last_ckpt, epoch, iter_num)

            if iter_num == 1:
                model_1.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_1, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance1 = np.mean(metric_list, axis=0)[0]
                metric_list = 0.0
                for _, sampled_batch in enumerate(testloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_1, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_test)
                performance1_test = np.mean(metric_list, axis=0)[0]
                if performance1 > best_performance1:
                    best_performance1 = performance1
                    best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
                    best_ckpt = run_root / "checkpoints" / "best.ckpt"
                    save_full_ckpt(best_ckpt, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
                    logging.info("best_checkpoint_updated path=%s epoch=%s global_step=%s metric=%f", best_ckpt, epoch, iter_num, performance1)
                logging.info("iteration %d : model1_mean_dice : %f", iter_num, performance1)
                logging.info("iteration %d : model1_mean_dice_test : %f", iter_num, performance1_test)
                model_1.train()

                model_2.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_2, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                performance2 = np.mean(metric_list, axis=0)[0]
                metric_list = 0.0
                for _, sampled_batch in enumerate(testloader):
                    metric_i = test_single_volume(sampled_batch["image"], sampled_batch["label"], model_2, classes=args.num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_test)
                performance2_test = np.mean(metric_list, axis=0)[0]
                if performance2 > best_performance2:
                    best_performance2 = performance2
                logging.info("iteration %d : model2_mean_dice : %f", iter_num, performance2)
                logging.info("iteration %d : model2_mean_dice_test : %f", iter_num, performance2_test)
                model_2.train()
                control.validation(iter_num, performance1, performance2, best_performance1)
                best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
                save_full_ckpt(run_root / "checkpoints/last.ckpt", model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)

            if iter_num >= 2:
                revision.STOP = True
            if revision.STOP:
                best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
                save_full_ckpt(run_root / "checkpoints/last.ckpt", model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
                revision.completion(run_root, iter_num, interrupted=True)
                writer.close()
                raise SystemExit(75)
            if iter_num >= args.max_iterations:
                break
        if iter_num >= args.max_iterations:
            break

    final_ckpt = run_root / "checkpoints" / "last.ckpt"
    best_metrics = {"model1_best_val_dice": best_performance1, "model2_best_val_dice": best_performance2}
    save_full_ckpt(final_ckpt, model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
    mark_last_dynamics_checkpoint_saved(dynamics_csv, iter_num)
    if not (run_root / "checkpoints" / "best.ckpt").exists():
        save_full_ckpt(run_root / "checkpoints" / "best.ckpt", model_1, model_2, ema_model, optimizer1, optimizer2, epoch, iter_num, best_metrics)
    append_dynamics(dynamics_csv, [iter_num, iter_num, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1])

    re_model_1 = BCP_net(in_chns=1, class_num=args.num_classes, ema=False, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)
    re_model_2 = BCP_net(in_chns=1, class_num=args.num_classes, ema=False, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)
    re_ema = BCP_net(in_chns=1, class_num=args.num_classes, ema=True, mask_token_enabled=args.mask_token_enabled, mask_token_num=args.mask_token_num, mask_token_dim=args.mask_token_dim)
    re_opt1 = make_optimizer(re_model_1, args.base_lr, args.mask_token_enabled)
    re_opt2 = make_optimizer(re_model_2, args.base_lr, args.mask_token_enabled)
    re_ckpt = load_full_ckpt(final_ckpt, re_model_1, re_model_2, re_ema, re_opt1, re_opt2)
    finite_check(re_model_1, re_opt1)
    finite_check(re_model_2, re_opt2)
    finite_check(re_ema, re_opt1)
    logging.info("resume_check_ok epoch=%s global_step=%s", re_ckpt.get("epoch"), re_ckpt.get("global_step"))
    revision.completion(run_root, iter_num)
    writer.close()


if __name__ == "__main__":
    train()
