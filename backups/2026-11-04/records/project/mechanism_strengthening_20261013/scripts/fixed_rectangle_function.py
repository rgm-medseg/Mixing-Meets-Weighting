import numpy as np, torch
def generate_adamix_bcp_lite_v1_mask(img, logits_a, logits_b, min_side=0.18, max_side=0.38):
    """Frozen PROMISE12 AdaMix sampling/mixing rule; class-count independent."""
    (batch_size, _, img_x, img_y) = img.shape
    min_side = max(0.05, min(0.75, float(min_side)))
    max_side = max(min_side, min(0.9, float(max_side)))
    with torch.no_grad():
        conf_a = torch.softmax(logits_a.detach(), dim=1).max(dim=1)[0].mean()
        conf_b = torch.softmax(logits_b.detach(), dim=1).max(dim=1)[0].mean()
        teacher_conf = torch.clamp(0.5 * (conf_a + conf_b), 0.0, 1.0)
        side_fraction = min_side + (max_side - min_side) * teacher_conf.item()
    patch_x = 97
    patch_y = 97
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
    stats = {'teacher_conf': float(teacher_conf.detach().cpu()), 'side_fraction': float(side_fraction), 'patch_area_ratio': float(patch_x * patch_y / float(img_x * img_y)), 'class_histogram': class_histogram, 'foreground_coverage': float(1.0 - class_histogram[0]), 'finite': bool(torch.isfinite(teacher_conf).item() and np.isfinite(side_fraction))}
    batch_mask = mask.unsqueeze(0).expand(batch_size, -1, -1)
    return (batch_mask.long(), loss_mask.long(), stats)
