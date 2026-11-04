# labeled data
image, label, imageA1, imageA2 = next(iter_train_labeled_dataloader)
image, label, imageA1, imageA2 = image.to(device), label.to(device), imageA1.to(device), imageA2.to(device)
# unlabel data
uimage, _, uimageA1, uimageA2 = next(iter_train_unlabeled_dataloader)
uimage, uimageA1, uimageA2 = uimage.to(device), uimageA1.to(device), uimageA2.to(device)
# labeled auxiliary data
laimage, lalabel, laimageA1, laimageA2, lamask = next(iter_train_labeled_aux_dataloader)
laimage, lalabel, laimageA1, laimageA2, lamask = laimage.to(device), lalabel.to(device), laimageA1.to(device), laimageA2.to(device), lamask.to(device)
# unlabeled auxiliary data
uaimage, _, uaimageA1, uaimageA2, uamask = next(iter_train_unlabeled_aux_dataloader)
uaimage, uaimageA1, uaimageA2, uamask = uaimage.to(device), uaimageA1.to(device), uaimageA2.to(device), uamask.to(device)

optimizer1.zero_grad()
optimizer2.zero_grad()
with torch.no_grad():
    # student 1
    s1_pred_a = student1(torch.cat([laimage, uaimage]))
    s1_pred_la_logits, s1_pred_ua_logits = s1_pred_a['out'].chunk(2)
    s1_pred_la_probs = torch.softmax(s1_pred_la_logits, dim=1)
    s1_pred_la_pseudo = torch.argmax(s1_pred_la_probs, dim=1).clone()
    s1_pred_la_conf = s1_pred_la_probs.max(dim=1)[0].clone()
    s1_pred_ua_probs = torch.softmax(s1_pred_ua_logits, dim=1)
    s1_pred_ua_pseudo = torch.argmax(s1_pred_ua_probs, dim=1).clone()
    s1_pred_ua_conf = s1_pred_ua_probs.max(dim=1)[0].clone()

    s1_pred_lu = student1(torch.cat([image, uimage]))
    s1_pred_l_logits, s1_pred_u_logits = s1_pred_lu['out'].chunk(2)
    s1_pred_l_probs = torch.softmax(s1_pred_l_logits, dim=1) # 8 4 256 256
    s1_pred_l_pseudo = torch.argmax(s1_pred_l_probs, dim=1) # 8 256 256
    s1_pred_l_conf = s1_pred_l_probs.max(dim=1)[0]
    s1_pred_u_probs = torch.softmax(s1_pred_u_logits, dim=1) # 8 4 256 256
    s1_pred_u_pseudo = torch.argmax(s1_pred_u_probs, dim=1) # 8 256 256
    s1_pred_u_conf = s1_pred_u_probs.max(dim=1)[0]

    s1_pred_uA1_logits = student1(uimageA1)['out']

    # student 2
    s2_pred_a = student2(torch.cat([laimage, uaimage]))
    s2_pred_la_logits, s2_pred_ua_logits = s2_pred_a['out'].chunk(2)
    s2_pred_la_probs = torch.softmax(s2_pred_la_logits, dim=1)
    s2_pred_la_pseudo = torch.argmax(s2_pred_la_probs, dim=1).clone()
    s2_pred_la_conf = s2_pred_la_probs.max(dim=1)[0].clone()
    s2_pred_ua_probs = torch.softmax(s2_pred_ua_logits, dim=1)
    s2_pred_ua_pseudo = torch.argmax(s2_pred_ua_probs, dim=1).clone()
    s2_pred_ua_conf = s2_pred_ua_probs.max(dim=1)[0].clone()

    s2_pred_lu = student2(torch.cat([image, uimage]))
    s2_pred_l_logits, s2_pred_u_logits = s2_pred_lu['out'].chunk(2)
    s2_pred_l_probs = torch.softmax(s2_pred_l_logits, dim=1) # 8 4 256 256
    s2_pred_l_pseudo = torch.argmax(s2_pred_l_probs, dim=1) # 8 256 256
    s2_pred_l_conf = s2_pred_l_probs.max(dim=1)[0]
    s2_pred_u_probs = torch.softmax(s2_pred_u_logits, dim=1) # 8 4 256 256
    s2_pred_u_pseudo = torch.argmax(s2_pred_u_probs, dim=1) # 8 256 256
    s2_pred_u_conf = s2_pred_u_probs.max(dim=1)[0]

    s2_pred_uA1_logits = student2(uimageA1)['out']

# AdaMix
s1_imageA1, s1_label, s1_pred_l_conf = s1_adamix_l(oimage=imageA1, aimage=laimageA1, olabel=label.squeeze(1).long(), alabel=lalabel.squeeze(1).long(), oconf=s1_pred_l_conf, aconf=s1_pred_la_conf, prediction=s1_pred_l_logits, cur_step=idx + len(pbar) * (epoch-1))
s1_uimageA1, s2_pred_u_pseudo, s2_pred_u_conf = s1_adamix_u(oimage=uimageA1, aimage=uaimageA1, olabel=s2_pred_u_pseudo, alabel=s2_pred_ua_pseudo, oconf=s2_pred_u_conf, aconf=s2_pred_ua_conf, prediction=s1_pred_uA1_logits, cur_step=idx + len(pbar) * (epoch-1))

s2_imageA1, s2_label, s2_pred_l_conf = s2_adamix_l(oimage=imageA1, aimage=laimageA1, olabel=label.squeeze(1).long(), alabel=lalabel.squeeze(1).long(), oconf=s2_pred_l_conf, aconf=s2_pred_la_conf, prediction=s2_pred_l_logits, cur_step=idx + len(pbar) * (epoch-1))
s2_uimageA1, s1_pred_u_pseudo, s1_pred_u_conf = s2_adamix_u(oimage=uimageA1, aimage=uaimageA1, olabel=s1_pred_u_pseudo, alabel=s1_pred_ua_pseudo, oconf=s1_pred_u_conf, aconf=s1_pred_ua_conf, prediction=s2_pred_uA1_logits, cur_step=idx + len(pbar) * (epoch-1))

s1_pred_l_logits = student1(s1_imageA1)['out']
s1_pred_uA1_logits = student1(s1_uimageA1)['out']
s2_pred_l_logits = student2(s2_imageA1)['out']
s2_pred_uA1_logits = student2(s2_uimageA1)['out']


# supervised path
loss_s = (criterion_s(s1_pred_l_logits, s1_label.long()) + criterion_s(s2_pred_l_logits, s2_label.long())) / 2. 
# unsupervised path
loss_u = (criterion_u(s1_pred_uA1_logits, s2_pred_u_pseudo.detach(), pixel_mask=(s2_pred_u_conf >= conf_threshold).float()) \
        + criterion_u(s2_pred_uA1_logits, s1_pred_u_pseudo.detach(), pixel_mask=(s1_pred_u_conf >= conf_threshold).float())) / 2.

loss = (loss_s + loss_u) / 2.

loss.backward()
optimizer1.step()
optimizer2.step()
included_ratio = (s1_pred_u_conf >= conf_threshold).sum() / s1_pred_u_conf.numel()

