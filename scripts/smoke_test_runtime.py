#!/usr/bin/env python3
from __future__ import annotations

import torch

from models.pointnet2_icesat_dual import CGTCNet, compute_losses


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    B, L, P, F = 2, 128, 16, 6

    features_obs = torch.randn(B, L, P, F, device=device)
    masks_obs = (torch.rand(B, L, P, device=device) > 0.2).float()
    ground_anchor_gt = torch.randn(B, L, device=device)
    anchor_valid_mask = (torch.rand(B, L, device=device) > 0.5).float()

    cls_gt_strong = torch.randint(low=-1, high=4, size=(B, L, P), device=device)
    cls_gt_weak = torch.randint(low=-1, high=4, size=(B, L, P), device=device)
    ground_line_gt = torch.randn(B, L, device=device)
    has_ground_mask = (torch.rand(B, L, device=device) > 0.3).float()

    model = CGTCNet(in_ch=6, d_model=64, seq_backbone="tcn").to(device)
    out = model(features_obs, masks_obs, ground_anchor_gt, anchor_valid_mask)

    batch = {
        "cls_gt_strong": cls_gt_strong,
        "cls_gt_weak": cls_gt_weak,
        "ground_line_gt": ground_line_gt,
        "has_ground_mask": has_ground_mask,
        "ground_anchor_gt": ground_anchor_gt,
        "anchor_valid_mask": anchor_valid_mask,
    }
    losses = compute_losses(out, batch)
    loss_total = losses["loss_total"]
    loss_total.backward()
    print(f"[SMOKE] forward/backward ok, loss_total={loss_total.item():.4f}")


if __name__ == "__main__":
    main()
