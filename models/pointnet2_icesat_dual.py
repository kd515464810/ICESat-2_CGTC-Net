#!/usr/bin/env python3
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

STRONG_IGNORE = -1
WEAK_IGNORE = -1


class PointBinEncoder(nn.Module):
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.point_mlp = nn.Sequential(
            nn.Linear(in_ch, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )

    def forward(self, features_obs: torch.Tensor, masks_obs: torch.Tensor):
        # features_obs: [B, L, P, F]
        B, L, P, _ = features_obs.shape
        point_feat = self.point_mlp(features_obs)
        masked = point_feat.masked_fill(masks_obs.unsqueeze(-1) <= 0, -1e4)
        bin_feat = masked.max(dim=2).values
        bin_feat = torch.where(torch.isfinite(bin_feat), bin_feat, torch.zeros_like(bin_feat))
        return point_feat, bin_feat


class TCNBackbone(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(d_model, d_model, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, 5, padding=2),
            nn.ReLU(inplace=True),
            nn.Conv1d(d_model, d_model, 9, padding=4),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x.transpose(1, 2)).transpose(1, 2)


class MambaLikeFallback(nn.Module):
    """Fallback when mamba_ssm is unavailable; keeps interface stable."""

    def __init__(self, d_model: int):
        super().__init__()
        try:
            from mamba_ssm import Mamba  # type: ignore

            self.block = Mamba(d_model=d_model, d_state=16, d_conv=4, expand=2)
            self.using_real_mamba = True
        except Exception:
            self.block = nn.GRU(d_model, d_model, batch_first=True, bidirectional=False)
            self.using_real_mamba = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.using_real_mamba:
            return self.block(x)
        out, _ = self.block(x)
        return out


class MultiScaleUNet1D(nn.Module):
    def __init__(self, in_ch: int, d_model: int):
        super().__init__()
        self.enc1 = nn.Conv1d(in_ch, d_model, 3, padding=1)
        self.enc2 = nn.Conv1d(d_model, d_model, 5, padding=2)
        self.dec = nn.Conv1d(d_model * 2, d_model, 3, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, C]
        x = x.transpose(1, 2)
        e1 = F.relu(self.enc1(x), inplace=True)
        e2 = F.relu(self.enc2(F.avg_pool1d(e1, kernel_size=2, stride=2)), inplace=True)
        e2u = F.interpolate(e2, size=e1.shape[-1], mode="linear", align_corners=False)
        out = F.relu(self.dec(torch.cat([e1, e2u], dim=1)), inplace=True)
        return out.transpose(1, 2)


class AnchorPropagation(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.anchor_proj = nn.Linear(2, d_model)
        self.gate = nn.Sequential(nn.Linear(d_model, d_model), nn.Sigmoid())

    def forward(self, ground_anchor_gt: torch.Tensor, anchor_valid_mask: torch.Tensor):
        inp = torch.stack([ground_anchor_gt, anchor_valid_mask], dim=-1)
        feat = self.anchor_proj(inp)
        gate = self.gate(feat) * anchor_valid_mask.unsqueeze(-1)
        return feat * gate, gate


class FusionDecoder(nn.Module):
    def __init__(self, d_model: int):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Linear(d_model * 3, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, d_model),
            nn.ReLU(inplace=True),
        )
        self.line_head = nn.Linear(d_model, 1)
        self.conf_head = nn.Sequential(nn.Linear(d_model, 1), nn.Sigmoid())

    def forward(self, cls_ctx: torch.Tensor, canopy_ctx: torch.Tensor, anchor_ctx: torch.Tensor):
        x = self.fuse(torch.cat([cls_ctx, canopy_ctx, anchor_ctx], dim=-1))
        return self.line_head(x).squeeze(-1), self.conf_head(x).squeeze(-1)


class CGTCNet(nn.Module):
    def __init__(self, in_ch: int = 6, d_model: int = 128, seq_backbone: str = "tcn", num_classes: int = 4):
        super().__init__()
        self.encoder = PointBinEncoder(in_ch, d_model)
        self.seq_backbone = seq_backbone
        if seq_backbone == "mamba":
            self.backbone = MambaLikeFallback(d_model)
        else:
            self.backbone = TCNBackbone(d_model)

        self.point_cls_head = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.ReLU(inplace=True),
            nn.Linear(d_model, num_classes),
        )

        self.canopy_stream = MultiScaleUNet1D(5, d_model)
        self.anchor_stream = AnchorPropagation(d_model)
        self.decoder = FusionDecoder(d_model)

    def forward(self, features_obs: torch.Tensor, masks_obs: torch.Tensor, ground_anchor_gt: torch.Tensor, anchor_valid_mask: torch.Tensor):
        point_feat, bin_feat = self.encoder(features_obs, masks_obs)
        seq_ctx = self.backbone(bin_feat)

        B, L, P, D = point_feat.shape
        seq_expand = seq_ctx.unsqueeze(2).expand(B, L, P, D)
        point_logits = self.point_cls_head(torch.cat([point_feat, seq_expand], dim=-1))
        point_probs = F.softmax(point_logits, dim=-1)

        # canopy envelope from predicted canopy/top classes + z distribution
        z = features_obs[..., 1]
        canopy_prob = point_probs[..., 1] + point_probs[..., 2]
        canopy_mask = canopy_prob * masks_obs
        c_density = canopy_mask.mean(dim=2)
        z_mean = (z * masks_obs).sum(dim=2) / (masks_obs.sum(dim=2) + 1e-6)
        z_max = (z.masked_fill(masks_obs <= 0, -1e4)).max(dim=2).values
        z_q95 = torch.quantile(z.masked_fill(masks_obs <= 0, 0.0), 0.95, dim=2)
        thickness = (z_max - z_mean).clamp(min=0.0)
        canopy_feats = torch.stack([z_mean, z_max, z_q95, c_density, thickness], dim=-1)

        canopy_ctx = self.canopy_stream(canopy_feats)
        anchor_ctx, anchor_gate = self.anchor_stream(ground_anchor_gt, anchor_valid_mask)
        ground_line, confidence = self.decoder(seq_ctx, canopy_ctx, anchor_ctx)

        return {
            "point_logits": point_logits,
            "point_probs": point_probs,
            "ground_line": ground_line,
            "confidence": confidence,
            "anchor_gate": anchor_gate.squeeze(-1),
            "seq_ctx": seq_ctx,
        }


@dataclass
class LossWeights:
    cls_strong: float = 1.0
    cls_weak: float = 0.2
    line: float = 1.0
    slope: float = 0.3
    curv: float = 0.2
    anchor: float = 0.5
    shape: float = 0.1


class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, ignore_index: int = -1):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        valid = target != self.ignore_index
        if valid.sum() == 0:
            return logits.new_tensor(0.0)
        l = logits[valid]
        t = target[valid]
        ce = F.cross_entropy(l, t, reduction="none")
        pt = torch.exp(-ce)
        return ((1 - pt) ** self.gamma * ce).mean()


def soft_dtw_approx(x: torch.Tensor, y: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    # Lightweight differentiable shape proxy via smoothed absolute sequence mismatch.
    diff = (x - y).abs()
    diff = torch.log1p(torch.exp(5 * diff)) / 5.0
    valid = mask > 0
    if valid.sum() == 0:
        return x.new_tensor(0.0)
    return diff[valid].mean()


def compute_losses(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor], weights: LossWeights | None = None) -> Dict[str, torch.Tensor]:
    if weights is None:
        weights = LossWeights()

    point_logits = outputs["point_logits"]
    cls_gt_strong = batch["cls_gt_strong"].to(point_logits.device)
    cls_gt_weak = batch["cls_gt_weak"].to(point_logits.device)

    focal = FocalLoss(ignore_index=STRONG_IGNORE)
    loss_cls_strong = focal(point_logits, cls_gt_strong)

    weak_valid = cls_gt_weak != WEAK_IGNORE
    if weak_valid.sum() > 0:
        logp = F.log_softmax(point_logits[weak_valid], dim=-1)
        soft_targets = F.one_hot(cls_gt_weak[weak_valid], num_classes=point_logits.shape[-1]).float()
        loss_cls_weak = F.kl_div(logp, soft_targets, reduction="batchmean")
    else:
        loss_cls_weak = point_logits.new_tensor(0.0)

    pred_line = outputs["ground_line"]
    gt_line = batch["ground_line_gt"].to(pred_line.device)
    has_ground = batch["has_ground_mask"].to(pred_line.device)
    anchor_gt = batch["ground_anchor_gt"].to(pred_line.device)
    anchor_valid = batch["anchor_valid_mask"].to(pred_line.device)

    line_diff = F.smooth_l1_loss(pred_line, gt_line, reduction="none")
    loss_line = (line_diff * has_ground).sum() / (has_ground.sum() + 1e-6)

    pred_slope = pred_line[:, 1:] - pred_line[:, :-1]
    gt_slope = gt_line[:, 1:] - gt_line[:, :-1]
    slope_mask = has_ground[:, 1:] * has_ground[:, :-1]
    slope_diff = F.smooth_l1_loss(pred_slope, gt_slope, reduction="none")
    loss_slope = (slope_diff * slope_mask).sum() / (slope_mask.sum() + 1e-6)

    pred_curv = pred_slope[:, 1:] - pred_slope[:, :-1]
    gt_curv = gt_slope[:, 1:] - gt_slope[:, :-1]
    curv_mask = slope_mask[:, 1:] * slope_mask[:, :-1]
    curv_diff = F.smooth_l1_loss(pred_curv, gt_curv, reduction="none")
    loss_curv = (curv_diff * curv_mask).sum() / (curv_mask.sum() + 1e-6)

    anchor_diff = F.smooth_l1_loss(pred_line, anchor_gt, reduction="none")
    loss_anchor = (anchor_diff * anchor_valid).sum() / (anchor_valid.sum() + 1e-6)

    loss_shape = soft_dtw_approx(pred_line, gt_line, has_ground)

    total = (
        weights.cls_strong * loss_cls_strong
        + weights.cls_weak * loss_cls_weak
        + weights.line * loss_line
        + weights.slope * loss_slope
        + weights.curv * loss_curv
        + weights.anchor * loss_anchor
        + weights.shape * loss_shape
    )

    return {
        "loss_total": total,
        "loss_cls_strong": loss_cls_strong,
        "loss_cls_weak": loss_cls_weak,
        "loss_line": loss_line,
        "loss_slope": loss_slope,
        "loss_curv": loss_curv,
        "loss_anchor": loss_anchor,
        "loss_shape": loss_shape,
    }
