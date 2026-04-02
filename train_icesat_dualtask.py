#!/usr/bin/env python3
from __future__ import annotations

import argparse
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from data_utils.ICESatDualTaskDataLoader import ICESatDualTaskDataset
from models.pointnet2_icesat_dual import CGTCNet, compute_losses


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        if isinstance(v, torch.Tensor):
            out[k] = v.to(device)
        else:
            out[k] = v
    return out


def run_epoch(model, loader, optimizer, device, train=True):
    meter = defaultdict(float)
    n = 0
    model.train(train)
    for batch in loader:
        batch = to_device(batch, device)
        outputs = model(
            batch["features_obs"].float(),
            batch["masks_obs"].float(),
            batch["ground_anchor_gt"].float(),
            batch["anchor_valid_mask"].float(),
        )
        losses = compute_losses(outputs, batch)

        if train:
            optimizer.zero_grad()
            losses["loss_total"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()

        for k, v in losses.items():
            meter[k] += float(v.item())
        pred = outputs["ground_line"]
        gt = batch["ground_line_gt"].float()
        mask = batch["has_ground_mask"].float()
        rmse = torch.sqrt((((pred - gt) ** 2) * mask).sum() / (mask.sum() + 1e-6))
        meter["val_rmse"] += float(rmse.item())
        n += 1

    for k in list(meter.keys()):
        meter[k] /= max(n, 1)
    return meter


def maybe_prepare_data(args):
    if not args.prepare_if_missing:
        return

    train_pkl = Path(args.train_pkl)
    val_pkl = Path(args.val_pkl)
    if train_pkl.exists() and val_pkl.exists():
        return

    raw_dir = Path(args.raw_dir)
    out_dir = Path(args.processed_dir)
    if not raw_dir.exists():
        raise FileNotFoundError(f"Raw data dir not found: {raw_dir}")

    cmd = [
        sys.executable,
        "data_utils/prepare_icesat_dualtask.py",
        "--raw_dir",
        str(raw_dir),
        "--out_dir",
        str(out_dir),
        "--bin_size_m",
        str(args.bin_size_m),
        "--seq_length",
        str(args.seq_length),
        "--stride",
        str(args.stride),
        "--max_points_per_bin",
        str(args.max_points_per_bin),
        "--seed",
        str(args.seed),
    ]
    print("[Data] train/val pkl not found, running preprocessing:")
    print(" ", " ".join(cmd))
    subprocess.run(cmd, check=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train_pkl", type=str, default="data/processed/train.pkl")
    parser.add_argument("--val_pkl", type=str, default="data/processed/val.pkl")
    parser.add_argument("--raw_dir", type=str, default="data/raw")
    parser.add_argument("--processed_dir", type=str, default="data/processed")
    parser.add_argument("--prepare_if_missing", action="store_true")
    parser.add_argument("--bin_size_m", type=float, default=0.5)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max_points_per_bin", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)

    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--d_model", type=int, default=128)
    parser.add_argument("--seq_backbone", type=str, default="tcn", choices=["tcn", "mamba"])
    parser.add_argument("--save_path", type=str, default="checkpoints/cgtc_net.pt")
    parser.add_argument("--mamba_layers", type=int, default=3)
    args = parser.parse_args()

    maybe_prepare_data(args)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    train_ds = ICESatDualTaskDataset(args.train_pkl, split="train", augment=True)
    val_ds = ICESatDualTaskDataset(args.val_pkl, split="val", augment=False)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=2)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=2)

    model = CGTCNet(in_ch=6, d_model=args.d_model, seq_backbone=args.seq_backbone, mamba_layers=args.mamba_layers).to(device)
    if args.seq_backbone == "mamba":
        print(f"[Backbone] mamba requested, real_mamba={model.using_real_mamba}")
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)

    best_rmse = 1e9
    for epoch in range(1, args.epochs + 1):
        tr = run_epoch(model, train_loader, optimizer, device, train=True)
        va = run_epoch(model, val_loader, optimizer, device, train=False)

        print(
            f"Epoch {epoch:03d} "
            f"loss_total={tr['loss_total']:.4f} "
            f"cls_s={tr['loss_cls_strong']:.4f} cls_w={tr['loss_cls_weak']:.4f} "
            f"line={tr['loss_line']:.4f} slope={tr['loss_slope']:.4f} curv={tr['loss_curv']:.4f} "
            f"anchor={tr['loss_anchor']:.4f} shape={tr['loss_shape']:.4f} "
            f"val_rmse={va['val_rmse']:.4f}"
        )

        if va["val_rmse"] < best_rmse:
            best_rmse = va["val_rmse"]
            Path(args.save_path).parent.mkdir(parents=True, exist_ok=True)
            torch.save({"model": model.state_dict(), "args": vars(args)}, args.save_path)
            print(f"Saved best checkpoint to {args.save_path}")


if __name__ == "__main__":
    main()
