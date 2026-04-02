#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from data_utils.ICESatDualTaskDataLoader import ICESatDualTaskDataset
from models.pointnet2_icesat_dual import CGTCNet


def to_device(batch, device):
    out = {}
    for k, v in batch.items():
        out[k] = v.to(device) if isinstance(v, torch.Tensor) else v
    return out


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_pkl", type=str, default="data/processed/test.pkl")
    parser.add_argument("--checkpoint", type=str, default="checkpoints/cgtc_net.pt")
    parser.add_argument("--out_dir", type=str, default="outputs/test")
    parser.add_argument("--batch_size", type=int, default=4)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pred_csv = out_dir / "predictions.csv"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location="cpu")
    model_args = ckpt.get("args", {})
    model = CGTCNet(in_ch=6, d_model=model_args.get("d_model", 128), seq_backbone=model_args.get("seq_backbone", "tcn"))
    model.load_state_dict(ckpt["model"], strict=True)
    model.to(device).eval()

    ds = ICESatDualTaskDataset(args.test_pkl, split="test", augment=False)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False)

    rows = []
    idx_offset = 0
    with torch.no_grad():
        for batch in loader:
            bsz = batch["features_obs"].shape[0]
            batch = to_device(batch, device)
            out = model(batch["features_obs"].float(), batch["masks_obs"].float(), batch["ground_anchor_gt"].float(), batch["anchor_valid_mask"].float())

            pred = out["ground_line"].cpu()
            conf = out["confidence"].cpu()
            gt = batch["ground_line_gt"].cpu()
            anchor = batch["ground_anchor_gt"].cpu()
            valid = batch["anchor_valid_mask"].cpu()

            for bi in range(bsz):
                L = pred.shape[1]
                for t in range(L):
                    rows.append(
                        {
                            "sample_idx": idx_offset + bi,
                            "bin_idx": t,
                            "pred_ground_line": float(pred[bi, t]),
                            "gt_ground_line": float(gt[bi, t]),
                            "ground_anchor": float(anchor[bi, t]),
                            "anchor_valid": float(valid[bi, t]),
                            "confidence": float(conf[bi, t]),
                        }
                    )

                fig = plt.figure(figsize=(10, 4))
                x = torch.arange(L)
                plt.plot(x, pred[bi], label="pred")
                plt.plot(x, gt[bi], label="gt")
                am = valid[bi] > 0
                if am.any():
                    plt.scatter(x[am], anchor[bi][am], s=10, label="anchor")
                plt.title(f"sample_{idx_offset + bi}")
                plt.legend()
                plt.tight_layout()
                fig.savefig(out_dir / f"sample_{idx_offset + bi}.png", dpi=150)
                plt.close(fig)
            idx_offset += bsz

    with pred_csv.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()) if rows else ["sample_idx"])
        writer.writeheader()
        writer.writerows(rows)

    print(f"Saved {pred_csv} and {idx_offset} visualizations under {out_dir}")


if __name__ == "__main__":
    main()
