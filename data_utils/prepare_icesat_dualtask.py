#!/usr/bin/env python3
"""Prepare ICESat-2 dual-task windows for CGTC-Net.

Pipeline highlights:
- split by file first (prevents leakage)
- bin along-track observations into fixed 1D bins
- keep DEM_2M only for ground_line_gt
- keep ORIGINAL_xinDEM for ground anchors and strong ground labels
- map ORIGINAL+classed_pc_flag=1 to weak-only (no strong ground)
"""

from __future__ import annotations

import argparse
import json
import math
import pickle
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

WEAK_IGNORE = -1
STRONG_IGNORE = -1

CLASS_GROUND = 0
CLASS_CANOPY = 1
CLASS_TOP_CANOPY = 2
CLASS_UNKNOWN = 3
NUM_CLASSES = 4


def weak_class_from_flag(flag: int) -> int:
    if flag == 1:
        return CLASS_GROUND
    if flag == 2:
        return CLASS_CANOPY
    if flag == 3:
        return CLASS_TOP_CANOPY
    return WEAK_IGNORE


def strong_class_from_source(source_type: str, classed_pc_flag: int) -> int:
    source_type = str(source_type)
    if source_type == "ORIGINAL_xinDEM":
        return CLASS_GROUND
    if source_type == "ORIGINAL":
        if classed_pc_flag == 2:
            return CLASS_CANOPY
        if classed_pc_flag == 3:
            return CLASS_TOP_CANOPY
        return STRONG_IGNORE
    if source_type == "DEM_LINE_CONSISTENT_TERRAIN":
        return CLASS_GROUND
    return STRONG_IGNORE


def robust_fill_smooth(y: np.ndarray, mask: np.ndarray, passes: int = 3, kernel: int = 9) -> np.ndarray:
    """Simple robust 1D smoother (LOWESS-like iterative median/mean blend)."""
    y = y.astype(np.float32).copy()
    out = y.copy()
    if mask.sum() == 0:
        return out

    idx = np.arange(len(y), dtype=np.float32)
    obs_x = idx[mask]
    obs_y = y[mask]
    out[~mask] = np.interp(idx[~mask], obs_x, obs_y)

    for _ in range(passes):
        pad = kernel // 2
        padded = np.pad(out, (pad, pad), mode="edge")
        smooth = np.empty_like(out)
        for i in range(len(out)):
            w = padded[i : i + kernel]
            med = np.median(w)
            mean = np.mean(w)
            smooth[i] = 0.7 * med + 0.3 * mean
        resid = np.abs(out - smooth)
        scale = np.median(resid) + 1e-6
        weight = 1.0 / (1.0 + (resid / (2.0 * scale)) ** 2)
        out = weight * out + (1 - weight) * smooth
    return out


@dataclass
class PrepConfig:
    bin_size_m: float = 0.5
    seq_length: int = 128
    stride: int = 64
    max_points_per_bin: int = 32
    train_ratio: float = 0.7
    val_ratio: float = 0.15
    seed: int = 42


def split_files(files: List[Path], cfg: PrepConfig) -> Dict[str, List[Path]]:
    rng = random.Random(cfg.seed)
    files = list(files)
    rng.shuffle(files)
    n = len(files)
    n_train = int(round(n * cfg.train_ratio))
    n_val = int(round(n * cfg.val_ratio))
    train = files[:n_train]
    val = files[n_train : n_train + n_val]
    test = files[n_train + n_val :]
    if not test and val:
        test = [val.pop()]
    return {"train": train, "val": val, "test": test}


def to_bin_idx(x: np.ndarray, bin_size_m: float) -> np.ndarray:
    return np.floor(x / bin_size_m).astype(np.int32)


def build_window(df: pd.DataFrame, start_bin: int, cfg: PrepConfig) -> Dict[str, np.ndarray]:
    L, P = cfg.seq_length, cfg.max_points_per_bin
    end_bin = start_bin + L

    feats = np.zeros((L, P, 6), dtype=np.float32)
    masks = np.zeros((L, P), dtype=np.float32)
    cls_gt_strong = np.full((L, P), STRONG_IGNORE, dtype=np.int64)
    cls_gt_weak = np.full((L, P), WEAK_IGNORE, dtype=np.int64)

    ground_line_raw = np.full((L,), np.nan, dtype=np.float32)
    ground_anchor = np.full((L,), 0.0, dtype=np.float32)
    has_ground = np.zeros((L,), dtype=np.float32)
    anchor_valid = np.zeros((L,), dtype=np.float32)

    obs_df = df[df["is_observed"]].copy()
    obs_df = obs_df[(obs_df["bin_idx"] >= start_bin) & (obs_df["bin_idx"] < end_bin)]

    dem_df = df[df["source_type"].astype(str).eq("DEM_2M")]
    dem_df = dem_df[(dem_df["bin_idx"] >= start_bin) & (dem_df["bin_idx"] < end_bin)]

    anchor_df = df[df["source_type"].astype(str).eq("ORIGINAL_xinDEM")]
    anchor_df = anchor_df[(anchor_df["bin_idx"] >= start_bin) & (anchor_df["bin_idx"] < end_bin)]

    z_values = obs_df["z"].to_numpy(dtype=np.float32)
    if len(z_values) == 0:
        return {}
    z_mean = float(z_values.mean())
    z_std = float(z_values.std() + 1e-6)

    for rel_bin in range(L):
        b = start_bin + rel_bin
        rows = obs_df[obs_df["bin_idx"] == b]
        if rows.empty:
            continue
        rows = rows.head(P)
        density = min(len(rows) / float(P), 1.0)
        center_x = (b + 0.5) * cfg.bin_size_m

        for j, (_, r) in enumerate(rows.iterrows()):
            x_offset = float(r["x"] - center_x)
            z_norm = float((r["z"] - z_mean) / z_std)
            weak = weak_class_from_flag(int(r["classed_pc_flag"]))
            weak_onehot = np.zeros((4,), dtype=np.float32)
            if weak >= 0:
                weak_onehot[weak] = 1.0
            feats[rel_bin, j, 0] = x_offset
            feats[rel_bin, j, 1] = z_norm
            feats[rel_bin, j, 2] = weak_onehot[0]
            feats[rel_bin, j, 3] = weak_onehot[1]
            feats[rel_bin, j, 4] = weak_onehot[2]
            feats[rel_bin, j, 5] = density
            masks[rel_bin, j] = 1.0
            cls_gt_weak[rel_bin, j] = weak
            cls_gt_strong[rel_bin, j] = strong_class_from_source(str(r["source_type"]), int(r["classed_pc_flag"]))

    for rel_bin in range(L):
        b = start_bin + rel_bin
        dm = dem_df[dem_df["bin_idx"] == b]
        if not dm.empty:
            ground_line_raw[rel_bin] = float(np.median(dm["z"].to_numpy(dtype=np.float32)))
            has_ground[rel_bin] = 1.0
        an = anchor_df[anchor_df["bin_idx"] == b]
        if not an.empty:
            ground_anchor[rel_bin] = float(np.median(an["z"].to_numpy(dtype=np.float32)))
            anchor_valid[rel_bin] = 1.0

    gt_mask = ~np.isnan(ground_line_raw)
    ground_line = robust_fill_smooth(np.nan_to_num(ground_line_raw, nan=0.0), gt_mask)
    ground_line_norm = (ground_line - z_mean) / z_std

    anchor_norm = ground_anchor.copy()
    anchor_norm[anchor_valid > 0] = (anchor_norm[anchor_valid > 0] - z_mean) / z_std

    return {
        "features_obs": feats,
        "masks_obs": masks,
        "cls_gt_strong": cls_gt_strong,
        "cls_gt_weak": cls_gt_weak,
        "ground_line_gt": ground_line_norm.astype(np.float32),
        "ground_anchor_gt": anchor_norm.astype(np.float32),
        "has_ground_mask": has_ground,
        "anchor_valid_mask": anchor_valid,
        "z_mean": np.float32(z_mean),
        "z_std": np.float32(z_std),
    }


def preprocess_file(csv_path: Path, cfg: PrepConfig) -> List[Dict[str, np.ndarray]]:
    df = pd.read_csv(csv_path)
    required = {"alongtrack_distance", "elevation", "classed_pc_flag", "source_type"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"{csv_path} missing required columns: {missing}")

    x = df["alongtrack_distance"].to_numpy(dtype=np.float32)
    x = x - x.min()
    df["x"] = x
    df["z"] = df["elevation"].to_numpy(dtype=np.float32)
    df["bin_idx"] = to_bin_idx(df["x"].to_numpy(dtype=np.float32), cfg.bin_size_m)

    observed_sources = {"ORIGINAL", "ORIGINAL_xinDEM", "DEM_LINE_CONSISTENT_TERRAIN"}
    df["is_observed"] = df["source_type"].astype(str).isin(observed_sources)

    min_bin = int(df["bin_idx"].min())
    max_bin = int(df["bin_idx"].max())
    windows = []
    for start in range(min_bin, max_bin - cfg.seq_length + 2, cfg.stride):
        win = build_window(df, start, cfg)
        if win:
            win["file_id"] = csv_path.stem
            win["start_bin"] = np.int32(start)
            windows.append(win)
    return windows


def save_split(windows: List[Dict[str, np.ndarray]], out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("wb") as f:
        pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw_dir", type=Path, default=Path("data/raw"))
    parser.add_argument("--out_dir", type=Path, default=Path("data/processed"))
    parser.add_argument("--bin_size_m", type=float, default=0.5)
    parser.add_argument("--seq_length", type=int, default=128)
    parser.add_argument("--stride", type=int, default=64)
    parser.add_argument("--max_points_per_bin", type=int, default=32)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = PrepConfig(
        bin_size_m=args.bin_size_m,
        seq_length=args.seq_length,
        stride=args.stride,
        max_points_per_bin=args.max_points_per_bin,
        seed=args.seed,
    )

    files = sorted(args.raw_dir.glob("*.csv"))
    if not files:
        raise FileNotFoundError(f"No CSV files under {args.raw_dir}")

    split = split_files(files, cfg)
    manifest = {}
    for name, split_files_list in split.items():
        all_windows: List[Dict[str, np.ndarray]] = []
        for f in split_files_list:
            all_windows.extend(preprocess_file(f, cfg))
        out_path = args.out_dir / f"{name}.pkl"
        save_split(all_windows, out_path)
        manifest[name] = {
            "num_files": len(split_files_list),
            "num_windows": len(all_windows),
            "out": str(out_path),
        }
        print(f"[{name}] files={len(split_files_list)} windows={len(all_windows)} -> {out_path}")

    with (args.out_dir / "manifest.json").open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)


if __name__ == "__main__":
    main()
