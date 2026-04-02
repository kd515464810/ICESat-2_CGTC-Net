#!/usr/bin/env python3
from __future__ import annotations

import pickle
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import Dataset


class ICESatDualTaskDataset(Dataset):
    def __init__(self, pkl_path: str | Path, split: str = "train", augment: bool = True):
        self.pkl_path = Path(pkl_path)
        self.split = split
        self.augment = augment and split == "train"
        with self.pkl_path.open("rb") as f:
            self.samples: List[Dict[str, np.ndarray]] = pickle.load(f)

    def __len__(self) -> int:
        return len(self.samples)

    def _augment(self, sample: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        out = {k: (v.copy() if isinstance(v, np.ndarray) else v) for k, v in sample.items()}
        feats = out["features_obs"]
        masks = out["masks_obs"]
        valid = masks > 0
        if np.random.rand() < 0.8:
            feats[..., 1][valid] += np.random.normal(0.0, 0.02, size=feats[..., 1][valid].shape)
        if np.random.rand() < 0.5:
            drop = (np.random.rand(*masks.shape) < 0.05) & valid
            masks[drop] = 0.0
            feats[drop] = 0.0
            out["masks_obs"] = masks
            out["features_obs"] = feats
        return out

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        sample = self.samples[idx]
        if self.augment:
            sample = self._augment(sample)

        tensor_sample: Dict[str, torch.Tensor] = {}
        for key in [
            "features_obs",
            "masks_obs",
            "cls_gt_strong",
            "cls_gt_weak",
            "ground_line_gt",
            "ground_anchor_gt",
            "has_ground_mask",
            "anchor_valid_mask",
            "z_mean",
            "z_std",
            "start_bin",
        ]:
            if key not in sample:
                continue
            arr = sample[key]
            if key in ["cls_gt_strong", "cls_gt_weak"]:
                tensor_sample[key] = torch.from_numpy(arr.astype(np.int64))
            else:
                tensor_sample[key] = torch.from_numpy(np.asarray(arr))

        tensor_sample["file_id"] = sample.get("file_id", "")
        return tensor_sample
