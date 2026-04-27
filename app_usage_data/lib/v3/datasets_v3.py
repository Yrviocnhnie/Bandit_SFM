"""v3 torch Datasets for TaskAModelV3 and TaskBModelV3.

Same API as v2 datasets + extra per-sample keys: `history_category`,
`history_loc`, `long_category`, `long_loc`, `last_app_idx`.
"""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TaskADatasetV3(Dataset):
    def __init__(self, tensors: dict):
        self.history_app = torch.as_tensor(tensors["history_app"], dtype=torch.long)
        self.history_feat = torch.as_tensor(tensors["history_feat"], dtype=torch.float32)
        self.history_mask = torch.as_tensor(tensors["history_mask"], dtype=torch.bool)
        self.history_category = torch.as_tensor(tensors["history_category"], dtype=torch.long)
        self.history_loc = torch.as_tensor(tensors["history_loc"], dtype=torch.long)
        self.long_app = torch.as_tensor(tensors["long_app"], dtype=torch.long)
        self.long_feat = torch.as_tensor(tensors["long_feat"], dtype=torch.float32)
        self.long_mask = torch.as_tensor(tensors["long_mask"], dtype=torch.bool)
        self.long_dt_bin = torch.as_tensor(tensors["long_dt_bin"], dtype=torch.long)
        self.long_category = torch.as_tensor(tensors["long_category"], dtype=torch.long)
        self.long_loc = torch.as_tensor(tensors["long_loc"], dtype=torch.long)
        self.profile = torch.as_tensor(tensors["profile"], dtype=torch.float32)
        self.last_app_idx = torch.as_tensor(tensors["last_app_idx"], dtype=torch.long)
        self.target_app = torch.as_tensor(tensors["target_app"], dtype=torch.long)

    def __len__(self):
        return int(self.target_app.shape[0])

    def __getitem__(self, i):
        return {
            "history_app": self.history_app[i],
            "history_feat": self.history_feat[i],
            "history_mask": self.history_mask[i],
            "history_category": self.history_category[i],
            "history_loc": self.history_loc[i],
            "long_app": self.long_app[i],
            "long_feat": self.long_feat[i],
            "long_mask": self.long_mask[i],
            "long_dt_bin": self.long_dt_bin[i],
            "long_category": self.long_category[i],
            "long_loc": self.long_loc[i],
            "profile": self.profile[i],
            "last_app_idx": self.last_app_idx[i],
            "target_app": self.target_app[i],
        }


class TaskADatasetV3(Dataset):  # noqa: F811
    def __init__(self, tensors):
        self.t = {}
        for k, v in tensors.items():
            if k == "target_app" or k.startswith("history_") or k.startswith("long_"):
                dtype = torch.long if (k.endswith("_app") or k.endswith("_category")
                                       or k.endswith("_loc") or k.endswith("_dt_bin")
                                       or k == "target_app" or k == "last_app_idx") else (
                    torch.bool if k.endswith("_mask") else torch.float32
                )
                self.t[k] = torch.as_tensor(v, dtype=dtype)
            elif k in ("profile", "side_hour_fourier"):
                self.t[k] = torch.as_tensor(v, dtype=torch.float32)
            elif k == "last_app_idx":
                self.t[k] = torch.as_tensor(v, dtype=torch.long)
            else:
                self.t[k] = torch.as_tensor(v)

    def __len__(self):
        return int(self.t["target_app"].shape[0])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.t.items()}


class TaskBDatasetV3(Dataset):
    def __init__(self, tensors, win_counts):
        self.tensors = {}
        for k, v in tensors.items():
            if k == "win_counts":
                continue
            if k.endswith(("_mask",)):
                self.tensors[k] = torch.as_tensor(v, dtype=torch.bool)
            elif k.endswith(("_feat", "profile", "side_hour_fourier")):
                self.tensors[k] = torch.as_tensor(v, dtype=torch.float32)
            elif k in ("history_app", "history_category", "history_loc",
                       "long_app", "long_category", "long_loc", "long_dt_bin",
                       "last_app_idx", "target_app"):
                self.tensors[k] = torch.as_tensor(v, dtype=torch.long)
            else:
                self.tensors[k] = torch.as_tensor(v)
        self.tensors["win_counts"] = torch.as_tensor(win_counts, dtype=torch.float32)

    def __len__(self):
        return int(self.tensors["win_counts"].shape[0])

    def __getitem__(self, i):
        return {k: v[i] for k, v in self.tensors.items()}
