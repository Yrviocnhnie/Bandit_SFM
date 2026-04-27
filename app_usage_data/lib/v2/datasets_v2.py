"""v2 datasets: TaskADataset and TaskBDataset packaging short+long history + profile."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


class TaskADataset(Dataset):
    def __init__(self, tensors: dict):
        self.history_app = torch.as_tensor(tensors["history_app"], dtype=torch.long)
        self.history_feat = torch.as_tensor(tensors["history_feat"], dtype=torch.float32)
        self.history_mask = torch.as_tensor(tensors["history_mask"], dtype=torch.bool)
        self.long_app = torch.as_tensor(tensors["long_app"], dtype=torch.long)
        self.long_feat = torch.as_tensor(tensors["long_feat"], dtype=torch.float32)
        self.long_mask = torch.as_tensor(tensors["long_mask"], dtype=torch.bool)
        self.long_dt_bin = torch.as_tensor(tensors["long_dt_bin"], dtype=torch.long)
        self.profile = torch.as_tensor(tensors["profile"], dtype=torch.float32)
        self.side_hour_fourier = torch.as_tensor(tensors["side_hour_fourier"], dtype=torch.float32)
        self.target_app = torch.as_tensor(tensors["target_app"], dtype=torch.long)

    def __len__(self):
        return int(self.target_app.shape[0])

    def __getitem__(self, i):
        return {
            "history_app": self.history_app[i],
            "history_feat": self.history_feat[i],
            "history_mask": self.history_mask[i],
            "long_app": self.long_app[i],
            "long_feat": self.long_feat[i],
            "long_mask": self.long_mask[i],
            "long_dt_bin": self.long_dt_bin[i],
            "profile": self.profile[i],
            "side_hour_fourier": self.side_hour_fourier[i],
            "target_app": self.target_app[i],
        }


class TaskBDataset(torch.utils.data.Dataset):
    def __init__(self, tensors: dict, win_counts: np.ndarray):
        self.history_app = torch.as_tensor(tensors["history_app"], dtype=torch.long)
        self.history_feat = torch.as_tensor(tensors["history_feat"], dtype=torch.float32)
        self.history_mask = torch.as_tensor(tensors["history_mask"], dtype=torch.bool)
        self.long_app = torch.as_tensor(tensors["long_app"], dtype=torch.long)
        self.long_feat = torch.as_tensor(tensors["long_feat"], dtype=torch.float32)
        self.long_mask = torch.as_tensor(tensors["long_mask"], dtype=torch.bool)
        self.long_dt_bin = torch.as_tensor(tensors["long_dt_bin"], dtype=torch.long)
        self.profile = torch.as_tensor(tensors["profile"], dtype=torch.float32)
        self.side_hour_fourier = torch.as_tensor(tensors["side_hour_fourier"], dtype=torch.float32)
        self.win_counts = torch.as_tensor(win_counts, dtype=torch.float32)

    def __len__(self):
        return int(self.win_counts.shape[0])

    def __getitem__(self, i):
        return {
            "history_app": self.history_app[i],
            "history_feat": self.history_feat[i],
            "history_mask": self.history_mask[i],
            "long_app": self.long_app[i],
            "long_feat": self.long_feat[i],
            "long_mask": self.long_mask[i],
            "long_dt_bin": self.long_dt_bin[i],
            "profile": self.profile[i],
            "side_hour_fourier": self.side_hour_fourier[i],
            "win_counts": self.win_counts[i],
        }


