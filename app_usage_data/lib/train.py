"""Training helpers: class weights, Task-B target counts, dataset adaptor."""
from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset


def class_weights(target_indices: np.ndarray, V: int, style: str = "sqrt_inv_freq") -> np.ndarray:
    counts = np.bincount(target_indices, minlength=V).astype(np.float64) + 1.0
    counts[:3] = counts.max()
    if style == "sqrt_inv_freq":
        w = 1.0 / np.sqrt(counts)
    elif style == "inv_freq":
        w = 1.0 / counts
    else:
        w = np.ones(V, dtype=np.float64)
    w = w / w.mean()
    return w.astype(np.float32)


def build_window_target_counts(target_ts_ns: np.ndarray, target_app_idx: np.ndarray,
                               horizon_ns: int, V: int) -> np.ndarray:
    """For each target event, count apps of subsequent target events in [t, t+horizon]."""
    M = len(target_ts_ns)
    counts = np.zeros((M, V), dtype=np.float32)
    j = 0
    for i in range(M):
        t_end = target_ts_ns[i] + horizon_ns
        # Start from i (include current event)
        k = i
        while k < M and target_ts_ns[k] <= t_end:
            a = int(target_app_idx[k])
            counts[i, a] += 1.0
            k += 1
    return counts


class TargetWindowDataset(Dataset):
    """Yields (history tensors, target app, target hour Fourier, window counts)."""

    def __init__(self, hist_dict: dict, window_counts: np.ndarray | None = None):
        self.h_app = torch.as_tensor(hist_dict["history_app"], dtype=torch.long)
        self.h_feat = torch.as_tensor(hist_dict["history_feat"], dtype=torch.float32)
        self.h_mask = torch.as_tensor(hist_dict["history_mask"], dtype=torch.bool)
        self.t_app = torch.as_tensor(hist_dict["target_app"], dtype=torch.long)
        self.side_h = torch.as_tensor(hist_dict["target_hour_fourier"], dtype=torch.float32)
        if window_counts is not None:
            self.win = torch.as_tensor(window_counts, dtype=torch.float32)
        else:
            self.win = None

    def __len__(self):
        return int(self.t_app.shape[0])

    def __getitem__(self, i):
        out = {
            "history_app": self.h_app[i],
            "history_feat": self.h_feat[i],
            "history_mask": self.h_mask[i],
            "target_app": self.t_app[i],
            "side_hour_fourier": self.side_h[i],
        }
        if self.win is not None:
            out["win_counts"] = self.win[i]
        return out


import torch  # placed here to avoid import order warnings
