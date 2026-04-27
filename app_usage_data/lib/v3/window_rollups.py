"""v3 multi-window behavioural rollups (causal, anchor-indexed)."""
from __future__ import annotations

from typing import Sequence, Tuple

import numpy as np

from .categories import NUM_CATEGORIES

DEFAULT_WINDOWS_SEC: Tuple[int, ...] = (900, 1800, 3600, 7200, 21600)
TOP_K_APPS = 5
SCALAR_FIELDS = 5
PER_WINDOW_DIM = SCALAR_FIELDS + NUM_CATEGORIES + TOP_K_APPS


def total_window_dim(n_windows: int) -> int:
    return n_windows * PER_WINDOW_DIM


def _log1p(x) -> float:
    return float(np.log1p(max(0.0, float(x))))


def build_window_aggregates(
    anchor_ts_ns: np.ndarray,
    stream_ts_ns: np.ndarray,
    stream_app_idx: np.ndarray,
    stream_cat_idx: np.ndarray,
    stream_dur_s: np.ndarray,
    windows_sec: Sequence[int] = DEFAULT_WINDOWS_SEC,
) -> np.ndarray:
    """Return (M, n_windows * PER_WINDOW_DIM) aggregates for each anchor.

    Args:
        anchor_ts_ns:     (M,) int64 anchor timestamps in ns
        stream_ts_ns:     (N,) int64 timestamps of TARGET events, sorted asc
        stream_app_idx:   (N,) app id per target event
        stream_cat_idx:   (N,) category id per target event
        stream_dur_s:     (N,) dwell duration in seconds (>=0)
        windows_sec:      tuple of window sizes in seconds
    """
    anchors = np.asarray(anchor_ts_ns, dtype=np.int64)
    ts = np.asarray(stream_ts_ns, dtype=np.int64)
    apps = np.asarray(stream_app_idx, dtype=np.int64)
    cats = np.asarray(stream_cat_idx, dtype=np.int64)
    durs = np.asarray(stream_dur_s, dtype=np.float32)

    if len(ts) > 1 and not np.all(ts[:-1] <= ts[1:]):
        order = np.argsort(ts, kind="mergesort")
        ts = ts[order]
        apps = apps[order]
        cats = cats[order]
        durs = durs[order]

    M = len(anchors)
    windows = list(windows_sec)
    W = len(windows)
    out = np.zeros((M, W * PER_WINDOW_DIM), dtype=np.float32)

    right_idx = np.searchsorted(ts, anchors, side="left")
    for wi, w_sec in enumerate(windows):
        base = wi * PER_WINDOW_DIM
        w_ns = int(w_sec) * 1_000_000_000
        left_idx = np.searchsorted(ts, anchors - w_ns, side="left")
        for i in range(M):
            lo = int(left_idx[i])
            hi = int(right_idx[i])
            if hi <= lo:
                continue
            win_apps = apps[lo:hi]
            win_cats = cats[lo:hi]
            win_durs = np.maximum(durs[lo:hi], 0.0)
            n = hi - lo

            u_apps = int(np.unique(win_apps).size)
            u_cats = int(np.unique(win_cats).size)
            n_trans = max(0, n - 1)
            n_self = int(np.sum(win_apps[1:] == win_apps[:-1])) if n > 1 else 0
            n_switch = n_trans - n_self

            out[i, base + 0] = _log1p(u_apps)
            out[i, base + 1] = _log1p(u_cats)
            out[i, base + 2] = _log1p(n_trans)
            out[i, base + 3] = _log1p(n_self)
            out[i, base + 4] = _log1p(n_switch)

            # dominant category one-hot (by dwell duration)
            cat_weight = np.zeros(NUM_CATEGORIES, dtype=np.float64)
            for k in range(n):
                c = int(win_cats[k])
                if 0 <= c < NUM_CATEGORIES:
                    cat_weight[c] += float(win_durs[k])
            if cat_weight.sum() > 0:
                dom = int(np.argmax(cat_weight))
                out[i, base + SCALAR_FIELDS + dom] = 1.0
            # top-K app duration shares in this window
            app_weight: dict = {}
            for k in range(n):
                av = int(win_apps[k])
                app_weight[av] = app_weight.get(av, 0.0) + float(win_durs[k])
            total = sum(app_weight.values())
            if total > 0:
                top = sorted(app_weight.values(), reverse=True)[:TOP_K_APPS]
                for si, sv in enumerate(top):
                    out[i, base + SCALAR_FIELDS + NUM_CATEGORIES + si] = float(sv / total)
    return out


TOP_K_APPS = 5
