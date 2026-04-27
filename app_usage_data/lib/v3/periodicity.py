"""v4 periodicity priors: same-hour-yesterday and same-hour-last-week dominant app."""
from __future__ import annotations

from collections import Counter

import numpy as np

SECS_24H = 24 * 3600
SECS_7D = 7 * 24 * 3600
DEFAULT_LOOKBACKS_S = (SECS_24H, SECS_7D)
DEFAULT_HALF_WINDOW_S = 30 * 60


def build_periodicity_for_anchors(
    anchor_ts_ns: np.ndarray,
    stream_ts_ns: np.ndarray,
    stream_app_idx: np.ndarray,
    topk_app_indices: np.ndarray,
    lookbacks_seconds=DEFAULT_LOOKBACKS_S,
    half_window_seconds: int = DEFAULT_HALF_WINDOW_S,
) -> np.ndarray:
    """For each anchor and each lookback L, return a one-hot over {top-K apps + OTHER}
    indicating the most-frequent app in [t-L-half, t-L+half].

    Returns (M, len(lookbacks) * (K + 1)) float32. Empty windows yield all-zero rows.
    """
    M = int(len(anchor_ts_ns))
    K = int(len(topk_app_indices))
    nL = int(len(lookbacks_seconds))
    out = np.zeros((M, nL * (K + 1)), dtype=np.float32)
    if len(stream_ts_ns) == 0:
        return out

    anchors = np.asarray(anchor_ts_ns, dtype=np.int64)
    stream_ts = np.asarray(stream_ts_ns, dtype=np.int64)
    stream_app = np.asarray(stream_app_idx, dtype=np.int64)

    if len(stream_ts) > 1 and not np.all(stream_ts[:-1] <= stream_ts[1:]):
        order = np.argsort(stream_ts, kind="mergesort")
        stream_ts = stream_ts[order]
        stream_app = stream_app[order]

    slot_for_app = {int(topk_app_indices[i]): i for i in range(K)}
    other_slot = K
    half_ns = int(half_window_seconds) * 1_000_000_000

    for li in range(nL):
        lookback_ns = int(lookbacks_seconds[li]) * 1_000_000_000
        offset = li * (K + 1)
        for mi in range(M):
            center = int(anchors[mi]) - lookback_ns
            lo_ns = center - half_ns
            hi_ns = center + half_ns
            l_idx = int(np.searchsorted(stream_ts, lo_ns, side="left"))
            r_idx = int(np.searchsorted(stream_ts, hi_ns, side="left"))
            if r_idx <= l_idx:
                continue
            counter = Counter(int(stream_app[j]) for j in range(l_idx, r_idx))
            if not counter:
                continue
            best_app = counter.most_common(1)[0][0]
            slot = slot_for_app.get(int(best_app), K)  # K = OTHER slot
            out[mi, offset + int(slot)] = 1.0
    return out
