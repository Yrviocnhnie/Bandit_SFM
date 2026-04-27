"""v4 per-app recency feature builder."""
from __future__ import annotations

import numpy as np

CLIP_SECONDS = 7 * 24 * 3600  # 7 days


def build_recency_for_anchors(
    anchor_ts_ns: np.ndarray,
    stream_ts_ns: np.ndarray,
    stream_app_idx: np.ndarray,
    topk_app_indices: np.ndarray,
    clip_seconds: int = CLIP_SECONDS,
) -> np.ndarray:
    """For each anchor time and each of K top apps, return log1p(seconds since
    the most recent use of that app strictly before the anchor), clipped at
    `clip_seconds`. Returns (M, K) float32."""
    anchors = np.asarray(anchor_ts_ns, dtype=np.int64)
    stream_ts = np.asarray(stream_ts_ns, dtype=np.int64)
    stream_app = np.asarray(stream_app_idx, dtype=np.int64)
    if len(stream_ts) > 1 and not np.all(stream_ts[:-1] <= stream_ts[1:]):
        order = np.argsort(stream_ts, kind="mergesort")
        stream_ts = stream_ts[order]
        stream_app = stream_app[order]

    M = len(anchors)
    K = len(topk_app_indices)
    clip_max = float(clip_seconds)
    log_clip = float(np.log1p(clip_max))
    out = np.full((M, K), log_clip, dtype=np.float32)

    for k_idx in range(K):
        target_app = int(topk_app_indices[k_idx])
        mask = stream_app == target_app
        ts_app = stream_ts[mask]
        if len(ts_app) == 0:
            continue
        ins = np.searchsorted(ts_app, anchors, side="left")
        for mi in range(M):
            ip = int(ins[mi])
            if ip <= 0:
                continue
            gap_ns = int(anchors[mi]) - int(ts_app[ip - 1])
            if gap_ns < 0:
                continue
            gap_sec = gap_ns / 1e9
            if gap_sec > clip_max:
                gap_sec = clip_max
            out[mi, k_idx] = float(np.log1p(gap_sec))
    return out
