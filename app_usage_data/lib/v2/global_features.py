"""Profile statistics and long-history builders for v2."""
from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd


TOP_K = 8
PROFILE_DIM = TOP_K * 4 + 4 + 2  # 38
DT_BUCKETS = 6
RECENT_24H_SEC = 24 * 3600
RECENT_7D_SEC = 7 * 24 * 3600


def log_bucket_dt(dt_sec):
    arr = np.asarray(dt_sec, dtype=np.float64)
    out = np.zeros_like(arr, dtype=np.int64)
    out[arr >= 60] = 1
    out[arr >= 600] = 2
    out[arr >= 3600] = 3
    out[arr >= 6 * 3600] = 4
    out[arr >= 24 * 3600] = 5
    return out


def _normalize_rows(matrix):
    s = matrix.sum(axis=-1, keepdims=True)
    s = np.where(s == 0, 1.0, s)
    return (matrix / s).astype(np.float32)


def fit_profile_stats(df_train, vocab):
    V = len(vocab)
    target_rows = df_train[df_train["is_target_event"].astype(bool)]
    raw_apps = target_rows["app_label_clean"].fillna("<UNK>").astype(str)
    apps = np.array([vocab.get(a, vocab["<UNK>"]) for a in raw_apps], dtype=np.int64)
    hours = target_rows["hour"].to_numpy().astype(int)
    wdays = target_rows["weekday"].to_numpy().astype(int)

    hour_counts = np.zeros((24, V), dtype=np.float64)
    wday_counts = np.zeros((7, V), dtype=np.float64)
    overall = np.zeros(V, dtype=np.float64)
    for h, w, a in zip(hours, wdays, apps):
        if 0 <= h < 24:
            hour_counts[h, a] += 1.0
        if 0 <= w < 7:
            wday_counts[w, a] += 1.0
        overall[a] += 1.0

    hour_counts[:, :3] = 0.0
    wday_counts[:, :3] = 0.0
    overall[:3] = 0.0

    hour_freq = _normalize_rows(hour_counts)
    wday_freq = _normalize_rows(wday_counts)

    top8_hour = np.argsort(-hour_freq, axis=1)[:, :TOP_K].astype(np.int64)
    top8_wday = np.argsort(-wday_freq, axis=1)[:, :TOP_K].astype(np.int64)

    s_overall = overall.sum()
    if s_overall == 0:
        s_overall = 1.0
    overall_prob = (overall / s_overall).astype(np.float32)
    global_top8 = np.argsort(-overall_prob)[:TOP_K].astype(np.int64)

    return {
        "fit_split": "train",
        "V": int(V),
        "hour_freq": hour_freq,
        "wday_freq": wday_freq,
        "top8_hour": top8_hour,
        "top8_wday": top8_wday,
        "global_top8": global_top8,
    }


def save_stats(stats, path):
    with open(path, "wb") as f:
        pickle.dump(stats, f)


def load_stats(path):
    with open(path, "rb") as f:
        stats = pickle.load(f)
    assert stats.get("fit_split") == "train", "profile stats must be fit on train split"
    return stats


def _rolling_slice(
    anchor_ns,
    target_ts_ns,
    target_apps,
    lookback_ns,
    slice_idx,
    V,
):
    """Causal rolling histogram sliced by slice_idx. Returns (M, K) float32."""
    M = len(anchor_ns)
    K = len(slice_idx)
    out = np.zeros((M, K), dtype=np.float32)
    order = np.argsort(anchor_ns)
    n = len(target_ts_ns)
    right = 0
    left = 0
    running = np.zeros(int(V), dtype=np.float64)
    for j in range(M):
        t = int(anchor_ns[order[j]])
        lo = t - int(lookback_ns)
        while right < n and int(target_ts_ns[right]) < t:
            running[int(target_apps[right])] += 1.0
            right += 1
        while left < n and int(target_ts_ns[left]) < lo:
            running[int(target_apps[left])] -= 1.0
            left += 1
        total = running.sum()
        if total > 0:
            out[int(order[j])] = (running[slice_idx] / total).astype(np.float32)
    return out


def _assemble_profile(
    anchor_hour,
    anchor_wd,
    anchor_ns,
    target_ts_full,
    target_app_full,
    session_pos_val,
    start_val,
    stats,
):
    V = int(stats["V"])
    hour_freq = stats["hour_freq"]
    wday_freq = stats["wday_freq"]
    top8_hour = stats["top8_hour"]
    top8_wday = stats["top8_wday"]
    global_top8 = stats["global_top8"]

    M = len(anchor_hour)
    f1 = np.zeros((M, TOP_K), dtype=np.float32)
    f2 = np.zeros((M, TOP_K), dtype=np.float32)
    for i in range(M):
        h = int(anchor_hour[i]) % 24
        w = int(anchor_wd[i]) % 7
        f1[i] = hour_freq[h][top8_hour[h]]
        f2[i] = wday_freq_at_w(stats, w)[top8_wday[w]] if False else wday_freq[w][top8_wday[w]]

    target_ns = target_ts_full.astype("datetime64[ns]").astype(np.int64)
    target_apps = target_app_full_placeholder = target_app_full
    f3 = _rolling_slice(
        anchor_ns=anchor_ns,
        target_ts_ns=target_ns,
        target_apps=target_app_full,
        lookback_ns=int(RECENT_24H_SEC) * 1_000_000_000,
        slice_idx=global_top8,
        V=V,
    )
    f4 = _rolling_slice(
        anchor_ns=anchor_ns,
        target_ts_ns=target_ns,
        target_apps=target_app_full,
        lookback_ns=int(RECENT_7D_SEC) * 1_000_000_000,
        slice_idx=global_top8,
        V=V,
    )

    h_arr = np.asarray(anchor_hour, dtype=float)
    f5 = np.stack([
        np.sin(2 * np.pi * h_arr / 24.0),
        np.cos(2 * np.pi * h_arr / 24.0),
        np.sin(2 * np.pi * h_arr / 12.0),
        np.cos(2 * np.pi * h_arr / 12.0),
    ], axis=-1).astype(np.float32)

    f6 = np.stack([
        np.asarray(session_pos_val, dtype=np.float32),
        np.asarray(start_val, dtype=np.float32),
    ], axis=-1).astype(np.float32)

    out = np.concatenate([f1, f2, f3, f4, f5, f6], axis=1).astype(np.float32)
    assert out.shape[1] == PROFILE_DIM, f"profile dim mismatch: {out.shape[1]} != {PROFILE_DIM}"
    return out


def f1_slot(x):
    return x


def wday_freq_at_w(stats, w):
    return stats["wday_freq"][w]


def _in_session_position(enc):
    """Derive a per-row in-session index (0..N-1) from the enriched session_id array."""
    sid = enc.session_id
    n = len(sid)
    positions = np.zeros(n, dtype=np.int64)
    run = -1
    last_sid = -1
    for i in range(n):
        cur = int(sid[i])
        if cur != last_sid:
            run = 0
            last_sid = cur
        else:
            run = run + 1 if False else run
            run = run + 1
        positions[i] = run
    return positions


def build_profile_for_targets(enc, stats, target_ts_full, target_app_full):
    pos = np.nonzero(enc.is_target)[0]
    M = len(pos)
    anchor_hour = enc.hour[pos].astype(int)
    anchor_wd = enc.weekday[pos].astype(int)
    anchor_ns = enc.ts[pos].astype("datetime64[ns]").astype(np.int64)

    # in-session position per target event
    sid_arr = enc.session_id
    positions = np.zeros(len(enc.app_idx), dtype=np.int64)
    run = 0
    last = None
    for i in range(len(enc.app_idx)):
        s = int(sid_arr[i])
        if s != last:
            run = 0
            last = s
        else:
            run = run + 1
        positions[i] = run
    pos_vals = positions[pos].astype(np.float32)
    session_pos_val = np.minimum(pos_vals, 31.0) / 32.0
    is_start = (pos_vals == 0).astype(np.float32)

    return _assemble_profile(
        anchor_hour=anchor_hour,
        anchor_wd=anchor_wd,
        anchor_ns=anchor_ns,
        target_ts_full=target_ts_full,
        target_app_full=target_app_full,
        session_pos_val=session_pos_val,
        start_val=is_start,
        stats=stats,
    )


def build_profile_for_targets_v2(enc, stats, target_ts_full, target_app_full):
    pos = np.nonzero(enc.is_target)[0]
    anchor_hour = enc.hour[pos].astype(int)
    anchor_wd = enc.weekday[pos].astype(int)
    anchor_ns = enc.ts[pos].astype("datetime64[ns]").astype(np.int64)

    # in-session position
    sid_arr = enc.session_id
    N = len(enc.app_idx)
    in_pos = np.zeros(N, dtype=np.int64)
    run = -1
    last = None
    for i in range(N):
        cur = int(sid_arr[i])
        if cur != last:
            run = 0
            last = cur
        else:
            run += 1
        in_pos[i] = run
    pos_vals = in_pos[np.array(pos, dtype=np.int64)].astype(np.float32)
    session_pos_val = np.minimum(pos_vals, 31.0) / 32.0
    start_val = (pos_vals == 0).astype(np.float32)

    return _assemble_profile(
        anchor_hour=anchor_hour,
        anchor_wd=anchor_wd,
        anchor_ns=anchor_ns,
        target_ts_full=target_ts_full,
        target_app_full=target_app_full,
        session_pos_val=session_pos_val,
        start_val=start_val,
        stats=stats,
    )


def build_profile_for_anchors(enc, anchor_ts, stats, target_ts_full, target_app_full):
    ats = pd.to_datetime(anchor_ts).to_numpy()
    anchor_ns = ats.astype("datetime64[ns]").astype(np.int64)
    anchor_hour = np.array([pd.Timestamp(a).hour for a in ats], dtype=np.int64)
    anchor_wd = np.array([pd.Timestamp(a).weekday() for a in ats], dtype=np.int64)
    M = len(ats)
    session_pos_val = np.zeros(M, dtype=np.float32)
    start_val = np.zeros(M, dtype=np.float32)
    return _assemble_profile(
        anchor_hour=anchor_hour,
        anchor_wd=anchor_wd,
        anchor_ns=anchor_ns,
        target_ts_full=target_ts_full,
        target_app_full=target_app_full,
        session_pos_val=session_pos_val,
        start_val=start_val,
        stats=stats,
    )


def build_long_history_for_targets(enc, k_long: int = 64, pad_idx: int = 0):
    ts_i64 = enc.ts.astype("datetime64[ns]").astype(np.int64)
    D = enc.feat.shape[1]
    target_pos = np.nonzero(enc.is_target)[0]
    M = len(target_pos)
    h_app = np.full((M, k_long), pad_idx, dtype=np.int64)
    h_feat = np.zeros((M, k_long, D), dtype=np.float32)
    h_mask = np.zeros((M, k_long), dtype=bool)
    h_dt = np.zeros((M, k_long), dtype=np.int64)

    for r in range(M):
        pos_i = int(target_pos[r])
        start = max(0, pos_i - k_long)
        take = pos_i - start
        if take > 0:
            h_app[r, -take:] = enc.app_idx[start:pos_i]
            h_feat[r, -take:, :] = enc.feat[start:pos_i]
            h_mask[r, -take:] = True
            dt_s = (ts_i64[pos_i] - ts_i64[start:pos_i])
            h_dt[r, -take:] = log_bucket_dt(dt_s.astype(np.float64) / 1e9)

    return {"long_app": h_app, "long_feat": h_feat, "long_mask": h_mask, "long_dt_bin": h_dt}


def build_long_history_for_anchors(enc, anchor_ts, k_long: int = 64, pad_idx: int = 0):
    ts_i64 = enc.ts.astype("datetime64[ns]").astype(np.int64)
    anchor_i64 = pd.to_datetime(anchor_ts).to_numpy().astype("datetime64[ns]").astype(np.int64)
    D = enc.feat.shape[1]
    M = len(anchor_ts)
    h_app = np.full((M, k_long), pad_idx, dtype=np.int64)
    h_feat = np.zeros((M, k_long, D), dtype=np.float32)
    h_mask = np.zeros((M, k_long), dtype=bool)
    h_dt = np.zeros((M, k_long), dtype=np.int64)
    ins = np.searchsorted(ts_i64, anchor_i64, side="left")
    for i in range(M):
        ip = int(ins[i])
        start = max(0, ip - k_long)
        take = ip - start
        if take > 0:
            h_app[i, -take:] = enc.app_idx[start:ip]
            h_feat[i, -take:, :] = enc.feat[start:ip]
            h_mask[i, -take:] = True
            dt_s = (anchor_i64[i] - ts_i64[start:ip]).astype(np.float64) / 1e9
            h_dt[i, -take:] = log_bucket_dt(dt_s)
    return {"long_app": h_app, "long_feat": h_feat, "long_mask": h_mask, "long_dt_bin": h_dt}
