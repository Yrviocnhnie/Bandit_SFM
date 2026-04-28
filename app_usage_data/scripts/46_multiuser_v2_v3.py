"""Train v2, v3 R4, v3 R6 per user.

For each user under artifacts/multiuser/<set>/<uid>/:
  - Loads splits + vocab.
  - Fits per-user v3 stats: app->category map, location vocab (from
    device_state_update_payload), v2 profile stats, Markov-1 prior. Train only.
  - Trains a shared-backbone v3 dual-head model (Task A + Task B) under one
    of three configs:
      v2     -> no v3 features at all
      v3_r4  -> full v3 features (cat + loc + daypart + windows), no Markov
      v3_r6  -> full v3 features + Markov-1 prior fusion on Task B sigmoid
  - Evaluates on val and test, saves per-user JSON + an aggregate JSON.

Hyperparameters: AdamW lr=1e-3, wd=1e-4, batch=256, 12 epochs, patience 4
on val Hit@5 (Task A), matching v1 multi-user runs.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import features as F
from lib import train as TR
from lib.v2 import global_features as GF
from lib.v3 import categories as CAT
from lib.v3 import location as LOC
from lib.v3 import recency as REC
from lib.v3 import periodicity as PER
from lib.v3 import models_v3 as M3
from lib.v3 import features_v3 as FV
from lib.v3.daypart import daypart_onehot
from lib.v3.window_rollups import build_window_aggregates


HISTORY_K = 16
LONG_K = 64
WIN_NS = 900 * 1_000_000_000
BATCH = 256
LR = 1e-3
WD = 1e-4
SEED = 7
LABEL_SMOOTHING = 0.05
EPOCHS = 12
PATIENCE = 4


def task_a_metrics(probs, targets):
    n = len(targets)
    if n == 0:
        return {"hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "n": 0}
    order = np.argsort(-probs, axis=1)
    h1 = float((order[:, 0] == targets).mean())
    h5 = float((order[:, :5] == targets[:, None]).any(axis=1).mean())
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        w = np.where(order[i] == int(targets[i]))[0]
        ranks[i] = int(w[0]) if len(w) > 0 else int(probs.shape[1] - 1)
    mrr = float((1.0 / (ranks + 1.0)).mean())
    return {"hit_at_1": h1, "hit_at_5": h5, "mrr": mrr, "n": int(n)}


def task_b_metrics(sigs, win_counts):
    base = {
        "event_hit_at_1": 0.0,
        "event_hit_at_3": 0.0,
        "event_hit_at_5": 0.0,
        "event_hit_at_dyn": 0.0,
        "recall_at_5": 0.0,
        "recall_at_dyn": 0.0,
        "coverage_at_5": 0.0,
        "n": 0,
        "n_anchors": 0,
        "avg_g_size": 0.0,
    }
    n = len(win_counts)
    if n == 0:
        return base
    V = win_counts.shape[1]
    order = np.argsort(-sigs, axis=1)
    total_events = 0
    hit1 = 0
    hit3 = 0
    hit5 = 0
    hit_dyn = 0
    rec5_list = []
    rec_dyn_list = []
    cov5_list = []
    g_sizes = []
    for i in range(n):
        ws = win_counts[i]
        tot = int(ws.sum())
        if tot == 0:
            continue
        gt = set()
        for a in range(V):
            if ws[a] > 0:
                gt.add(int(a))
        if len(gt) == 0:
            continue
        gsize = len(gt)
        g_sizes.append(gsize)
        total_events += tot
        ord_i = order[i]
        top1 = ord_i[:1]
        top3 = ord_i[:3]
        top5 = ord_i[:5]
        kd = max(1, gsize)
        td = ord_i[:kd]
        for a in top1:
            hit1 += int(ws[int(a)])
        for a in top3:
            hit3 += int(ws[int(a)])
        for a in top5:
            hit5 += int(ws[int(a)])
        for a in td:
            hit_dyn += int(ws[int(a)])
        top5_set = set(int(a) for a in top5)
        td_set = set(int(a) for a in td)
        denom = float(gsize)
        rec5_list.append(len(top5_set & gt) / denom)
        rec_dyn_list.append(len(td_set & gt) / denom)
        cov5_list.append(1.0 if gt.issubset(top5_set) else 0.0)
    if total_events == 0:
        base["n"] = int(n)
        return base
    return {
        "event_hit_at_1": float(hit1 / total_events),
        "event_hit_at_3": float(hit3 / total_events),
        "event_hit_at_5": float(hit5 / total_events),
        "event_hit_at_dyn": float(hit_dyn / total_events),
        "recall_at_5": float(np.mean(rec5_list)) if rec5_list else 0.0,
        "recall_at_dyn": float(np.mean(rec_dyn_list)) if rec_dyn_list else 0.0,
        "coverage_at_5": float(np.mean(cov5_list)) if cov5_list else 0.0,
        "n": int(n),
        "n_anchors": int(len(g_sizes)),
        "avg_g_size": float(np.mean(g_sizes)) if g_sizes else 0.0,
    }


def cw_inv_sqrt(idx, V):
    counts = np.bincount(idx, minlength=V).astype(np.float32) + 1.0
    counts[:3] = counts.max()
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return w.astype(np.float32)


def build_app_to_cat(vocab):
    V = len(vocab)
    out = np.zeros(V, dtype=np.int64)
    other_idx = CAT.CAT_IDX["other_app"]
    for label, idx in vocab.items():
        if idx == 0:
            out[0] = CAT.CAT_IDX.get("<PAD>", 0)
            continue
        cat_name = CAT.APP_TO_CATEGORY.get(label, "other_app")
        out[idx] = CAT.CAT_IDX.get(cat_name, other_idx)
    return out


def fit_markov_log_prior(train_target_app, V, alpha=0.5):
    counts = np.zeros((V, V), dtype=np.float64)
    seq = train_target_app.astype(int)
    for i in range(1, len(seq)):
        a = int(seq[i - 1])
        b = int(seq[i])
        if a >= 3 and b >= 3:
            counts[a, b] += 1.0
    table = counts + float(alpha)
    table[:, :3] = 0.0
    table[:3, :] = 0.0
    rows = table.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    probs = (table / rows).astype(np.float64)
    return np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32)


def build_target_stream(train_df, val_df, test_df, vocab, app_to_cat):
    full = pd.concat([train_df, val_df, test_df], ignore_index=True)
    full = full.sort_values("event_ts").reset_index(drop=True)
    mask = full["is_target_event"].astype(bool).to_numpy()
    rare = vocab.get("<RARE>", 2)
    labels = full.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    apps = np.array([vocab.get(a, rare) for a in labels], dtype=np.int64)
    cats = app_to_cat[np.clip(apps, 0, len(app_to_cat) - 1)]
    ts = pd.to_datetime(full.loc[mask, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)
    if "seconds_to_next_event" in full.columns:
        durs = full.loc[mask, "seconds_to_next_event"].fillna(60.0).to_numpy().astype(np.float32)
        durs = np.clip(durs, 0.0, 3600.0)
    else:
        durs = np.full(int(mask.sum()), 60.0, dtype=np.float32)
    return {"ts_ns": ts, "app": apps, "cat": cats, "dur_s": durs}


def last_app_for_targets(target_ts_ns, stream_ts_ns, stream_app_idx, default_idx=1):
    target = np.asarray(target_ts_ns, dtype=np.int64)
    ins = np.searchsorted(stream_ts_ns, target, side="left")
    out = np.full(len(target), default_idx, dtype=np.int64)
    for i in range(len(target)):
        ip = int(ins[i])
        if ip > 0:
            out[i] = int(stream_app_idx[ip - 1])
    return out


def prep_split(df, vocab, scaler, profile_stats, app_to_cat, loc_stats, stream,
               use_category, use_loc, use_daypart, use_windows,
               use_recency=False, use_periodicity=False,
               drop_scene=False, drop_f1234=False, drop_long_windows=False, drop_n_trans=False,
               bg_mask_prior=False, bg_scalars=False):
    V = len(vocab)
    enc = F.encode_events(df, vocab, scaler["mean"], scaler["std"])
    if drop_scene:
        # Zero device_state_scene one-hot at cols 17..21 of the 28-d numeric pack
        enc.feat[:, 17:22] = 0.0
    short = F.build_history_for_targets(enc, history_k=HISTORY_K)
    long = GF.build_long_history_for_targets(enc, k_long=LONG_K)

    target_pos = np.nonzero(enc.is_target)[0]
    a_hour = enc.hour[target_pos].astype(int)
    a_wd = enc.weekday[target_pos].astype(int)
    a_ts = enc.ts[target_pos].astype("datetime64[ns]").astype(np.int64)

    cat_per_row = app_to_cat[np.clip(enc.app_idx, 0, len(app_to_cat) - 1)]
    if not use_category:
        cat_per_row = np.zeros_like(cat_per_row)
    if use_loc and loc_stats is not None:
        loc_per_row = LOC.attach_loc_id(df, loc_stats)
    else:
        loc_per_row = np.zeros(len(enc.app_idx), dtype=np.int64)

    hist_cat, hist_loc = FV.build_history_cat_loc_for_targets(
        enc, cat_per_row=cat_per_row, loc_per_row=loc_per_row, history_k=HISTORY_K,
    )
    long_cl = FV.build_long_cat_loc_for_targets(
        enc, cat_per_row=cat_per_row, loc_per_row=loc_per_row, k_long=LONG_K,
    )

    v2_profile = GF.build_profile_for_targets(enc, profile_stats, stream["ts_ns"], stream["app"])
    if drop_f1234:
        v2_profile = v2_profile[:, 32:].astype(np.float32)  # keep F5+F6 only (6 dims)
    parts = [v2_profile]
    if use_daypart:
        parts.append(daypart_onehot(a_hour, a_wd))
    if use_windows:
        windows_sec = (900, 1800, 3600) if drop_long_windows else (900, 1800, 3600, 7200, 21600)
        wagg = build_window_aggregates(
            anchor_ts_ns=a_ts,
            stream_ts_ns=stream["ts_ns"],
            stream_app_idx=stream["app"],
            stream_cat_idx=stream["cat"],
            stream_dur_s=stream["dur_s"],
            windows_sec=windows_sec,
        )
        if drop_n_trans:
            for w_i in range(len(windows_sec)):
                wagg[:, w_i * 21 + 2] = 0.0
        parts.append(wagg)
    if use_recency:
        from lib.v3 import recency as _REC
        top8 = profile_stats["global_top8"].astype(np.int64)
        parts.append(_REC.build_recency_for_anchors(a_ts, stream["ts_ns"], stream["app"], top8))
    if use_periodicity:
        from lib.v3 import periodicity as _PER
        top8 = profile_stats["global_top8"].astype(np.int64)
        parts.append(_PER.build_periodicity_for_anchors(a_ts, stream["ts_ns"], stream["app"], top8))
    profile = np.concatenate(parts, axis=1).astype(np.float32)

    # ----- v5 BG features (optional) ---------------------------------
    bg_mask_arr = None
    if bg_mask_prior or bg_scalars:
        from lib.v3.bg_state import reconstruct_bg, per_target_arrays
        snaps = reconstruct_bg(df, vocab)
        bg_arrs = per_target_arrays(snaps, V)
        if bg_mask_prior:
            bg_mask_arr = bg_arrs["bg_mask"]
        if bg_scalars:
            rec = bg_arrs["bg_recency"]
            mask = bg_arrs["bg_mask"]
            inf_inactive = np.where(mask > 0, rec, np.inf)
            bg_rec_min = np.where(np.isinf(inf_inactive).all(axis=1),
                                   0.0, np.min(inf_inactive, axis=1)).astype(np.float32)
            scalars = np.stack([
                bg_arrs["bg_count"].astype(np.float32),
                bg_rec_min,
                bg_arrs["time_since_screen_on"].astype(np.float32),
            ], axis=1).astype(np.float32)
            profile = np.concatenate([profile, scalars], axis=1).astype(np.float32)

    last_app = last_app_for_targets(a_ts, stream["ts_ns"], stream["app"])

    target_ts_ns = short["target_ts"].astype("datetime64[ns]").astype(np.int64)
    win = TR.build_window_target_counts(target_ts_ns, short["target_app"], int(WIN_NS), V)

    return {
        "history_app": short["history_app"],
        "history_feat": short["history_feat"],
        "history_mask": short["history_mask"],
        "history_category": hist_cat,
        "history_loc": hist_loc.astype(np.int64),
        "long_app": long["long_app"],
        "long_feat": long["long_feat"],
        "long_mask": long["long_mask"],
        "long_dt_bin": long["long_dt_bin"],
        "long_category": long_cl["long_category"],
        "long_loc": long_cl["long_loc"].astype(np.int64),
        "profile": profile,
        "last_app_idx": last_app,
        "target_app": short["target_app"],
        "win_counts": TR.build_window_target_counts(
            short["target_ts"].astype("datetime64[ns]").astype(np.int64),
            short["target_app"], int(WIN_NS), V),
        "bg_mask": (bg_mask_arr if bg_mask_arr is not None
                    else np.zeros((short["target_app"].shape[0], V), dtype=np.float32)),
    }


class SharedBackboneV3(nn.Module):
    def __init__(self, cfg, markov_log_prior=None, bg_mask_prior=False):
        super().__init__()
        self.cfg = cfg
        self.app_emb = nn.Embedding(cfg.vocab_size, cfg.app_emb_dim, padding_idx=0)
        self.cat_emb = (nn.Embedding(cfg.num_categories, cfg.cat_emb_dim, padding_idx=0)
                        if cfg.use_category else None)
        self.loc_emb = (nn.Embedding(cfg.num_locations, cfg.loc_emb_dim, padding_idx=0)
                        if cfg.use_loc else None)
        if cfg.use_local:
            self.local_enc = M3.LocalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_global:
            self.global_enc = M3.GlobalEncoderV3(cfg, self.app_emb, self.cat_emb, self.loc_emb)
        if cfg.use_profile:
            self.profile_enc = M3.ProfileEncoderV3(cfg.profile_dim, cfg.d_profile, cfg.dropout)
        self.fusion = M3.GatedFusionV3(cfg)
        self.head_a = nn.Linear(cfg.d_fused, cfg.vocab_size)
        self.head_b_sig = nn.Linear(cfg.d_fused, cfg.vocab_size)
        self.head_b_rate = nn.Linear(cfg.d_fused, cfg.vocab_size)
        self.use_markov_prior = bool(cfg.use_markov_prior)
        if self.use_markov_prior and markov_log_prior is not None:
            self.register_buffer("markov_log_prior", markov_log_prior.float())
            self.alpha_markov = nn.Parameter(torch.tensor(0.5))
        else:
            self.markov_log_prior = None
            self.alpha_markov = None
        # v5: BG-mask logit prior
        self.bg_mask_prior = bool(bg_mask_prior)
        if self.bg_mask_prior:
            # Task A: log-prior with strong α; Task B: linear bonus
            self.alpha_bg_a = nn.Parameter(torch.tensor(0.5))
            self.alpha_bg_b = nn.Parameter(torch.tensor(0.5))
        else:
            self.alpha_bg_a = None
            self.alpha_bg_b = None

    def forward(self, batch):
        h_local = h_global = h_profile = None
        if self.cfg.use_local:
            h_local = self.local_enc(
                batch["history_app"], batch["history_feat"], batch["history_mask"],
                history_category=batch.get("history_category") if self.cfg.use_category else None,
                history_loc=batch.get("history_loc") if self.cfg.use_loc else None,
            )
        if self.cfg.use_global:
            h_global = self.global_enc(
                batch["long_app"], batch["long_feat"], batch["long_mask"], batch["long_dt_bin"],
                long_category=batch.get("long_category") if self.cfg.use_category else None,
                long_loc=batch.get("long_loc") if self.cfg.use_loc else None,
            )
        else:
            h_global = None
        if self.cfg.use_profile:
            h_profile = self.profile_enc(batch["profile"])
        else:
            h_profile = None
        fused = self.fusion(h_local=h_local, h_global=h_global, h_profile=h_profile)
        logits_a = self.head_a(fused)
        sig = self.head_b_sig(fused)
        rate = self.head_b_rate(fused)
        if self.use_markov_prior and self.markov_log_prior is not None:
            la = batch["last_app_idx"].long()
            alpha = torch.clamp(self.alpha_markov, 0.0, 2.0)
            sig = sig + alpha * self.markov_log_prior[la]
        # v5: BG-mask logit prior
        if self.bg_mask_prior and "bg_mask" in batch:
            bg = batch["bg_mask"].float()
            alpha_a = torch.clamp(self.alpha_bg_a, 0.0, 5.0)
            alpha_b = torch.clamp(self.alpha_bg_b, -2.0, 5.0)
            logits_a = logits_a + alpha_a * torch.log(bg + 1e-3)
            sig = sig + alpha_b * bg
        return {"logits_a": logits_a, "logits_b_sig": sig, "log_rate_b": rate}


def to_torch_dict(d):
    """Convert numpy arrays to torch tensors for the SharedBackboneV3 forward pass."""
    out = {}
    for k, v in d.items():
        if k.endswith("_mask"):
            out[k] = torch.as_tensor(v, dtype=torch.bool)
        elif k in ("history_feat", "long_feat", "profile"):
            out[k] = torch.as_tensor(v, dtype=torch.float32)
        elif k == "win_counts":
            out[k] = torch.as_tensor(v, dtype=torch.float32)
        else:
            out[k] = torch.as_tensor(v, dtype=torch.long)
    return out


def slice_t(t, idx):
    return {k: v[idx] for k, v in t.items()}


def evaluate(model, t, V):
    model.eval()
    n = int(t["target_app"].shape[0])
    if n == 0:
        return (task_a_metrics(np.zeros((0, V), dtype=np.float32), np.zeros(0, dtype=int)),
                task_b_metrics(np.zeros((0, V), dtype=np.float32), np.zeros((0, V), dtype=np.float32)))
    pa_chunks = []
    sb_chunks = []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            sl = slice(i, j)
            b = {k: v[sl] for k, v in t.items()}
            out = model(b)
            pa_chunks.append(torch.softmax(out["logits_a"], dim=-1).cpu().numpy())
            sb_chunks.append(torch.sigmoid(out["logits_b_sig"]).cpu().numpy())
    pa = np.concatenate(pa_chunks, axis=0)
    sb = np.concatenate(sb_chunks, axis=0)
    return (task_a_metrics(pa, t["target_app"].numpy()),
            task_b_metrics(sb, t["win_counts"].numpy()))


def config_flags(config_name):
    """Resolve a config name to feature/markov flags.

    Configs (matching REPORT_v4 §2 + §6.5):
      v2              = no v3 features, no Markov
      v3_r4           = full v3 features (cat+loc+daypart+windows), no Markov
      v3_r6           = R4 + Markov fusion on Task B head
      v3_r6_lite      = no v3 features (v2 backbone) + Markov
      v4              = R4 + recency + periodicity + Markov
      v4_trim         = v4 with FEATURES_v2 drops (full proposal)
      v3_r6_arch_trim = R6 with drops only (no rec/per)
    """
    v5_set = ("v5_e1", "v5_e2", "v5_e4")
    is_r4plus = config_name in (("v3_r4", "v3_r6", "v4", "v4_trim", "v3_r6_arch_trim") + v5_set)
    use_category = is_r4plus
    use_loc = is_r4plus
    use_daypart = is_r4plus
    use_windows = is_r4plus
    use_markov = config_name in (("v3_r6", "v3_r6_lite", "v4", "v4_trim", "v3_r6_arch_trim") + v5_set)
    use_recency = config_name in ("v4", "v4_trim")
    use_periodicity = config_name in ("v4", "v4_trim")
    drops = config_name in (("v4_trim", "v3_r6_arch_trim") + v5_set)
    bg_mask_prior = config_name in v5_set
    bg_scalars = config_name in ("v5_e2", "v5_e4")
    wider = config_name == "v5_e4"
    return {
        "use_category": use_category, "use_loc": use_loc,
        "use_daypart": use_daypart, "use_windows": use_windows,
        "use_markov": use_markov,
        "use_recency": use_recency, "use_periodicity": use_periodicity,
        "drop_scene": drops, "drop_f1234": drops,
        "drop_long_windows": drops, "drop_n_trans": drops,
        "bg_mask_prior": bg_mask_prior,
        "bg_scalars": bg_scalars,
        "wider": wider,
    }


def train_user_dir(user_dir, config_name):
    """config_name in v2 / v3_r4 / v3_r6 / v3_r6_lite / v4 / v4_trim / v3_r6_arch_trim"""
    f = config_flags(config_name)
    use_category = f["use_category"]
    use_loc = f["use_loc"]
    use_daypart = f["use_daypart"]
    use_windows = f["use_windows"]
    use_markov = f["use_markov"]
    use_recency = f["use_recency"]
    use_periodicity = f["use_periodicity"]
    drop_scene = f["drop_scene"]
    drop_f1234 = f["drop_f1234"]
    drop_long_windows = f["drop_long_windows"]
    drop_n_trans = f["drop_n_trans"]
    bg_mask_prior = f.get("bg_mask_prior", False)
    bg_scalars = f.get("bg_scalars", False)
    wider = f.get("wider", False)

    train_df = pd.read_parquet(user_dir / "splits/train.parquet")
    val_df = pd.read_parquet(user_dir / "splits/val.parquet")
    test_df = pd.read_parquet(user_dir / "splits/test.parquet")
    with open(user_dir / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)

    if int(train_df["is_target_event"].astype(bool).sum()) < 50:
        return {"error": "too few train targets", "V": V}

    scaler = F.fit_dt_scaler(train_df)
    profile_stats = GF.fit_profile_stats(train_df, vocab)
    app_to_cat = build_app_to_cat(vocab)
    if use_loc:
        loc_stats = LOC.fit_location_vocab(train_df, top_k=15)
    else:
        loc_stats = None
    num_locations = len(loc_stats["vocab"]) if loc_stats is not None else 1
    stream = build_target_stream(train_df, val_df, test_df, vocab, app_to_cat)

    kw = dict(use_category=use_category, use_loc=use_loc,
              use_daypart=use_daypart, use_windows=use_windows,
              use_recency=use_recency, use_periodicity=use_periodicity,
              drop_scene=drop_scene, drop_f1234=drop_f1234,
              drop_long_windows=drop_long_windows, drop_n_trans=drop_n_trans,
              bg_mask_prior=bg_mask_prior, bg_scalars=bg_scalars)
    tr = prep_split(train_df, vocab, scaler, profile_stats, app_to_cat, loc_stats, stream, **kw)
    va = prep_split(val_df, vocab, scaler, profile_stats, app_to_cat, loc_stats, stream, **kw)
    te = prep_split(test_df, vocab, scaler, profile_stats, app_to_cat, loc_stats, stream, **kw)

    if tr["target_app"].shape[0] == 0 or va["target_app"].shape[0] == 0:
        return {"error": "empty split", "V": V}

    tr_t = to_torch_dict(tr)
    va_t = to_torch_dict(va)
    te_t = to_torch_dict(te)

    profile_dim = tr["profile"].shape[1]
    cfg = M3.ConfigV3(
        vocab_size=V,
        num_categories=CAT.NUM_CATEGORIES,
        num_locations=max(num_locations, 4),
        profile_dim=profile_dim,
        use_category=use_category,
        use_loc=use_loc,
        use_local=True,
        use_global=True,
        use_profile=True,
        use_markov_prior=use_markov,
    )
    if wider:
        cfg.d_local = 128
        cfg.app_emb_dim = 64

    log_prior_tensor = None
    if use_markov:
        log_prior_np = fit_markov_log_prior(tr["target_app"].astype(np.int64), V)
        log_prior_tensor = torch.as_tensor(log_prior_np, dtype=torch.float32)

    torch.manual_seed(SEED)
    model = SharedBackboneV3(cfg=cfg, markov_log_prior=log_prior_tensor,
                             bg_mask_prior=bg_mask_prior) if use_markov else SharedBackboneV3(
        cfg=cfg, bg_mask_prior=bg_mask_prior)

    n_params = sum(p.numel() for p in model.parameters())
    cw = torch.as_tensor(cw_inv_sqrt(tr["target_app"], V), dtype=torch.float32)

    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)

    n_train = int(tr["target_app"].shape[0])
    indices = np.arange(n_train)
    rng = np.random.default_rng(SEED)

    best_h5 = -1.0
    best_eh5 = -1.0
    best_state = None
    best_alpha = None
    patience = 0
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(indices)
        loss_sum = 0.0
        n_batches = 0
        for i in range(0, n_train, BATCH):
            idx = torch.from_numpy(indices[i:i + BATCH]).long()
            b = {k: v[idx] for k, v in tr_t.items()}
            out = model(b)
            ce = Fnn.cross_entropy(out["logits_a"], b["target_app"], weight=cw,
                                   label_smoothing=LABEL_SMOOTHING)
            wc = b["win_counts"]
            y_bin = (wc > 0).float()
            bce = Fnn.binary_cross_entropy_with_logits(out["logits_b_sig"], y_bin)
            log_rate = out["log_rate_b"].clamp(-10, 10)
            pois = (torch.exp(log_rate) - wc * log_rate).mean()
            loss = ce + 0.5 * bce + 0.2 * pois
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n_batches += 1
        m_va_a, m_va_b = evaluate(model, va_t, V)
        log_entry = {"epoch": epoch, "loss": loss_sum / max(1, n_batches),
                     "val_h1": m_va_a["hit_at_1"], "val_h5": m_va_a["hit_at_5"],
                     "val_eh5": m_va_b["event_hit_at_5"]}
        if model.alpha_markov is not None:
            log_entry["alpha_markov"] = float(model.alpha_markov.item())
        log.append(log_entry)
        if m_va_a["hit_at_5"] > best_h5:
            best_h5 = m_va_a["hit_at_5"]
            best_eh5 = m_va_b["event_hit_at_5"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_alpha = float(model.alpha_markov.item()) if model.alpha_markov is not None else None
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                break

    if best_state is not None:
        model.load_state_dict(best_state)
        if best_alpha is not None and model.alpha_markov is not None:
            with torch.no_grad():
                model.alpha_markov.data.fill_(best_alpha)

    val_a, val_b = evaluate(model, va_t, V)
    test_a, test_b = evaluate(model, te_t, V)

    return {
        "config": config_name,
        "V": V,
        "n_params": int(n_params),
        "n_train": int(tr["target_app"].shape[0]),
        "n_val": int(va["target_app"].shape[0]),
        "n_test": int(te["target_app"].shape[0]),
        "alpha_markov_final": best_alpha,
        "val_a": val_a,
        "val_b": val_b,
        "test_a": test_a,
        "test_b": test_b,
        "log": log,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--config", required=True,
                   choices=["v2", "v3_r4", "v3_r6", "v3_r6_lite",
                            "v4", "v4_trim", "v3_r6_arch_trim",
                            "v5_e1", "v5_e2", "v5_e4"])
    p.add_argument("--out-name", default=None,
                   help="filename suffix; defaults to '<config>.json'")
    p.add_argument("--max-users", type=int, default=None)
    args = p.parse_args()

    config_name = args.config
    out_name = args.out_name if args.out_name is not None else f"{config_name}.json"

    art = ROOT / "artifacts" / "multiuser"
    udirs = []
    for s in sorted(art.iterdir()):
        if not s.is_dir():
            continue
        for u in sorted(s.iterdir()):
            if (u / "vocab.json").exists():
                udirs.append((s.name, u.name, u))
    if args.max_users is not None:
        udirs = udirs[: args.max_users]

    print(f"[v2/v3 multiuser] config={config_name}  users={len(udirs)}  epochs={EPOCHS}")
    all_results = []
    t0 = time.time()
    for set_name, uid, ud in udirs:
        ts = time.time()
        try:
            r = train_user_dir(ud, config_name)
            r["set"] = set_name
            r["uid"] = uid
            r["config"] = config_name
            with open(ud / out_name, "w") as f:
                json.dump(r, f, indent=2)
            if "error" in r:
                print(f"  {set_name}/{uid}  ERROR: {r['error']}")
            else:
                print(f"  {set_name}/{uid}  V={r['V']}  test A.h1={r['test_a']['hit_at_1']:.3f}  "
                      f"test B.eh5={r['test_b']['event_hit_at_5']:.3f}  "
                      f"({time.time() - ts:.1f}s)")
            all_results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {set_name}/{uid}  EXCEPTION: {e}")

    out_agg = art / f"{config_name}_aggregate.json"
    with open(out_agg, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDONE. {len(all_results)} users; total {time.time() - t0:.1f}s")
    print(f"Aggregated to {out_agg}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
