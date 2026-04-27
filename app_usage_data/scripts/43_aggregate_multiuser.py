"""Compute Task B metrics for all 4 baselines (MFU, MRU, HourMFU, Markov-1) per user.
Adds them to per-user baselines.json and produces a clean aggregate JSON.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
WIN_NS = 900 * 1_000_000_000


def app_idx(df, vocab):
    rare = vocab.get("<RARE>", 2)
    m = df["is_target_event"].astype(bool).values
    labels = df.loc[m, "app_label_clean"].fillna("<UNK>").astype(str).values
    return np.array([vocab.get(a, rare) for a in labels], dtype=np.int64)


def target_ts_int(df):
    m = df["is_target_event"].astype(bool).values
    return pd.to_datetime(df.loc[m, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)


def fit_mfu(seq, V):
    counts = np.bincount(seq.astype(int), minlength=V).astype(np.float32)
    counts[:3] = 0.0
    s = counts.sum()
    return counts / s if s > 0 else counts


def fit_hourmfu(hours, apps, V, alpha=0.5):
    table = np.zeros((24, V), dtype=np.float32)
    for h, a in zip(hours, apps):
        if 0 <= int(h) < 24 and int(a) >= 3:
            table[int(h), int(a)] += 1.0
    table = table + alpha
    table[:, :3] = 0.0
    rs = table.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (table / rs).astype(np.float32)


def fit_markov(seq, V, alpha=0.5):
    counts = np.zeros((V, V), dtype=np.float64)
    for i in range(1, len(seq)):
        a, b = int(seq[i - 1]), int(seq[i])
        if a >= 3 and b >= 3:
            counts[a, b] += 1.0
    sm = counts + alpha
    sm[:, :3] = 0.0
    sm[:3, :] = 0.0
    rs = sm.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (sm / rs).astype(np.float32)


def build_win(eval_ts, eval_app, V):
    M = len(eval_ts)
    counts = np.zeros((M, V), dtype=np.float32)
    for i in range(M):
        t_end = int(eval_ts[i]) + 900_000_000_000
        k = i
        while k < M and int(eval_ts[k]) <= t_end:
            counts[i, int(eval_app[k])] += 1.0
            k += 1
    return counts


def task_b_metrics(scores, win):
    n = len(win)
    if n == 0:
        return {"event_hit_at_5": 0.0, "recall_at_5": 0.0, "coverage_at_5": 0.0, "n": 0}
    topk = np.argsort(-scores_safe(scores, n), axis=1)[:, :5] if False else np.argsort(-scores, axis=1)[:, :5]
    total_events = 0
    total_hit = 0
    recs, covs = [], []
    for i in range(n):
        tot = int(win[i].sum())
        if tot == 0:
            continue
        tset = set(int(x) for x in topk[i])
        total_events += tot
        for a in range(win.shape[1]):
            if a in tset:
                total_hit += int(win[i, a])
        gt = set(int(a) for a in range(win.shape[1]) if win[i, a] > 0)
        if gt:
            recs.append(len(tset & gt) / max(1, len(gt)))
            covs.append(1.0 if gt.issubset(tset) else 0.0)
    return {
        "event_hit_at_5": total_hit / max(1, total_events),
        "recall_at_5": float(np.mean(recs)) if recs else 0.0,
        "coverage_at_5": float(np.mean(covs)) if covs else 0.0,
        "n": n,
    }


def main():
    art = ROOT / "artifacts" / "multiuser"
    udirs = []
    for s in sorted(art.iterdir()):
        if s.is_dir():
            for u in sorted(s.iterdir()):
                if (u / "vocab.json").exists():
                    udirs.append((s.name, u.name, u))
    print(f"[task-B baselines] {len(udirs)} users")
    out = []
    for sname, uid, ud in udirs:
        train = pd.read_parquet(ud / "splits/train.parquet")
        val = pd.read_parquet(ud / "splits/val.parquet")
        test = pd.read_parquet(ud / "splits/test.parquet")
        with open(ud / "vocab.json") as f:
            vocab = json.load(f)
        V = len(vocab)

        tr_app = app_idx(train, vocab)
        te_app = app_idx(test, vocab)
        tr_ts = target_ts_int(train)
        te_ts = target_ts_int(test)
        tr_hours = pd.to_datetime(train.loc[train["is_target_event"].astype(bool).values, "event_ts"]).dt.hour.to_numpy()

        full_ts = np.concatenate([tr_ts, target_ts_int(val), te_ts])
        full_app = np.concatenate([tr_app, app_idx(val, vocab), te_app])
        ord_ = np.argsort(full_ts)
        full_ts = full_ts[ord_]
        full_app = full_app[ord_]

        mfu_dist = fit_mfu(tr_app, V)
        hmfu_table = fit_hourmfu(tr_hours, tr_app, V)
        markov_table = fit_markov(tr_app, V)

        # Build test window counts (Task B ground truth)
        win_test = build_win(te_ts, te_app, V)
        n_test = len(te_ts)

        # MFU: same prediction for every anchor
        scores_mfu = np.broadcast_to(mfu_dist, (n_test, V)).copy()

        # MRU: top-1 = last app, others zero
        scores_mru = np.zeros((n_test, V), dtype=np.float32)
        ins = np.searchsorted(full_ts, te_ts, side="left")
        for i in range(n_test):
            ip = int(ins[i])
            if ip > 0:
                la = int(full_app[ip - 1])
                if la >= 3:
                    scores_mru[i, la] = 1.0
        # tie-break with mfu
        scores_mru = scores_mru + 1e-6 * mfu_dist[None, :]

        # HourMFU
        hours_eval = np.array([int(pd.Timestamp(t).hour) for t in te_ts], dtype=np.int64)
        scores_hmfu = hmfu_table[hours_eval]

        # Markov-1: P(app | last_app)
        scores_mk = np.zeros((n_test, V), dtype=np.float32)
        for i in range(n_test):
            ip = int(np.searchsorted(full_ts, te_ts[i], side="left"))
            if ip > 0 and 3 <= int(full_app[ip - 1]) < V:
                scores_mk[i] = markov_table[int(full_app[ip - 1])]
            else:
                scores_mk[i] = mfu_dist

        result = {
            "set": sname, "uid": uid, "V": V,
            "n_test": n_test,
            "task_b_test_MFU": task_b_metrics(scores_mfu, win_test),
            "task_b_test_MRU": task_b_metrics(scores_mru, win_test),
            "task_b_test_HourMFU": task_b_metrics(scores_hmfu, win_test),
            "task_b_test_Markov-1": task_b_metrics(scores_mk, win_test),
        }
        with open(ud / "baselines_b.json", "w") as f:
            json.dump(result_dict_safe(result), f, indent=2)
        out.append(result)
        print(f"  {sname}/{uid}: MFU eh@5={result['task_b_test_MFU']['event_hit_at_5']:.3f}  HourMFU={result['task_b_test_HourMFU']['event_hit_at_5']:.3f}  Markov={result['task_b_test_Markov-1']['event_hit_at_5']:.3f}")

    with open(art / "task_b_baselines_aggregate.json", "w") as f:
        json.dump(out, f, indent=2)
    print(f"\nDONE")


def result_dict_safe(d):
    return d


def markov_get(table, last_app, V):
    return table[last_app]


def scores_safe(scores, n):
    return scores


def target_ts_int_to_pd(ts):
    return pd.to_datetime(ts)


if __name__ == "__main__":
    sys.exit(main() or 0)
