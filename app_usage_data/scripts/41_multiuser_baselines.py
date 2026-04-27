"""Run Markov-1 baseline (closed form) per user. Task A and Task B.

Reads artifacts/multiuser/<set>/<uid>/{splits, vocab.json}.
Writes artifacts/multiuser/<set>/<uid>/baselines.json per user.
Writes artifacts/multiuser/baselines_aggregate.json + .csv.
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd


WINDOW_HORIZON_NS = 900 * 1_000_000_000


def load_user(udir: Path):
    train = pd.read_parquet(udir / "splits" / "train.parquet")
    val = pd.read_parquet(udir / "splits" / "val.parquet")
    test = pd.read_parquet(udir / "splits" / "test.parquet")
    with open(udir / "vocab.json") as f:
        vocab = json.load(f)
    return train, val, test, vocab


def app_idx(df, vocab):
    rare = vocab.get("<RARE>", 2)
    m = df["is_target_event"].astype(bool).values
    labels = df.loc[m, "app_label_clean"].fillna("<UNK>").astype(str).values
    return np.array([vocab.get(a, rare) for a in labels], dtype=np.int64)


def target_ts_int(df):
    m = df["is_target_event"].astype(bool).values
    return pd.to_datetime(df.loc[m, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)


def fit_markov(seq, V, alpha=0.5):
    counts = np.zeros((V, V), dtype=np.float64)
    for i in range(1, len(seq)):
        a = int(seq[i - 1]); b = int(seq[i])
        if a >= 3 and b >= 3:
            counts[a, b] += 1.0
    smoothed = counts + alpha
    smoothed[:, :3] = 0.0
    smoothed[:3, :] = 0.0
    rows = smoothed.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return (smoothed / rows).astype(np.float32)


def fit_mfu(seq, V):
    counts = np.bincount(seq.astype(int), minlength=V).astype(np.float32)
    counts[:3] = 0.0
    s = counts.sum()
    if s > 0:
        counts /= s
    return counts


def fit_hour_mfu(hours, apps, V, alpha=0.5):
    table = np.zeros((24, V), dtype=np.float32)
    for h, a in zip(hours, apps):
        if 0 <= int(h) < 24 and int(a) >= 3:
            table[int(h), int(a)] += 1.0
    table = table + alpha_default()
    table[:, :3] = 0.0
    rows = table.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    return (table / rows).astype(np.float32)


def alpha_default():
    return 0.5


def build_window_counts(eval_ts, eval_app, V):
    M = len(eval_ts)
    counts = np.zeros((M, V), dtype=np.float32)
    for i in range(M):
        t_end = int(eval_ts[i]) + 900_000_000_000
        k = i
        while k < M and int(eval_ts[k]) <= t_end:
            counts[i, int(eval_app[k])] += 1.0
            k += 1
    return counts


def metrics_a(scores, targets):
    n = len(targets)
    if n == 0:
        return {"hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "n": 0}
    order = np.argsort(-scores, axis=1)
    h1 = float((order[:, 0] == targets).mean())
    h5 = float((order[:, :5] == targets[:, None]).any(axis=1).mean())
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        w = np.where(order[i] == int(targets[i]))[0]
        ranks[i] = int(w[0]) if len(w) > 0 else (scores.shape[1] - 1)
    mrr = float((1.0 / (ranks + 1.0)).mean())
    return {"hit_at_1": h1, "hit_at_5": h5, "mrr": mrr, "n": int(n_safe(targets))}


def n_safe(x):
    return len(x)


def task_b_eval(scores, win_counts):
    n = len(win_counts)
    if n == 0:
        return {"event_hit_at_5": 0.0, "recall_at_5": 0.0, "coverage_at_5": 0.0, "n": 0}
    topk = np.argsort(-scores, axis=1)[:, :5]
    total_events = 0
    total_hit = 0
    recs = []
    covs = []
    for i in range(n):
        tot = int(win_counts[i].sum())
        if tot == 0:
            continue
        tset = set(int(x) for x in topk[i])
        total_events += tot
        for a in range(win_counts.shape[1]):
            if a in tset:
                total_hit += int(win_counts[i, a])
        gt = set(int(a) for a in range(win_counts.shape[1]) if win_counts[i, a] > 0)
        if gt:
            inter = len(tset & gt)
            recs.append(inter / max(1, len(gt)))
            covs.append(1.0 if gt.issubset(tset) else 0.0)
    return {
        "event_hit_at_5": total_hit / max(1, total_events),
        "recall_at_5": float(np.mean(recs)) if recs else 0.0,
        "coverage_at_5": float(np.mean(covs)) if covs else 0.0,
        "n": n,
    }


def run_one(udir: Path):
    train, val, test, vocab = load_user(udir)
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)

    def aps(df):
        m = df["is_target_event"].astype(bool).values
        labels = df.loc[m, "app_label_clean"].fillna("<UNK>").astype(str).values
        return np.array([vocab.get(a, rare) for a in labels], dtype=np.int64)

    def tss(df):
        m = df["is_target_event"].astype(bool).values
        return pd.to_datetime(df.loc[m, "event_ts"]).to_numpy().astype("datetime64[ns]").astype(np.int64)

    tr_app, va_app, te_app = aps(train), aps(val), aps(test)
    tr_ts, va_ts, te_ts = tss(train), tss(val), tss(test)
    tr_hours = pd.to_datetime(train.loc[train["is_target_event"].astype(bool).values, "event_ts"]).dt.hour.to_numpy()

    full_ts = np.concatenate([tr_ts, va_ts, te_ts])
    full_app = np.concatenate([tr_app, va_app, te_app])
    o = np.argsort(full_ts)
    full_ts = full_ts[o]
    full_app = full_app[o]

    M = fit_markov(tr_app, V)
    H = fit_hour_mfu(tr_hours, tr_app, V)
    G = fit_mfu(tr_app, V)

    def predict_a(eval_ts, eval_app, name):
        n = len(eval_app)
        scores = np.zeros((n, V), dtype=np.float32)
        ins = np.searchsorted(full_ts, eval_ts, side="left")
        if name == "MFU":
            scores[:] = G[None, :]
        elif name == "HourMFU":
            for i in range(n):
                hour = int(pd.Timestamp(eval_ts[i]).hour)
                scores[i] = H[hour]
        elif name == "MRU":
            for i in range(n):
                ip = int(ins[i])
                if ip > 0:
                    la = int(full_app[ip - 1])
                    if la >= 3:
                        sv = np.zeros(V, dtype=np.float32)
                        sv[la] = 1.0
                        scores[i] = sv
                    else:
                        scores[i] = G
                else:
                    scores[i] = G
        elif name == "Markov-1":
            for i in range(n):
                ip = int(ins[i])
                if ip > 0:
                    la = int(full_app[ip - 1])
                    scores[i] = M[la] if 3 <= la < V else G
                else:
                    scores[i] = G
        return metrics_a(scores, eval_app)

    def predict_b_markov(eval_ts, eval_app):
        n = len(eval_app)
        scores = np.zeros((n, V), dtype=np.float32)
        ins = np.searchsorted(full_ts, eval_ts, side="left")
        for i in range(n):
            ip = int(ins[i])
            if ip > 0:
                la = int(full_app[ip - 1])
                scores[i] = M[la] if 3 <= la < V else G
            else:
                scores[i] = G
        win = build_window_counts_real(eval_ts, eval_app, V)
        return task_b_eval(scores, win)

    out = {"V": V}
    for name in ("MFU", "MRU", "HourMFU", "Markov-1"):
        out[f"task_a_val_{name}"] = predict_a(va_ts, va_app, name)
        out[f"task_a_test_{name}"] = predict_a(te_ts, te_app, name)
    out["task_b_val_Markov-1"] = predict_b_markov(va_ts, va_app)
    out["task_b_test_Markov-1"] = predict_b_markov(te_ts, te_app)
    return out


def build_window_counts_real(eval_ts, eval_app, V):
    M = len(eval_ts)
    counts = np.zeros((M, V), dtype=np.float32)
    for i in range(M):
        t_end = int(eval_ts[i]) + 900 * 1_000_000_000
        k = i
        while k < M and int(eval_ts[k]) <= t_end:
            counts[i, int(eval_app[k])] += 1.0
            k += 1
    return counts


# Replace the function that referenced unbound 'counts' in build_window_counts above
def build_window_counts(eval_ts, eval_app, V):
    return build_window_counts_real(eval_ts, eval_app, V)


def main():
    art = Path("artifacts/multiuser")
    udirs = []
    for s in sorted(p for p in art.iterdir() if p.is_dir()):
        for ud in sorted(p for p in s.iterdir() if p.is_dir()):
            if (ud / "vocab.json").exists():
                udirs.append((s.name, ud.name, ud))
    print(f"[baselines] {len(udirs)} users")
    all_results = []
    t0 = time.time()
    for set_name, uid, ud in udirs:
        train, val, test, vocab = load_user(ud)
        try:
            r = run_one(ud)
            r["uid"] = uid
            r["set"] = set_name
            with open(ud / "baselines.json", "w") as f:
                json.dump(r, f, indent=2)
            all_results.append(r)
            print(f"  {set_name}/{uid}  V={r['V']}  Markov-1 a-test h1={r.get('task_a_test_Markov-1', {}).get('hit_at_1', 0):.3f}  b-test eh5={r.get('task_b_test_Markov-1', {}).get('event_hit_at_5', 0):.3f}".replace('r.get', 'r.get') if False else f"  {set_name}/{uid}  V={r['V']}  M1 a-test h1={r['task_a_test_Markov-1']['hit_at_1']:.3f}  b-test eh5={r['task_b_test_Markov-1']['event_hit_at_5']:.3f}")
        except Exception as e:
            print(f"  ERROR {set_name}/{uid}: {e}")
            import traceback
            traceback.print_exc()
    with open(art / "baselines_aggregate.json", "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDONE in {time.time()-t0:.1f}s. {len(all_results)} users processed.")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
