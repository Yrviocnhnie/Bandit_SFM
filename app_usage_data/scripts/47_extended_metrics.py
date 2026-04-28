"""Recompute Task B baselines per user with extended metrics.

Baselines: MFU, MRU-5, HourMFU, Markov-1.

Per baseline on the test set:
    EH@1, EH@3, EH@5            micro event-hit (count-weighted)
    EH@dyn                      EH at K = max(1, |G(anchor)|)
    Recall@5, Recall@dyn        macro mean over non-empty anchors
    Coverage@5                  macro mean over non-empty anchors
    n_anchors, avg_g_size

Inputs:  artifacts/multiuser/<set>/<uid>/{splits/*.parquet, vocab.json}
Output:  artifacts/multiuser/task_b_baselines_extended.json
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
ART = ROOT / "artifacts" / "multiuser"
WIN_NS = 900 * 1_000_000_000
SPECIAL = 3


# ----- Data helpers ---------------------------------------------------------

def filter_target_rows(df):
    mask = df["is_target_event"].astype(bool).values
    return df.loc[mask]


def labels_to_idx(labels, vocab):
    rare = vocab.get("<RARE>", 2)
    return np.array([vocab.get(str(x), rare) for x in labels], dtype=np.int64)


def event_ts_ns(series):
    return pd.to_datetime(series).to_numpy().astype("datetime64[ns]").astype(np.int64)


def df_to_targets(df, vocab):
    sub = filter_target_rows(df)
    labels = sub["app_label_clean"].fillna("<UNK>").astype(str).values
    apps = labels_to_idx(labels, vocab)
    ts = event_ts_ns(sub["event_ts"])
    return apps, ts


# ----- Baseline fitters -----------------------------------------------------

def fit_mfu(seq, V):
    counts = np.bincount(seq.astype(int), minlength=V).astype(np.float32)
    counts[:SPECIAL] = 0.0
    s = counts.sum()
    if s > 0:
        return counts / s
    return counts


def fit_hourmfu(hours, apps, V, alpha=0.5):
    table = np.zeros((24, V), dtype=np.float64)
    h_list = [int(h) for h in hours.tolist()]
    a_list = [int(a) for a in apps.tolist()]
    for h, a in zip(h_list, a_list):
        if 0 <= h < 24 and a >= SPECIAL:
            table[h, a] += 1.0
    table = table + alpha
    table[:, :SPECIAL] = 0.0
    rs = table.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (table / rs).astype(np.float32)


def fit_markov(seq, V, alpha=0.5):
    counts = np.zeros((V, V), dtype=np.float64)
    s_list = [int(x) for x in seq.tolist()]
    for i in range(1, len(s_list)):
        a = s_list[i - 1]
        b = s_list[i]
        if a >= SPECIAL and b >= SPECIAL:
            counts[a, b] += 1.0
    counts = counts + alpha
    counts[:, :SPECIAL] = 0.0
    counts[:SPECIAL, :] = 0.0
    rs = counts.sum(axis=1, keepdims=True)
    rs[rs == 0] = 1.0
    return (counts / rs).astype(np.float32)


def build_win(eval_ts, eval_app, V):
    M = len(eval_ts)
    win = np.zeros((M, V), dtype=np.float32)
    if M == 0:
        return win
    ts = eval_ts.astype(np.int64)
    ap = eval_app.astype(np.int64)
    for i in range(M):
        t_end = int(ts[i]) + WIN_NS
        k = i
        while k < M and int(ts[k]) <= t_end:
            a = int(ap[k])
            if 0 <= a < V:
                win[i, a] += 1.0
            k += 1
    return win


# ----- Extended metrics -----------------------------------------------------

def metrics(scores, win):
    """Compute extended Task B metrics for one (scores, win) pair.

    scores: (M, V) array of model scores (higher = more likely).
    win:    (M, V) integer counts of events per app in the 15-min window.

    Returns dict with EH@{1,3,5,dyn}, Recall@{5,dyn}, Cov@5, n_anchors, avg_g_size.
    """
    base = {
        "event_hit_at_1": 0.0,
        "event_hit_at_3": 0.0,
        "event_hit_at_5": 0.0,
        "event_hit_at_dyn": 0.0,
        "recall_at_5": 0.0,
        "recall_at_dyn": 0.0,
        "coverage_at_5": 0.0,
        "n_anchors": 0,
        "avg_g_size": 0.0,
    }
    if win.shape[0] == 0 or win.shape[1] == 0:
        return base

    M = win.shape[0]
    V = win.shape[1]
    order = np.argsort(-scores, axis=1)

    total_events = 0
    hit1 = 0
    hit3 = 0
    hit5 = 0
    hit_dyn = 0
    rec5_list = []
    rec_dyn_list = []
    cov5_list = []
    g_sizes = []
    n_used = 0

    for i in range(M):
        ws = win[i]
        tot_i = int(ws.sum())
        if tot_i == 0:
            continue

        gt_set = set()
        for a in range(V):
            if ws[a] > 0:
                gt_set.add(int(a))
        gsize = len(gt_set)
        if gsize == 0:
            continue

        n_used += 1
        g_sizes.append(gsize)
        total_events += tot_i

        ord_i = order[i]
        top1 = ord_i[:1].tolist()
        top3 = ord_i[:3].tolist()
        top5 = ord_i[:5].tolist()
        if gsize >= 1:
            K_dyn = gsize
        else:
            K_dyn = 1
        top_dyn = ord_i[:K_dyn].tolist()

        for a in top1:
            hit1 += int(ws[int(a)])
        for a in top3:
            hit3 += int(ws[int(a)])
        for a in top5:
            hit5 += int(ws[int(a)])
        for a in top_dyn:
            hit_dyn += int(ws[int(a)])

        top5_set = set(int(a) for a in top5)
        top_dyn_set = set(int(a) for a in top_dyn)

        denom = float(gsize)
        rec5_list.append(len(top5_set & gt_set) / denom)
        rec_dyn_list.append(len(top_dyn_set & gt_set) / denom)
        if gt_set.issubset(top5_set):
            cov5_list.append(1.0)
        else:
            cov5_list.append(0.0)

    if n_used == 0:
        return base

    if total_events > 0:
        denom_events = total_events
    else:
        denom_events = 1
    return {
        "event_hit_at_1": float(hit1) / float(denom_events),
        "event_hit_at_3": float(hit3) / float(denom_events),
        "event_hit_at_5": float(hit5) / float(denom_events),
        "event_hit_at_dyn": float(hit_dyn) / float(denom_events),
        "recall_at_5": float(np.mean(rec5_list)),
        "recall_at_dyn": float(np.mean(rec_dyn_list)),
        "coverage_at_5": float(np.mean(cov5_list)),
        "n_anchors": int(n_used),
        "avg_g_size": float(np.mean(g_sizes)),
    }


# ----- Summary --------------------------------------------------------------

def summarise(out):
    print("\n=== Summary across users ===")
    cohorts = {"top2000": [], "M_beta_Top30": []}
    for r in out:
        if r["set"] in cohorts:
            cohorts[r["set"]].append(r)

    rows_all = list(out)
    rows_top2000 = cohorts["top2000"]
    rows_mbeta = cohorts["M_beta_Top30"]

    def cohort_block(rows, label):
        if len(rows) == 0:
            print(f"\n[{label}] (no users)")
            return
        print(f"\n[{label}] n_users={len(rows)}")
        for base_key in ["task_b_test_MFU", "task_b_test_MRU", "task_b_test_HourMFU", "task_b_test_Markov1"]:
            eh5 = [r[base_key]["event_hit_at_5"] for r in rows]
            eh_dyn = [r[base_key]["event_hit_at_dyn"] for r in rows]
            avg_g = [r[base_key]["avg_g_size"] for r in rows]
            print(f"  {base_key:25s} mean EH@5={np.mean(eh5):.4f} median={np.median(eh5):.4f}  "
                  f"mean EH@dyn={np.mean(eh_dyn):.4f} median={np.median(eh_dyn):.4f}  "
                  f"mean avg_g_size={np.mean(avg_g):.3f}")

    cohort_block(rows_all, "ALL")
    cohort_block(rows_top2000, "top2000")
    cohort_block(rows_mbeta, "M_beta_Top30")


# ----- Per-user processing --------------------------------------------------

def discover_users():
    out = []
    for set_dir in sorted(ART.iterdir()):
        if not set_dir.is_dir():
            continue
        for user_dir in sorted(set_dir.iterdir()):
            if not user_dir.is_dir():
                continue
            if (user_dir / "vocab.json").exists():
                out.append((set_dir.name, user_dir.name, user_dir))
    return out


def process_user(sname, uid, ud):
    with open(ud / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)

    train = pd.read_parquet(ud / "splits/train.parquet")
    val = pd.read_parquet(ud / "splits/val.parquet")
    test = pd.read_parquet(ud / "splits/test.parquet")

    train_apps, train_ts = df_to_targets(train, vocab)
    val_apps, val_ts = df_to_targets(val, vocab)
    test_apps, test_ts = df_to_targets(test, vocab)

    train_target_df = filter_target_rows(train)
    train_hours = pd.to_datetime(train_target_df["event_ts"]).dt.hour.to_numpy()

    # Merged train+val+test target stream (sorted by timestamp) for MRU/Markov lookup.
    all_ts = np.concatenate([train_ts, val_ts, test_ts])
    all_app = np.concatenate([train_apps, val_apps, test_apps])
    ord_full = np.argsort(all_ts, kind="stable")
    full_ts = all_ts[ord_full]
    full_app = all_app[ord_full]

    # Fit baselines on train.
    mfu_dist = fit_mfu(train_apps, V)
    hour_table = fit_hourmfu(train_hours, train_apps, V)
    markov_table = fit_markov(train_apps, V)

    # Test window matrix (Task B GT).
    win_test = build_win(test_ts, test_apps, V)
    n_test = len(test_ts)

    # MFU: same dist for every anchor.
    scores_mfu = np.broadcast_to(mfu_dist, (n_test, V)).copy()

    # MRU-5: top-5 most recent distinct apps before each anchor.
    scores_mru = np.zeros((n_test, V), dtype=np.float32)
    ins = np.searchsorted(full_ts, test_ts, side="left")
    K_MRU = 5
    for i in range(n_test):
        seen = []
        j = int(ins[i]) - 1
        while j >= 0 and len(seen) < K_MRU:
            a = int(full_app[j])
            if a >= SPECIAL and a not in seen:
                seen.append(a)
            j -= 1
        rank = 0
        for a in seen:
            scores_mru[i, a] = float(K_MRU - rank)
            rank += 1

    # HourMFU: row indexed by anchor's hour-of-day.
    test_hours = pd.to_datetime(pd.Series(test_ts)).dt.hour.to_numpy().astype(np.int64)
    test_hours = np.clip(test_hours, 0, 23)
    scores_hmfu = hour_table[test_hours]

    # Markov-1: row indexed by last_app strictly before anchor.
    scores_mk = np.zeros((n_test, V), dtype=np.float32)
    for i in range(n_test):
        ip = int(np.searchsorted(full_ts, test_ts[i], side="left"))
        last_app = -1
        if ip > 0:
            last_app = int(full_app[ip - 1])
        if last_app >= SPECIAL and last_app < V:
            scores_mk[i] = markov_table[last_app]
        else:
            scores_mk[i] = mfu_dist

    return {
        "scores_mfu": scores_mfu,
        "scores_mru": scores_mru,
        "scores_hmfu": scores_hmfu,
        "scores_mk": scores_mk,
        "win_test": win_test,
        "V": V,
        "n_test": n_test,
    }


def main():
    udirs = discover_users()
    print(f"Found {len(udirs)} users", flush=True)

    out = []
    for sname, uid, ud in udirs:
        bundle = process_user(sname, uid, ud)
        res = {
            "set": sname,
            "uid": uid,
            "V": bundle["V"],
            "n_test": bundle["n_test"],
            "task_b_test_MFU": metrics(bundle["scores_mfu"], bundle["win_test"]),
            "task_b_test_MRU": metrics(bundle["scores_mru"], bundle["win_test"]),
            "task_b_test_HourMFU": metrics(bundle["scores_hmfu"], bundle["win_test"]),
            "task_b_test_Markov1": metrics(bundle["scores_mk"], bundle["win_test"]),
        }
        out.append(res)
        mfu_eh5 = res["task_b_test_MFU"]["event_hit_at_5"]
        mru_eh5 = res["task_b_test_MRU"]["event_hit_at_5"]
        hmf_eh5 = res["task_b_test_HourMFU"]["event_hit_at_5"]
        mk_eh5 = res["task_b_test_Markov1"]["event_hit_at_5"]
        print(f"  {sname}/{uid}: n_test={bundle['n_test']} MFU={mfu_eh5:.3f} MRU={mru_eh5:.3f} "
              f"HourMFU={hmf_eh5:.3f} Markov={mk_eh5:.3f}", flush=True)

    out_path = ART / "task_b_baselines_extended.json"
    with open(out_path, "w") as fh:
        json.dump(out, fh, indent=2)
    print(f"\nWrote {out_path}")
    summarise(out)
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)

