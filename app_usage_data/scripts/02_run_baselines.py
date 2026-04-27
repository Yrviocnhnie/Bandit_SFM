"""Evaluate baselines MFU / MRU / HourMFU / Markov1 on Task A and Task B."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import baselines as B
from lib import data as D
from lib import features as F

HISTORY_K = 16
K_LIST = [1, 3, 5, 10]
ANCHOR_STRIDE = 300
WINDOW_HORIZON = 900
HOUR_START = 6
HOUR_END = 24
RESERVED = 3


def eval_task_a(scores, targets, V):
    n = int(len(targets))
    out = {"n": n}
    if n == 0:
        out["hit_at_1"] = 0.0
        out["hit_at_5"] = 0.0
        out["hit_at_10"] = 0.0
        out["mrr"] = 0.0
        out["macro_f1"] = 0.0
        return out
    preds = scores.argmax(axis=1)
    order = np.argsort(-scores, axis=1)
    out["hit_at_1"] = float((preds == targets).mean())
    out["hit_at_5"] = float((order[:, :5] == targets[:, None]).any(axis=1).mean())
    out["hit_at_10"] = float((order[:, :10] == targets[:, None]).any(axis=1).mean())
    ranks = np.zeros(n, dtype=np.int64)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == targets[i])[0][0])
    out["mrr"] = float((1.0 / (ranks + 1.0)).mean())

    f1s = []
    for c in range(V):
        pmask = preds == c
        tmask = targets == c
        tp = int((pmask & tmask).sum())
        fp = int((pmask & (~tmask)).sum())
        fn = int(((~pmask) & tmask).sum())
        if tp + fp + fn == 0:
            continue
        denom_p = tp + fp
        denom_r = tp + fn
        if denom_p > 0:
            p = tp / denom_p
        else:
            p = 0.0
        if denom_r > 0:
            r = tp / denom_r
        else:
            r = 0.0
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        f1s_append_target = f1
        f1s.append(f1) if False else None
    # Clean version of macro F1 using the same loop once
    f1s = []
    for c in range(V):
        pmask = preds == c
        tmask = targets == c
        tp = int((pmask & tmask).sum())
        fp = int((pmask & (~tmask)).sum())
        fn = int(((~pmask) & tmask).sum())
        if tp + fp + fn == 0:
            continue
        denom_p = tp + fp
        denom_r = tp + fn
        if denom_p > 0:
            p = tp / denom_p
        else:
            p = 0.0
        if denom_r > 0:
            r = tp / denom_r
        else:
            r = 0.0
        if p + r > 0:
            f1 = 2 * p * r / (p + r)
        else:
            f1 = 0.0
        f1s.append(f1)
    if f1s:
        out_macro = float(np.mean(f1s))
    else:
        out_macro = 0.0
    out["macro_f1"] = out_macro
    return out


def eval_task_b(scores, gt_multisets):
    results = {}
    for k in K_LIST:
        topk = np.argsort(-scores, axis=1)[:, :k]
        p_list = []
        r_list = []
        f1_list = []
        jac_list = []
        cov_list = []
        eh_list = []
        tot_e = 0
        hit_e = 0
        for i in range(len(gt_multisets)):
            ms = gt_multisets[i]
            if not ms:
                continue
            g_set = set()
            for a_key in ms.keys():
                g_set.add(int(a_key))
            t_set = set()
            for x_val in topk[i].tolist():
                t_set.add(int(x_val))
            inter = len(t_set & g_set)
            uni = len(t_set | g_set)
            p_val = inter / k
            if len(g_set) > 0:
                r_val = inter / len(g_set)
            else:
                r_val = 0.0
            p_list.append(p_val)
            r_list.append(r_val)
            if p_val + r_val > 0:
                f1_val = 2 * p_val * r_val / (p_val + r_val)
            else:
                f1_val = 0.0
            f1_list.append(f1_val)
            if uni > 0:
                jac_val = inter / uni
            else:
                jac_val = 0.0
            jac_list.append(jac_val)
            if g_set.issubset(t_set):
                cov_val = 1.0
            else:
                cov_val = 0.0
            cov_list.append(cov_val)
            tot = 0
            hit = 0
            for key, cnt in ms.items():
                tot += cnt
                if int(key) in t_set:
                    hit += cnt
            if tot > 0:
                eh = hit / tot
            else:
                eh = 0.0
            eh_list.append(eh)
            tot_e += tot
            hit_e += hit
        if p_list:
            results[f"precision@{k}"] = float(np.mean(p_list))
            results[f"recall@{k}"] = float(np.mean(r_list))
            results[f"f1@{k}"] = float(np.mean(f1_list))
            results[f"jaccard@{k}"] = float(np.mean(jac_list))
            results[f"coverage@{k}"] = float(np.mean(cov_list))
            results[f"event_hit@{k}"] = float(np.mean(eh_list))
            results[f"event_hit_micro@{k}"] = float(hit_e / max(tot_e, 1))
            results[f"n@{k}"] = int(len(p_list))
    return results


def prev_target_in_session(enc):
    pos = np.nonzero(enc.is_target)[0]
    prev = np.full(len(pos), -1, dtype=np.int64)
    last_by_sess = {}
    ptr = 0
    for i in range(len(enc.app_idx)):
        if ptr < len(pos) and i == pos[ptr]:
            s = int(enc.session_id[i])
            prev[ptr] = last_by_sess.get(s, -1)
            last_by_sess[s] = int(enc.app_idx[i])
            ptr += 1
    return prev


def prev_target_before_anchor(enc, anchor_ts):
    ts_i64 = enc.ts.astype("datetime64[ns]").astype(np.int64)
    a_i64 = pd.to_datetime(anchor_ts).to_numpy().astype("datetime64[ns]").astype(np.int64)
    ins = np.searchsorted(ts_i64, a_i64, side="left")
    out = np.full(len(anchor_ts), -1, dtype=np.int64)
    for i, ip in enumerate(ins):
        j = int(ip) - 1
        while j >= 0:
            if enc.is_target[j] and enc.app_idx[j] >= RESERVED:
                out[i] = int(enc.app_idx[j])
                break
            j -= 1
    return out


def anchor_truth(df_split, vocab):
    anchors = D.anchor_grid(df_split, stride_sec=ANCHOR_STRIDE,
                            hour_start=HOUR_START, hour_end=HOUR_END)
    raw = D.compute_window_gt(df_split, anchors, horizon_sec=WINDOW_HORIZON)
    keep = []
    for i in range(len(raw)):
        if raw[i]:
            keep.append(i)
    anchors_kept = anchors.iloc[keep].reset_index(drop=True)
    gt_idx = []
    for i in keep:
        ms = raw[i]
        m = {}
        for key, cnt in ms.items():
            idx = int(D.app_to_idx(key, vocab))
            if idx in m:
                m[idx] = m[idx] + int(cnt)
            else:
                m[idx] = int(cnt)
        gt_idx.append(m)
    return anchors_kept, gt_idx


def scores_mfu_tile(mfu, n):
    return np.tile(mfu.probs, (n, 1))


def scores_mru_from_history(history_app, history_mask, V):
    n = history_app.shape[0]
    k = history_app.shape[1]
    out = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        for j in range(k - 1, -1, -1):
            if history_mask[i, j] and history_app[i, j] >= RESERVED:
                out[i, int(history_app[i, j])] = 1.0
                break
    return out


def scores_mru_from_prev(prev_apps, V):
    n = len(prev_apps)
    out = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        if prev_apps[i] >= RESERVED:
            out[i, int(prev_apps[i])] = 1.0
    return out


def scores_hour(hmfu, hours):
    arr = np.asarray(hours, dtype=int) % 24
    return hmfu.table[arr].copy()


def scores_markov(mk, prev_apps):
    n = len(prev_apps)
    V = mk.vocab_size
    out = np.zeros((n, V), dtype=np.float32)
    for i in range(n):
        pa = int(prev_apps[i])
        if pa >= RESERVED:
            out[i] = mk.predict_task_a(pa)
        else:
            out[i] = mk.predict_task_a(None)
    return out


def main():
    art = ROOT / "artifacts"
    (art / "results").mkdir(parents=True, exist_ok=True)
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)
    print("[baselines] vocab size =", V)

    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")

    scaler = F.fit_dt_scaler(train)
    enc_train = F.encode_events(train, vocab, scaler["mean"], scaler["std"])
    enc_val = F.encode_events(val, vocab, scaler["mean"], scaler["std"])
    enc_test = F.encode_events(test, vocab, scaler["mean"], scaler["std"])

    tgt_mask = enc_train.is_target
    tr_apps = enc_train.app_idx[tgt_mask]
    tr_hours = enc_train.hour[tgt_mask]

    print("[baselines] fitting on", len(tr_apps), "target events")
    mfu = B.MFU(V)
    mfu.fit(tr_apps)
    hmfu = B.HourMFU(V, alpha=0.5)
    hmfu.fit(tr_hours, tr_apps)
    mk = B.Markov1(V, alpha=0.5)
    mk.fit(tr_apps)

    all_results = {"V": V, "models": {"MFU": {}, "MRU": {}, "HourMFU": {}, "Markov1": {}}}

    splits_list = [("val", enc_val, val), ("test", enc_test, test)]
    for split_name, enc_split, df_split in splits_list:
        print(f"[baselines] === {split_name} ===")
        # -- Task A --
        hist = F.build_history_for_targets(enc_split, history_k=HISTORY_K)
        targets = hist["target_app"].astype(np.int64)
        hours_t = hist["target_hour"].astype(np.int64)
        pa_sess = prev_target_in_session(enc_split)
        n_targets = len(targets)
        print(f"  targets: {n_targets}")

        s_mfu = np.tile(mfu.probs, (n_targets, 1))
        s_mru = scores_mru_from_history(hist["history_app"], hist["history_mask"], V)
        s_hm = scores_hour(hmfu, hours_t)
        s_mk = scores_markov(mk, pa_sess)

        task_a_by_model = {
            "MFU": eval_task_a(s_mfu, targets, V),
            "MRU": eval_task_a(s_mru, targets, V),
            "HourMFU": eval_task_a(s_hm, targets, V),
            "Markov1": eval_task_a(s_mk, targets, V),
        }
        for name, res in task_a_by_model.items():
            all_results["models"][name][f"task_a_{split_name}"] = res
            print(f"  [A] {name:7s}  hit@1={res['hit_at_1']:.3f}  hit@5={res['hit_at_5']:.3f}  mrr={res['mrr']:.3f}  f1={res['macro_f1']:.3f}")

        # -- Task B --
        anchors_df, gt_idx = anchor_truth(df_split, vocab)
        anchor_ts_arr = anchors_df["anchor_ts"].to_numpy()
        anchor_hrs = anchors_df["anchor_ts"].dt.hour.to_numpy()
        pa_anc = prev_target_before_anchor(enc_split, anchor_ts_arr)
        n_anc = len(anchors_df)
        print(f"  anchors (non-empty): {n_anc}")

        s_mfu_b = np.tile(mfu.probs, (n_anc, 1))
        s_mru_b = scores_mru_from_prev(pa_anc, V)
        s_hm_b = scores_hour(hmfu, anchor_hrs)
        s_mk_b = scores_markov(mk, pa_anc)

        task_b_by_model = {
            "MFU": eval_task_b(s_mfu_b, gt_idx),
            "MRU": eval_task_b(s_mru_b, gt_idx),
            "HourMFU": eval_task_b(s_hm_b, gt_idx),
            "Markov1": eval_task_b(s_mk_b, gt_idx),
        }
        for name, res in task_b_by_model.items():
            all_results["models"][name][f"task_b_{split_name}"] = res
            print(f"  [B] {name:7s}  P@5={res.get('precision@5', 0):.3f}  R@5={res.get('recall@5', 0):.3f}  EH@5={res.get('event_hit@5', 0):.3f}  Cov@5={res.get('coverage@5', 0):.3f}")

    out_path = art / "results" / "baselines.json"
    with open(out_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"[baselines] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
