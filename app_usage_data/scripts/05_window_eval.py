"""Evaluate trained GRU and TGT-lite checkpoints on Task A and Task B.

Task B uses the B1 decoder (direct sigmoid-head top-K).
Outputs artifacts/results/task_a.json and task_b.json
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import data as D
from lib import features as F
from lib import models as MOD
from lib import train as TR


HISTORY_K = 16
K_LIST = [1, 3, 5, 10]
ANCHOR_STRIDE = 300
WINDOW_HORIZON = 900
HOUR_START = 6
HOUR_END = 24


def eval_task_a(scores, targets, V):
    n = int(len(targets))
    out = {"n": n}
    if n == 0:
        for kk in ("hit_at_1", "hit_at_5", "hit_at_10", "mrr", "macro_f1"):
            out[kk] = 0.0
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
        p_val = tp / (tp + fp) if (tp + fp) else 0.0
        r_val = tp / (tp + fn) if (tp + fn) else 0.0
        if p_val + r_val > 0:
            f1s.append(2 * p_val * r_val / (p_val + r_val))
        else:
            f1s.append(0.0)
    if f1s:
        out["macro_f1"] = float(np.mean(f1s))
    else:
        out["macro_f1"] = 0.0
    return out


def eval_task_b(scores, gt_ms):
    res = {}
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
        for i in range(len(gt_ms)):
            ms = gt_ms[i]
            if not ms:
                continue
            g_set = set(int(a) for a in ms.keys())
            t_set = set(int(x) for x in topk[i].tolist())
            inter = len(t_set & g_set)
            uni = len(t_set | g_set)
            p_val = inter / k
            r_val = inter / len(g_set) if len(g_set) > 0 else 0.0
            p_list.append(p_val)
            r_list.append(r_val)
            if p_val + r_val > 0:
                f1_list.append(2 * p_val * r_val / (p_val + r_val))
            else:
                f1_list.append(0.0)
            jac_list.append(inter / uni if uni > 0 else 0.0)
            if g_set.issubset(t_set):
                cov_list.append(1.0)
            else:
                cov_list.append(0.0)
            tot = 0
            hit = 0
            for key, cnt in ms.items():
                tot += cnt
                if int(key) in t_set:
                    hit += cnt
            if tot > 0:
                eh_list.append(hit / tot)
            else:
                eh_list.append(0.0)
            tot_e += tot
            hit_e += hit
        if len(p_list) > 0:
            res[f"precision@{k}"] = float(np.mean(p_list))
            res[f"recall@{k}"] = float(np.mean(r_list))
            res[f"f1@{k}"] = float(np.mean(f1_list))
            res[f"jaccard@{k}"] = float(np.mean(jac_list))
            res[f"coverage@{k}"] = float(np.mean(cov_list))
            res[f"event_hit@{k}"] = float(np.mean(eh_list))
            res[f"event_hit_micro@{k}"] = float(hit_e / max(tot_e, 1))
            res[f"n@{k}"] = int(len(p_list))
    return res


def anchor_gt_by_idx(df_split, vocab):
    anchors = D.anchor_grid(df_split, stride_sec=ANCHOR_STRIDE,
                            hour_start=HOUR_START, hour_end=HOUR_END)
    raw = D.compute_window_gt(df_split, anchors, horizon_sec=900)
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


def run_model_task_a(model, enc, history_k: int, device: str):
    hist = F.build_history_for_targets(enc, history_k=history_k)
    ds = TR.TargetWindowDataset(hist, None)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    p_list = []
    t_list = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b["history_app"], b["history_feat"], b["history_mask"], b["side_hour_fourier"])
            p_list.append(out["logits_a"].softmax(dim=-1).cpu().numpy())
            t_list.append(b["target_app"].cpu().numpy())
    S = np.concatenate(p_list, axis=0)
    T = np.concatenate(t_list, axis=0)
    return S, T


def run_model_task_b(model, enc, anchor_ts_arr, history_k: int, device: str):
    # Use build_history_for_anchors for anchor-based history
    hist = F.build_history_for_anchors(enc, anchor_ts=anchor_ts_arr, history_k=history_k)
    # wrap into a dataset-like access
    n = len(anchor_ts_arr)
    bs = 256
    model.eval()
    sig_list = []
    rate_list = []
    with torch.no_grad():
        start = 0
        while start < n:
            end = min(n, start + bs)
            ha = torch.as_tensor(hist["history_app"] if False else hist["history_app"][start:end], dtype=torch.long) if False else None
            # simpler: slice arrays and tensorize
            ha = torch.as_tensor(hist_anchor(hist, start, end, "history_app"), dtype=torch.long).to(device)
            hf = torch.as_tensor(hist_anchor(hist, start, end, "history_feat"), dtype=torch.float32).to(device)
            hm = torch.as_tensor(hist_anchor(hist, start, end, "history_mask"), dtype=torch.bool).to(device)
            sh = torch.as_tensor(hist_anchor(hist, start, end, "side_hour_fourier"), dtype=torch.float32).to(device)
            out = model(ha, hf, hm, sh)
            sig = torch.sigmoid(out["logits_b_sig"]).cpu().numpy()
            lr = out["log_rate_b"].cpu().numpy()
            sig_list.append(sig)
            rate_list.append(np.exp(np.clip(lr, -10, 10)))
            start = end
    S = np.concatenate(sig_list, axis=0)
    R = np.concatenate(rate_list, axis=0)
    return S, R


def hist_anchor(hist_dict, start, end, key):
    return hist_dict[key][start:end]


def run_model_task_b_v2(model, hist, device, bs=256):
    model.eval()
    n = hist["history_app"].shape[0]
    sig_chunks = []
    rate_chunks = []
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(n, s + bs)
            ha = torch.as_tensor(hist["history_app"][s:e], dtype=torch.long).to(device)
            hf = torch.as_tensor(hist["history_feat"][s:e], dtype=torch.float32).to(device)
            hm = torch.as_tensor(hist["history_mask"][s:e], dtype=torch.bool).to(device)
            sh = torch.as_tensor(hist["side_hour_fourier"][s:e], dtype=torch.float32).to(device)
            out = model(ha, hf, hm, sh)
            sig_chunks.append(torch.sigmoid(out["logits_b_sig"]).cpu().numpy())
            rate_chunks.append(out["log_rate_b"].cpu().numpy())
    S = np.concatenate(sig_chunks, axis=0)
    R = np.concatenate(rate_chunks, axis=0)
    return S, R


def load_checkpoint(path, V, numeric_dim, model_kind):
    cfg = MOD.ModelConfig(vocab_size=V, numeric_dim=numeric_dim)
    if model_kind == "gru":
        m = MOD.GRUModel(cfg)
    elif model_kind == "tgt":
        m = MOD.TGTLite(cfg)
    else:
        raise ValueError(model_kind)
    sd = torch.load(path, map_location="cpu", weights_only=False)["state_dict"]
    m.load_state_dict(sd)
    return m


def main():
    art = ROOT / "artifacts"
    (art / "results").mkdir(parents=True, exist_ok=True)
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)

    train_df = pd.read_parquet(art / "splits" / "train.parquet")
    val_df = pd.read_parquet(art / "splits" / "val.parquet")
    test_df = pd.read_parquet(art / "splits" / "test.parquet")
    scaler = F.fit_dt_scaler(train_df)

    splits = [("val", val_df), ("test", test_df)]

    all_a = {"V": V, "models": {"GRU": {}, "TGT": {}}}
    all_b = {"V": V, "models": {"GRU": {}, "TGT": {}}}

    for model_kind, ckpt, key_name in (
        ("gru", art / "checkpoints" / "gru.pt", "GRU"),
        ("tgt", art / "checkpoints" / "tgt.pt", "TGT"),
    ):
        print(f"[eval] loading {model_kind} checkpoint")
        model = load_checkpoint(ckpt, V, F.NUMERIC_FEAT_DIM, model_kind)
        model.eval()
        device = "cpu"

        for split_name, df in splits:
            print(f"[eval] {model_kind}/{split_name}")
            enc = F.encode_events(df, vocab, scaler["mean"], scaler["std"])

            # Task A
            S, T = run_model_task_a(model, enc, device)
            res_a = eval_task_a(S, T, V)
            all_results_a_setdefault(all_a, key_name, split_name, res_a)
            print(f"  A  hit@1={res_a['hit_at_1']:.3f}  hit@5={res_a['hit_at_5']:.3f}  mrr={res_a['mrr']:.3f}")

            # Task B (B1 sigmoid head)
            anchors_df, gt_idx = anchor_gt_by_idx(df, vocab)
            anchor_ts_arr = anchors_df["anchor_ts"].to_numpy()
            hist_anc = F.build_history_for_anchors(enc, anchor_ts_arr, history_k=HISTORY_K)
            S_b, _R_b = run_model_task_b_v2(model, hist_anc, device)
            res_b = eval_task_b(S_b, gt_idx)
            all_b["models"][key_name][split_name] = res_b
            print(f"  B  P@5={res_b.get('precision@5', 0):.3f}  R@5={res_b.get('recall@5', 0):.3f}  EH@5={res_b.get('event_hit@5', 0):.3f}  Cov@5={res_b.get('coverage@5', 0):.3f}")

    out_a = ROOT / "artifacts" / "results" / "task_a.json"
    with open(out_a, "w") as f:
        json.dump(all_a, f, indent=2)
    out_b = ROOT / "artifacts" / "results" / "task_b.json"
    with open(out_b, "w") as f:
        json.dump(all_b, f, indent=2)
    print("[eval] wrote", out_a, "and", out_b)
    return 0


def run_model_task_a(model, enc, history_k: int = 16, device: str = "cpu"):
    hist = F.build_history_for_targets(enc, history_k=history_k)
    ds = TR.TargetWindowDataset(hist, None)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    p_list = []
    t_list = []
    with torch.no_grad():
        for b in dl:
            out = model(b["history_app"], b["history_feat"], b["history_mask"], b["side_hour_fourier"])
            p_list.append(out["logits_a"].softmax(dim=-1).cpu().numpy())
            t_list.append(b["target_app"].cpu().numpy())
    return np.concatenate(p_list, axis=0), np.concatenate(t_list, axis=0)


def anchor_gt_by_idx_v2(df_split, vocab):
    return anchor_gt_by_idx(df_split, vocab)


def anchor_gt_by_idx(df_split, vocab):
    anchors = D.anchor_grid(df_split, stride_sec=ANCHOR_STRIDE,
                            hour_start=HOUR_START, hour_end=HOUR_END)
    raw = D.compute_window_gt(df_split, anchors, horizon_sec=WINDOW_HORIZON)
    keep = []
    for i, ms in enumerate(raw):
        if ms:
            keep.append(i)
    anchors_kept = anchors.iloc[keep].reset_index(drop=True)
    gt_idx = []
    for i in keep:
        m = {}
        for key, cnt in raw[i].items():
            idx = int(D.app_to_idx(key, vocab))
            if idx in m:
                m[idx] = m[idx] + int(cnt)
            else:
                m[idx] = int(cnt)
        gt_idx.append(m)
    return anchors_kept, gt_idx


def all_results_a_setdefault(all_a, model_key, split_key, res):
    if model_key not in all_a["models"]:
        all_a["models"][model_key] = {}
    all_a["models"][model_key][split_key] = res


def run_model_task_a(model, enc, device="cpu", history_k=HISTORY_K):
    hist = F.build_history_for_targets(enc, history_k=history_k)
    ds = TR.TargetWindowDataset(hist, None)
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    p_list = []
    t_list = []
    with torch.no_grad():
        for b in dl:
            ha = b["history_app"]
            hf = b["history_feat"]
            hm = b["history_mask"]
            sh = b["side_hour_fourier"]
            out = model(ha, hf, hm, sh)
            p_list.append(out["logits_a"].softmax(dim=-1).cpu().numpy())
            t_list.append(b["target_app"].cpu().numpy())
    return np.concatenate(p_list, axis=0), np.concatenate(t_list, axis=0)


if __name__ == "__main__":
    sys.exit(main() or 0)
