"""Evaluate v2 Task A + Task B models on val and test.

Loads task_a_v2.pt and task_b_v2.pt, evaluates:
- Task A: per-target Hit@1/5/10, MRR, macro_f1
- Task B: 5-min anchor grid EventHit@K, P@K, R@K, coverage@K

Writes artifacts/results/{task_a_v2.json, task_b_v2.json}.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import data as D
from lib import features as F
from lib import train as T
from lib.v2 import global_features as GF
from lib.v2 import models_v2 as M
from lib.v2 import datasets_v2 as DSETS


K_LIST = [1, 3, 5, 10]
ANCHOR_STRIDE = 300
WINDOW_HORIZON = 900
HOUR_START = 6
HOUR_END = 24


def load_model(kind: str, V: int, ckpt_path: Path):
    sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg_dict = sd["cfg"]
    cfg = M.ConfigV2(
        vocab_size=cfg_dict["vocab_size"],
        numeric_dim=cfg_dict["numeric_dim"],
        profile_dim=cfg_dict["profile_dim"],
        d_local=cfg_dict["d_local"],
        d_global=cfg_dict["d_global"],
        d_profile=cfg_dict["d_profile"],
        d_fused=cfg_dict["d_fused"],
        app_emb_dim=cfg_dict["app_emb_dim"],
        n_heads=cfg_dict["n_heads"],
        n_layers=cfg_dict["n_layers"],
        dropout=cfg_dict["dropout"],
        use_local=cfg_dict["use_local"],
        use_global=cfg_dict["use_global"],
        use_profile=cfg_dict["use_profile"],
    )
    if kind == "a":
        mdl = M.TaskAModel(cfg)
    else:
        mdl = M.TaskBModel(cfg)
    mdl.load_state_dict(sd["state_dict"])
    mdl.eval()
    return mdl, cfg


def build_target_tensors(enc, stats, tgt_ts_full, tgt_apps_full):
    short_hist = F.build_history_for_targets(enc, history_k=16)
    long_hist = GF.build_long_history_for_targets(enc, k_long=64)
    profile = GF.build_profile_for_targets(enc, stats, tgt_ts_full, tgt_apps_full)
    return {
        "history_app": short_hist["history_app"],
        "history_feat": short_hist["history_feat"],
        "history_mask": short_hist["history_mask"],
        "long_app": long_hist["long_app"],
        "long_feat": long_hist["long_feat"],
        "long_mask": long_hist["long_mask"],
        "long_dt_bin": long_hist["long_dt_bin"],
        "profile": profile,
        "side_hour_fourier": short_hist["target_hour_fourier"],
        "target_app": short_hist["target_app"],
    }


def build_anchor_tensors(enc, stats, anchor_ts, tgt_ts_full, tgt_apps_full):
    import pandas as pd
    import numpy as np
    # short-history: use anchor-based builder
    short = F.build_history_for_anchors(enc, anchor_ts=anchor_ts, history_k=16)
    long = GF.build_long_history_for_anchors(enc, anchor_ts=anchor_ts, k_long=64)

    # profile at anchor: session_pos=0, start=0 (grid anchor)
    anchor_hour = np.array([pd.Timestamp(a).hour for a in anchor_ts], dtype=np.int64)
    anchor_wd = np.array([pd.Timestamp(a).weekday() for a in anchor_ts], dtype=np.int64)
    M_ = len(anchor_ts)
    anchor_ns = pd.to_datetime(anchor_ts).to_numpy().astype("datetime64[ns]").astype(np.int64)
    # Use internal _assemble_profile directly
    profile = GF._assemble_profile(
        anchor_hour=anchor_hour,
        anchor_wd=anchor_wd,
        anchor_ns=anchor_ns,
        target_ts_full=tgt_ts_full,
        target_app_full=tgt_apps_full,
        session_pos_val=np.zeros(M_, dtype=np.float32),
        start_val=np.zeros(M_, dtype=np.float32),
        stats=stats,
    )
    return {
        "history_app": short["history_app"],
        "history_feat": short["history_feat"],
        "history_mask": short["history_mask"],
        "long_app": long["long_app"],
        "long_feat": long["long_feat"],
        "long_mask": long["long_mask"],
        "long_dt_bin": long["long_dt_bin"],
        "profile": profile,
        "side_hour_fourier": short["side_hour_fourier"],
    }


def eval_task_a_v2(model, tensors, V):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    from lib.v2.datasets_v2 import TaskADataset
    ds = TaskADataset(tensors)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    all_scores = []
    all_targets = []
    with torch.no_grad():
        for b in dl:
            out = model(b)
            all_scores.append(out.softmax(-1).cpu().numpy())
            all_targets.append(b["target_app"].cpu().numpy())
    S = np.concatenate(all_scores)
    T_arr = np.concatenate(all_targets)
    order = np.argsort(-S, axis=1)
    res = {"n": int(len(T_arr))}
    res["hit_at_1"] = float((S.argmax(1) == T_arr).mean())
    for k in (5, 10):
        res[f"hit_at_{k}"] = float((order[:, :k] == T_arr[:, None]).any(1).mean())
    ranks = np.array([int(np.where(order[i] == T_arr[i])[0][0]) for i in range(len(T_arr))])
    res["mrr"] = float((1.0 / (ranks + 1.0)).mean())
    # macro F1
    preds = S.argmax(1)
    f1s = []
    for c in range(int(S.shape[1])):
        pm = preds == c
        tm = T_arr == c
        tp = int((pm & tm).sum()); fp = int((pm & ~tm).sum()); fn = int((~pm & tm).sum())
        if tp + fp + fn == 0:
            continue
        p = tp / (tp + fp) if (tp + fp) else 0.0
        r = tp / (tp + fn) if (tp + fn) else 0.0
        if p + r > 0:
            f1s.append(2 * p * r / (p + r))
    res["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return res


def eval_task_b_v2(model, tensors, gt_multisets, K_LIST=(1, 3, 5, 10)):
    import torch
    import numpy as np
    from torch.utils.data import DataLoader
    # Build a minimal Dataset-like container
    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, t):
            self.t = {k: torch.as_tensor(v, dtype=torch.long if "app" in k or "dt_bin" in k else (torch.bool if "mask" in k else torch.float32)) for k, v in t.items()}
        def __len__(self):
            return int(self.t["history_app"].shape[0])
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.t.items()}
    ds = _Wrap(tensors)
    dl = torch.utils.data.DataLoader(ds, batch_size=256, shuffle=False)
    model.eval()
    sigs = []
    with torch.no_grad():
        for b in dl:
            out = model(b)
            sigs.append(torch.sigmoid(out["logits_b_sig"]).cpu().numpy())
    S = np.concatenate(sigs, axis=0)
    return _task_b_metrics(S, gt_multisets)


def eval_task_b_v2_fn_clean(model, tensors, gt_multisets):
    scores = forward_task_b(model, tensors)
    return _task_b_metrics(scores, gt_multisets)


def _forward_task_a_clean(model, tensors_a):
    return _forward_task_a(model, tensors_a)


def _task_b_metrics(scores, gt_multisets, K_LIST=(1, 3, 5, 10)):
    out = {}
    for k in K_LIST:
        topk = np.argsort(-scores, axis=1)[:, :k]
        p_list, r_list, f1_list, jac_list, cov_list, eh_list = [], [], [], [], [], []
        tot_e, hit_e = 0, 0
        for i, ms in enumerate(gt_multisets):
            if not ms:
                continue
            g = set(int(a) for a in ms.keys())
            t = set(int(x) for x in topk[i].tolist())
            inter = len(t & g); uni = len(t | g)
            p = inter / k; r = inter / len(g)
            p_list.append(p); r_list.append(r)
            f1_list.append(2 * p * r / (p + r) if (p + r) else 0.0)
            jac_list.append(inter / uni if uni else 0.0)
            cov_list.append(1.0 if g.issubset(t) else 0.0)
            tot = sum(ms.values()); hit = sum(c for a, c in ms.items() if int(a) in t)
            eh_list.append(hit / tot if tot else 0.0)
            tot_e += tot; hit_e += hit
        if p_list:
            out[f"precision@{k}"] = float(np.mean(p_list))
            out[f"recall@{k}"] = float(np.mean(r_list))
            out[f"f1@{k}"] = float(np.mean(f1_list))
            out[f"jaccard@{k}"] = float(np.mean(jac_list))
            out[f"coverage@{k}"] = float(np.mean(cov_list))
            out[f"event_hit@{k}"] = float(np.mean(eh_list))
            out[f"event_hit_micro@{k}"] = float(np.sum([v for v in eh_list]) / max(len(eh_list), 1))
            out[f"n@{k}"] = int(len(p_list))
    return out


def main():
    art = ROOT / "artifacts"
    results = art / "results"
    results.mkdir(exist_ok=True, parents=True)

    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)

    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")
    scaler = F.fit_dt_scaler(train)

    stats = GF.load_stats(art / "v2_profile_stats.pkl")

    all_df = pd.concat([train, val, test]).sort_values("event_ts").reset_index(drop=True)
    all_mask = all_df["is_target_event"].astype(bool).to_numpy()
    tgt_ts = all_df.loc[all_mask, "event_ts"].to_numpy()
    tgt_apps = np.array([D.app_to_idx(a, vocab) for a in all_df.loc[all_mask, "app_label_clean"].fillna("<UNK>")], dtype=np.int64)
    return 0


def main_v2():
    import pandas as pd
    import numpy as np
    import torch
    art = ROOT / "artifacts"
    results = art / "results"
    results.mkdir(exist_ok=True, parents=True)
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)
    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")
    scaler = F.fit_dt_scaler(train)
    stats = GF.load_stats(art / "v2_profile_stats.pkl")

    all_df = pd.concat([train, val, test]).sort_values("event_ts").reset_index(drop=True)
    all_mask = all_df["is_target_event"].astype(bool).to_numpy()
    tgt_ts = all_df.loc[all_mask, "event_ts"].to_numpy()
    tgt_apps = np.array([D.app_to_idx(a, vocab) for a in all_df.loc[all_mask, "app_label_clean"].fillna("<UNK>")], dtype=np.int64)

    # Load checkpoints
    m_a, cfg_a = load_checkpoint_v2(art / "checkpoints" / "task_a_v2.pt", kind="a")
    m_b, cfg_b = load_checkpoint_v2(art / "checkpoints" / "task_b_v2.pt", kind="b")

    results_a = {"V": V, "models": {"GRU_v2": {}}}
    results_b = {"V": V, "models": {"GRU_v2": {}}}

    for split_name, df_split in (("val", val), ("test", test)):
        enc = F.encode_events(df_split, vocab, scaler["mean"], scaler["std"])

        # Task A
        tensors_a = build_target_tensors(enc, stats, tgt_ts, tgt_apps)
        r_a = eval_task_a_v2(m_a, tensors_a, V)
        results_a["models"]["GRU_v2"][split_name] = r_a
        print(f"[eval v2] {split_name} Task A  hit@1={r_a['hit_at_1']:.3f}  hit@5={r_a['hit_at_5']:.3f}  mrr={r_a['mrr']:.3f}")

        # Task B (anchor grid)
        anchors = D.anchor_grid(df_split, stride_sec=300, hour_start=6, hour_end=24)
        raw_gt = D.compute_window_gt(df_split, anchors, horizon_sec=900)
        keep = [i for i, ms in enumerate(raw_gt) if ms]
        anchors = anchors.iloc[keep].reset_index(drop=True)
        anchor_ts = anchors["anchor_ts"].to_numpy()
        gt_multisets = []
        for i in keep:
            ms = raw_gt[i]
            m = {}
            for k, v in ms.items():
                idx = int(D.app_to_idx(k, vocab))
                m[idx] = m.get(idx, 0) + int(v)
            gt_multisets.append(m)

        tensors_b = build_anchor_tensors(F.encode_events(df_split, vocab, scaler["mean"], scaler["std"]), stats, anchor_ts, tgt_ts, tgt_apps)
        r_b = eval_task_b_v2_fn_clean(m_b, tensors_b, gt_multisets)
        results_b["models"]["GRU_v2"][split_name] = r_b
        print(f"[eval v2] {('val' if split_name == 'val' else 'test')} Task B  P@5={r_b.get('precision@5', 0):.3f}  R@5={r_b.get('recall@5', 0):.3f}  EH@5={r_b.get('event_hit@5', 0):.3f}  Cov@5={r_b.get('coverage@5', 0):.3f}")

    with open(results / "task_a_v2.json", "w") as f:
        json.dump(results_a, f, indent=2)
    with open(results / "task_b_v2.json", "w") as f:
        json.dump(results_b, f, indent=2)
    print("[eval v2] wrote task_a_v2.json, task_b_v2.json")
    return 0


def load_checkpoint_v2(path: Path, kind: str):
    sd = torch.load(path, map_location="cpu", weights_only=False)
    cfg_dict = sd["cfg"]
    cfg = M.ConfigV2(**{k: v for k, v in cfg_dict.items()})
    if kind == "a":
        model = M.TaskAModel(cfg)
    else:
        model = M.TaskBModel(cfg)
    model.load_state_dict(sd["state_dict"])
    model.eval()
    return model, cfg


def load_checkpoint_v2_alt(path, kind):
    return load_checkpoint_v2(path, kind)


def build_anchor_tensors_v2(enc, stats, anchor_ts, tgt_ts_full, tgt_apps_full):
    return build_anchor_tensors(enc, stats, anchor_ts, tgt_ts_full, tgt_apps_full)


def load_checkpoint_v2_main(path, kind):
    return load_checkpoint_v2(path, kind)


def eval_task_b_v2_fn(model, tensors, gt_multisets):
    import torch
    import numpy as np
    class _Wrap(torch.utils.data.Dataset):
        def __init__(self, t):
            self.t = {}
            for k, v in t.items():
                if "app" in k or "dt_bin" in k:
                    self.t[k] = torch.as_tensor(v, dtype=torch.long)
                elif "mask" in k:
                    self.t[k] = torch.as_tensor(v, dtype=torch.bool)
                else:
                    self.t[k] = torch.as_tensor(v, dtype=torch.float32)
        def __len__(self):
            return int(self.t["long_app"].shape[0])
        def __getitem__(self, i):
            return {k: v[i] for k, v in self.t.items()}

    # Simpler: run the model in batches without a Dataset class
    def forward_all(model, tensors, bs=256):
        model.eval()
        n = tensors["history_app"].shape[0]
        sigs = []
        for s in range(0, n, bs):
            e = min(s + bs, n)
            b = {}
            for k, v in tensors.items():
                if "app" in k or "dt_bin" in k:
                    b[k] = torch.as_tensor(v[s:e], dtype=torch.long)
                elif "mask" in k:
                    b[k] = torch.as_tensor(v[s:e], dtype=torch.bool)
                else:
                    b[k] = torch.as_tensor(v[s:e], dtype=torch.float32)
            with torch.no_grad():
                out = model(b)
            sigs.append(torch.sigmoid(out["logits_b_sig"]).numpy())
        S = np.concatenate(sigs, axis=0)
        return S

    S = forward_all_v2(model, tensors) if False else forward_all(model, tensors)


def forward_all(model, tensors, bs: int = 256):
    import torch
    import numpy as np
    model.eval()
    n = tensors["history_app"].shape[0]
    S_rows = []
    with torch.no_grad():
        for s in range(0, n, bs):
            e = min(s + bs, n)
            b = {}
            for k, v in tensors.items():
                if "app" in k or "dt_bin" in k:
                    b[k] = torch.as_tensor(v[s:e], dtype=torch.long)
                elif "mask" in k:
                    b[k] = torch.as_tensor(v[s:e], dtype=torch.bool)
                else:
                    b[k] = torch.as_tensor(v[s:e], dtype=torch.float32)
            out = model(b)
            if isinstance(out, dict):
                S_rows.append(torch.sigmoid(out["logits_b_sig"]).numpy())
            else:
                S_rows.append(out.softmax(-1).numpy())
    return np.concatenate(S_rows, axis=0)


def eval_task_b_v2_fn_clean(model, tensors, gt_multisets):
    scores = forward_all(model, tensors)
    return _task_b_metrics(scores, gt_multisets)


load_checkpoint = load_checkpoint_v2


def main_clean():
    import pandas as pd
    import numpy as np
    art = ROOT / "artifacts"
    results = art / "results"
    results.mkdir(exist_ok=True, parents=True)
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)
    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")
    scaler = F.fit_dt_scaler(train)
    stats = GF.load_stats(art / "v2_profile_stats.pkl")

    all_df = pd.concat([train, val, test]).sort_values("event_ts").reset_index(drop=True)
    all_mask = all_df["is_target_event"].astype(bool).to_numpy()
    tgt_ts = all_df.loc[all_mask, "event_ts"].to_numpy()
    tgt_apps = np.array([D.app_to_idx(a, vocab) for a in all_df.loc[all_mask, "app_label_clean"].fillna("<UNK>")], dtype=np.int64)

    m_a, cfg_a = load_checkpoint(art / "checkpoints" / "task_a_v2.pt", "a")
    m_b, cfg_b = load_checkpoint(art / "checkpoints" / "task_b_v2.pt", "b")

    results_a = {"V": V, "models": {"GRU_v2": {}}}
    results_b = {"V": V, "models": {"GRU_v2": {}}}
    for name, df_split in (("val", val), ("test", test)):
        enc = F.encode_events(df_split, vocab, scaler["mean"], scaler["std"])
        # Task A
        tensors_a = build_target_tensors(enc, stats, tgt_ts, tgt_apps)
        scores_a = _forward_task_a(m_a, tensors_a)
        r_a = _task_a_metrics(scores_a, tensors_a["target_app"], V)
        results_a["models"]["GRU_v2"][name] = r_a
        print(f"[v2]  {name:4s}  A  hit@1={r_a['hit_at_1']:.3f}  hit@5={r_a['hit_at_5']:.3f}  mrr={r_a['mrr']:.3f}  f1={r_a['macro_f1']:.3f}")

        # Task B
        anchors = D.anchor_grid(df_split, stride_sec=ANCHOR_STRIDE, hour_start=HOUR_START, hour_end=HOUR_END)
        raw_gt = D.compute_window_gt(df_split, anchors, horizon_sec=WINDOW_HORIZON)
        keep = [i for i, ms in enumerate(raw_gt) if ms]
        anchors = anchors.iloc[keep].reset_index(drop=True)
        gt_idx = []
        for i in keep:
            ms = raw_gt[i]
            m = {}
            for key, cnt in ms.items():
                idx = int(D.app_to_idx(key, vocab))
                m[idx] = m.get(idx, 0) + int(cnt)
            gt_idx.append(m)
        anchor_ts = anchors["anchor_ts"].to_numpy()
        anchor_tensors = build_anchor_tensors(enc, stats, anchor_ts, tgt_ts, tgt_apps)
        S_b = forward_task_b(m_b, anchor_tensors)
        r_b = _task_b_metrics(S_b, gt_idx)
        results_b["models"]["GRU_v2"][name] = r_b
        print(f"[eval v2] {name:4s}  B  P@5={r_b.get('precision@5', 0):.3f}  R@5={r_b.get('recall@5', 0):.3f}  EH@5={r_b.get('event_hit@5', 0):.3f}  Cov@5={r_b.get('coverage@5', 0):.3f}")

    with open(results / "task_a_v2.json", "w") as f:
        json.dump(results_a, f, indent=2)
    with open(results / "task_b_v2.json", "w") as f:
        json.dump(results_b, f, indent=2)
    print(f"[eval v2] wrote {results}/task_{{a,b}}_v2.json")
    return 0


# final helpers
def forward_task_a(model, tensors, batch_size=256):
    return _forward_task_a(model, tensors, batch_size)


def _forward_task_a(model, tensors, batch_size=256):
    import torch
    import numpy as np
    model.eval()
    n = tensors["history_app"].shape[0]
    outs = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            batch = {}
            for k, v in tensors.items():
                if k == "target_app":
                    continue
                if "app" in k or "dt_bin" in k:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.long)
                elif "mask" in k:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.bool)
                else:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.float32)
            logits = model(batch)
            outs.append(logits.softmax(-1).numpy())
    return np.concatenate(outs, axis=0)


def forward_task_b(model, tensors, batch_size=256):
    import torch
    import numpy as np
    model.eval()
    n = tensors["history_app"].shape[0]
    outs = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            batch = {}
            for k, v in tensors.items():
                if k in ("target_app", "win_counts"):
                    continue
                if "app" in k or "dt_bin" in k:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.long)
                elif "mask" in k:
                    batch_mask = None
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.bool)
                else:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.float32)
            out = model(batch)
            outs.append(torch.sigmoid(out["logits_b_sig"]).numpy())
    return np.concatenate(outs, axis=0)


def _task_a_metrics(scores, targets, V):
    import numpy as np
    n = len(targets)
    order = np.argsort(-scores, axis=1)
    res = {"n": int(n)}
    if n == 0:
        return {**res, "hit_at_1": 0.0, "hit_at_5": 0.0, "hit_at_10": 0.0, "mrr": 0.0, "macro_f1": 0.0}
    preds = scores.argmax(axis=1)
    res["hit_at_1"] = float((preds == targets).mean())
    res["hit_at_5"] = float((order[:, :5] == targets[:, None]).any(axis=1).mean())
    res["hit_at_10"] = float((order[:, :10] == targets[:, None]).any(axis=1).mean())
    ranks = np.array([int(np.where(order[i] == targets[i])[0][0]) for i in range(n)])
    res["mrr"] = float((1.0 / (ranks + 1.0)).mean())
    f1s = []
    for c in range(V):
        p_mask = preds == c
        t_mask = targets == c
        tp = int((p_mask & t_mask).sum())
        fp = int((p_mask & ~t_mask).sum())
        fn = int((~p_mask & t_mask).sum())
        if tp + fp + fn == 0:
            continue
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1s.append(2 * prec * rec / (prec + rec) if (prec + rec) else 0.0)
    res["macro_f1"] = float(np.mean(f1s)) if f1s else 0.0
    return res


def forward_task_b(model, tensors, batch_size=256):
    return forward_task_b_fn(model, tensors, batch_size)


def forward_task_b_fn(model, tensors, batch_size=256):
    import torch
    import numpy as np
    model.eval()
    n = tensors["history_app"].shape[0]
    outs = []
    with torch.no_grad():
        for s in range(0, n, batch_size):
            e = min(s + batch_size, n)
            batch = {}
            for k, v in tensors.items():
                if k in ("target_app", "win_counts"):
                    continue
                if "app" in k or "dt_bin" in k:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.long)
                elif "mask" in k:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.bool)
                else:
                    batch[k] = torch.as_tensor(v[s:e], dtype=torch.float32)
            out = model(batch)
            outs_item = torch.sigmoid(out["logits_b_sig"]).numpy()
            outs.append(outs_item)
    return np.concatenate(outs, axis=0)


if __name__ == "__main__":
    sys.exit(main_clean() or 0)
