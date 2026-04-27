"""Train TaskBModel (v2) with BCE + Poisson loss, early-stop on val EventHit@5."""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import data as D
from lib import features as F
from lib import train as T
from lib.v2 import global_features as GF
from lib.v2 import models_v2 as M2
from lib.v2 import datasets_v2 as DSETS


HISTORY_K = 16
LONG_K = 64
WINDOW_HORIZON = 900
BATCH = 256
LR = 1e-3
WD = 1e-4
PATIENCE = 6
SEED = 7


def task_b_loss(out, target_counts):
    y_bin = (target_counts > 0).float()
    bce = Fnn.binary_cross_entropy_with_logits(out["logits_b_sig"], y_bin)
    log_rate = out["log_rate_b"].clamp(max=8.0)
    rate = torch.exp(log_rate)
    poisson = (rate - target_counts * log_rate).mean()
    return bce + 0.25 * poisson


def eval_event_hit5(model, ds, device, gt_for_targets):
    """Approximate EventHit@5: for each TRAINING-style target, use the same win_counts
    as ground truth (equivalent to evaluating on a target-event-indexed variant of Task B).
    This is a cheap proxy used only for checkpoint selection; the full anchor-grid eval
    runs in scripts/12_eval_v2.py."""
    import numpy as np
    from torch.utils.data import DataLoader
    model.eval()
    V = ds.t_vocab_size if hasattr(ds, "t_vocab_size") else (ds.history_app.shape[0], )
    dl = DataLoader(ds, batch_size=256, shuffle=False)
    sig_all = []
    wc_all = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b)
            sig_all.append(torch.sigmoid(out["logits_b_sig"]).cpu().numpy())
            wc_all.append(b["win_counts"].cpu().numpy())
    sig = np.concatenate(sig_all, axis=0)
    wc = np.concatenate(wc_all, axis=0)
    topk = np.argsort(-sig, axis=1)[:, :5]
    hit_events = 0
    total_events = 0
    for i in range(len(wc)):
        t_set = set(int(x) for x in topk[i].tolist())
        tot = int(wc[i].sum())
        if tot == 0:
            continue
        hit = 0
        for a in range(wc.shape[1]):
            if a in t_set:
                hit += int(wc[i, a])
        total_events += tot
        hit_events += hit
    return hit_events / max(total_events, 1)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-local", action="store_true")
    parser.add_argument("--no-global", action="store_true")
    parser.add_argument("--no-profile", action="store_true")
    parser.add_argument("--out", default="task_b_v2.pt")
    parser.add_argument("--epochs", type=int, default=25)
    args = parser.parse_args()

    art = ROOT / "artifacts"
    with open(art / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)

    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")

    scaler = F.fit_dt_scaler(train)
    stats = GF.fit_profile_stats(train, vocab)
    GF.save_stats(stats, art / "v2_profile_stats.pkl")

    all_df = pd.concat([train, val, test], ignore_index=True).sort_values("event_ts").reset_index(drop=True)
    all_mask = all_df["is_target_event"].astype(bool).to_numpy()
    tgt_ts = all_df.loc[all_mask, "event_ts"].to_numpy()
    tgt_apps = np.array([D.app_to_idx(a, vocab) for a in all_df.loc[all_mask, "app_label_clean"].fillna("<UNK>")], dtype=np.int64)

    def prep(split_df):
        enc = F.encode_events(split_df, vocab, scaler["mean"], scaler["std"])
        short = F.build_history_for_targets(enc, history_k=16)
        long = GF.build_long_history_for_targets(enc, k_long=64)
        profile = GF.build_profile_for_targets(enc, stats, tgt_ts, tgt_apps)
        target_ts_i64 = short["target_ts"].astype("datetime64[ns]").astype(np.int64)
        target_app_idx = short["target_app"]
        win_counts = T.build_window_target_counts(
            target_ts_i64, target_app_idx, int(WINDOW_HORIZON) * 1_000_000_000, V
        )
        tensors = {
            "history_app": short["history_app"],
            "history_feat": short["history_feat"],
            "history_mask": short["history_mask"],
            "long_app": long["long_app"],
            "long_feat": long["long_feat"],
            "long_mask": long["long_mask"],
            "long_dt_bin": long["long_dt_bin"],
            "profile": profile,
            "side_hour_fourier": short["target_hour_fourier"],
            "target_app": short["target_app"],
        }
        return tensors, win_counts

    tr_tensors, tr_win = prep(train)
    va_tensors, va_win = prep(val)

    from lib.v2 import datasets_v2 as DSETS2
    ds_tr = DSETS2.TaskBDataset(tr_tensors, tr_win)
    ds_va = DSETS2.TaskBDataset(va_tensors, va_win)

    cfg = M.ConfigV2(
        vocab_size=V,
        use_local=not args.no_local,
        use_global=not args.no_global,
        use_profile=not args.no_profile,
    )
    torch.manual_seed(7)
    model = M.TaskBModel(cfg)
    print(f"TaskBModel params = {sum(p.numel() for p in model.parameters())}  "
          f"(local={cfg.use_local}, global={cfg.use_global}, profile={cfg.use_profile})")

    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=256, shuffle=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=256, shuffle=False)

    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    best = 0.0
    best_state = None
    patience = 0
    log = []

    device = "cpu"
    for epoch in range(1, args.epochs + 1):
        model.train()
        t0 = time.time()
        loss_sum = 0.0
        n = 0
        for b in dl_tr:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b)
            loss = task_b_loss(out, b["win_counts"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n += 1
        # eval Task B on val (per-target, so EventHit on win_counts is a proxy)
        eh5 = eval_eh5_per_target(model, ds_va)
        log.append({"epoch": epoch, "loss": loss_sum / max(n, 1), "val_eh5": eh5})
        print(f"  ep{epoch:02d}  loss={loss_sum/max(n,1):.4f}  val EH@5 (per-target)={eh5:.4f}  ({time.time()-t0:.1f}s)")
        if eh5 > best:
            best = eh5
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  early-stop")
                break

    ckpt_path = ROOT / "artifacts" / "checkpoints" / args.out
    torch.save({"state_dict": best_state, "cfg": vars(cfg), "best_eh5": best}, ckpt_path)
    (ROOT / "artifacts" / "results").mkdir(exist_ok=True, parents=True)
    with open(ROOT / "artifacts" / "results" / "task_b_v2_train.json", "w") as f:
        json.dump({"best_eh5": best, "log": log, "cfg": vars(cfg)}, f, indent=2, default=str)
    print(f"[train B v2] best EH@5 = {best:.4f}")
    return 0


def task_b_loss(out, win_counts):
    y_bin = (win_counts > 0).float()
    bce = Fnn.binary_cross_entropy_with_logits(out["logits_b_sig"], y_bin)
    log_rate = out["log_rate_b"].clamp(-10, 10)
    rate = torch.exp(log_rate)
    pois = (rate - win_counts * log_rate).mean()
    return bce + 0.25 * pois


def eval_eh5(model, ds_val):
    from torch.utils.data import DataLoader
    import numpy as np
    dl = DataLoader(ds_val, batch_size=256, shuffle=False)
    model.eval()
    sig_all = []
    win_all = []
    with torch.no_grad():
        for b in dl:
            out = model(b)
            sig_all.append(torch.sigmoid(out["logits_b_sig"]).cpu().numpy())
            win_all.append(b["win_counts"].cpu().numpy())
    sig = np.concatenate(sig_all, axis=0)
    wc = np.concatenate(win_all, axis=0)
    topk = np.argsort(-sig, axis=1)[:, :5]
    total = wc.sum()
    hit = 0.0
    for i in range(len(wc)):
        t_set = set(int(x) for x in topk[i].tolist())
        for a_idx in range(wc.shape[1]):
            if a_idx in t_set:
                hit += float(wc[i, a_idx])
    return float(hit / max(total, 1.0))


def eval_eh5_v2(model, ds_val):
    return eval_eh5(model, ds_val)


def eval_eh5_per_target(model, ds_val):
    return eval_eh5(model, ds_val)


def task_b_loss_fn(out, win_counts):
    return task_b_loss(out, win_counts)


def _not_used():
    pass


# Import M here to avoid circular
from lib.v2 import models_v2 as M


if __name__ == "__main__":
    sys.exit(main() or 0)
