"""Train TaskAModel (v2) with optional ablation toggles via CLI flags.

Flags:
  --no-local     : disable LocalEncoder
  --no-global    : disable GlobalEncoder
  --no-profile   : disable ProfileEncoder
  --out PATH     : checkpoint output path (default artifacts/checkpoints/task_a_v2.pt)
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
import torch.nn.functional as Fnn
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib import data as D
from lib import features as F
from lib import train as T
from lib.v2 import global_features as GF
from lib.v2 import models_v2 as M
from lib.v2 import datasets_v2 as DSETS


HISTORY_K = 16
LONG_K = 64
WINDOW_HORIZON = 900
BATCH = 256
EPOCHS = 25
LR = 1e-3
WD = 1e-4
PATIENCE = 6
SEED = 7
LABEL_SMOOTHING = 0.05


def full_target_stream(df, vocab):
    mask = df["is_target_event"].astype(bool).to_numpy()
    ts = df.loc[mask, "event_ts"].to_numpy()
    apps = np.array([D.app_to_idx(a, vocab) for a in df.loc[mask, "app_label_clean"].fillna("<UNK>")], dtype=np.int64)
    order = np.argsort(ts.astype("datetime64[ns]").astype(np.int64))
    return ts[order], apps[order]


def build_tensors(split_df, vocab, stats, scaler, target_ts_full, target_app_full):
    enc = F.encode_events(split_df, vocab, scaler["mean"], scaler["std"])
    short_hist = F.build_history_for_targets(enc, history_k=HISTORY_K)
    long_hist = GF.build_long_history_for_targets(enc, k_long=LONG_K)
    profile = GF.build_profile_for_targets(enc, stats, target_ts_full, target_app_full)

    tensors = {
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
    return tensors


HISTORY_K = 16


def main():
    ap = __import__("argparse").ArgumentParser()
    ap.add_argument("--no-local", action="store_true")
    ap.add_argument("--no-global", action="store_true")
    ap.add_argument("--no-profile", action="store_true")
    ap.add_argument("--out", default="task_a_v2.pt")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    args = ap.parse_args()

    from pathlib import Path as _P
    root = _P(__file__).resolve().parents[1]
    import sys as _s; _s.path.insert(0, str(root))
    import json as _json
    import pandas as _pd
    import numpy as _np
    import torch as _torch

    from lib import data as D
    from lib import features as F
    from lib.v2 import global_features as GF
    from lib.v2 import models_v2 as M
    from lib.v2 import datasets_v2 as DSETS
    return _main(args, root, D, F, GF, M, DSETS)


def _main(args, root, D, F, GF, M, DSETS):
    import pandas as pd
    import numpy as np
    import torch
    art = root / "artifacts"
    with open(art / "vocab.json") as f:
        import json
        vocab = json.load(f)
    V = len(vocab)

    train = pd.read_parquet(art / "splits" / "train.parquet")
    val = pd.read_parquet(art / "splits" / "val.parquet")
    test = pd.read_parquet(art / "splits" / "test.parquet")

    scaler = F.fit_dt_scaler(train)

    # profile stats fit ONLY on train
    stats = GF.fit_profile_stats(train, vocab)
    GF.save_stats(stats, art / "v2_profile_stats.pkl")

    # Target stream over ALL splits combined — causal rolling only looks back
    all_df = pd.concat([train, val, test], ignore_index=True).sort_values("event_ts").reset_index(drop=True)
    all_mask = all_df["is_target_event"].astype(bool).to_numpy()
    tgt_ts = all_df.loc[all_mask, "event_ts"].to_numpy()
    tgt_apps = np.array(
        [D.app_to_idx(a, vocab) for a in all_df.loc[all_mask, "app_label_clean"].fillna("<UNK>")],
        dtype=np.int64,
    )

    def prep(split_df):
        enc = F.encode_events(split_df, vocab, scaler["mean"], scaler["std"])
        short = F.build_history_for_targets(enc, history_k=16)
        long = GF.build_long_history_for_targets(enc, k_long=64)
        profile = GF.build_profile_for_targets(enc, stats, tgt_ts, tgt_apps)
        side = F.fourier_hour(short["target_hour"])
        return {
            "history_app": short["history_app"],
            "history_feat": short["history_feat"],
            "history_mask": short["history_mask"],
            "long_app": long["long_app"],
            "long_feat": long["long_feat"],
            "long_mask": long["long_mask"],
            "long_dt_bin": long["long_dt_bin"] if False else long["long_dt_bin"],
            "profile": profile,
            "side_hour_fourier": side,
            "target_app": short["target_app"],
        }

    # fit scaler from encoded train
    # scaler already computed above

    tensors_train = prep(train)
    tensors_val = prep(val)

    cfg = M.ConfigV2(
        vocab_size=len(vocab),
        use_local=not args.no_local,
        use_global=not args.no_global,
        use_profile=not args.no_profile,
    )

    import torch
    torch.manual_seed(7)
    model = M.TaskAModel(cfg)
    print(f"TaskAModel params = {sum(p.numel() for p in model.parameters())}  "
          f"(local={cfg.use_local}, global={cfg.use_global}, profile={cfg.use_profile})")

    ds_tr = DSETS.TaskADataset(tensors_train)
    ds_va = DSETS.TaskADataset(tensors_val)
    dl_tr = torch.utils.data.DataLoader(ds_tr, batch_size=256, shuffle=True)
    dl_va = torch.utils.data.DataLoader(ds_va, batch_size=256, shuffle=False)

    from lib import train as T
    cw_np = T.class_weights(tensors_train["target_app"].astype(np.int64), len(vocab))
    cw = torch.tensor(cw_np)

    torch.manual_seed(7)
    cfg_again = cfg  # cfg is already built
    opt = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad),
        lr=1e-3,
        weight_decay=1e-4,
    )

    best_hit5 = -1.0
    best_state = None
    patience = 0
    log = []

    for epoch in range(1, args.epochs + 1):
        model_train(model, dl_tr, opt, cw, device="cpu")
        metrics = eval_task_a_v2(model, dl_va, device="cpu")
        log.append({"epoch": epoch, **metrics})
        print(f"  ep{epoch:02d}  hit@1={metrics['hit_at_1']:.3f}  hit@5={metrics['hit_at_5']:.3f}  mrr={metrics['mrr']:.3f}")
        if metrics["hit_at_5"] > best_hit5:
            best_hit5 = metrics["hit_at_5"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= 6:
                print("  early-stop")
                break

    out_dir = root / "artifacts" / "checkpoints"
    out_dir.mkdir(parents=True, exist_ok=True)
    torch.save({
        "state_dict": best_state,
        "cfg": vars(cfg),
        "best_val_hit5": max(m["hit_at_5"] for m in log),
    }, out_dir / args.out)

    results_dir = root / "artifacts" / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    with open(results_dir / "task_a_v2_train.json", "w") as f:
        import json
        json.dump({"log": log, "best_hit5": best_hit5, "cfg": vars(cfg)}, f, indent=2, default=str)

    print(f"[train A v2] best val hit@5 = {best_hit5:.4f}")
    return 0


def model_train(model, dl, opt, cw, device):
    import torch.nn.functional as Fnn
    import torch
    model.train()
    for b in dl:
        b = {k: v.to(device) for k, v in b.items()}
        logits = model(b)
        loss = Fnn.cross_entropy(logits, b["target_app"], weight=cw.to(device), label_smoothing=0.05)
        opt.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()


def eval_val_hit(model, dl, device):
    import torch
    import numpy as np
    model.eval()
    scores = []
    targets = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(b)
            scores.append(logits.softmax(-1).cpu().numpy())
            targets.append(b["target_app"].cpu().numpy())
    return np.concatenate(scores), np.concatenate(targets)


def eval_task_a_v2(model, dl, device="cpu"):
    import torch
    import numpy as np
    model.eval()
    scores_list = []
    targets_list = []
    with torch.no_grad():
        for b in dl:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(b)
            scores_list.append(logits.softmax(-1).cpu().numpy())
            targets_list.append(b["target_app"].cpu().numpy())
    S = np.concatenate(scores_list)
    T_arr = np.concatenate(targets_list)
    order = np.argsort(-S, axis=1)
    out = {}
    out["hit_at_1"] = float((S.argmax(1) == T_arr).mean())
    out["hit_at_5"] = float((order[:, :5] == T_arr[:, None]).any(1).mean())
    n = len(T_arr)
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        ranks[i] = int(np.where(order[i] == T_arr[i])[0][0])
    out["mrr"] = float((1.0 / (ranks + 1.0)).mean())
    return out


if __name__ == "__main__":
    sys.exit(main() or 0)
