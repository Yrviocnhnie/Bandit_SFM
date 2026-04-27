"""Train v4 model: v3 features + per-app recency + periodicity priors.

Supports both Task A and Task B via --task {a,b}. New flags:
  --no-recency       drop per-app recency (default ON)
  --no-periodicity   drop periodicity priors (default ON)
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
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import features as F
from lib import train as TR
from lib.v2 import global_features as GF
from lib.v3 import categories as CAT
from lib.v3 import features_v3 as FV
from lib.v3 import models_v3 as M3
from lib.v3 import prep as PREP
from lib.v3.daypart import daypart_onehot
from lib.v3 import markov_prior as MK
from lib.v3 import recency as REC
from lib.v3 import periodicity as PER
from lib.v3.datasets_v3 import TaskADatasetV3, TaskBDatasetV3
from lib.v3.location import load_location_vocab
from lib.v3.window_rollups import build_window_aggregates

HISTORY_K = 16
LONG_K = 64
WINDOW_HORIZON_SEC = 900
BATCH = 256
LR = 1e-3
WD = 1e-4
PATIENCE = 6
SEED = 7
POISSON_WEIGHT = 0.25


def prep_split(df, loc_ids, vocab, scaler, profile_stats, stream, app_to_cat,
               use_category, use_loc, use_daypart, use_windows):
    enc = F.encode_events(df, vocab, scaler["mean"], scaler["std"])
    short = F.build_history_for_targets(enc, history_k=HISTORY_K)
    long = GF.build_long_history_for_targets(enc, k_long=LONG_K)
    v2p = GF.build_profile_for_targets(enc, profile_stats, stream["ts_ns"], stream["app"])

    target_pos = np.nonzero(enc.is_target)[0]
    a_h = enc.hour[target_pos].astype(int)
    a_w = enc.weekday[target_pos].astype(int)
    a_ts = enc.ts[target_pos].astype("datetime64[ns]").astype(np.int64)

    cat_pr = app_to_cat[np.clip(enc.app_idx, 0, len(app_to_cat) - 1)]
    if not use_category:
        cat_pr = np.zeros_like(cat_pr)
    if use_loc:
        loc_pr = np.asarray(loc_ids, dtype=np.int64)
    else:
        loc_pr = np.zeros(len(enc.app_idx), dtype=np.int64)

    hist_cat, hist_loc = FV.build_history_cat_loc_for_targets(
        enc, cat_per_row=cat_pr, loc_per_row=loc_pr, history_k=HISTORY_K,
    )
    long_cl = FV.build_long_cat_loc_for_targets(
        enc, cat_per_row=cat_pr, loc_per_row=loc_pr, k_long=LONG_K,
    )
    parts = [v2p]
    if use_daypart:
        parts.append(daypart_onehot(a_h, a_w))
    if use_windows:
        parts.append(build_window_aggregates(
            anchor_ts_ns=a_ts, stream_ts_ns=stream_ts_ns,
            stream_app_idx=stream_app, stream_cat_idx=stream_cat, stream_dur_s=stream_dur,
        ))
    profile = np.concatenate(parts, axis=1).astype(np.float32)
    return prep_return_block(short, long, hist_cat, hist_loc, long_cl, profile, last_app=last_app)


def prep_return_block(short, long, hist_cat, hist_loc, long_cl, profile, last_app):
    return {
        "history_app": short["history_app"],
        "history_feat": short["history_feat"],
        "history_mask": short["history_mask"],
        "history_category": hist_cat,
        "history_loc": hist_loc,
        "long_app": long["long_app"],
        "long_feat": long["long_feat"],
        "long_mask": long["long_mask"],
        "long_dt_bin": long["long_dt_bin"],
        "long_category": long_cl["long_category"],
        "long_loc": long_cl["long_loc"],
        "profile": profile,
        "last_app_idx": last_app,
        "target_app": short["target_app"],
        "side_hour_fourier": short["target_hour_fourier"],
    }


def evaluate_task_a(model, loader, device="cpu"):
    model.eval()
    S_list, T_list = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            logits = model(b)
            S_list.append(torch.softmax(logits, dim=-1).cpu().numpy())
            T_list.append(b["target_app"].cpu().numpy())
    S = np.concatenate(S_list, axis=0)
    T = np.concatenate(T_list, axis=0)
    order = np.argsort(-S, axis=1)
    hit1 = float((order[:, 0] == T).mean())
    hit5 = float((order[:, :5] == T[:, None]).any(axis=1).mean())
    ranks = np.zeros(len(T), dtype=int)
    for i in range(len(T)):
        r = np.where(order[i] == T[i])[0]
        ranks[i] = int(r[0]) if len(r) else int(S.shape[1] - 1)
    mrr = float((1.0 / (ranks + 1.0)).mean())
    return {"hit_at_1": hit1, "hit_at_5": hit5, "mrr": mrr, "n_samples": int(len(T))}


def task_b_loss(out, win_counts):
    y_bin = (win_counts > 0).float()
    bce = Fnn.binary_cross_entropy_with_logits(out["logits_b_sig"], y_bin)
    log_rate = out["log_rate_b"].clamp(-10, 10)
    rate = torch.exp(log_rate)
    poisson = (rate - win_counts * log_rate).mean()
    return bce + 0.25 * poisson


def evaluate_task_b(model, loader, device="cpu"):
    model.eval()
    sig_list, wc_list = [], []
    with torch.no_grad():
        for b in loader:
            b = {k: v.to(device) for k, v in b.items()}
            out = model(b)
            sig_list.append(torch.sigmoid(out["logits_b_sig"]).cpu().numpy())
            wc_list.append(b["win_counts"].cpu().numpy())
    sig = np.concatenate(sig_list, axis=0)
    wc = np.concatenate(wc_list, axis=0)
    topk = np.argsort(-sig, axis=1)[:, :5]
    total_events = 0
    total_hit = 0
    precisions = []
    recalls = []
    covs = []
    for i in range(len(wc)):
        tset = set(int(x) for x in topk[i])
        tot = int(wc[i].sum())
        if tot == 0:
            continue
        total_events += tot
        for a_idx in range(wc.shape[1]):
            if a_idx in tset:
                total_hit += int(wc[i, a_idx])
        gt = set(int(a) for a in range(wc.shape[1]) if wc[i, a] > 0)
        inter = len(tset & gt)
        precisions.append(inter / 5.0)
        recalls.append(inter / max(1, len(gt)))
        covs.append(1.0 if gt.issubset(tset) else 0.0)
    return {
        "event_hit_at_5": total_hit / max(1, total_events),
        "precision_at_5": float(np.mean(precisions)) if precisions else 0.0,
        "recall_at_5": float(np.mean(recalls)) if recalls else 0.0,
        "coverage_at_5": float(np.mean(covs)) if covs else 0.0,
        "n_samples": int(len(wc)),
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--no-category", dest="use_category", action="store_false")
    p.add_argument("--no-loc", dest="use_loc", action="store_false")
    p.add_argument("--no-daypart", dest="use_daypart", action="store_false")
    p.add_argument("--no-windows", dest="use_windows", action="store_false")
    p.add_argument("--no-local", dest="use_local", action="store_false")
    p.add_argument("--no-global", dest="use_global", action="store_false")
    p.add_argument("--no-profile", dest="use_profile", action="store_false")
    p.add_argument("--use-markov", dest="use_markov", action="store_true")
    p.add_argument("--task", choices=["a", "b"], default="a")
    p.add_argument("--no-recency", dest="use_recency", action="store_false")
    p.add_argument("--no-periodicity", dest="use_periodicity", action="store_false")
    p.add_argument("--out", default=None)
    p.add_argument("--tag", default=None)
    p.add_argument("--epochs", type=int, default=25)
    p.set_defaults(
        use_category=True, use_loc=True, use_daypart=True, use_windows=True,
        use_local=True, use_global=True, use_profile=True, use_markov=False,
        use_recency=True, use_periodicity=True,
    )
    args = p.parse_args()

    if args.task not in ("a", "b"):
        args.task = "a"
    if args.out is None:
        args.out = f"task_{args.task}_v4.pt"
    if args.tag is None:
        args.tag = f"task_{args.task}_v4"

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    art = ROOT / "artifacts"
    v3d = art / "v3"

    train = pd.read_parquet(art / "splits" / "train.parquet").sort_values("event_ts").reset_index(drop=True)
    val = pd.read_parquet(art / "splits" / "val.parquet").sort_values("event_ts").reset_index(drop=True)
    test = pd.read_parquet(art / "splits" / "test.parquet").sort_values("event_ts").reset_index(drop=True)

    vocab = PREP.load_vocab(art)
    V = len(vocab)

    scaler = F.fit_dt_scaler(train)
    profile_stats = GF.fit_profile_stats(train, vocab)
    top8_indices = np.asarray(profile_stats["global_top8"], dtype=np.int64)
    app_to_cat = CAT.build_app_to_cat_idx(vocab)
    loc_stats = load_location_vocab(v3d / "loc_vocab.pkl")
    num_locations = len(loc_stats["vocab"])

    loc_ids = {
        "train": np.load(v3d / "loc_ids_train.npy"),
        "val": np.load(v3d / "loc_ids_val.npy"),
        "test": np.load(v3d / "loc_ids_test.npy"),
    }

    stream = PREP.build_target_stream(train, val, test, vocab, app_to_cat)

    # build tensors per split (using the same prep logic inline, no closures)
    tensors = {}
    for split_name, df, ids in (("train", train, loc_ids["train"]),
                                ("val", val, loc_ids["val"]),
                                ("test", test, loc_ids["test"])):
        enc = F.encode_events(df, vocab, scaler["mean"], scaler["std"])
        short = F.build_history_for_targets(enc, history_k=HISTORY_K)
        long = GF.build_long_history_for_targets(enc, k_long=LONG_K)
        v2p = GF.build_profile_for_targets(enc, profile_stats, stream["ts_ns"], stream["app"])
        target_pos = np.nonzero(enc.is_target)[0]
        a_h = enc.hour[target_pos].astype(int)
        a_w = enc.weekday[target_pos].astype(int)
        a_ts = enc.ts[target_pos].astype("datetime64[ns]").astype(np.int64)

        cat_per_row = app_to_cat[np.clip(enc.app_idx, 0, len(app_to_cat) - 1)]
        if not args.use_category:
            cat_per_row = np.zeros_like(cat_per_row)
        if args.use_loc:
            loc_per_row = np.asarray(ids, dtype=np.int64)
        else:
            loc_per_row = np.zeros(len(enc.app_idx), dtype=np.int64)

        hist_cat, hist_loc = FV.build_history_cat_loc_for_targets(
            enc, cat_per_row=cat_per_row, loc_per_row=loc_per_row, history_k=HISTORY_K,
        )
        long_cl = FV.build_long_cat_loc_for_targets(
            enc, cat_per_row=cat_per_row, loc_per_row=loc_per_row, k_long=LONG_K,
        )
        parts = [v2p]
        if args.use_daypart:
            parts.append(daypart_onehot(a_h, a_w))
        if args.use_windows:
            parts.append(build_window_aggregates(
                anchor_ts_ns=a_ts, stream_ts_ns=stream["ts_ns"],
                stream_app_idx=stream["app"], stream_cat_idx=stream["cat"],
                stream_dur_s=stream["dur_s"],
            ))
        if args.use_recency:
            recency_arr = REC.build_recency_for_anchors(
                a_ts, stream["ts_ns"], stream["app"], top8_indices,
            )
            parts.append(recency_arr)
        if args.use_periodicity:
            periodicity_arr = PER.build_periodicity_for_anchors(
                a_ts, stream["ts_ns"], stream["app"], top8_indices,
            )
            parts.append(periodicity_arr)
        profile_full = np.concatenate(parts, axis=1).astype(np.float32)
        last_app = PREP.last_target_app_for_anchors(a_ts_view(a_ts), stream["ts_ns"], stream["app"])
        tensors[split_name] = {
            "history_app": short["history_app"],
            "history_feat": short["history_feat"],
            "history_mask": short["history_mask"],
            "history_category": hist_cat,
            "history_loc": np.asarray(hist_loc, dtype=np.int64),
            "long_app": long["long_app"],
            "long_feat": long["long_feat"],
            "long_mask": long["long_mask"],
            "long_dt_bin": long["long_dt_bin"],
            "long_category": long_cl["long_category"],
            "long_loc": long_cl["long_loc"],
            "profile": profile_full,
            "last_app_idx": last_app,
            "target_app": short["target_app"],
            "side_hour_fourier": short["target_hour_fourier"],
        }
        # Task B window counts
        target_ts_ns_sorted = np.sort(enc.ts[np.nonzero(enc.is_target)[0]].astype("datetime64[ns]").astype(np.int64))
        # compute per-target win_counts using the stored target_app and target_ts from build_history_for_targets
        target_ts_ns = short["target_ts"].astype("datetime64[ns]").astype(np.int64)
        win_counts = TR.build_window_target_counts(target_ts_ns, short["target_app"],
                                                    int(WINDOW_HORIZON_SEC) * 1_000_000_000, V)
        tensors[split_name]["win_counts"] = win_counts

    profile_dim = tensors["train"]["profile"].shape[1]
    cfg = M3.ConfigV3(
        vocab_size=V,
        num_categories=CAT.NUM_CATEGORIES,
        num_locations=num_locations,
        profile_dim=profile_dim,
        use_category=args.use_category,
        use_loc=args.use_loc,
        use_local=args.use_local,
        use_global=args.use_global,
        use_profile=args.use_profile,
    )

    if args.task == "b":
        ds_tr = TaskBDatasetV3(tensors["train"], tensors["train"]["win_counts"])
        ds_va = TaskBDatasetV3(tensors["val"], tensors["val"]["win_counts"])
    else:
        ds_tr = TaskADatasetV3(tensors["train"])
        ds_va = TaskADatasetV3(tensors["val"])
    dl_tr = DataLoader(ds_tr, batch_size=BATCH, shuffle=True)
    dl_va = DataLoader(ds_va, batch_size=BATCH, shuffle=False)

    # --- Markov prior (Task B only)
    mk_log_prior = None
    if args.task == "b" and args.use_markov:
        mk_stats = MK.load_markov_prior(v3d / "markov_prior.pkl")
        mk_log_prior = torch.from_numpy(mk_stats["log_prior"].astype(np.float32))
        cfg.use_markov_prior = True
    else:
        cfg.use_markov_prior = False

    if args.task == "b":
        model = M3.TaskBModelV3(cfg, markov_log_prior=mk_log_prior)
    else:
        model = M3.TaskAModelV3(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[v4 task_{args.task}] params={n_params}  profile_dim={profile_dim}  "
          f"flags: cat={args.use_category} loc={args.use_loc} day={args.use_daypart} "
          f"win={args.use_windows} rec={args.use_recency} per={args.use_periodicity} mk={args.use_markov}")

    cw = torch.tensor(TR.class_weights(tensors["train"]["target_app"], V)) if args.task == "a" else None
    opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WD)
    device = "cpu"
    best_score = 0.0
    best_state = None
    patience = 0
    log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        train_loss = 0.0
        n_batches = 0
        for b in dl_tr:
            b = {k: v.to(device) for k, v in b.items()}
            if args.task == "a":
                logits = model(b)
                loss = Fnn.cross_entropy(logits, b["target_app"], weight=cw, label_smoothing=0.05)
            else:
                out = model(b)
                loss = task_b_loss(out, b["win_counts"])
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            train_loss += float(loss.item())
            n_batches += 1
        if args.task == "b":
            eval_metrics = evaluate_task_b(model, dl_va)
            score_key = "event_hit_at_5"
            score_summary = f"val EH@5={eval_metrics['event_hit_at_5']:.4f}  R@5={eval_metrics['recall_at_5']:.3f}  Cov@5={eval_metrics['coverage_at_5']:.3f}"
        else:
            eval_metrics = evaluate_task_a(model, dl_va)
            score_key = "hit_at_5"
            score_summary = f"val H@1={eval_metrics['hit_at_1']:.4f}  H@5={eval_metrics['hit_at_5']:.4f}  MRR={eval_metrics['mrr']:.4f}"
        log.append({"epoch": epoch, "train_loss": train_loss / max(1, n_batches), **eval_metrics})
        print(f"  ep{epoch:02d}  loss={train_loss / max(1, n_batches):.4f}  {score_summary}")
        if eval_metrics[score_key] > best_score + 1e-6:
            best_score = eval_metrics[score_key]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  early-stop")
                break

    # Save checkpoint
    out_ckpt = art / "checkpoints" / args.out
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    flags_dict = {
        "task": args.task,
        "use_category": args.use_category, "use_loc": args.use_loc,
        "use_daypart": args.use_daypart, "use_windows": args.use_windows,
        "use_local": args.use_local, "use_global": args.use_global,
        "use_profile": args.use_profile, "use_markov": args.use_markov,
        "use_recency": args.use_recency, "use_periodicity": args.use_periodicity,
    }
    torch.save({"state_dict": best_state, "cfg": vars(cfg),
                "best_score": best_score, "flags": flags_dict}, out_ckpt)
    print(f"[v4 task_{args.task}] saved {out_ckpt}  best val score={best_score:.4f}  "
          f"elapsed={time.time() - t0:.1f}s")

    model.load_state_dict(best_state)
    if args.task == "b":
        val_metrics = evaluate_task_b(model, dl_va)
        ds_te = TaskBDatasetV3(tensors["test"], tensors["test"]["win_counts"])
        dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False)
        test_metrics = evaluate_task_b(model, dl_te)
    else:
        val_metrics = evaluate_task_a(model, dl_va)
        ds_te = TaskADatasetV3(tensors["test"])
        dl_te = DataLoader(ds_te, batch_size=BATCH, shuffle=False)
        test_metrics = evaluate_task_a(model, dl_te)

    flags_dump = {"task": args.task, "use_category": args.use_category, "use_loc": args.use_loc,
                  "use_daypart": args.use_daypart, "use_windows": args.use_windows,
                  "use_local": args.use_local, "use_global": args.use_global,
                  "use_profile": args.use_profile, "use_markov": args.use_markov,
                  "use_recency": args.use_recency, "use_periodicity": args.use_periodicity}
    results_dir = art / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    out_results = results_dir / f"{args.tag}.json"
    with open(out_results, "w") as f:
        json.dump({
            "tag": args.tag,
            "flags": flags_dump,
            "n_params": n_params,
            "profile_dim": profile_dim,
            "val": eval_metrics,
            "val_final_reloaded": val_metrics,
            "test": test_metrics,
            "train_log": log,
            "alpha_markov_final": float(model.alpha_markov.item()) if (args.task == "b" and args.use_markov and hasattr(model, 'alpha_markov') and model.alpha_markov is not None) else None,
        }, f, indent=2)
    if args.task == "b":
        print(f"[v4 task_b] best val EH@5={best_score:.4f}  test EH@5={test_metrics['event_hit_at_5']:.4f}")
    else:
        print(f"[v4 task_a] best val H@5={best_score:.4f}  test H@1={test_metrics['hit_at_1']:.4f}")
    return 0


def a_ts_from_enc(enc):
    pos = np.nonzero(enc.is_target)[0]
    return enc.ts[pos].astype("datetime64[ns]").astype(np.int64)


def a_ts_view(at):
    return at


def _unused_evaluate_task_a_v0(model, loader, device="cpu"):
    return None


if __name__ == "__main__":
    sys.exit(main() or 0)
