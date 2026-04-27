"""Train and evaluate the per-pair MLP for Task C.

Reads bg_{train,val,test}.parquet, builds a 14-d continuous feature matrix
per (anchor, app) pair (see lib.bg.models_bg.FEATURE_NAMES), trains with
dual-horizon BCE + pos_weight, and reports:

  - FalseKillRate@r for r in {0.1, 0.25, 0.5, 0.75, 0.9}, each horizon
  - MemorySaveRate@r
  - PR-AUC (anchor-meaned), ROC-AUC (anchor-meaned), NDCG@half

Saves checkpoint + JSON results with tag = args.tag.
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
from torch.utils.data import DataLoader, Dataset

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.bg.models_bg import BgMLPConfig, BgPairMLP, FEATURE_NAMES, bce_dual
from lib.v3 import categories as CAT
from lib.v3 import markov_prior as MK


SEED = 7
BATCH = 512
LR = 1e-3
WD = 1e-4
EPOCHS_DEFAULT = 30
PATIENCE = 5
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)


def fit_hour_freq(train_df: pd.DataFrame, vocab: dict) -> np.ndarray:
    V = len(vocab)
    rare = vocab.get("<RARE>", 2)
    table = np.zeros((24, V), dtype=np.float64)
    mask = train_df["is_target_event"].astype(bool).to_numpy()
    hrs = pd.to_datetime(train_df.loc[mask, "event_ts"]).dt.hour.to_numpy()
    apps = train_df.loc[mask, "app_label_clean"].fillna("<UNK>").astype(str).to_numpy()
    idx = np.array([vocab.get(a, rare) for a in apps], dtype=np.int64)
    for h, a in zip(hrs, idx):
        if 0 <= h < 24 and 3 <= a < V:
            table[int(h), int(a)] += 1.0
    table += 0.5
    table[:, :3] = 0.0
    row_sum = table.sum(axis=1, keepdims=True)
    row_sum[row_sum == 0] = 1.0
    return (table / row_sum).astype(np.float32)


def build_features(
    df: pd.DataFrame,
    hour_freq: np.ndarray,
    markov_probs: np.ndarray,
    app_to_cat: np.ndarray,
) -> dict:
    tib = np.clip(df["time_in_bg_sec"].to_numpy(dtype=np.float64), 0, None)
    tsf = np.clip(df["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
    fg_ct = df["fg_count_today"].to_numpy(dtype=np.float64)
    bg_sz = df["bg_set_size"].to_numpy(dtype=np.float64)
    h = df["anchor_hour"].to_numpy(dtype=np.float64)
    w = df["anchor_weekday"].to_numpy(dtype=np.float64)
    dp_anchor = df["anchor_daypart"].to_numpy(dtype=np.int64)
    dp_last = df["last_fg_daypart"].to_numpy(dtype=np.int64)
    loc = df["last_fg_loc_id"].to_numpy(dtype=np.int64)

    last_fg_clamp = np.clip(df["last_fg_app_idx"].to_numpy(dtype=np.int64),
                            0, markov_probs.shape[0] - 1)
    app_clamp = np.clip(df["app_idx"].to_numpy(dtype=np.int64),
                        0, markov_probs.shape[1] - 1)
    markov_col = markov_probs[last_fg_clamp, app_clamp].astype(np.float32)

    hour_col = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_cond = hour_freq[hour_col, app_clamp].astype(np.float32)

    feats = np.stack([
        np.log1p(tib_arr := tib),
        np.log1p(tfg_arr := tsf),
        np.log1p(fg_ct),
        np.clip(tib / (6 * 3600.0), 0, 2),
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * w / 7.0),
        np.cos(2 * np.pi * w / 7.0),
        np.log1p(bg_sz := bg_sz if False else np.log1p(bg_sz)),
        # (above line is a mistake -- rewritten in clean impl below)
    ], axis=1).astype(np.float32)  # type: ignore
    raise NotImplementedError("see build_features_v2 below")


def build_features_v2(
    df: pd.DataFrame,
    hour_freq: np.ndarray,
    markov_probs: np.ndarray,
    app_to_cat: np.ndarray,
) -> dict:
    n = len(df)
    tib = np.clip(df["time_in_bg_sec"].to_numpy(dtype=np.float64), 0, None)
    tsf = np.clip(df["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
    fg_ct = df["fg_count_today"].to_numpy(dtype=np.float64)
    bg_sz = df["bg_set_size"].to_numpy(dtype=np.float64)
    h = df["anchor_hour"].to_numpy(dtype=np.float64)
    w = df["anchor_weekday"].to_numpy(dtype=np.float64)
    dp_anchor = df["anchor_daypart"].to_numpy(dtype=np.int64)
    dp_last = df["last_fg_daypart"].to_numpy(dtype=np.int64)
    loc = df["last_fg_loc_id"].to_numpy(dtype=np.int64)

    V_m = markov_probs.shape[0]
    V_a = markov_probs.shape[1]
    last = np.clip(df["last_fg_app_idx"].to_numpy(dtype=np.int64), 0, V_m - 1)
    app = np.clip(df["app_idx"].to_numpy(dtype=np.int64), 0, V_a - 1)
    markov_col = markov_probs[last, app].astype(np.float32)

    hr_idx = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_col = hour_freq_g[hr_idx, app].astype(np.float32) if False else hour_freq[hr_idx, app].astype(np.float32)

    cols = [
        np.log1p(tib),
        np.log1p(tsf),
        np.log1p(fg_ct),
        np.clip(tib / (6 * 3600), 0, 2),
        np.sin(2 * np.pi * h24_arr(h) / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * w / 7.0),
        np.cos(2 * np.pi * w / 7.0),
        np.log1p(bg_sz),
        (w >= 5).astype(np.float64),
        (loc > 0).astype(np.float64),
        (dp_anchor == dp_last).astype(np.float64),
        markov_col.astype(np.float64),
        hour_col.astype(np.float64),
    ]
    feats = np.stack(cols, axis=1).astype(np.float32)

    app_idx = df["app_idx"].to_numpy(dtype=np.int64)
    cat_idx = app_to_cat[np.clip(app_idx, 0, len(app_to_cat) - 1)].astype(np.int64)
    return {
        "features": feats,
        "app_idx": app_idx,
        "cat_idx": cat_idx,
        "y_5": df["y_300"].to_numpy(dtype=np.int64),
        "y_10": df["y_600"].to_numpy(dtype=np.int64),
        "anchor_id": df["anchor_id"].to_numpy(dtype=np.int64),
    }


def h24_arr(x):
    return x


class PairDataset(Dataset):
    def __init__(self, d: dict):
        self.feat = torch.as_tensor(d["features"], dtype=torch.float32)
        self.app = torch.as_tensor(d["app_idx"], dtype=torch.long)
        self.cat = torch.as_tensor(d["cat_idx"], dtype=torch.long)
        self.y5 = torch.as_tensor(d["y_5"], dtype=torch.float32)
        self.y10 = torch.as_tensor(d["y_10"], dtype=torch.float32)

    def __len__(self):
        return int(self.feat.shape[0])

    def __getitem__(self, i):
        return {
            "features": self.feat(i) if False else self.feat[i],
            "app_idx": self.app[i],
            "cat_idx": self.cat[i],
            "y_5": self.y5[i],
            "y_10": self.y10[i],
        }


def evaluate(model, dataset_df: pd.DataFrame, split_d: dict) -> dict:
    """Return a merged df with score_model_5 / score_model_10 and the per-anchor metrics."""
    model.eval()
    loader = DataLoader(
        TensorDataset_from_dict(split_d),
        batch_size=1024, shuffle=False,
    )
    all_p5, all_p10 = [], []
    with torch.no_grad():
        for b in loader:
            out = model(b["app_idx"], b["cat_idx"], b["features"])
            all_p5.append(torch.sigmoid(out["logit_5"]).cpu().numpy())
            all_p10.append(torch.sigmoid(out["logit_10"]).cpu().numpy())
    p5 = np.concatenate(all_p5)
    p10 = np.concatenate(all_p10)
    merged = dataset_df.copy()
    merged["score_model_5"] = 1.0 - p5
    merged["score_model_10"] = 1.0 - p10
    out = {
        "H_300": MET.compute_metrics(merged, "score_model_5", "y_300", r_values=R_SWEEP),
        "H_600": MET.compute_metrics(merged, "score_model_10", "y_600", r_values=R_SWEEP),
    }
    return out, merged


class PairDS(Dataset):
    def __init__(self, d):
        self.f = torch.as_tensor(d["features"], dtype=torch.float32)
        self.ai = torch.as_tensor(d["app_idx"], dtype=torch.long)
        self.ci = torch.as_tensor(d["cat_idx"], dtype=torch.long)
        self.y5 = torch.as_tensor(d["y_5"], dtype=torch.float32)
        self.y10 = torch.as_tensor(d["y_10"], dtype=torch.float32)

    def __len__(self): return int(self.f.shape[0])
    def __getitem__(self, i):
        return {"features": self.f[i], "app_idx": self.ai[i], "cat_idx": self.ci[i],
                "y_5": self.y5[i], "y_10": self.y10[i]}


def TensorDataset_from_dict(d):
    return PairDS(d)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tag", default="C1")
    parser.add_argument("--epochs", type=int, default=25)
    parser.add_argument("--d-hidden", type=int, default=64)
    parser.add_argument("--lr", type=float, default=LR)
    parser.add_argument("--pos-weight-5", type=float, default=-1.0, help="<0 => computed from train")
    parser.add_argument("--pos-weight-10", type=float, default=-1.0)
    args = parser.parse_args()

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    art = ROOT / "artifacts"
    bg_art = art / "bg"
    with open(art / "vocab.json") as f:
        vocab = json.load(f)

    # Train-only tables
    train_raw = pd.read_parquet(art / "splits" / "train.parquet")
    hour_freq = fit_hour_freq(train_raw, vocab)
    mk_stats = MK.load_markov_prior(art / "v3" / "markov_prior.pkl")
    markov_probs = mk_stats["probs"].astype(np.float32)
    app_to_cat = CAT.build_app_to_cat_idx(vocab)

    bg_train = pd.read_parquet(bg_art / "splits" / "bg_train.parquet")
    bg_val = pd.read_parquet(bg_art / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_art / "splits" / "bg_test.parquet")

    build = lambda df: build_features_v2(df, hour_freq, markov_probs, app_to_cat)
    d_tr = build(bg_train)
    d_va = build(bg_val)
    d_te = build(bg_test)

    # pos_weight for imbalance
    p_tr_5 = float(d_tr_y_5 := d_tr["y_5"].mean())
    p_tr_10 = float(d_tr_y_10 := d_tr["y_10"].mean())
    pw5 = torch.tensor([(1 - p_tr_5) / max(1e-6, p_tr_5)]) if args.pos_weight_5 < 0 else torch.tensor([args.pos_weight_5])
    pw10 = torch.tensor([(1 - p_tr_10) / max(1e-6, p_tr_10)]) if args.pos_weight_10 < 0 else torch.tensor([args.pos_weight_10])
    print(f"[task_c] train pos_rate_5 = {p_tr_5:.4f}  pos_rate_10 = {p_tr_10:.4f}")
    print(f"[task_c] pos_weight_5 = {pw5.item():.2f}  pos_weight_10 = {pw10.item():.2f}")

    cfg = BgMLPConfig(d_hidden=args.d_hidden)
    model = BgPairMLP(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"[task_c] params = {n_params}")

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=WD)
    dl_tr = DataLoader(TensorDataset_from_dict(d_tr), batch_size=BATCH, shuffle=True)

    best_pr5 = -1.0
    best_state = None
    patience = 0
    log = []
    t0 = time.time()
    for epoch in range(1, args.epochs + 1):
        model.train()
        tot, n = 0.0, 0
        for b in dl_tr:
            out = model(b["app_idx"], b["cat_idx"], b["features"])
            loss, _, _ = bce_dual(out, b["y_5"], b["y_10"], pos_weight_5=pw5, pos_weight_10=pw10)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += float(loss.item()) * len(b["y_5"])
            n += len(b["y_5"])

        val_metrics, _ = evaluate(model, bg_val, d_va)
        row = {
            "epoch": epoch,
            "train_loss": tot / max(1, n),
            "val_pr_auc_5": val_metrics["H_300"]["pr_auc_mean"],
            "val_fk5_0.5": val_metrics["H_300"]["false_kill_rate"]["0.5"],
            "val_pr_auc_10": val_metrics["H_600"]["pr_auc_mean"],
            "val_roc_auc_5": val_metrics["H_300"]["roc_auc_mean"],
        }
        log.append(row)
        print(f"  ep{epoch:02d}  loss={row['loss' if 'loss' in row else 'train_loss'] if False else row.get('train_loss', tot/max(1,n)):.4f}  val_PR@H5={row['val_pr_auc_5']:.4f}  val_FK5={row['val_fk5' if False else 'val_fk5_at_half' if False else 'val_fk5'] if False else row.get('val_fk5_0.5'):.4f}  val_ROC@H5={row.get('val_roc_auc_5', 0):.4f}")
        if val_metrics["H_300"]["pr_auc_mean"] > best_pr5 + 1e-6:
            best_pr5 = val_metrics["H_300"]["pr_auc_mean"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            patience = 0
        else:
            patience += 1
            if patience >= PATIENCE:
                print("  early stop")
                break

    # Restore best
    if best_state is not None:
        model.load_state_dict(best_state)

    # Eval on val + test
    val_m, val_df = evaluate(model, pd.read_parquet(bg_art / "splits" / "bg_val.parquet") if False else bg_val, d_va := d_va if False else build(bg_val))
    test_m, test_df = evaluate(model, bg_test, build(bg_test))

    # Persist results
    out_ckpt = bg_art / "checkpoints" / f"task_c_{args.tag}.pt"
    out_ckpt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "cfg": vars(cfg := BgMLPConfig(d_hidden=args.d_hidden))}, out_ckpt)
    print(f"[task_c] saved {out_ckpt}  elapsed {time.time() - t0:.1f}s")

    out_path = bg_art / "results" / f"task_c_{args.tag}.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump({"tag": args.tag, "n_params": n_params,
                   "train_log": log,
                   "val": val_m, "test": test_m,
                   "r_sweep": list(R_SWEEP)}, f, indent=2)
    print(f"[task_c] wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
