"""Train ONE global Task C model on pooled multi-user bg data.

Same architectures + recipes as the single-user `36_train_c3_grid.py`, but:
  - data is pooled across 22 users (artifacts/bg_multi/splits/bg_*.parquet)
  - vocab is the pooled global vocab (artifacts/bg_multi/vocab.json, V≈243)
  - per-user feature stats are looked up by user_id_idx at feature-build time
  - listwise loss / metric groupbys use a global anchor id (user_id_idx * 1e6 + anchor_id)

Trains 4 production picks:
  - task_c_multi_c3p3          (C3.3 baseline MLP, dropout 0.2)
  - task_c_multi_c3pro_reg     (C3.3 + dropout 0.3 + label smoothing + cosine LR + SWA)
  - task_c_multi_c3pro_listwise (C3.3 + listwise softmax NLL aux loss)
  - task_c_multi_c3pro_wide    (C3.3 + wider arch — 32-d emb, 128→64 trunk, GELU)

Outputs:
  artifacts/bg_multi/checkpoints/task_c_multi_<tag>.pt
  artifacts/bg_multi/results/task_c_multi_<tag>.json
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg import metrics_bg as MET
from lib.bg_multi.helpers import (
    GLOBAL_ANCHOR_OFFSET, load_per_user_stats, make_global_anchor_id, stack_per_user,
)
from lib.v3 import categories as CAT


# ============================================================================
# Import single-user modules via importlib (mirrors 41_threshold_metrics.py)
# ============================================================================
def _load_module(name: str, path: Path):
    if name in sys.modules:
        return sys.modules[name]
    spec = importlib.util.spec_from_file_location(name, path)
    m = importlib.util.module_from_spec(spec)
    sys.modules[name] = m
    spec.loader.exec_module(m)
    return m


_TRAINER = _load_module("trainer_h60_for_train_multi", ROOT / "scripts" / "34_train_task_c_h60.py")
_GRID = _load_module("grid_for_train_multi", ROOT / "scripts" / "36_train_c3_grid.py")


Y_COL = "y_3600"
R_SWEEP = (0.1, 0.25, 0.5, 0.75, 0.9)


# ============================================================================
# Multi-user feature builders
# ============================================================================
def build_c2_multi(bg_df: pd.DataFrame, hour_freq_stack: np.ndarray,
                   markov_probs_stack: np.ndarray) -> np.ndarray:
    """Same as `_TRAINER.build_c2`, but Markov + hour_freq lookup is per-user.

    hour_freq_stack: (U, 24, V)
    markov_probs_stack: (U, V, V)

    Returns just the (N, num_features) feature tensor (no other keys).
    """
    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    n = len(bg_df)

    tib = np.clip(bg_df["time_in_bg_sec"].to_numpy(dtype=np.float64), 0, None)
    tsf = np.clip(bg_df["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
    fg_today = bg_df["fg_count_today"].to_numpy(dtype=np.float64)
    fg_1h = bg_df["fg_count_last_3600s"].to_numpy(dtype=np.float64)
    fg_6h = bg_df["fg_count_last_21600s"].to_numpy(dtype=np.float64)
    rec_rank = bg_df["recency_rank_in_bg"].to_numpy(dtype=np.float64)
    h = bg_df["anchor_hour"].to_numpy(dtype=np.float64)
    w = bg_df["anchor_weekday"].to_numpy(dtype=np.float64)
    dp_anchor = bg_df["anchor_daypart"].to_numpy(dtype=np.int64)
    dp_last = bg_df["last_fg_daypart"].to_numpy(dtype=np.int64)

    V = markov_probs_stack.shape[1]
    last = np.clip(bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64), 0, V - 1)
    app = np.clip(bg_df["app_idx"].to_numpy(dtype=np.int64), 0, V - 1)

    # Per-user gather: markov_probs_stack[uid_i, last_i, app_i]
    markov_col = markov_probs_stack[uid, last, app].astype(np.float64)
    hr_idx = np.clip(h.astype(np.int64) % 24, 0, 23)
    hour_col = hour_freq_stack[uid, hr_idx, app].astype(np.float64)

    bg_rec_min = np.clip(bg_df["bg_recency_min_sec"].to_numpy(dtype=np.float64), 0, None)
    bg_rec_norm = np.log1p(bg_rec_min) / np.log1p(6 * 3600.0)

    tso = bg_df["time_since_screen_on_sec"].to_numpy(dtype=np.float64)
    tso_clip = np.where(tso < 0, 3600.0, np.clip(tso, 0, 3600.0))
    tso_norm = np.log1p(tso_clip) / np.log1p(3600.0)

    pk_idx = bg_df["prev_killed_app_idx"].to_numpy(dtype=np.int64)
    pk_age = bg_df["prev_killed_age_sec"].to_numpy(dtype=np.float64)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    pk_match = ((pk_idx == a_idx) & (pk_idx > 0) & (pk_age >= 0)).astype(np.float64)

    cols = [
        np.log1p(tib),
        np.log1p(tsf),
        rec_rank,
        np.log1p(fg_today),
        np.log1p(fg_1h),
        np.log1p(fg_6h),
        np.sin(2 * np.pi * h / 24.0),
        np.cos(2 * np.pi * h / 24.0),
        np.sin(2 * np.pi * w / 7.0),
        np.cos(2 * np.pi * w / 7.0),
        (dp_anchor == dp_last).astype(np.float64),
        markov_col,
        hour_col,
        bg_rec_norm,
        tso_norm,
        pk_match,
    ]
    return np.stack(cols, axis=1).astype(np.float32)


def compute_c31_arrays_multi(bg_df: pd.DataFrame,
                              fg_timeline_per_user: dict[int, dict],
                              per_app_inter_stack: np.ndarray) -> dict[str, np.ndarray]:
    """Multi-user version of `_TRAINER.compute_c31_arrays`.

    Loops per user, slices bg_df, calls the single-user function, scatters back.

    fg_timeline_per_user: {uid_idx: {app_idx: sorted ts_ns ndarray}}
    per_app_inter_stack:  (U, V) per-user inter-FG mean
    """
    n = len(bg_df)
    out_overdue = np.zeros(n, dtype=np.float32)
    out_fg24 = np.zeros(n, dtype=np.float32)
    out_fg7d = np.zeros(n, dtype=np.float32)
    out_cnt24 = np.zeros(n, dtype=np.float32)
    out_cnt7d = np.zeros(n, dtype=np.float32)

    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    # iterate per user_id_idx to slice + dispatch
    for uid_idx in np.unique(uid):
        mask = uid == uid_idx
        if not mask.any():
            continue
        sub_idx = np.where(mask)[0]
        sub_df = bg_df.iloc[sub_idx].reset_index(drop=True)
        ftl = fg_timeline_per_user.get(int(uid_idx), {})
        per_app_inter = per_app_inter_stack[int(uid_idx)]
        extras = _TRAINER.compute_c31_arrays(sub_df, ftl, per_app_inter)
        out_overdue[sub_idx] = extras["overdue_ratio"]
        out_fg24[sub_idx] = extras["was_fg_24h_ago"]
        out_fg7d[sub_idx] = extras["was_fg_7d_ago"]
        out_cnt24[sub_idx] = extras["log_fg_count_last_24h"]
        out_cnt7d[sub_idx] = extras["log_fg_count_last_7d"]
    return {
        "overdue_ratio": out_overdue,
        "was_fg_24h_ago": out_fg24,
        "was_fg_7d_ago": out_fg7d,
        "log_fg_count_last_24h": out_cnt24,
        "log_fg_count_last_7d": out_cnt7d,
    }


def add_c32_features_multi(bg_df, app_to_cat,
                           app_lifetime_share_stack,    # (U, V)
                           app_lifetime_kill_rate_stack, # (U, V)
                           cat_lifetime_share_stack):    # (U, V_cat)
    """Same 9 columns as `_GRID.add_c32_features`, with per-user lookups for
    app_share / app_kill_rate / cat_share. The bg-set composition stats are
    intra-anchor and don't depend on per-user fits — reused via groupby."""
    n = len(bg_df)
    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    last_idx = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    cat = _GRID.cat_for(a_idx, app_to_cat)
    last_cat = _GRID.cat_for(last_idx, app_to_cat)

    cat_match = (cat == last_cat).astype(np.float32)
    is_sys = np.isin(cat, np.array(_GRID.SYSTEM_CAT_IDS, dtype=np.int64)).astype(np.float32)

    V = app_lifetime_share_stack.shape[1]
    a_clip = np.clip(a_idx, 0, V - 1)
    a_share = app_lifetime_share_stack[uid, a_clip].astype(np.float32)
    a_kill = app_lifetime_kill_rate_stack[uid, a_clip].astype(np.float32)
    V_cat = cat_lifetime_share_stack.shape[1]
    cat_clip = np.clip(cat, 0, V_cat - 1)
    c_share = cat_lifetime_share_stack[uid, cat_clip].astype(np.float32)

    # Bg-set composition (groupby per global anchor — uses pooled df's anchor groups)
    bg_recency_mean = np.zeros(n, dtype=np.float32)
    bg_recency_max = np.zeros(n, dtype=np.float32)
    bg_unique_cat_cnt = np.zeros(n, dtype=np.float32)
    df_local = bg_df.assign(_cat=cat, _row=np.arange(n))
    # Group by the pair (user_id_idx, anchor_id) so two users' anchor=0 don't merge.
    for _, sub in df_local.groupby(["user_id_idx", "anchor_id"], sort=False):
        rs = sub["_row"].to_numpy()
        tsf = np.clip(sub["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
        log_tsf = np.log1p(tsf)
        bg_recency_mean[rs] = float(log_tsf.mean())
        bg_recency_max[rs] = float(log_tsf.max())
        bg_unique_cat_cnt[rs] = float(np.unique(sub["_cat"].to_numpy()).size)

    log_bg_size = np.log1p(bg_df["bg_set_size"].to_numpy(dtype=np.float64)).astype(np.float32)

    return np.stack([
        cat_match, is_sys, a_share, a_kill, c_share,
        bg_recency_mean, bg_recency_max, bg_unique_cat_cnt, log_bg_size,
    ], axis=1).astype(np.float32)


def add_c33_cols_multi(bg_df, app_to_cat, cat_markov_probs_stack):
    """Same as `_GRID.add_c33_cols`, with per-user cat_markov lookup."""
    n = len(bg_df)
    uid = bg_df["user_id_idx"].to_numpy(dtype=np.int64)
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    last_idx = bg_df["last_fg_app_idx"].to_numpy(dtype=np.int64)
    V_cat = cat_markov_probs_stack.shape[1]
    cat = np.clip(_GRID.cat_for(a_idx, app_to_cat).astype(np.int64), 0, V_cat - 1)
    last_cat = np.clip(_GRID.cat_for(last_idx, app_to_cat).astype(np.int64), 0, V_cat - 1)
    cat_markov = cat_markov_probs_stack[uid, last_cat, cat].astype(np.float32)

    t_since_cat = np.zeros(n, dtype=np.float32)
    bg_in_cat = np.zeros(n, dtype=np.float32)
    df = bg_df.assign(_cat=cat, _row=np.arange(n))
    for _, sub in df.groupby(["user_id_idx", "anchor_id"], sort=False):
        cats_here = sub["_cat"].to_numpy()
        tsf = np.clip(sub["time_since_fg_sec"].to_numpy(dtype=np.float64), 0, None)
        rs = sub["_row"].to_numpy()
        for j, c in enumerate(cats_here):
            same = cats_here == c
            n_same = int(same.sum())
            if n_same > 0:
                t_min = float(tsf[same].min())
                t_since_cat[rs[j]] = float(np.log1p(t_min) / np.log1p(6 * 3600.0))
                bg_in_cat[rs[j]] = float(n_same - 1)
            else:
                t_since_cat[rs[j]] = 0.0
                bg_in_cat[rs[j]] = 0.0
    return np.stack([cat_markov, t_since_cat, bg_in_cat], axis=1).astype(np.float32)


def build_schema_data_multi(schema: str, bg_df: pd.DataFrame, ctx: dict) -> dict:
    """Multi-user equivalent of `_GRID.build_schema_data`. Currently supports c3.3."""
    if schema not in ("c2", "c3.1", "c3.2", "c3.3"):
        raise NotImplementedError(f"schema {schema!r} not supported in multi-user trainer")

    feats = build_c2_multi(bg_df, ctx["hour_freq_stack"], ctx["markov_probs_stack"])
    if schema != "c2":
        c31 = compute_c31_arrays_multi(
            bg_df, ctx["fg_timeline_per_user"], ctx["per_app_inter_stack"],
        )
        feats = np.concatenate([feats, np.stack([
            c31["overdue_ratio"], c31["was_fg_24h_ago"], c31["was_fg_7d_ago"],
            c31["log_fg_count_last_24h"], c31["log_fg_count_last_7d"],
        ], axis=1).astype(np.float32)], axis=1)
    if schema == "c3.1":
        return _wrap_features(feats, bg_df, ctx)
    feats = np.concatenate([feats, add_c32_features_multi(
        bg_df, ctx["app_to_cat"],
        ctx["app_lifetime_share_stack"],
        ctx["app_lifetime_kill_rate_stack"],
        ctx["cat_lifetime_share_stack"],
    )], axis=1)
    if schema == "c3.2":
        return _wrap_features(feats, bg_df, ctx)
    feats = np.concatenate([feats, add_c33_cols_multi(
        bg_df, ctx["app_to_cat"], ctx["cat_markov_probs_stack"],
    )], axis=1)
    return _wrap_features(feats, bg_df, ctx)


def _wrap_features(feats: np.ndarray, bg_df: pd.DataFrame, ctx: dict) -> dict:
    a_idx = bg_df["app_idx"].to_numpy(dtype=np.int64)
    cat_idx = ctx["app_to_cat"][np.clip(a_idx, 0, len(ctx["app_to_cat"]) - 1)].astype(np.int64)
    return {
        "features": feats,
        "app_idx": a_idx,
        "cat_idx": cat_idx,
        "y": bg_df[Y_COL].to_numpy(dtype=np.int64),
        # Global anchor id — collision-free across users (used by listwise loss).
        "anchor_id": make_global_anchor_id(bg_df).astype(np.int64),
    }


# ============================================================================
# Multi-user training context
# ============================================================================
def build_ctx_multi(art_dir: Path) -> dict:
    """Load pooled vocab + uid map, all per-user stats, and stack tensors."""
    bg_multi = art_dir / "bg_multi"
    with open(bg_multi / "vocab.json") as f:
        vocab = json.load(f)
    with open(bg_multi / "uid_to_idx.json") as f:
        uid_to_idx = json.load(f)

    # Load per-user stats
    per_user: dict[str, dict] = {}
    for uid in uid_to_idx.keys():
        p = bg_multi / "per_user" / uid / "stats.pkl"
        per_user[uid] = load_per_user_stats(p)

    app_to_cat = CAT.build_app_to_cat_idx(vocab)

    # Stack the regular per-user arrays
    hour_freq_stack             = stack_per_user(per_user, "hour_freq",              uid_to_idx)
    markov_probs_stack          = stack_per_user(per_user, "markov_probs",           uid_to_idx)
    per_app_inter_stack         = stack_per_user(per_user, "per_app_inter_fg_mean",  uid_to_idx)
    app_lifetime_share_stack    = stack_per_user(per_user, "app_lifetime_share",     uid_to_idx)
    app_lifetime_kill_rate_stack = stack_per_user(per_user, "app_lifetime_kill_rate", uid_to_idx)
    cat_lifetime_share_stack    = stack_per_user(per_user, "cat_lifetime_share",     uid_to_idx)
    cat_markov_probs_stack      = stack_per_user(per_user, "cat_markov_probs",       uid_to_idx)

    # fg_timeline is a per-user dict-of-arrays (variable shape) — keep as nested dict
    fg_timeline_per_user: dict[int, dict] = {}
    for uid, idx in uid_to_idx.items():
        fg_timeline_per_user[int(idx)] = per_user[uid]["fg_timeline"]

    return {
        "vocab": vocab,
        "uid_to_idx": uid_to_idx,
        "app_to_cat": app_to_cat,
        "hour_freq_stack": hour_freq_stack,
        "markov_probs_stack": markov_probs_stack,
        "per_app_inter_stack": per_app_inter_stack,
        "app_lifetime_share_stack": app_lifetime_share_stack,
        "app_lifetime_kill_rate_stack": app_lifetime_kill_rate_stack,
        "cat_lifetime_share_stack": cat_lifetime_share_stack,
        "cat_markov_probs_stack": cat_markov_probs_stack,
        "fg_timeline_per_user": fg_timeline_per_user,
    }


def evaluate_multi(model: torch.nn.Module, bg_df: pd.DataFrame, data_d: dict) -> dict:
    """Forward + per-(user, anchor) Track A metrics."""
    model.eval()
    f = torch.as_tensor(data_d["features"].copy(), dtype=torch.float32)
    a = torch.as_tensor(data_d["app_idx"].copy(), dtype=torch.long)
    probs = []
    with torch.no_grad():
        for s in range(0, f.shape[0], 4096):
            logits = model(a[s:s + 4096], f[s:s + 4096])
            probs.append(torch.sigmoid(logits).cpu().numpy())
    p = np.concatenate(probs)
    merged = bg_df.copy()
    merged["score_model"] = 1.0 - p
    # Use global anchor id so two users' anchor=k don't merge in groupby
    merged["_global_aid"] = make_global_anchor_id(merged)
    merged_for_metric = merged.rename(columns={"anchor_id": "_per_user_aid",
                                                 "_global_aid": "anchor_id"})
    return MET.compute_metrics(merged_for_metric, "score_model", Y_COL,
                                r_values=R_SWEEP)


# ============================================================================
# Recipe registry + main
# ============================================================================
RECIPES = {
    "c3p3":           {"schema": "c3.3", "wide": False, "dropout": 0.2,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0},
    "c3pro_reg":      {"schema": "c3.3", "wide": False, "dropout": 0.3,
                       "label_smooth": 0.05, "swa": True, "listwise_lambda": 0.0},
    "c3pro_listwise": {"schema": "c3.3", "wide": False, "dropout": 0.2,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.5},
    "c3pro_wide":     {"schema": "c3.3", "wide": True,  "dropout": 0.3,
                       "label_smooth": 0.0, "swa": False, "listwise_lambda": 0.0},
}


def make_model(recipe: dict, num_features: int, vocab_size: int) -> torch.nn.Module:
    if recipe["wide"]:
        return _GRID.make_wide_model(num_features=num_features, vocab_size=vocab_size,
                                      dropout=recipe["dropout"])
    return _GRID.make_baseline_model(num_features=num_features, vocab_size=vocab_size,
                                      dropout=recipe["dropout"])


def train_recipe(recipe_name: str, ctx: dict,
                 bg_train: pd.DataFrame, bg_val: pd.DataFrame, bg_test: pd.DataFrame,
                 epochs: int, patience: int, seed: int) -> dict:
    """Train one recipe end-to-end. Returns a result dict (without state_dict here)."""
    rec = RECIPES[recipe_name]
    schema = rec["schema"]
    print(f"[52] building features for schema={schema!r} ...")
    t0 = time.time()
    d_tr = build_schema_data_multi(schema, bg_train, ctx)
    d_va = build_schema_data_multi(schema, bg_val, ctx)
    d_te = build_schema_data_multi(schema, bg_test, ctx)
    print(f"[52]   feature build: {time.time() - t0:.1f}s   "
          f"feature_dim={d_tr['features'].shape[1]}  "
          f"train_rows={d_tr['features'].shape[0]:,}")

    # Use existing single-user train_one (it operates on the data_d dict + bg_df)
    print(f"[52] training recipe={recipe_name} ...")
    result = _GRID.train_one(
        d_tr, d_va, d_te, bg_val, bg_test, ctx["vocab"],
        tag=recipe_name,
        model_fn=lambda num_features, vocab_size: make_model(
            rec, num_features=num_features, vocab_size=vocab_size,
        ),
        listwise_lambda=rec["listwise_lambda"],
        label_smooth=rec["label_smooth"],
        epochs=epochs, patience=patience, dropout=rec["dropout"],
        swa=rec["swa"], seed=seed, log_prefix=f"[{recipe_name}] ",
    )
    # Replace the (single-user) val/test metrics with multi-user-grouped ones
    state = result.get("state_dict")
    model = make_model(rec, num_features=d_tr["features"].shape[1], vocab_size=len(ctx["vocab"]))
    if state is not None:
        model.load_state_dict(state, strict=False)
    result["val"] = evaluate_multi(model, bg_val, d_va)
    result["test"] = evaluate_multi(model, bg_test, d_te)
    return result


def save_run(recipe_name: str, result: dict, art_dir: Path):
    state = result.pop("state_dict", None)
    ckpt_dir = art_dir / "bg_multi" / "checkpoints"
    res_dir = art_dir / "bg_multi" / "results"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    res_dir.mkdir(parents=True, exist_ok=True)
    if state is not None:
        torch.save({"state_dict": state, "tag": recipe_name},
                   ckpt_dir / f"task_c_multi_{recipe_name}.pt")
    with open(res_dir / f"task_c_multi_{recipe_name}.json", "w") as f:
        json.dump({**result, "recipe": recipe_name}, f, indent=2)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--art-dir", type=str, default=str(ROOT / "artifacts"))
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--patience", type=int, default=4)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--recipes", type=str, default="all",
                        help='comma list of recipes or "all"; choices=' + ",".join(RECIPES))
    parser.add_argument("--smoke", action="store_true",
                        help="2-epoch run on a 30 %% sample for quick verification")
    args = parser.parse_args()

    art_dir = Path(args.art_dir)
    bg_multi = art_dir / "bg_multi"
    bg_train = pd.read_parquet(bg_multi / "splits" / "bg_train.parquet")
    bg_val = pd.read_parquet(bg_multi / "splits" / "bg_val.parquet")
    bg_test = pd.read_parquet(bg_multi / "splits" / "bg_test.parquet")
    print(f"[52] pooled rows: train={len(bg_train):,}  val={len(bg_val):,}  test={len(bg_test):,}")

    if args.smoke:
        print("[52] SMOKE — sampling 30 % of bg_train")
        bg_train = bg_train.sample(frac=0.3, random_state=7).reset_index(drop=True)
        args.epochs = 2

    print(f"[52] building multi-user ctx ...")
    ctx = build_ctx_multi(art_dir)
    print(f"[52]   V_pool={len(ctx['vocab'])}  "
          f"U={ctx['markov_probs_stack'].shape[0]}  "
          f"V_cat={ctx['cat_markov_probs_stack'].shape[1]}")

    if args.recipes == "all":
        recipes = list(RECIPES.keys())
    else:
        recipes = [r.strip() for r in args.recipes.split(",") if r.strip()]
    for r in recipes:
        if r not in RECIPES:
            print(f"[52] unknown recipe: {r}; skipping", file=sys.stderr)
            continue

    summary = {}
    for r in recipes:
        if r not in RECIPES:
            continue
        print(f"\n{'=' * 70}\n[52] === recipe: {r} ===\n{'=' * 70}")
        t0 = time.time()
        result = train_recipe(r, ctx, bg_train, bg_val, bg_test,
                              epochs=args.epochs, patience=args.patience, seed=args.seed)
        save_run(r, result, art_dir)
        summary[r] = {
            "elapsed_sec": time.time() - t0,
            "feature_dim": result.get("feature_dim"),
            "n_params": result.get("n_params"),
            "best_val_pr": result.get("best_val_pr"),
            "test_pr": float(result["test"]["pr_auc_mean"]),
            "test_roc": float(result["test"]["roc_auc_mean"]),
            "test_fk_05": float(result["test"]["false_kill_rate"]["0.5"]),
            "test_msr_05": float(result["test"]["memory_save_rate"]["0.5"]),
        }
        print(f"[52] recipe={r} done in {summary[r]['elapsed_sec']:.0f}s  "
              f"test PR={summary[r]['test_pr']:.4f}  ROC={summary[r]['test_roc']:.4f}  "
              f"FK@.5={summary[r]['test_fk_05']:.4f}  MSR@.5={summary[r]['test_msr_05']:.4f}")

    with open(bg_multi / "results" / "task_c_multi_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\n[52] DONE  summary -> {bg_multi / 'results' / 'task_c_multi_summary.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
