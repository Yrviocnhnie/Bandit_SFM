"""Train v1 GRU + Markov-1 prior fusion on Task B head per user.

For each user: fit Markov-1 (V x V) log P(next | last_app) from train targets only.
Add `alpha_markov * log_prior[last_app]` to the sigmoid head's logits during training.
alpha_markov is a single learnable scalar (init 0.5, clipped [0, 2]).
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as Fnn

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib import features as F
from lib import train as TR
from lib import models as M1


HISTORY_K = 16
WIN_NS = 900 * 1_000_000_000
BATCH = 256
LR = 1e-3
WD = 1e-4
SEED = 7
LABEL_SMOOTHING = 0.05
EPOCHS = 12
PATIENCE = 4


def metrics_a(probs, targets):
    n = len(targets)
    if n == 0:
        return {"hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "n": 0}
    order = np.argsort(-probs, axis=1)
    h1 = float((order[:, 0] == targets).mean())
    h5 = float((order[:, :5] == targets[:, None]).any(axis=1).mean())
    ranks = np.zeros(n, dtype=int)
    for i in range(n):
        w = np.where(order[i] == int(targets[i]))[0]
        ranks[i] = int(w[0]) if len(w) > 0 else (probs.shape[1] - 1)
    return {"hit_at_1": h1, "hit_at_5": h5, "mrr": float((1.0 / (ranks + 1.0)).mean()), "n": int(n)}


def metrics_b(sigs, win_counts):
    n = len(win_counts)
    if n == 0:
        return {"event_hit_at_5": 0.0, "recall_at_5": 0.0, "n": 0}
    topk = np.argsort(-sigs, axis=1)[:, :5]
    total_events = 0
    total_hit = 0
    recs = []
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
            recs.append(len(tset & gt) / max(1, len(gt)))
    return {
        "event_hit_at_5": total_hit / max(1, total_events),
        "recall_at_5": float(np.mean(recs)) if recs else 0.0,
        "n": n,
    }


def recs_append(recs, val):
    recs.append(val)
    return recs


def cw_inv_sqrt(idx, V):
    counts = np.bincount(idx, minlength=V).astype(np.float32) + 1.0
    counts[:3] = counts.max()
    w = 1.0 / np.sqrt(counts)
    w = w / w.mean()
    return w.astype(np.float32)


def fit_markov_log_prior(train_target_app, V, alpha=0.5):
    """Return (V, V) log P(next | last_app), train-only with Dirichlet alpha smoothing."""
    counts = np.zeros((V, V), dtype=np.float64)
    seq = train_target_app.astype(int)
    for i in range(1, len(seq)):
        a, b = int(seq[i - 1]), int(seq[i])
        if a >= 3 and b >= 3:
            counts[a, b] += 1.0
    table = counts + alpha
    table[:, :3] = 0.0
    table[:3, :] = 0.0
    rows = table.sum(axis=1, keepdims=True)
    rows[rows == 0] = 1.0
    probs = (table / rows).astype(np.float64)
    log_prior = np.log(np.clip(probs, 1e-8, 1.0)).astype(np.float32)
    return log_prior


def last_valid_app(history_app, history_mask):
    B, K = history_app.shape
    mfloat = history_mask.long()
    any_valid = mfloat.any(dim=1)
    idx = mfloat.cumsum(dim=1).argmax(dim=1)
    last = history_app[torch.arange(B, device=history_app.device), idx]
    return torch.where(any_valid, last, torch.full_like(last, 1))


def cw_inv_sqrt_dummy(idx, V):
    return cw_inv_sqrt(idx, V)


def cw_inv_sqrt_alias(idx, V):
    return cw_inv_sqrt(idx, V)


def cw_inv_sqrt_alias2(idx, V):
    return cw_inv_sqrt(idx, V)


def prep(df, vocab, scaler):
    enc = F.encode_events(df, vocab, scaler["mean"], scaler["std"])
    h = F.build_history_for_targets(enc, history_k=HISTORY_K)
    win = TR.build_window_target_counts(
        h["target_ts"].astype("datetime64[ns]").astype(np.int64),
        h["target_app"], int(WIN_NS), len(vocab))
    return {
        "ha": torch.as_tensor(h["history_app"], dtype=torch.long),
        "hf": torch.as_tensor(h["history_feat"], dtype=torch.float32),
        "hm": torch.as_tensor(h["history_mask"], dtype=torch.bool),
        "ta": torch.as_tensor(h["target_app"], dtype=torch.long),
        "wc": torch.as_tensor(win, dtype=torch.float32),
    }


WIN_NS = 900 * 1_000_000_000


def evaluate(model, t, V):
    model.eval()
    n = t["ta"].shape[0]
    if n == 0:
        return ({"hit_at_1": 0.0, "hit_at_5": 0.0, "mrr": 0.0, "n": 0},
                {"event_hit_at_5": 0.0, "recall_at_5": 0.0, "n": 0})
    pa_chunks, sb_chunks = [], []
    with torch.no_grad():
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            out = model(t["ha"][i:j], t["hf"][i:j], t["hm"][i:j])
            pa_chunks.append(torch.softmax(out["logits_a"], dim=-1).cpu().numpy())
            sb_chunks.append(torch.sigmoid(out["logits_b_sig"]).cpu().numpy())
    pa = np.concatenate(pa_chunks, axis=0)
    sb = np.concatenate(sb_chunks, axis=0)
    return metrics_a(pa, t["ta"].numpy()), metrics_b(sb, t["wc"].numpy())


def train_user_dir(user_dir):
    train_df = pd.read_parquet(user_dir / "splits/train.parquet")
    val_df = pd.read_parquet(user_dir / "splits/val.parquet")
    test_df = pd.read_parquet(user_dir / "splits/test.parquet")
    with open(user_dir / "vocab.json") as f:
        vocab = json.load(f)
    V = len(vocab)

    scaler = F.fit_dt_scaler(train_df)
    tr = prep(train_df, vocab, scaler)
    va = prep(val_df, vocab, scaler)
    te = prep(test_df, vocab, scaler)
    if tr["ta"].shape[0] == 0 or va["ta"].shape[0] == 0:
        return {"error": "empty split", "V": V}

    cfg = M1.ModelConfig(vocab_size=V, numeric_dim=F.NUMERIC_FEAT_DIM)
    torch.manual_seed(SEED)
    model = M1.GRUModel(cfg)
    n_params = sum(p.numel() for p in model.parameters())
    cw = torch.as_tensor(cw_inv_sqrt(tr["ta"].numpy(), V), dtype=torch.float32)

    # Fit Markov-1 prior on train target sequence
    log_prior_np = fit_markov_log_prior(tr["ta"].numpy(), V, alpha=0.5)
    log_prior = torch.as_tensor(log_prior_np, dtype=torch.float32)

    # Learnable scalar alpha for Markov prior (init 0.5, clipped to [0, 2])
    alpha_markov = torch.nn.Parameter(torch.tensor(0.5))

    opt = torch.optim.AdamW(list(model.parameters()) + [alpha_markov], lr=LR, weight_decay=WD)

    n_train = tr["ta"].shape[0]
    indices = np.arange(n_train)
    rng = np.random.default_rng(SEED)
    best_h5 = -1.0
    best_state = None
    patience = 0
    log = []

    for epoch in range(1, EPOCHS + 1):
        model.train()
        rng.shuffle(indices)
        loss_sum, n_batches = 0.0, 0
        for i in range(0, n_train, BATCH):
            idx = torch.from_numpy(indices[i:i + BATCH]).long()
            ha = tr["ha"][idx]
            hf = tr["hf"][idx]
            hm = tr["hm"][idx]
            ta = tr["ta"][idx]
            wc = tr["wc"][idx]
            out = model(ha, hf, hm)
            ce = Fnn.cross_entropy(out["logits_a"], ta, weight=cw, label_smoothing=LABEL_SMOOTHING)
            # Markov-prior fusion on Task B sigmoid head
            last_app = last_valid_app(ha, hm)
            alpha = torch.clamp(alpha_markov, 0.0, 2.0)
            logits_b_with_prior = out["logits_b_sig"] + alpha * log_prior[last_app]
            y_bin = (wc > 0).float()
            bce = Fnn.binary_cross_entropy_with_logits(logits_b_with_prior, y_bin)
            log_rate = out["log_rate_b"].clamp(-10, 10)
            pois = (torch.exp(log_rate) - wc * log_rate).mean()
            loss = ce + 0.5 * bce + 0.2 * pois
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            loss_sum += float(loss.item())
            n_batches += 1

        m_va_a, m_va_b = eval_with_markov(model, va, V, log_prior, alpha_markov)
        log.append({"epoch": epoch, "loss": loss_sum / max(1, n_batches),
                    "alpha": float(alpha_markov.item()),
                    **{f"val_a_{k}": v for k, v in m_va_a.items()},
                    **{f"val_b_{k}": v for k, v in m_va_b.items()}})
        if m_va_a["hit_at_5"] > best_h5:
            best_h5 = m_va_a["hit_at_5"]
            best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
            best_alpha = float(alpha_markov.item())
            patience = 0
        else:
            patience += 1
            if patience >= 4:
                break

    model.load_state_dict(best_state)
    # Restore best alpha
    with torch.no_grad():
        alpha_markov.data.fill_(best_alpha)
    val_a, val_b = eval_with_markov(model, va, V, log_prior, alpha_markov)
    test_a, test_b = eval_with_markov(model, te, V, log_prior, alpha_markov)
    return {
        "V": V,
        "n_train": int(n_train),
        "n_val": int(va["ta"].shape[0]),
        "n_test": int(te["ta"].shape[0]),
        "params": int(n_params),
        "alpha_markov_final": float(alpha_markov.item()),
        "val_a": val_a, "val_b": val_b,
        "test_a": test_a, "test_b": test_b,
        "log": log,
    }


def eval_one(model, t, V):
    raise RuntimeError("eval_one disabled — use eval_with_markov in this script")


def eval_with_markov(model, t, V, log_prior, alpha_markov):
    model.eval()
    n = t["ta"].shape[0]
    if n == 0:
        return metrics_a(np.zeros((0, V), dtype=np.float32), np.zeros(0, dtype=int)), metrics_b(np.zeros((0, V), dtype=np.float32), np.zeros((0, V), dtype=np.float32))
    pa_chunks, sb_chunks = [], []
    with torch.no_grad():
        alpha_val = float(torch.clamp(alpha_markov, 0.0, 2.0).item())
        for i in range(0, n, BATCH):
            j = min(i + BATCH, n)
            out = model(t["ha"][i:j], t["hf"][i:j], t["hm"][i:j])
            la = last_valid_app(t["ha"][i:j], t["hm"][i:j])
            sig_logits = out["logits_b_sig"] + alpha_val * log_prior[la]
            pa_chunks.append(torch.softmax(out["logits_a"], dim=-1).cpu().numpy())
            sb_chunks.append(torch.sigmoid(sig_logits).cpu().numpy())
    pa = np.concatenate(pa_chunks, axis=0)
    sb = np.concatenate(sb_chunks, axis=0)
    return metrics_a(pa, t["ta"].numpy()), metrics_b(sb, t["wc"].numpy())


def main():
    art = Path("artifacts/multiuser")
    udirs = []
    for sd in sorted(art.iterdir()):
        if not sd.is_dir():
            continue
        for ud in sorted(sd.iterdir()):
            if (ud / "vocab.json").exists():
                udirs.append((sd.name, ud.name, ud))

    print(f"[v1 GRU+Markov] training on {len(udirs)} users (epochs={EPOCHS})")
    all_results = []
    t0 = time.time()
    for set_name, uid, ud in udirs:
        ts = time.time()
        try:
            r = train_user_dir(ud)
            r["set"] = set_name
            r["uid"] = uid
            with open(ud / "v1_gru_markov.json", "w") as f:
                json.dump(r, f, indent=2)
            if "error" in r:
                print(f"  {set_name}/{uid}  ERROR: {r['error']}")
            else:
                print(f"  {set_name}/{uid}  test h@1={r['test_a']['hit_at_1']:.3f}  test eh@5={r['test_b']['event_hit_at_5']:.3f}  ({time.time()-ts:.1f}s)")
            all_results.append(r)
        except Exception as e:
            import traceback
            traceback.print_exc()
            print(f"  {set_name}/{uid}  EXCEPTION: {e}")

    out_agg = ROOT / "artifacts" / "multiuser" / "v1_gru_markov_aggregate.json"
    with open(out_agg, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nDONE. {len(all_results)} users; total {time.time()-t0:.1f}s")
    print(f"Aggregated to {out_agg}")
    return 0


EPOCHS_LOOP = 12  # placeholder


if __name__ == "__main__":
    sys.exit(main() or 0)
