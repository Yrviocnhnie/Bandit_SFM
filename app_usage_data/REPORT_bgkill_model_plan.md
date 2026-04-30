# Task C — Model Architecture Plan (H = 60 min)

End-to-end architecture plan for the next iteration of the background-suspension model. **Pinned to a single horizon (H = 60 min)** — the multi-horizon multi-task design has been simplified to one focal head plus optional auxiliary supervision.

Companion to:
- `REPORT_bgkill_data.md` — data pipeline (2 h staleness window, single-horizon labels)
- `REPORT_bgkill_features_review.md` — feature plan (~50 numeric + app_emb, cumulative C3.1 → C3.6 bundles)
- `REPORT_bgkill_v2.md` — current production model (C2)

---

## TL;DR

- **Focus:** single horizon **H = 60 min** (positive rate ~25 %, mild class imbalance — much easier than H=5's 4 %).
- **C1 and C2 re-trained on the new data** as learned baselines, then **six new model variants** (C3.1 → C3.6) each adding one feature bundle on top of C2-rerun. Plus a final **C3-Pro** with architectural improvements.
- **Architectural levers** (in priority): listwise loss → capacity expansion → set-attention over B(t) → auxiliary multi-horizon supervision → regularization → ensembling.
- **Expected ROC-AUC@H=60 ceiling:** ~0.92 (vs C2's expected ~0.85 at H=60).

---

## 1. Setup

| Quantity | Value |
|---|---|
| Training rows (2 h staleness) | 25,013 |
| Anchors with `\|B(t)\|` ≥ 1 | 6,164 |
| Mean `\|B(t)\|` | 4.06 |
| Headline horizon | **H = 60 min** |
| Positive rate at H=60 (train, 2 h window) | ~25.23 % |
| `pos_weight` for BCE | ~3.0 |
| Numeric features (full C3 schema) | ~50 |
| App embedding | 16 → 32-d (learnable) |
| Total positive label-instances | ~6,300 (vs C2's 1,312 at H=5) |
| Current C2 baseline at H=60 | ROC-AUC ≈ 0.85 (extrapolated; not yet trained on this horizon) |

The H=60 horizon means the **class imbalance is no longer the primary obstacle**. Multi-horizon supervision is now an *auxiliary* tool, not a necessity.

---

## 2. Two parallel tracks of model variants

### Track 1 — Feature ablations (legacy baselines + C3.1 through C3.6)

All variants in Track 1 share the **same MLP architecture** (single sigmoid head at H=60, no listwise loss, no set-attention). Each is **trained from scratch on the new 2 h-window, single-horizon (H=60) data**. Only the input feature schema changes across rows — this isolates the *features* lever from any architectural changes.

| Variant | Adds (cumulative) | What it tests |
|---|---|---|
| **C1 (re-run)** | original C1 schema (14 numeric + 16-d app_emb + 4-d cat_emb) | Legacy baseline at the new H=60 / 2 h-window setting |
| **C2 (re-run)** | original C2 schema (15 numeric + 16-d app_emb, no cat_emb) | Reference point — every C3.k bundle is a cumulative ablation against this |
| **C3.1** | + Hourly habit (`overdue_ratio`, `was_fg_24h_ago`, `was_fg_7d_ago`, `log_fg_count_last_24h`, `log_fg_count_last_7d`) | Periodicity / lifetime rhythm — **expected biggest lift at H=60** |
| **C3.2** | + Identity priors + BG-set composition | App-level priors and BG-set summary signals |
| **C3.3** | + Category-aware (`cat_markov_prob`, `time_since_category_last_used`, `bg_apps_in_same_category`) | Category-level transitions and competition |
| **C3.4** | + 2-step Markov (`markov_prob_2step`) | Higher-order context |
| **C3.5** | + Session-state (`session_elapsed`, `event_count`, `unique_apps`, `dom_cat`, `prev_fg_was_reuse`) | Active-session signals |
| **C3.6** | + Active-session intensity (sub-5-min event burst features) | Sub-minute activity (lowest priority at H=60) |

Stop adding bundles when val ROC-AUC@H=60 fails to improve by ≥ 0.5 pp. The feature stack at C3.k that produces the best validation lift is then taken as the **C3-final** feature schema.

#### Why retrain C1 and C2 instead of re-using the old numbers

Comparing the new C3.k variants against the *old* C1/C2 numbers (which were trained at H=5/10 on 6 h-window data) would conflate three independent changes — new data window, new horizon, new features. Retraining C1 and C2 on the new setup with a single H=60 head gives a clean reference point so the leaderboard isolates each effect:

- **C1 (re-run) → C2 (re-run)**: effect of the prior schema audit (drop wday, age_bg_over_6h, loc_match, log_bg_set_size, cat_emb).
- **C2 (re-run) → C3.k**: effect of each newly-added feature bundle.
- **C3-final → C3-Pro**: effect of architectural changes layered on the winning feature schema.

### Track 2 — Architectural innovations (C3-Pro)

Once C3-final is fixed, layer architectural improvements on top:

| Variant | Adds | Tests |
|---|---|---|
| **C3-final** | best feature stack from track 1 | Feature ceiling |
| **C3-Pro-A** | + listwise loss (per-anchor softmax NLL) | Ranking-aware training |
| **C3-Pro-B** | + capacity (32-d app_emb, 128→64 trunk, GELU, dropout 0.3) | More representation power |
| **C3-Pro-C** | + auxiliary supervision (predict `last_fg_app`, predict `is_reuse`) | Multi-task regularization |
| **C3-Pro-D** | + set-attention over apps in B(t) | Within-anchor interaction |
| **C3-Pro** (final) | best of A/B/C/D combined + 5-seed ensemble | Production candidate |

---

## 3. Architectural levers, ranked by leverage at H=60

### Lever 1: Listwise loss (highest leverage)

C2 trains *per-pair BCE* — every `(anchor, app)` row is i.i.d. But the OS evicts the **top-K within each anchor**, so what matters is *ranking quality*, not absolute calibration.

```
Loss = L_BCE(per pair) + λ · L_listwise(per anchor)

L_listwise(anchor): for each anchor with apps a_1, ..., a_K and true positives a*:
    softmax over [logit(a_1), ..., logit(a_K)]
    L = -Σ_{a∈a*} log P(softmax = a)
```

Implementation: per-anchor batching, padded `B(t)` to max length, masked softmax. **Largest single ROC-AUC lever** outside of features.

**Expected lift:** +1.5–3 pp over single-horizon BCE.

### Lever 2: Capacity expansion (cheap)

C2's trunk (31→64→32, ~5k params) is undersized for the C3 schema (~50 features + 16-d app_emb).

```
input:    32-d app_emb ⊕ ~50 numeric features ≈ 82-d
trunk:    Linear(82 → 128) → GELU → LayerNorm → Drop(0.3)
          Linear(128 → 64) → GELU → LayerNorm → Drop(0.3)
head:     Linear(64 → 1)  → sigmoid  → P(used in 60 min)
```

Changes: 16-d → 32-d app emb, 64-d → 128-d hidden, ReLU → GELU, dropout 0.2 → 0.3. ~25k params.

**Expected lift:** +0.5 to +1 pp.

### Lever 3: Set-attention over B(t) (medium effort)

Each anchor has up to ~10 apps in `B(t)`. Currently each app is encoded independently. A 1-layer self-attention encoder lets each app's representation incorporate context from all other apps in the same anchor.

```
After per-pair encoder:
  Pad anchor batches to max K_max (e.g. 10).
  1 transformer block: 4 heads × d=64, mask padded positions.
  Output: context-aware h_pair' (64-d).
```

Adds ~30k params. Lets the model reason about competition: "of these 5 apps, which is the most disposable?"

**Expected lift:** +1 to +2 pp.

### Lever 4: Auxiliary multi-horizon supervision

Instead of using H=5/10/30/60 as primary heads, use H=60 as the **only primary head** and add the others as **auxiliary** heads with small loss weights:

```
L = L_main(H=60) + 0.1 · L_aux(H=30) + 0.05 · L_aux(H=10) + 0.05 · L_aux(H=5)
```

The shorter horizons regularize the trunk and provide denser positive supervision (5-min head fires for 2.7% of rows; 60-min for 25%). The shorter horizons **don't** drive the prediction — they just give the trunk extra gradient signal.

**Expected lift:** +0.3 to +0.8 pp.

### Lever 5: Auxiliary "what's next" tasks (multi-task)

Use the same backbone to predict additional signals from the same row:
- `aux_pred_last_fg_app` — what was the last FG app? (V-way classification)
- `aux_pred_is_reuse` — will the next FG event be a reuse from B(t)? (binary anchor-level)
- `aux_pred_session_continues` — will the active session continue past anchor + 5 min? (binary)

Each adds a small head (~50 params) with a low loss weight (0.05–0.1). Forces the trunk to encode information beyond just "kill score".

**Expected lift:** +0.3 to +1 pp.

### Lever 6: Regularization

For ~25 k rows + ~50–80 k params:
- Dropout 0.3 on every hidden layer
- Weight decay 1e-3 (was 1e-4)
- Label smoothing ε = 0.05 on positive class
- AdamW + cosine LR schedule, 5 % warmup
- Gradient clip 1.0
- Stochastic Weight Averaging (SWA) over the last 25 % of training

**Expected lift:** +0.3 to +0.8 pp robustness.

### Lever 7: Ensembling

5-seed average. Free at inference (just sum logits).

**Expected lift:** +0.5 to +1 pp.

---

## 4. Recommended architecture: C3-Pro

```
INPUT (per-pair)
  app_idx  → embedding(50, 32)             [learnable identity]
  cat_idx  → embedding(11, 8)               [learnable category]
  ~50 numeric features                       [C3 schema, see features doc]
  ────────────────────────────
  total: ~90-d input

ENCODER (per-pair)
  Linear(90 → 128) → GELU → LayerNorm → Dropout(0.3)
  Linear(128 → 64) → GELU → LayerNorm → Dropout(0.3)
  → h_pair (64-d)

[OPTIONAL] SET-ATTENTION over B(t)
  pad anchor rows to max K=10
  1 layer × 4-head self-attention, d=64, FFN 128
  mask padded slots
  → h_pair_ctx (64-d)

PRIMARY HEAD (H=60)
  Linear(64 → 1)  → logit_60

[AUX HEADS — small loss weight]
  Linear(64 → V) → aux_pred_last_fg_app (cross-entropy)
  Linear(64 → 1) → aux_pred_is_reuse (BCE)

LOSS
  L = BCE(logit_60, y_3600, pos_weight ≈ 3.0)
    + λ_list · per_anchor_softmax_NLL(logit_60, anchor_id)
    + 0.1 · aux_pred_last_fg_app loss
    + 0.05 · aux_pred_is_reuse loss
```

**Total params (approx):**
- without set-attention: ~35 k
- with set-attention: ~75 k

Both trainable on CPU in 2–5 min.

---

## 5. Training recipe

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3, cosine schedule with 5 % warmup |
| Weight decay | 1e-3 |
| Batch size | 256 (per-pair) or 64 anchors (per-anchor for listwise loss) |
| Max epochs | 50 |
| Early-stop patience | 5 on val PR-AUC@H=60 |
| Dropout | 0.3 throughout |
| Label smoothing | ε = 0.05 on positives |
| Gradient clip | max-norm 1.0 |
| `pos_weight` | ~3.0 (computed from train positive rate) |
| Listwise loss weight λ | 0.5 |
| Aux loss weights | 0.05–0.10 each |
| SWA | last 25 % of epochs |
| Ensemble | 5 seeds (averaged logits) |
| Device | CPU (single user; full pipeline ~10 min) |

---

## 6. Phased rollout

Two passes — first pin down the feature schema, then layer architecture improvements on top.

### Pass A — feature ablation (C3.1 → C3.6)

Each row is **one trained model**, single sigmoid head at H=60, same MLP architecture as C2 (kept simple to isolate feature effect):

| Stage | Cumulative features | Effort | Expected ROC-AUC@H=60 |
|---|---|---|---|
| **C2 (re-run)** | C2 schema (15) | 30 min | baseline |
| **C3.1** | + hourly habit (5 features) | 30 min | +1.5 to +2.5 pp |
| **C3.2** | + identity & BG comp (8 features) | 30 min | +0.5 to +1 pp |
| **C3.3** | + category-aware (3 features) | 30 min | +0.3 to +0.8 pp |
| **C3.4** | + 2-step Markov (1 feature) | 30 min | +0 to +1 pp |
| **C3.5** | + session-state (5 features + 11-d one-hot) | 1 h | +0.3 to +0.8 pp |
| **C3.6** | + active-session intensity (5 features) | 30 min | +0 to +0.3 pp |

After C3.6 (or wherever val plateaus), pick the winning feature schema as **C3** for Tier B.

### Pass B — architectural innovations on top of C3

Each row uses C3 features (whichever stage won Pass A); architecture varies.

| Variant | What changes vs C3 | Effort | Expected lift |
|---|---|---|---|
| **C3** (baseline for Pass B) | 15+k feature MLP, single H=60 head | — | reference |
| **C3-Pro-Listwise** | + per-anchor softmax NLL (λ = 0.5) | 1 day | +1.5 to +3 pp |
| **C3-Pro-Wide** | + 32-d app_emb + 128→64 trunk + GELU | 0.5 day | +0.5 to +1 pp |
| **C3-Pro-Aux** | + auxiliary tasks (predict_last_fg, is_reuse) | 0.5 day | +0.3 to +1 pp |
| **C3-Pro-Set** | + set-attention over B(t) | 2 days | +1 to +2 pp |
| **C3-Pro-Reg** | + dropout 0.3, WD 1e-3, label smoothing, SWA, cosine LR | 0.5 day | +0.3 to +0.8 pp |
| **C3-Pro-Ensemble** | 5-seed ensemble of best of above | 0.5 day | +0.5 to +1 pp |
| **C3-Pro** (final) | all of the above combined | — | (final candidate) |

Stop a path if it doesn't lift val ROC-AUC@H=60 by ≥ 0.5 pp.

---

## 7. Loss function deep-dive

The combined loss for the most ambitious model (C3-Pro):

```
L = w_main · BCE_with_pos_weight(logit_60, y_3600)              [main head]
  + w_list · listwise_softmax_NLL(logit_60 grouped by anchor)   [ranking]
  + w_aux1 · BCE_aux(logit_30_aux, y_1800)                       [aux H=30]
  + w_aux2 · BCE_aux(logit_10_aux, y_600)                        [aux H=10]
  + w_aux3 · BCE_aux(logit_5_aux,  y_300)                        [aux H=5]
  + w_aux4 · CE(pred_last_fg_app, last_fg_app_idx)                [aux task]
  + w_aux5 · BCE(pred_is_reuse, is_reuse)                          [aux task]
```

Suggested weights:

| Term | Weight |
|---|---|
| `BCE(logit_60)` | 1.0 |
| `listwise_softmax_NLL` | 0.5 |
| `BCE(aux H=30)` | 0.1 |
| `BCE(aux H=10)` | 0.1 |
| `BCE(aux H=5)` | 0.05 |
| `CE(pred_last_fg_app)` | 0.1 |
| `BCE(pred_is_reuse)` | 0.05 |

(Tune via single-shot val sweep.)

The auxiliary multi-horizon (H=5/10/30) is *kept* but as *regularization*, not the main head — much simpler than the original 4-head monotonic design.

---

## 8. Expected results

If everything goes as planned:

| Model | Val ROC-AUC@H=60 (expected) |
|---|---|
| Random | ~0.50 (sanity floor) |
| LRU / TimeInBG | ~0.74 |
| LFU-hour | ~0.69 |
| Markov-inverse | ~0.78 |
| Hybrid LRU + Markov-inv | ~0.79 |
| C1 (re-run on H=60) | ~0.83 (estimate; legacy 14-feature schema) |
| C2 (re-run on H=60) | ~0.84 (estimate; current 15-feature schema) |
| C3.1–C3.6 (best) | ~0.86–0.88 |
| C3-Pro (final) | **~0.89–0.92** |

The H=60 task is structurally easier than H=5 (less imbalance, more positive supervision), so absolute ROC-AUC will be higher even before model improvements. The interesting comparison is the **lift over C2 retrained on H=60 data** — that quantifies the value of the feature + architecture work.

---

## 9. What I'd skip

| Option | Why skip |
|---|---|
| Mixture of Experts (per-app routing) | 50 vocab apps → too sparse; MoE can't recover its complexity. |
| Pretrained transformer backbone | Architectural gap is too wide for a 25k-row tabular task. |
| Reinforcement learning | Supervised signal is strong; deferred until label noise becomes the bottleneck. |
| Bayesian neural networks | Calibration via temperature scaling is 100× cheaper. |
| LSTM / GRU over event history | Sequence info is already encoded in time-since-X scalars. |
| Per-app interval distributions (variance, percentiles) | Diminishing returns at 25 k rows; mean is what's identifiable. |

---

## 10. Closed-form baselines

All six baselines are **retrained / re-fit on the new 2 h-window dataset** and **scored against the new H=60 label** (`y_3600`). They share the same val/test splits and metrics as every model variant. Train-only stats are guarded with `fit_split="train"`.

| Baseline | Score formula | Train-fit table | Notes |
|---|---|---|---|
| **Random** | `Uniform(0,1)` per row, seed=7 | none | Sanity floor; ROC-AUC should land ≈ 0.5. |
| **LRU** | `time_since_fg_sec` | none | Classic memory-manager heuristic. |
| **TimeInBG** | `time_in_bg_sec` | none | Near-duplicate of LRU; differs only on BG→FG→BG cycles. |
| **LFU-hour** | `1 − P(app \| anchor_hour)` (Dirichlet-smoothed) | 24×V table fit on train target events | Captures hourly habit. |
| **Markov-inverse** | `1 − P(app \| last_fg_app)` | V×V transition table, train target sequence | Strongest closed-form baseline at H=5 (v2). |
| **Hybrid LRU + Markov-inv** | `0.5·LRU_norm + 0.5·(1 − P_markov)_norm` (per-anchor min-max norm) | uses the Markov table | Tests whether the two strongest single signals combine. |

These six form the **closed-form floor** for the leaderboard. Each is a single line at score time; the only fitting needed is the HourMFU and Markov-1 tables on the new train split.

### Deferred (not in the immediate experiment matrix)

- **TaskB-inverse** using v5 E2 (`task_b_v4.pt`) — strongest available non-Task-C-trained baseline; requires the v3 anchor-time tensor builder (~4–6 h plumbing). Pin to follow-up after C3-Pro.
- **LightGBM** on C3 features — strong tabular comparator. Defer to "is our MLP architecture earning its keep?" later.
- **Stacked ensemble** of learned + closed-form scores (e.g. `0.6·C3-Pro + 0.3·LightGBM + 0.1·Markov-inverse`) — only meaningful once the individual leaderboard rows exist; free post-hoc lift typically ~+0.3–0.5 pp.

---

## 11. Files & code structure

Additive (no edits to existing v1/v2/v3/C1/C2):

- `lib/bg/models_bg.py::C3Model` — single-head MLP, configurable trunk size and embedding dims.
- `lib/bg/models_bg.py::C3ProModel` — adds set-attention layer (optional via flag), aux heads.
- `lib/bg/datasets_bg.py::AnchorBatchSampler` — yields per-anchor mini-batches for listwise loss.
- `lib/bg/losses.py` — `bce_with_pos_weight`, `listwise_softmax_nll`, `aux_loss_bundle`.
- `scripts/34_train_task_c_pro.py` — argparse-driven training (variant flags: `--feature-bundle C3.1..C3.6`, `--listwise`, `--wider`, `--set-attn`, `--aux-tasks`).
- `scripts/35_eval_task_c_pro.py` — eval matching `33_eval_task_c.py`'s metric set, single-horizon (H=60).

---

## 12. Summary

> **Plan:** drop multi-horizon multi-task in favour of a single H=60 head with optional auxiliary supervision. Run two parallel ablation tracks: feature bundles (C3.1 → C3.6) and architecture variants (C3-Pro family). Expected ROC-AUC@H=60 ceiling ~0.90 (vs C2's ~0.84 on the same data).
>
> **Highest-leverage levers:** (1) listwise loss, (2) the C3.1 hourly-habit feature bundle, (3) capacity expansion, (4) set-attention over B(t).
>
> **Quickest win:** train **C3.1** (just adds 5 hourly-habit features to C2) on the new 2 h-window H=60 data — likely +1.5 to +2.5 pp ROC-AUC for ~30 minutes of work.
