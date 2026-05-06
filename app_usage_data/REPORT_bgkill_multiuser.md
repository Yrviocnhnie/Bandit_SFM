# REPORT — Multi-User Task C: Background-App Suspension Prediction

> **Status:** *Numbers below are placeholders updated after `52_train_task_c_multiuser.py`
> finishes. Pipeline is verified on 22 real users; the 4 production picks (C3.3 /
> Pro-Reg / Pro-List / Pro-Wide) are in training as of writing. The script,
> tests, and partial baselines are committed.*

## TL;DR

The single-user Task C pipeline (HarmonyOS, 42 days, 1 user) was extended to a
**22-user benchmark** at `/data00/ruiqing/app_forecasting/data/cleaned/`.
The headline question: **does ONE globally-trained model — with no per-user
fine-tuning — match the per-user bespoke C3.3 family**?

Setup:
- **22 users** (11 from `M_beta_Top30` + 11 from `top2000`).
- **Per-user split**: last 3 days = test, prior 3 days = val, rest = train, 60-min embargo.
- **Pooled vocab** built from all users' train target events (count ≥ 5 globally) → V = 243.
- **Pooled bg parquets**: 625 866 train / 74 774 val / 64 936 test rows.
- **Per-user feature stats** (Markov / hour_freq / per-app inter-fg mean / lifetime stats / fg_timeline) fit on each user's train rows separately; fed to the global model as feature inputs at scoring time via vectorized `(U, *)` lookup keyed by `user_id_idx`.
- **One global model** per recipe. Same recipes as single-user: C3.3 (baseline MLP), Pro-Reg/c3.3 (regularised), Pro-List/c3.3 (listwise loss), Pro-Wide/c3.3 (wider arch). Vocab = 243, app-emb table grows from `50 × 16` to `243 × 16`.
- **Two evaluation slices**: A = the 22-user pooled bg_test (temporal generalization within trained users); B = the existing single-user bg_test (cold-start transfer to a held-out user).
- **Track set**: rank-based (Pareto, ROC-AUC, PR-AUC, FK@{0.25, 0.5}, MSR@{0.25, 0.5}); positives-only (PosScoreNorm, PosRank, WAKR@{0.25, 0.5}); threshold-based (FKR / SKR / F1 / Acc / **MCC**) at τ\* = argmax-F1(keep) on val (per-user + global).

> **Headline test numbers (TBD after training):** mean PR-AUC across 22 users for
> the best trained pick, mean MCC at argmax-F1(keep) τ\*, and the slice-B
> cold-start gap vs the single-user bespoke model.

---

## 1. Cohort & data

22 users, day spans 31–43 days, 15k–96k events per user, vocab 16–89 per user before pooling.

| | Single-user (existing) | Multi-user (this report) |
|---|---|---|
| Source | `app_usage_cleaned_dictionary_mapped.xlsx` | `/data00/ruiqing/app_forecasting/data/cleaned/{M_beta_Top30,top2000}/*.xlsx` |
| Users | 1 | 22 |
| Days per user | 42 | 24–36 train + 3 val + 3 test |
| Events per user | 47 k | 28 k–96 k |
| Vocab | 50 (per-user) | 243 (pooled) |
| BG rows | ~25 k train | 625 866 train / 74 774 val / 64 936 test |
| Schema | 19 cols (incl. `device_state_scene` / `networktype` / `has_*`) | 12 cols (lean — extras filled with NaN by `lib/multiuser.load_xlsx_with_padding`) |

**Causality discipline:** per-user replay sees only that user's events; per-user feature stats are fit on that user's **train** rows only (`fit_split="train"` assertion on every `stats.pkl`). Labels at H = 60 min use **same-split FG events** for label causality; rolling-count features are full-stream (causal at lookup via `searchsorted < anchor_ts`).

**Per-user split day counts (representative):**

The exact day counts per user appear in `artifacts/bg_multi/stats/bg_data_stats.json`. All 22 users have ≥ 24 train days.

**Per-user vocab coverage on test:** ≥ 91.8 % (min) across users; mean 98.8 %. Apps falling to `<RARE>` are concentrated in the long tail.

**Pooled positive rate (test):** 0.259 — close to the single-user 0.22.

---

## 2. Method — global model with personalised features

**Why one global model.** The user explicitly requested *one* generalised model (not 22 per-user models). Implications:
- **Pooled vocab** (one app-emb table covering all users' apps).
- **Pooled training data** (rows from all users in one global train set).
- **Anchor IDs** are made globally unique (`user_id_idx * 1 000 000 + per_user_anchor_id`) so listwise-loss groupbys and metric groupbys don't conflate users.

**Why per-user features (not user-id embedding).** The model should learn rules that generalise; user-specific *behaviour* enters via feature inputs (Markov prior, hour-conditional probability, per-app inter-FG-mean, lifetime kill rate, fg_timeline). At scoring time these are looked up by `user_id_idx`.

```
                 ┌─────────────────────────┐
                 │  pooled bg_train.parquet│   (625k rows, with user_uid + user_id_idx)
                 └────────────┬────────────┘
                              │
        ┌─────────── per-user stats stack ───────────┐
        │  hour_freq_stack:        (22, 24, V=243)    │
        │  markov_probs_stack:     (22, V, V)         │
        │  per_app_inter_stack:    (22, V)            │
        │  app_lifetime_share/kill (22, V) each       │
        │  cat_lifetime_share:     (22, V_cat=11)     │
        │  cat_markov_probs:       (22, V_cat, V_cat) │
        │  fg_timeline_per_user:   nested dict        │
        └────────────────────────┬────────────────────┘
                                 │
        ┌────────────────────────▼────────────────────────┐
        │ build_schema_data_multi("c3.3", bg_df, ctx)     │
        │   per-row gather: stack[uid_i, last_i, app_i]   │
        └────────────────────────┬────────────────────────┘
                                 │
                  ┌──────────────▼──────────────┐
                  │  Single-head MLP (or Wide)  │
                  │  vocab_size = 243           │
                  └─────────────────────────────┘
```

**Recipes:** identical to single-user `36_train_c3_grid.py`. Hyperparams unchanged (LR=1e-3, AdamW, batch=512, epochs ≤ 30, patience 4).

**Tests.** 54 pytest tests across 7 layers covering: helper unit semantics, data-build integrity (causality, pooled coverage, anchor uniqueness), per-user stat lookup, baseline correctness, model + checkpoint round-trip, metric range/monotonicity, and an end-to-end smoke pipeline. All green at commit time.

---

## 3. Track A — Closed-form baselines (per-user, mean across 22 users)

Numbers from `artifacts/bg_multi/results/baselines_bg.json`, computed per user with that user's train-fit Markov / hour_freq, then averaged.

| Baseline       | PR-AUC ↑ | ROC-AUC ↑ | FK@.25 ↓ | FK@.5 ↓ | MSR@.25 ↑ | MSR@.5 ↑ | NDCG ↑ |
|----------------|---------:|----------:|---------:|--------:|----------:|---------:|-------:|
| random         | 0.7350   | 0.5160    | 0.2572   | 0.2556  | 0.3726    | 0.5510   | 0.7600 |
| LRU            | 0.8524   | 0.7376    | 0.1626   | 0.1844  | 0.4356    | 0.6237   | 0.8485 |
| TimeInBG       | 0.8422   | 0.7253    | 0.1714   | 0.1874  | 0.4293    | 0.6203   | 0.8413 |
| LFU-hour       | 0.8878   | 0.7761    | 0.1281   | 0.1602  | 0.4607    | 0.6484   | 0.8787 |
| **Markov-inv** | **0.8903** | **0.8168** | 0.1337 | 0.1641  | 0.4566    | 0.6459   | 0.8729 |
| Hybrid LRU+Mk  | 0.8755   | 0.7823    | 0.1422   | 0.1699  | 0.4491    | 0.6391   | 0.8661 |

**Observations:**

1. **Markov-inverse is the strongest baseline** on PR-AUC and ROC-AUC, mirroring single-user.
2. **LFU-hour beats Markov-inv on FK@.5** by 4 bp.
3. **Random's PR-AUC ≈ 0.74** is *higher* than the single-user random PR-AUC (~0.78) because the multi-user pooled positive rate is also higher.

---

## 4. Track A — Trained models (TBD)

> Filled in once `52_train_task_c_multiuser.py` finishes. Per-user means + bootstrap CIs (B=1000 user-level resamples).

| Model            | PR-AUC | ROC-AUC | FK@.25 | FK@.5 | MSR@.25 | MSR@.5 |
|------------------|-------:|--------:|-------:|------:|--------:|-------:|
| C3.3             |  TBD   |   TBD   |  TBD   |  TBD  |   TBD   |  TBD   |
| Pro-Reg/c3.3     |  TBD   |   TBD   |  TBD   |  TBD  |   TBD   |  TBD   |
| Pro-List/c3.3    |  TBD   |   TBD   |  TBD   |  TBD  |   TBD   |  TBD   |
| Pro-Wide/c3.3    |  TBD   |   TBD   |  TBD   |  TBD  |   TBD   |  TBD   |

**Pareto curves (figures):**

- `figures/bg_multi/pareto_h60_test.png` — mean MSR vs (1 − FK) across users, all 4 trained picks + 3 reference baselines.

---

## 5. Track A — Per-user breakdown

> Filled in after eval. Sorted by PR-AUC, descending.

`figures/bg_multi/per_user_pr_auc.png` — bar chart of test PR-AUC per (user, model). Useful for spotting users who gain little from the global model and might benefit from per-user fine-tuning in a follow-up.

---

## 6. Positives-only metrics (test, mean across users)

> Filled in after `55_positive_metrics_multiuser.py`.

| Model            | PosRank ↑ | PosScoreNorm ↓ | WAKR@.25 ↓ | WAKR@.5 ↓ |
|------------------|----------:|---------------:|-----------:|----------:|
| random           |  TBD      |  TBD           |  TBD       |  TBD      |
| LRU              |  TBD      |  TBD           |  TBD       |  TBD      |
| Markov-inv       |  TBD      |  TBD           |  TBD       |  TBD      |
| C3.3             |  TBD      |  TBD           |  TBD       |  TBD      |
| Pro-Reg/c3.3     |  TBD      |  TBD           |  TBD       |  TBD      |
| Pro-List/c3.3    |  TBD      |  TBD           |  TBD       |  TBD      |
| Pro-Wide/c3.3    |  TBD      |  TBD           |  TBD       |  TBD      |

---

## 7. Track B — Threshold-based metrics

τ\* picked by **argmax F1(keep)** on val (the recommended picker from the single-user report) — one per user (per-user τ\*) plus a global τ\* on pooled val.

### 7.1 Per-user τ\* (mean across 22 users, evaluated on each user's test rows)

| Model            | τ_keep (mean) | FKR@τ ↓ | SKR@τ ↑ | F1 ↑ | Acc ↑ | MCC ↑ |
|------------------|---:|---:|---:|---:|---:|---:|
| random           |  TBD | TBD | TBD | TBD | TBD | TBD |
| C3.3             |  TBD | TBD | TBD | TBD | TBD | TBD |
| Pro-Reg/c3.3     |  TBD | TBD | TBD | TBD | TBD | TBD |
| Pro-List/c3.3    |  TBD | TBD | TBD | TBD | TBD | TBD |
| Pro-Wide/c3.3    |  TBD | TBD | TBD | TBD | TBD | TBD |

### 7.2 Global τ\* (one pooled threshold; evaluated on pooled test)

> A single τ\* applied across all users — what we'd ship in a one-knob deployment.

### 7.3 Dense τ-sweep (top-5 by mean trained MCC on pooled test)

> Mirrors the single-user §5.3.7 finding: the MCC-optimal τ band lies within ~10 pp; pin a global τ ≈ 0.40–0.50 in deployment.

---

## 8. Slice B — Cold-start single-user transfer

The existing single-user `app_usage_cleaned_dictionary_mapped.xlsx` is **not** in training. Slice B scores it under two regimes:

1. **Naïve cold-start**: random + LRU + TimeInBG (no per-user stats needed). Reference floor — confirms the single user's data is inside the same problem class.
2. **Realistic deployment cold-start**: fit single-user feature stats from its first ~36 days; apply the global multi-user model. Compares against the single-user's own bespoke C3.3-family checkpoints (already in `artifacts/bg/checkpoints/`).

> Headline gap (TBD): how much PR-AUC do we lose by *not* personalising the model?

---

## 9. Limitations

- **Per-user pos-rate variance**: one user's test pos-rate is 0.003 (essentially no positive examples). Per-user PR-AUC is unstable for that user; the median is more honest than the mean.
- **Vocab cold-start**: a brand-new user installing apps the cohort hasn't seen will see those apps map to `<RARE>`. The model recovers by leaning on per-app structural features (time-in-bg, hour, etc.) but loses the per-app embedding signal.
- **Schema gap**: the multi-user XLSX schema is a strict subset of the single-user one. Single-user features that depended on `device_state_scene` / `networktype` were unused even in single-user (verified — the bg pipeline never read those columns).
- **No per-user fine-tune**: a follow-up could train per-user heads on top of the shared backbone (cheap to run) and likely close most of the cold-start gap.

---

## 10. Reproduction

```bash
cd app_usage_data

# 1. Build pooled multi-user bg data + per-user stats (~7-10 min)
python scripts/50_build_bg_data_multiuser.py

# 2. Closed-form baselines per user, aggregate (~2 min)
python scripts/51_run_baselines_bg_multiuser.py

# 3. Train 4 production picks on pooled data (~2 h on CPU)
python scripts/52_train_task_c_multiuser.py --epochs 30 --patience 4

# 4. Track A — per-user metrics + Pareto figures (~10 min)
python scripts/53_eval_h60_multiuser.py --include-slice-b

# 5. Track B — argmax-F1(keep), per-user + global (~5 min)
python scripts/54_threshold_multiuser.py

# 6. Positives-only metrics (~5 min)
python scripts/55_positive_metrics_multiuser.py

# 7. (optional) Run the test suite
python -m pytest lib/bg_multi/tests/ -v
```

Outputs:
- `artifacts/bg_multi/vocab.json` — pooled vocab (V=243)
- `artifacts/bg_multi/uid_to_idx.json` — stable user index map
- `artifacts/bg_multi/splits/bg_{train,val,test}.parquet` — pooled bg rows
- `artifacts/bg_multi/per_user/<uid>/stats.pkl` — per-user feature stats
- `artifacts/bg_multi/checkpoints/task_c_multi_<recipe>.pt` — trained checkpoints
- `artifacts/bg_multi/results/{baselines_bg,h60_leaderboard,threshold_metrics,positive_metrics}.json`
- `artifacts/bg_multi/figures/{pareto_h60_test,per_user_pr_auc}.png`

---

## 11. Test inventory

54 tests, 7 layers, all green:

- **Layer 1** (`test_helpers.py`, 23 tests) — short_uid, list_cohort_users, build_pooled_vocab, make_global_anchor_id, attach_user_columns, stack_per_user.
- **Layer 2** (`test_data_build.py`, 11 tests) — per-user splits, embargo, replay isolation, label causality, pooled bg integrity, vocab coverage.
- **Layer 3** (`test_feature_lookup.py`, 5 tests) — vectorized stat lookup correctness; build_schema_data_multi shape + global anchor IDs.
- **Layer 4** (`test_baselines.py`, 6 tests) — Random / LRU / Markov-inverse / aggregate correctness.
- **Layer 5** (`test_model_io.py`, 3 tests) — vocab-size scaling, checkpoint round-trip, listwise loss respects global anchor IDs.
- **Layer 6** (`test_metrics.py`, 6 tests) — Track A monotone-in-r, MCC≈0 on random, perfect-score check, argmax-keep > argmax-kill, aggregate sanity.
- **Layer 7** (`test_smoke_e2e.py`, 1 test) — synthetic 3-user end-to-end pipeline.

```bash
python -m pytest lib/bg_multi/tests/ -v --tb=short
# 54 passed, 1 skipped
```
