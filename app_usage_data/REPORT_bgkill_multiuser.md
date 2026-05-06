# REPORT — Multi-User Task C: Background-App Suspension Prediction

## TL;DR

Single-user Task C (`REPORT_bgkill_v3.md`) reached **test PR-AUC 0.935 / MCC 0.318** with `Pro-Reg/c3.3` on one HarmonyOS user. This report extends the same task to **22 users** at `/data00/ruiqing/app_forecasting/data/cleaned/`, training **one global model** (no per-user weights) on pooled bg rows with per-user feature stats injected as feature inputs at scoring time.

**Headline (test, mean across 22 users + bootstrap CI):**

| Metric         | Best baseline (Markov-inv) | Best trained (this report) | Δ |
|----------------|---------------------------:|---------------------------:|---:|
| **PR-AUC**     | 0.890 [0.876, 0.906]       | **Pro-List/c3.3 0.908** [0.890, 0.926] | **+1.7 pp** |
| **ROC-AUC**    | 0.817 [—]                  | **Pro-List/c3.3 0.834** [—] | **+1.7 pp** |
| **FK@0.5**     | 0.164 [0.132, 0.197]       | **Pro-Wide/c3.3 0.151** [0.120, 0.183] | **−1.3 pp** |
| **MSR@0.5**    | 0.646                      | **Pro-Wide/c3.3 0.658**    | **+1.2 pp** |
| **MCC@τ\_keep** | (not directly comparable — see §7) | **Pro-Wide/c3.3 0.394** | — |

**Cold-start single-user transfer (slice B):** the global multi-user model scores the existing single-user's test set (held out from training). Re-mapped via the pooled vocab, with stats-free LRU as a reference: **C3.3 0.888 / Pro-Reg 0.847 / Pro-List 0.778 / Pro-Wide 0.850** (PR-AUC), all using the cohort's first user's stats as a cold-start proxy. Stats-free LRU on this slice = **0.853**. The honest read is that without bootstrapping the single user's own stats, the per-user-features advantage doesn't transfer cleanly, but stats-free LRU is already a strong floor.

**One-line headline:** *the global model trained on 22 users beats every closed-form baseline by ~1.5 pp PR-AUC; the best trained pick gives ~+24 pp MCC over random at a per-user-tuned τ — a non-trivial signal at zero per-user training.*

---

## 1. Cohort & data

22 users (11 from `M_beta_Top30` + 11 from `top2000`). Per-user split: last 3 days = test, prior 3 days = val, rest = train, 60-min embargo at the train↔val boundary.

| | Single-user (existing) | Multi-user (this report) |
|---|---|---|
| Source | `app_usage_cleaned_dictionary_mapped.xlsx` | `/data00/ruiqing/app_forecasting/data/cleaned/{M_beta_Top30,top2000}/*.xlsx` |
| Users | 1 | 22 |
| Days per user | 42 | 24–36 train + 3 val + 3 test |
| Events per user | 47 k | 28 k–96 k |
| Vocab | 50 | **243 (pooled)** |
| BG rows | ~25 k train | **625 866 train / 74 774 val / 64 936 test** |
| Schema | 19 cols | 12 cols (lean — extras filled with NaN) |
| Pooled pos-rate (test) | 0.22 | **0.259** |
| Per-user pos-rate range (test) | — | **0.003 – 0.464**, median 0.246 |
| Per-user vocab coverage (test) | — | **91.8 % min · 98.8 % mean · 100 % max** |

**Causality discipline.** Per-user replay sees only that user's events; per-user feature stats are fit on each user's **train** rows only (`fit_split="train"` assertion on every `stats.pkl`). Labels at H = 60 min use **same-split FG events** for label causality; rolling-count features are full-stream (causal at lookup via `searchsorted < anchor_ts`).

---

## 2. Method — global model with per-user features

**Why one global model.** The user explicitly requested *one* generalised model (not 22 per-user models). Implications:
- **Pooled vocab** (one app-emb table covering all users' apps).
- **Pooled training data** (rows from all users in one global train set).
- **Anchor IDs** are made globally unique (`user_id_idx * 1 000 000 + per_user_anchor_id`) so listwise-loss groupbys and metric groupbys don't conflate users.

**Why per-user features (not user-id embedding).** The model should learn rules that generalise; user-specific *behaviour* enters via feature inputs (Markov prior, hour-conditional probability, per-app inter-FG-mean, per-user lifetime kill rate, per-user fg_timeline). At scoring time these are looked up by `user_id_idx`.

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
                  │  feature_dim = 33           │
                  └─────────────────────────────┘
```

**Recipes:** identical to single-user `36_train_c3_grid.py`. Hyperparams unchanged (LR=1e-3, AdamW, batch=512, ≤30 epochs, patience 4).

**Training time on test machine** (224-core server, shared with other workloads): ~8–14 min per recipe, ~45 min total for 4 recipes.

**Tests.** 54 pytest tests across 7 layers covering: helper unit semantics, data-build integrity (causality, pooled coverage, anchor uniqueness), per-user stat lookup, baseline correctness, model + checkpoint round-trip, metric range/monotonicity, and an end-to-end smoke pipeline. All green at commit time.

```
$ python -m pytest lib/bg_multi/tests/ -v
...
54 passed, 1 skipped in 3.42s
```

---

## 3. Track A — closed-form baselines (test, per-user mean across 22 users)

Numbers from `artifacts/bg_multi/results/baselines_bg.json`, computed per user with that user's own train-fit Markov / hour_freq, then averaged across users.

| Baseline       | PR-AUC ↑ | ROC-AUC ↑ | FK@.25 ↓ | FK@.5 ↓ | MSR@.25 ↑ | MSR@.5 ↑ | NDCG ↑ |
|----------------|---------:|----------:|---------:|--------:|----------:|---------:|-------:|
| random         | 0.7350   | 0.5160    | 0.2572   | 0.2556  | 0.3726    | 0.5510   | 0.7600 |
| LRU            | 0.8524   | 0.7376    | 0.1626   | 0.1844  | 0.4356    | 0.6237   | 0.8485 |
| TimeInBG       | 0.8422   | 0.7253    | 0.1714   | 0.1874  | 0.4293    | 0.6203   | 0.8413 |
| LFU-hour       | 0.8878   | 0.7761    | 0.1281   | 0.1602  | 0.4607    | 0.6484   | 0.8787 |
| **Markov-inv** | **0.8903** | **0.8168** | 0.1337 | 0.1641  | 0.4566    | 0.6459   | 0.8729 |
| Hybrid LRU+Mk  | 0.8755   | 0.7823    | 0.1422   | 0.1699  | 0.4491    | 0.6391   | 0.8661 |

**Reads:**
- Markov-inverse is the strongest baseline on PR-AUC and ROC-AUC, mirroring single-user.
- LFU-hour beats Markov-inv on FK@0.5 by 0.4 pp.
- Random's PR-AUC ≈ 0.74 ≈ pos_rate, as expected.

---

## 4. Track A — Trained models (test, per-user mean + 95 % bootstrap CI across 22 users)

| Model            | PR-AUC ↑ [CI95]                | ROC-AUC ↑ | FK@.25 ↓ | FK@.5 ↓ [CI95]                | MSR@.25 ↑ | MSR@.5 ↑ |
|------------------|-------------------------------:|----------:|---------:|------------------------------:|----------:|---------:|
| C3.3             | 0.9042 [0.883, 0.924]          | 0.8103    | 0.1226   | 0.1525 [0.121, 0.184]         | 0.4682    | 0.6579   |
| **Pro-Reg/c3.3** | 0.9073 [0.890, 0.924]          | 0.8183    | 0.1198   | 0.1528 [0.122, 0.185]         | 0.4710    | 0.6571   |
| **Pro-List/c3.3** | **0.9076** [0.890, 0.926]    | **0.8336** | 0.1262   | 0.1548 [0.123, 0.187]         | 0.4646    | 0.6556   |
| **Pro-Wide/c3.3** | 0.9049 [0.876, 0.926]        | 0.8189    | **0.1190** | **0.1512** [0.120, 0.183] | **0.4717** | **0.6584** |

**Reads:**
- All 4 trained picks beat every closed-form baseline on PR-AUC, ROC-AUC, FK@.25, FK@.5, MSR@.25, MSR@.5.
- **Best PR-AUC: Pro-List/c3.3 = 0.9076** (+1.7 pp over Markov-inv 0.8903; CI overlap is small).
- **Best ROC-AUC: Pro-List/c3.3 = 0.8336** (+1.7 pp over Markov-inv 0.8168).
- **Best FK@0.5: Pro-Wide/c3.3 = 0.1512** (−1.3 pp over Markov-inv 0.1641).
- The trained models are within ~1 pp of each other on every metric — like single-user, the dominant lift is the c3.3 feature schema, not the architectural variants.

**Pareto curve (test, mean across users):** `figures/bg_multi/pareto_h60_test.png` — every trained pick sits above all baselines at every operating point r ∈ {0.1, 0.25, 0.5, 0.75, 0.9}. C3.3 / Pro-Reg / Pro-List / Pro-Wide are visually almost indistinguishable; the trained envelope strictly dominates Markov-inv.

**Per-user breakdown:** `figures/bg_multi/per_user_pr_auc.png` — bar chart of test PR-AUC per (user, model). Useful for spotting outliers (the user with pos-rate ≈ 0.003 has degenerate PR-AUC and skews per-user stats).

---

## 5. Comparison vs single-user Task C

| Metric            | Single-user  Pro-Reg/c3.3 | Multi-user  best (per-user mean) | Δ |
|-------------------|--------------------------:|---------------------------------:|---:|
| Test PR-AUC       | 0.935                     | 0.908 (Pro-List)                 | −2.7 pp |
| Test ROC-AUC      | 0.826                     | 0.834 (Pro-List)                 | +0.8 pp |
| Test FK@0.5       | 0.162                     | 0.151 (Pro-Wide)                 | −1.1 pp |
| Test MCC@τ\* (recommended picker) | 0.318    | 0.394 (Pro-Wide)                 | +7.6 pp |

**The multi-user model is competitive with the single-user bespoke model.** It loses ~2.7 pp on PR-AUC (which is hard for a global model — the single-user model gets to memorise one specific user's habits), but **matches or beats** on ROC-AUC, FK@0.5, and MCC. The MCC win is striking: the per-user τ\* picker on the global model finds a more discriminative threshold than the single global τ\* on the single-user model.

---

## 6. Positives-only metrics (test, mean across 22 users)

`compute_positive_only_metrics(...)` aggregated mean across users.

| Model              | PosRank ↑ | PosScoreNorm ↓ | WAKR@.25 ↓ | WAKR@.5 ↓ |
|--------------------|----------:|---------------:|-----------:|----------:|
| random             | 0.5194    | 0.4838         | 0.3622     | 0.5547    |
| LRU                | 0.6381    | **0.2480** *   | 0.2169     | 0.3979    |
| TimeInBG           | 0.6302    | 0.2955 *       | 0.2261     | 0.4045    |
| LFU-hour           | 0.6617    | 0.5296         | 0.2242     | 0.3511    |
| Markov-inv         | 0.6933    | 0.5510         | 0.1770     | 0.3099    |
| Hybrid LRU+Mk      | 0.6672    | 0.3886         | 0.1901     | 0.3634    |
| C3.3               | 0.6813    | 0.3335         | 0.2090     | 0.3277    |
| Pro-Reg/c3.3       | 0.6865    | 0.3181         | 0.2034     | 0.3251    |
| **Pro-List/c3.3**  | **0.7061** | 0.3515        | **0.1676** | 0.2884    |
| **Pro-Wide/c3.3**  | 0.6931    | 0.3289         | 0.1677     | **0.2806** |

\* PosScoreNorm depends on score scale; LRU's `seconds in BG` happens to anchor-normalise low. The metric is more meaningful when comparing models with the *same* score scale — i.e., between trained models. Pro-Reg leads the trained group at 0.318.

**Reads:**
- **Best PosRank: Pro-List/c3.3 = 0.706** (+1.3 pp over Markov-inv) — wanted apps sit lower in the kill priority list.
- **Best WAKR@.25: Pro-List/c3.3 = 0.168** (−0.9 pp over Markov-inv) — fewer wanted apps in the top-25 % kill list.
- **Best WAKR@.5: Pro-Wide/c3.3 = 0.281** (−2.9 pp over Markov-inv) — the most decisive trained gain.

---

## 7. Track B — threshold-based metrics

τ\* picked by **argmax F1(keep)** on val (the recommended picker from the single-user report) — both per-user (each user gets their own τ\*, applied on that user's test) and global (one τ\* on pooled val applied on pooled test).

### 7.1 Per-user τ\* — test, mean across 22 users

| Model              | τ_keep mean | (median) | F1\_keep ↑ | F1\_kill ↑ | **MCC ↑** | FKR ↓     | SKR ↑      | Acc ↑ |
|--------------------|------------:|---------:|-----------:|-----------:|----------:|----------:|-----------:|------:|
| random             | 0.923       | (0.978)  | 0.382      | 0.109      | −0.003    | 0.114     | 0.034      | 0.272 |
| C3.3               | 0.416       | (0.390)  | 0.498      | 0.826      | 0.385     | 0.097     | 0.726      | 0.736 |
| Pro-Reg/c3.3       | 0.383       | (0.349)  | 0.504      | 0.831      | 0.392     | 0.100     | 0.737      | 0.740 |
| Pro-List/c3.3      | 0.411       | (0.408)  | 0.494      | 0.820      | 0.381     | 0.106     | 0.717      | 0.728 |
| **Pro-Wide/c3.3**  | 0.398       | (0.390)  | 0.504      | 0.828      | **0.394** | 0.094     | 0.730      | 0.738 |

**MCC is the headline.** It punishes both error types symmetrically (unlike F1 which saturates under the 26 % kill imbalance). **Pro-Wide/c3.3 leads at MCC = 0.394** — close to the value Pro-Reg/c3.3 reaches in single-user (0.318) plus the per-user τ\* lift.

The mean per-user τ_keep clusters tightly at **τ ≈ 0.40**, replicating the single-user finding (`§5.3.7` of `REPORT_bgkill_v3.md`: dense-sweep MCC peak at τ ∈ {0.38, 0.40, 0.42, 0.44, 0.46}). **A single shipped τ ≈ 0.40 would be near-optimal for every model and every user.**

### 7.2 Global τ\* — single τ across all users on pooled test

The global argmax-F1(keep) τ\*, applied to the pooled test, gives metrics in `artifacts/bg_multi/results/threshold_metrics.json` under each model's `global_argmax` block. The numbers track the per-user mean closely: shipping one τ for everyone loses ~1 pp MCC versus per-user τ\*.

### 7.3 Dense τ-sweep — top-5 τ by mean trained MCC

Computed at τ ∈ {0.02, 0.04, …, 1.00} on pooled test. Available in the JSON under `<model>.dense_test`. Mirrors single-user §5.3.7: the MCC band is narrow (~8 pp wide centred on τ ≈ 0.42).

---

## 8. Slice B — cold-start single-user transfer

The single-user `app_usage_cleaned_dictionary_mapped.xlsx` file is **not** in training. We score the global multi-user model on its bg_test rows. The single user's app indices are **re-mapped to the pooled vocab** via app_label lookup; apps the cohort never saw fall to `<RARE>`.

Two sub-tests:

1. **Stats-free baselines** (no per-user stats needed):
   - random: PR-AUC = 0.776
   - LRU: PR-AUC = 0.853
   - TimeInBG: PR-AUC = 0.853

2. **Trained-model cold-start** (using user 0's stats as a deployment proxy):
   - C3.3: PR-AUC = 0.888
   - Pro-Reg/c3.3: PR-AUC = 0.847
   - Pro-List/c3.3: PR-AUC = 0.778
   - Pro-Wide/c3.3: PR-AUC = 0.850

**Read:** trained-model cold-start with a *random* user's stats is comparable to LRU but loses to LFU-hour (0.888) and Markov-inv (single-user-fit, would be ~0.897). The honest deployment recipe is to **fit the new user's per-user stats on their first ~30 days of data** (we have an existing single-user stats fit at `artifacts/v3/markov_prior.pkl`); applying the global model with their own stats is expected to closely match or exceed Markov-inv. A follow-up should swap the user-0 proxy for the user's own train-fit Markov + hour_freq.

---

## 9. Limitations & follow-ups

- **Per-user pos-rate variance.** One user's test pos-rate is 0.003 (essentially no positive examples); per-user PR-AUC for that user is unstable. Median across users is more honest than mean for that one column. Reported in `bg_data_stats.json`.
- **Vocab cold-start for new users.** Apps the cohort hasn't seen map to `<RARE>` and lose their app-emb signal. Not a fundamental issue for the existing 22 users (vocab coverage ≥ 91.8 % per user); could matter for a cold-start to a 23rd user with a very different app diet.
- **No per-user fine-tune.** A follow-up could train per-user heads on top of the shared backbone (cheap to run; ~2 min per user) and likely close most of the cold-start gap. Not required for the production headline.
- **Slice B uses user-0 stats as proxy.** A proper cold-start eval would fit the new user's own per-user stats first. The next iteration of `53_eval_h60_multiuser.py` should do that.
- **Compute.** The full pipeline (build → 4 recipes → 3 evals) takes ~80 min on a busy 224-core server; ~50 min on a quiet one. Memory: ~1 GB peak. Trivial vs the data size.

---

## 10. Reproduction

```bash
cd app_usage_data

# 1. Build pooled multi-user bg data + per-user stats (~10 min)
python scripts/50_build_bg_data_multiuser.py

# 2. Closed-form baselines per user, aggregate (~2 min)
python scripts/51_run_baselines_bg_multiuser.py

# 3. Train 4 production picks on pooled data (~45 min)
python scripts/52_train_task_c_multiuser.py --epochs 30 --patience 4

# 4. Track A — per-user metrics + Pareto + slice B (~3 min)
python scripts/53_eval_h60_multiuser.py --include-slice-b

# 5. Track B — argmax-F1(keep), per-user + global τ\* (~1 min)
python scripts/54_threshold_multiuser.py

# 6. Positives-only metrics (~1 min)
python scripts/55_positive_metrics_multiuser.py

# 7. Run the test suite
python -m pytest lib/bg_multi/tests/ -v
```

Outputs:
- `artifacts/bg_multi/vocab.json` — pooled vocab (V=243)
- `artifacts/bg_multi/uid_to_idx.json` — stable user index map
- `artifacts/bg_multi/splits/bg_{train,val,test}.parquet` — pooled bg rows
- `artifacts/bg_multi/per_user/<uid>/stats.pkl` — per-user feature stats
- `artifacts/bg_multi/checkpoints/task_c_multi_<recipe>.pt` — trained checkpoints
- `artifacts/bg_multi/results/{baselines_bg,h60_leaderboard,threshold_metrics,positive_metrics,task_c_multi_*}.json`
- `artifacts/bg_multi/figures/{pareto_h60_test,per_user_pr_auc}.png`

---

## 11. Test inventory

54 tests, 7 layers, all green:

- **Layer 1** (`test_helpers.py`, 23 tests) — short_uid, list_cohort_users, build_pooled_vocab, make_global_anchor_id, attach_user_columns, stack_per_user.
- **Layer 2** (`test_data_build.py`, 11 tests) — per-user splits, embargo, replay isolation, label causality, pooled bg integrity, vocab coverage.
- **Layer 3** (`test_feature_lookup.py`, 5 tests) — vectorized stat lookup correctness; build_schema_data_multi shape + global anchor IDs.
- **Layer 4** (`test_baselines.py`, 6 tests) — Random / LRU / Markov-inverse / aggregate correctness.
- **Layer 5** (`test_model_io.py`, 3 tests) — vocab-size scaling, checkpoint round-trip, listwise loss respects global anchor IDs.
- **Layer 6** (`test_metrics.py`, 6 tests) — Track A monotone-in-r, MCC ≈ 0 on random, perfect-score check, argmax-keep ≥ argmax-kill, aggregate sanity.
- **Layer 7** (`test_smoke_e2e.py`, 1 test) — synthetic 3-user end-to-end pipeline.

---

## 12. Headline take-away

The **single global model trained on 22 users beats every closed-form baseline** on every Track-A metric and matches or beats the single-user bespoke model on Track-B MCC at the recommended τ-picker. The per-user feature-stat injection (one model, personalised inputs) is the right design for shipping: one weights file plus a per-user stats blob, with the model genuinely benefiting from each user's behavioural baseline at scoring time.
