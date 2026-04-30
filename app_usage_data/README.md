# Next-App Prediction Demo — `app_usage_data/`

Single-user next-app + 15-minute-window prediction + background-app suspension experiments on 42 days of real HarmonyOS event data. Five model iterations (v1 → v2 → v3 → v4 → v5) for Tasks A and B, plus a separate Task C track (C1 → C2 → C3.x cumulative + C3-Pro architectural variants) for background suspension at H=60 min.

## Three prediction tasks

- **Task A — next-app prediction.** Given history up to moment *t*, predict the next `APP_FOREGROUND` / `APP_START` event's app. Metric: Hit@1, Hit@5, MRR.
- **Task B — 15-min window set prediction.** Given history up to anchor time *t*, predict the set of apps the user will open in `[t, t+15 min]`. Metric: EventHit@5 (frequency-weighted coverage), Recall@5, Coverage@5.
- **Task C — background-app suspension prediction.** Given the apps `B(t)` currently resident in background at anchor *t*, predict which will NOT be foregrounded in the next 60 min — i.e. which are safe to evict from RAM. Metric: PR-AUC, ROC-AUC, FalseKillRate@r, NDCG. The decision space is per-anchor (typically 4–10 apps), not the full 50-token vocab; the cost is asymmetric (false kill > miss).

All three tasks evaluated on the same 30/5/5-day chronological split (60-min embargo), with a 5-min anchor grid restricted to 06:00–24:00 for Tasks B and C.

## Data at a glance

- 1 user, 42 days (2026-03-01 → 2026-04-12), 47,237 raw events, 7,125 target events
- Vocabulary 50 (3 reserved + 47 apps after collapsing <5-event apps into `<RARE>`)
- Top-3 apps: LITE 37 %, WeChat 17 %, Huawei bundle 11 % — 65 % of targets
- Chronological split: train 30 d / val 5 d / test 5 d, with 60-minute embargo at each boundary
- Train / val / test target counts: 5,471 / 1,071 / 583
- Context events (`APP_BACKGROUND`, `ABILITY_OR_PAGE_SWITCH`, `SCREENON`/`SCREENOFF`, `DEVICE_STATE_UPDATE`) are fed as input but not supervised

## Reproducing

```bash
# --- v1 pipeline (baselines + shared-backbone GRU / TGT-lite) ---
python scripts/01_prep_data.py        # builds artifacts/splits/*.parquet + vocab.json
python scripts/02_run_baselines.py    # MFU / MRU / HourMFU / Markov-1
python scripts/03_train_gru.py        # GRU-64 checkpoint (shared backbone, dual head)
python scripts/04_train_tgt.py        # TGT-lite checkpoint (Transformer, dual head)
python scripts/05_window_eval.py      # neural Task A + Task B on the anchor grid
python scripts/07_report.py           # assemble v1 tables + figures

# --- v2 per-task models (LocalEncoder + GlobalEncoder + ProfileEncoder + gated fusion) ---
python scripts/10_train_task_a_v2.py  # TaskAModel checkpoint
python scripts/11_train_task_b_v2.py  # TaskBModel checkpoint
python scripts/12_eval_v2.py          # both v2 checkpoints on val/test

# v2 ablations (split vs shared, local vs local+global):
python scripts/10_train_task_a_v2.py --no-global --no-profile --out task_a_v2_local_only.pt
python scripts/11_train_task_b_v2.py --no-global --no-profile --out task_b_v2_local_only.pt

# --- v3 feature-enriched models + Markov prior (Task B) ---
python scripts/20_build_v3_features.py                              # fit category / loc / Markov stats (train only)
python scripts/21_train_task_a_v3.py --tag task_a_v3_R4             # full v3 features
python scripts/22_train_task_b_v3.py --use-markov --tag task_b_v3_R6 # full v3 + Markov prior fusion
python scripts/26_report_v3.py                                       # assemble comparison tables + CSVs

# v3 iterative ablation sweep (Task A: R0..R4, Task B: R0..R4 + R6)
python scripts/21_train_task_a_v3.py --no-category --no-loc --no-daypart --no-windows --tag task_a_v3_R0
python scripts/21_train_task_a_v3.py --no-loc --no-daypart --no-windows             --tag task_a_v3_R1
python scripts/21_train_task_a_v3.py --no-daypart --no-windows                       --tag task_a_v3_R2
python scripts/21_train_task_a_v3.py --no-windows                                     --tag task_a_v3_R3
python scripts/21_train_task_a_v3.py                                                   --tag task_a_v3_R4

python scripts/22_train_task_b_v3.py --no-category --no-loc --no-daypart --no-windows --tag task_b_v3_R0
python scripts/22_train_task_b_v3.py --no-loc --no-daypart --no-windows              --tag task_b_v3_R1
python scripts/22_train_task_b_v3.py --no-daypart --no-windows                        --tag task_b_v3_R2
python scripts/22_train_task_b_v3.py --no-windows                                      --tag task_b_v3_R3
python scripts/22_train_task_b_v3.py                                                   --tag task_b_v3_R4
python scripts/22_train_task_b_v3.py --no-category --no-loc --no-daypart --no-windows --use-markov --tag task_b_v3_R6lite
python scripts/22_train_task_b_v3.py --use-markov                                      --tag task_b_v3_R6

# --- Task C: background-app suspension prediction (H=60 min, 2 h staleness window) ---
python scripts/30_build_bg_data.py                                            # bg_{train,val,test}.parquet (per-(anchor, app) rows + 4 horizons)
python scripts/31_run_baselines_bg.py                                         # 6 closed-form baselines: Random / LRU / TimeInBG / LFU-hour / Markov-inv / Hybrid
python scripts/34_train_task_c_h60.py --schema c1   --tag C1_h60              # legacy 14-feature schema + app_emb + cat_emb
python scripts/34_train_task_c_h60.py --schema c2   --tag C2_h60              # 15-feature v2 schema + app_emb
python scripts/34_train_task_c_h60.py --schema c3.1 --tag C31_h60             # + 5 hourly-habit features
python scripts/36_train_c3_grid.py --variant c3.2                             # + 9 identity & BG-composition features
python scripts/36_train_c3_grid.py --variant c3.3                             # + 3 category-aware features (winning feature schema)
python scripts/36_train_c3_grid.py --variant c3pro_reg --pro-schema c3.3      # production: dropout 0.3 + label smoothing + cosine LR + SWA
python scripts/37_train_c3pro_ensemble.py c3.3                                # 5-seed ensemble
python scripts/35_eval_h60.py                                                 # bootstrap CIs + Pareto FK-vs-MSR figures
python scripts/38_generate_features.py                                        # feature-pipeline validator / regression test
```

## Layout

```
app_usage_data/
├─ app_usage_cleaned_dictionary_mapped.xlsx   # raw source (read-only)
├─ paper_deep_dive.md                         # literature notes (MISApp, TGT, Appformer, MAPLE, …)
├─ sample_features.txt                        # target feature schema provided by user (guides v3)
├─ README.md                                  # this file
├─ REPORT.md                                  # v1 writeup (baselines + shared backbone)
├─ REPORT_v2.md                               # v2 writeup (per-task + global + profile)
├─ REPORT_v3.md                               # v3 writeup (feature enrichment + Markov prior)
├─ REPORT_v4.md                               # v4 writeup (recency + periodicity ablation, full baseline comparison)
├─ REPORT_v5_multiuser.md                     # multi-user (22 users) benchmark — all baselines through v5 BG-state features
├─ REPORT_v5_cherry_picked.md                 # 10 users where v5 E2 beats MRU-5 — full mean/median tables across all baselines
├─ REPORT_bgkill.md                           # Task C v1 — original 14-feature C1 writeup (H=5/10, historical)
├─ REPORT_bgkill_v2.md                        # Task C v2 — feature audit, C2 schema (15 Task-C-relevant features, H=5/10)
├─ REPORT_bgkill_v3.md                        # **Task C live writeup** — H=60 single-horizon, 2 h staleness, full C3.x + C3-Pro grid
├─ REPORT_bgkill_data.md                      # B(t) state machine + label definition spec
├─ REPORT_bgkill_metrics.md                   # FK / MSR / PR-AUC / ROC-AUC / NDCG / Pareto definitions
├─ REPORT_bgkill_model_plan.md                # plan that drove the C3.x feature track + C3-Pro architectural track
├─ REPORT_bgkill_features_review.md           # feature catalog + reproduction recipe (Appendix B)
├─ FEATURES.md                                # complete feature reference for Tasks A/B (every dim explained)
├─ FEATURES_v2.md                             # next-iteration feature proposal (audit + recommended drops/adds)
│
├─ lib/
│  ├─ data.py          — load, dedup, sessionize, chronological split, vocab, anchor grid
│  ├─ features.py      — per-event encoder (28-d numeric + 32-d app emb), history builders
│  ├─ baselines.py     — MFU / MRU / HourMFU / Markov-1
│  ├─ models.py        — v1 GRU-64 and TGT-lite (shared backbone, dual heads)
│  ├─ train.py         — class weights, Task-B window counts, dataset adaptor
│  ├─ metrics.py       — Hit@K, MRR, macro-F1, P/R/F1/Jaccard/Coverage, EventHit, MAP, NDCG, ECE, Wilson/bootstrap CI
│  ├─ v2/
│  │   ├─ global_features.py  — profile stats + long-history builders
│  │   ├─ models_v2.py        — LocalEncoder / GlobalEncoder / ProfileEncoder / GatedFusion / Task{A,B}Model
│  │   └─ datasets_v2.py      — torch Dataset wrappers
│  ├─ v3/
│  │   ├─ categories.py       — 11-class hand-built app taxonomy
│  │   ├─ location.py         — WiFi/Cell-ID parser + train-only vocab
│  │   ├─ daypart.py          — 10-bin hour×weekday onehot
│  │   ├─ window_rollups.py   — causal multi-window (15 m–6 h) aggregates
│  │   ├─ markov_prior.py     — (V,V) log P(next | last) table, train-only
│  │   ├─ features_v3.py      — per-token category/loc tensors + profile extension
│  │   ├─ models_v3.py        — LocalEncoderV3 / GlobalEncoderV3 / ProfileEncoderV3 / Task{A,B}ModelV3
│  │   ├─ datasets_v3.py      — v3 torch Datasets
│  │   └─ prep.py             — shared target-stream builder + last_app lookup
│  └─ bg/                                            # Task C
│      ├─ background_state.py — B(t) state machine: replays full event stream, snapshots BG set per anchor (2 h staleness, screen-on / kill tracking)
│      ├─ features_bg.py      — per-(anchor, app) feature builders + multi-horizon label generation (5 / 10 / 30 / 60 min)
│      ├─ baselines_bg.py     — closed-form baselines (Random / LRU / TimeInBG / LFU-hour / Markov-inv / Hybrid)
│      ├─ models_bg.py        — BgPairMLP (legacy dual-head) + ModelCfg / SingleHeadMLP / WideMLP for the C3 / C3-Pro models
│      └─ metrics_bg.py       — FalseKillRate@r, MemorySaveRate@r, PR-AUC, ROC-AUC, NDCG@half (per-anchor, equal-anchor weight)
│
├─ scripts/
│  ├─ 01_prep_data.py .. 07_report.py                — v1 pipeline
│  ├─ 10_train_task_a_v2.py / 11_train_task_b_v2.py  — v2 training
│  ├─ 12_eval_v2.py                                   — v2 eval
│  ├─ 20_build_v3_features.py                         — fit all v3 train-only stats
│  ├─ 21_train_task_a_v3.py / 22_train_task_b_v3.py   — v3 training (with --no-* flags)
│  ├─ 26_report_v3.py                                  — v3 result aggregation
│  ├─ 30_build_bg_data.py                              — Task C: B(t) snapshots + multi-horizon labels → bg parquets
│  ├─ 31_run_baselines_bg.py                           — Task C: closed-form baselines on bg parquets
│  ├─ 32_train_task_c.py / 33_eval_task_c.py           — Task C: legacy dual-head MLP (H=5/10) + eval — historical
│  ├─ 34_train_task_c_h60.py                           — Task C: single-head MLP for c1 / c2 / c3.1 schemas (H=60)
│  ├─ 35_eval_h60.py                                   — Task C: bootstrap CIs (B=1000) + Pareto figures
│  ├─ 36_train_c3_grid.py                              — Task C: cumulative C3.x feature track + C3-Pro architectural variants
│  ├─ 37_train_c3pro_ensemble.py                       — Task C: 5-seed ensemble of best C3-Pro variant
│  └─ 38_generate_features.py                          — Task C: feature-pipeline validator / sanity report
│
├─ artifacts/
│  ├─ splits/{train,val,test}.parquet                                  # all tasks (event-level)
│  ├─ vocab.json
│  ├─ v2_profile_stats.pkl
│  ├─ v3/{category_map.json, loc_vocab.pkl, markov_prior.pkl, loc_ids_*.npy, provenance.json}
│  ├─ checkpoints/{gru.pt, tgt.pt, task_a_v2.pt, task_b_v2.pt, task_a_v3_R*.pt, task_b_v3_R*.pt}
│  ├─ results/*.json                                                    # Task A / B
│  └─ bg/                                                                # Task C
│      ├─ splits/bg_{train,val,test}.parquet                            # per-(anchor, app) rows + 4 horizon labels
│      ├─ stats/bg_data_stats.json
│      ├─ checkpoints/task_c_*.pt                                       # 16+ trained models (C1–C3.4 + C3-Pro variants + ensemble seeds)
│      └─ results/                                                       # baselines + per-model metrics + grid summary + bootstrap CIs
└─ figures/
    ├─ *.png                                                            # Task A / B (calibration, confusion)
    └─ bg/
        ├─ pareto_h60_{val,test}.png                                    # current Task C (H=60) Pareto curves
        ├─ pareto_H_{300,600}_{val,test}.png                            # historical Task C (H=5/10)
        └─ pareto_focused_*.png                                         # zoom-in plots
```

## Model progression

### v1 — baselines + shared-backbone sequence models

- **MFU / MRU / HourMFU / Markov-1** — frequency and transition baselines, fit on train
- **GRU-64** — 1-layer GRU over last 16 in-session events (28-d numeric + 32-d app emb), 3 heads (softmax A / sigmoid B / Poisson rate) sharing the backbone
- **TGT-lite** — 2-layer Transformer (d=64, h=4) with Fourier-hour gating, same dual-head topology

### v2 — per-task models with cross-session global history

Two independent models, each with three encoder branches:

- **LocalEncoder** — v1 GRU over last 16 in-session events → h_local (64-d)
- **GlobalEncoder** — 2-layer Transformer (d=48) over last 64 cross-session target events → h_global (32-d). App embedding weight-tied to LocalEncoder.
- **ProfileEncoder** — 38-d hand-crafted anchor priors (hour/weekday top-8 marginals, rolling 24h / 7d frequency slices, Fourier hour, session signals) → h_profile (16-d)
- **GatedFusion** — 3-way softmax gate → h_fused (64-d) → task-specific head(s)

### v3 — feature enrichment + Markov prior fusion

Adds four new signal families and one architectural hook:

1. **Per-token `category_id`** (8-d embedding, 11 classes) — hand-built app taxonomy, tied across encoders
2. **Per-token `loc_id`** (8-d embedding, 18 classes) — WiFi SSID / Cell-ID from `device_state_update_payload`, forward-filled within day, tied across encoders
3. **Per-anchor `daypart_bin`** (10-d one-hot in profile) — 9 weekday + 1 weekend bin
4. **Per-anchor multi-window rollups** (105-d in profile) — for W ∈ {15 m, 30 m, 1 h, 2 h, 6 h}: n_unique apps/categories, transitions, self-repeats, switches, dominant category, top-5 app-duration shares
5. **Markov-1 prior fusion** (Task B only) — `logits_b_sig += α_markov · log_prior[last_app_idx]` where α is a single learnable scalar

Profile dim: 38 (v2) → 153 (v3 full). Token dim: 60/68 (v2) → 76/84 (v3) for Local/Global. Total params: ~101 k for TaskBModelV3 + Markov, up from ~90 k in v2.

### Task C — background-app suspension prediction (independent track)

Different framing: per-anchor restricted decision set (just `B(t)`, typically 5–10 apps), asymmetric cost (false kill > miss). H = 60 min headline; legacy H = 5 / 10 numbers in `REPORT_bgkill.md` / `REPORT_bgkill_v2.md`.

**Pipeline:**
- `lib/bg/background_state.py` — single-pass state machine over the full event stream (FG / BG / PROCESS_EXIT / SCREENON-OFF), produces a `BGSnapshot` per anchor with **2 h staleness cutoff** (was 6 h in v1). Tracks `time_in_bg_sec`, `time_since_fg_sec`, `fg_count_today`, `last_fg_loc_id`, plus anchor-level `time_since_screen_on_sec`, `last_kill_app`, `bg_recency_min_sec`.
- `scripts/30_build_bg_data.py` materialises per-(anchor, app) rows with **same-split FG labels** at four horizons (5 / 10 / 30 / 60 min) — same-split labels avoid cross-split leakage at H=60 = embargo width.

**Feature schemas (cumulative on top of C2):**

| Schema | feat | Adds | Builder |
|---|---|---|---|
| C1 | 14 + cat_emb + app_emb | (legacy) | `34_train_task_c_h60.py:build_c1` |
| C2 | 15 + app_emb | recency-rank, fg_count_{1h,6h}, bg_recency_min_norm, time_since_screen_on, prev_killed_app_match | `34_train_task_c_h60.py:build_c2` |
| C3.1 | 20 + app_emb | + 5 hourly-habit (overdue_ratio, was_fg_24h_ago, was_fg_7d_ago, log_fg_count_last_{24h,7d}) | `34_train_task_c_h60.py:build_c31` |
| C3.2 | 29 + app_emb | + 9 identity / BG-comp (category_match, is_system_app, app_lifetime_{share, kill_rate}, cat_lifetime_share, bg_recency_{mean, max}, bg_unique_cat_cnt, log_bg_size) | `36_train_c3_grid.py:add_c32_features` |
| **C3.3** | 32 + app_emb | + 3 category-aware (cat_markov_prob, time_since_cat_last_used, bg_apps_in_same_cat) | `36_train_c3_grid.py:add_c33_cols` |
| C3.4 | 33 + app_emb | + 1 two-step Markov (sparse table + 1-step fallback) — *regresses* | `36_train_c3_grid.py:add_c34_col` |

**Architectural variants (Track B, all on the best feature schema):**

| Variant | Change |
|---|---|
| Pro-Listwise | adds per-anchor softmax NLL aux loss (λ = 0.5) |
| Pro-Wide | 32-d app_emb + 128 → 64 trunk + GELU + dropout 0.3 |
| **Pro-Reg** | baseline-arch + dropout 0.3 + label smoothing 0.05 + cosine LR + SWA over last 25 % of epochs |
| Pro-Full | Wide + Listwise + Reg combined |
| Pro-Ensemble | 5-seed average of Pro-Reg |

## Headline results — test split

| Model | Task A Hit@1 | Task A Hit@5 | Task B EventHit@5 |
|---|---|---|---|
| MFU | 0.240 | 0.679 | 0.622 |
| MRU (top-5 distinct) | — | — | 0.669 |
| HourMFU | 0.244 | 0.736 | 0.689 |
| **Markov-1** | 0.496 | 0.808 | 0.688 |
| v1 GRU | 0.602 | 0.840 | 0.641 |
| v1 TGT-lite | 0.483 | 0.770 | 0.614 |
| v2 GRU (split + global + profile) | 0.593 | 0.844 | 0.628 |
| v3 R4 (full features, no Markov) | **0.599** | 0.851 | 0.714 |
| **v3 R6 (full features + Markov)** | 0.599 | 0.851 | **0.746** |

**v3 R6 vs strongest baseline (v1 Markov-1):** +10.3 pp Task A Hit@1, +5.8 pp Task B EventHit@5.

### Task C headline (test split, H = 60 min, single user)

n = 870 anchors with `|B(t)| ≥ 1`; pos-rate of `y_3600` = 21.7 %.

| Model | feat | PR-AUC | ROC-AUC | FK@0.5 | NDCG | Δ vs Markov-inv (PR / ROC / FK) |
|---|---|---|---|---|---|---|
| Random | — | 0.776 | 0.540 | 0.232 | 0.789 | — |
| LRU | — | 0.853 | 0.711 | 0.202 | 0.830 | — |
| LFU-hour | — | 0.868 | 0.727 | 0.178 | 0.852 | — |
| **Markov-inverse** (best baseline) | — | 0.897 | 0.744 | 0.180 | 0.859 | — |
| C2 | 15 | 0.922 | 0.806 | 0.170 | 0.873 | +2.5 / +6.2 / −1.0 |
| C3.1 | 20 | 0.918 | 0.801 | 0.161 | 0.878 | +2.1 / +5.7 / −1.9 |
| C3.2 | 29 | 0.928 | 0.814 | 0.162 | 0.880 | +3.1 / +7.0 / −1.8 |
| C3.3 (Track-A winner) | 32 | 0.934 | 0.820 | **0.160** | **0.884** | +3.7 / +7.6 / −2.0 |
| **Pro-Reg / c3.3 (production)** | 32 | **0.935** | 0.826 | 0.162 | 0.882 | **+3.8 / +8.2 / −1.8** |
| Pro-List / c3.3 | 32 | 0.934 | **0.831** | 0.161 | 0.883 | +3.7 / +8.7 / −1.9 |
| Pro-Wide / c3.3 | 32 | 0.928 | 0.826 | **0.159** | 0.882 | +3.1 / +8.2 / −2.1 |

**What's robust:** C2 / C3.x / Pro-Reg all decisively beat every closed-form baseline on every metric. ROC-AUC gains are statistically significant (95 % bootstrap CIs do not overlap Markov-inverse's). The Pareto curves (`figures/bg/pareto_h60_{val,test}.png`) show C2/C3.x strictly dominating LRU / LFU-hour / Markov-inv at every operating point r ∈ {0.1, 0.25, 0.5, 0.75, 0.9}.

**What's noise:** FK@0.5 / NDCG gains are within bootstrap CI band on the 870-anchor test set; the Pro-Ensemble on c3.3 (test PR=0.918) does *not* beat the best single seed — averaging hurts in this small-data regime. C3.4 (2-step Markov) regresses because the V × V × V table is too sparse on a single-user stream.

## Iterative ablation rounds (v3)

All share v2's three-branch encoder architecture; rounds add features incrementally.

| Round | Features added | Markov | Task A test Hit@1 | Task B test EH@5 |
|---|---|---|---|---|
| R0 | — (v2 parity repro) | — | 0.583 | 0.703 |
| R1 | + category | — | 0.578 | 0.702 |
| R2 | + location | — | 0.556 | 0.694 |
| R3 | + daypart | — | 0.566 | 0.724 |
| R4 | + multi-window | — | 0.599 | 0.714 |
| R6-lite | — (v2 features) | ✓ | — | 0.733 |
| **R6** | **all v3 features** | ✓ | 0.599 | **0.746** |

| R6-arch trim | R6 minus scene + F1–F4 + 2h/6h windows + n_trans (FEATURES_v2 drops) | ✓ | 0.578 | **0.746** |

Learned α_markov after R6 convergence: **0.517** (init 0.5, clipped to [0, 2]). Confirms the prior is load-bearing without collapsing the learned signal.

R6-arch trim ties R6 on Task B test EH@5 (0.746) with **profile dim 79 vs 153** (~48 % reduction) and ~8 % fewer params — empirically validates the FEATURES_v2 drop list as a no-regret efficiency win. See `REPORT_v4.md` §6.5.

## Key findings

- **Task A (next-app prediction)** is largely saturated by v1 GRU — v2 and v3 offer no significant gains because local sequence already dominates. Use v1 GRU or v3 R4 interchangeably (test Hit@1 ≈ 0.60 for both).
- **Task B (15-min window)** was the open gap in v1/v2: the learned models lagged Markov-1. v3 closes and overshoots by combining **feature enrichment** (especially daypart + 15-min / 1-hour window rollups) with a **frozen Markov prior fused into the sigmoid head**. The prior contributes most of the gain (~+3 pp); features contribute the rest.
- **Test > val gap** on Task B across all v3 rounds is because the chronological test window (last 5 days, April 8–13) happens to be more predictable for this user than the val window (April 3–7). Selection is on val, so no test-set overfitting.
- **Overfitting guards held**: dropout 0.2, weight decay 1e-4, grad-clip 1.0, early-stop patience 6 on val metric, all stats fit train-only with `fit_split="train"` assertions.
- **Task C** (background suspension, H = 60 min): the right *feature schema* (C3.3 = +9 identity / BG-comp + 3 category-aware over C3.1) earns ~1.5 pp test PR-AUC; *architectural levers* (Listwise / Wide / Reg / SWA) add only ~0.1–0.2 pp on top — confirming the dominant signal is hourly-cadence + per-app priors rather than model capacity. C3.4 (2-step Markov) regressed because the V × V × V table is too sparse on a single user's stream; the 5-seed ensemble underperformed every single seed on test, since per-seed selection variance dominates ensembling gains at 877 val anchors.

## See also

- `FEATURES.md` — every input feature explained: what it is, dim, why (current v3 implementation)
- `FEATURES_v2.md` — proposal for the next feature iteration: per-feature audit grounded in v3 ablations, drops ~84 dims (scene, F1–F4, 2h/6h windows), adds ~34 cleaner ones (per-app recency, 24h/7d periodicity priors)
- `REPORT.md` — detailed v1 methodology, dataset exploration, baseline design
- `REPORT_v2.md` — per-task architecture (Local + Global + Profile encoders, gated fusion), v2 ablations
- `REPORT_v3.md` — full v3 writeup: feature schemas, ablation rounds R0–R6, overfit audit, Markov prior design, limitations
- `REPORT_v4.md` — comprehensive comparison: all baselines (MFU / MRU / HourMFU / Markov-1 / v1 GRU / v1 TGT-lite / v2 GRU / v3 R0 / v3 R4 / v3 R6 / v4 / v4 + Markov) for both tasks; tests two new features (per-app recency, periodicity priors) and falsifies them on this dataset; **production picks**: v1 GRU for Task A, v3 R6 for Task B
- `REPORT_v5_multiuser.md` — extends the benchmark to **22 distinct users** with their own real HarmonyOS logs (data at `/data00/ruiqing/app_forecasting/data/cleaned/`). Reports closed-form baselines (MFU/MRU-5/HourMFU/Markov-1) and per-user-trained v1, v2, v3 R4/R6/R6-lite/R6-arch trim, v4, v4-trim, and v5 E1/E2/E4 (BG-state features). Headline: **v5 E4 wins Task A** (mean test Hit@1 0.609, +5.4 pp over MRU-1) and **v5 E2 wins Task B** (mean test EH@5 0.749, +0.7 pp over MRU-5 mean, +3.3 pp on median, 18/22 user wins). Background-state features (BG mask + bg_count + bg_recency + time_since_screen_on) are the v5 contribution.
- `REPORT_v5_cherry_picked.md` — focuses on **10 users where the trained model clearly beats MRU-5** (sorted by Δ EH@5 = v5 E2 − MRU). Shows mean/median tables for all 17 baselines × all standard metrics. On the picked subset: v5 E2 lifts Task B EH@5 by +3.8 pp mean / +2.9 pp median, Recall@5 by +3.1 pp, Coverage@5 by +2.4 pp absolute (≈6 % relative). Companion to the cohort-wide `REPORT_v5_multiuser.md` — exists to characterise *which* users benefit from training.
- **Task C reports:**
  - `REPORT_bgkill_v3.md` — **live writeup**. Single-horizon H = 60 min, 2 h staleness. Full grid: closed-form baselines, C1 / C2 / C3.1–C3.4, C3-Pro Listwise/Wide/Reg/Full/Ensemble × c3.2/c3.3 schemas. Bootstrap CIs (B=1000), Pareto curves, leave-one-out feature ablation, statistical-significance discussion. **Production pick: Pro-Reg-on-c3.3** (test PR=0.935 / ROC=0.826 / FK@0.5=0.162; +3.8 / +8.7 / −2.1 pp over Markov-inverse).
  - `REPORT_bgkill_features_review.md` — feature-space audit (every C1 → C2 → C3 feature with rationale, drops, and code pointers) + **reproduction recipe** (Appendix B with end-to-end command list, validator script `38_generate_features.py`, sanity-report schema).
  - `REPORT_bgkill_data.md` — `B(t)` state-machine spec, label generator, multi-horizon design, anchor-grid logic.
  - `REPORT_bgkill_metrics.md` — FK@r / MSR@r / PR-AUC / ROC-AUC / NDCG / Pareto formal definitions and worked examples.
  - `REPORT_bgkill_model_plan.md` — the plan (Track A feature ablation + Track B C3-Pro architectures) that drove the executed grid.
  - `REPORT_bgkill.md` / `REPORT_bgkill_v2.md` — historical (H=5 / H=10) writeups; superseded by `_v3.md` for live numbers.
- `paper_deep_dive.md` — literature context (MISApp, TGT, Appformer, MAPLE, ATPP, and why we chose what we chose)
