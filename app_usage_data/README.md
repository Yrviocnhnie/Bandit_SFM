# Next-App Prediction Demo — `app_usage_data/`

Single-user next-app and 15-minute-window prediction experiments on 42 days of real HarmonyOS event data. Three successive model iterations (v1 → v2 → v3) with detailed ablations and honest reporting of what moved the needle.

## Two prediction tasks

- **Task A — next-app prediction.** Given history up to moment *t*, predict the next `APP_FOREGROUND` / `APP_START` event's app. Metric: Hit@1, Hit@5, MRR.
- **Task B — 15-min window set prediction.** Given history up to anchor time *t*, predict the set of apps the user will open in `[t, t+15 min]`. Metric: EventHit@5 (frequency-weighted coverage), Recall@5, Coverage@5.

Both tasks evaluated on the same 30/5/5-day chronological split (60-min embargo), with a 5-minute anchor grid restricted to 06:00–24:00 for Task B.

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
├─ FEATURES.md                                # complete feature reference (every dim explained)
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
│  └─ v3/
│      ├─ categories.py       — 11-class hand-built app taxonomy
│      ├─ location.py         — WiFi/Cell-ID parser + train-only vocab
│      ├─ daypart.py          — 10-bin hour×weekday onehot
│      ├─ window_rollups.py   — causal multi-window (15 m–6 h) aggregates
│      ├─ markov_prior.py     — (V,V) log P(next | last) table, train-only
│      ├─ features_v3.py      — per-token category/loc tensors + profile extension
│      ├─ models_v3.py        — LocalEncoderV3 / GlobalEncoderV3 / ProfileEncoderV3 / Task{A,B}ModelV3
│      ├─ datasets_v3.py      — v3 torch Datasets
│      └─ prep.py             — shared target-stream builder + last_app lookup
│
├─ scripts/
│  ├─ 01_prep_data.py .. 07_report.py                — v1 pipeline
│  ├─ 10_train_task_a_v2.py / 11_train_task_b_v2.py  — v2 training
│  ├─ 12_eval_v2.py                                   — v2 eval
│  ├─ 20_build_v3_features.py                         — fit all v3 train-only stats
│  ├─ 21_train_task_a_v3.py / 22_train_task_b_v3.py   — v3 training (with --no-* flags)
│  └─ 26_report_v3.py                                  — v3 result aggregation
│
├─ artifacts/
│  ├─ splits/{train,val,test}.parquet
│  ├─ vocab.json
│  ├─ v2_profile_stats.pkl
│  ├─ v3/{category_map.json, loc_vocab.pkl, markov_prior.pkl, loc_ids_{train,val,test}.npy, provenance.json}
│  ├─ checkpoints/{gru.pt, tgt.pt, task_a_v2.pt, task_b_v2.pt, task_a_v3_R*.pt, task_b_v3_R*.pt}
│  └─ results/*.json
└─ figures/*.png
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

## Headline results — test split

| Model | Task A Hit@1 | Task A Hit@5 | Task B EventHit@5 |
|---|---|---|---|
| MFU | 0.240 | 0.679 | 0.622 |
| MRU | — | — | 0.470 |
| HourMFU | 0.244 | 0.736 | 0.689 |
| **Markov-1** | 0.496 | 0.808 | 0.688 |
| v1 GRU | 0.602 | 0.840 | 0.641 |
| v1 TGT-lite | 0.483 | 0.770 | 0.614 |
| v2 GRU (split + global + profile) | 0.593 | 0.844 | 0.628 |
| v3 R4 (full features, no Markov) | **0.599** | 0.851 | 0.714 |
| **v3 R6 (full features + Markov)** | 0.599 | 0.851 | **0.746** |

**v3 R6 vs strongest baseline (v1 Markov-1):** +10.3 pp Task A Hit@1, +5.8 pp Task B EventHit@5.

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

Learned α_markov after R6 convergence: **0.517** (init 0.5, clipped to [0, 2]). Confirms the prior is load-bearing without collapsing the learned signal.

## Key findings

- **Task A (next-app prediction)** is largely saturated by v1 GRU — v2 and v3 offer no significant gains because local sequence already dominates. Use v1 GRU or v3 R4 interchangeably (test Hit@1 ≈ 0.60 for both).
- **Task B (15-min window)** was the open gap in v1/v2: the learned models lagged Markov-1. v3 closes and overshoots by combining **feature enrichment** (especially daypart + 15-min / 1-hour window rollups) with a **frozen Markov prior fused into the sigmoid head**. The prior contributes most of the gain (~+3 pp); features contribute the rest.
- **Test > val gap** on Task B across all v3 rounds is because the chronological test window (last 5 days, April 8–13) happens to be more predictable for this user than the val window (April 3–7). Selection is on val, so no test-set overfitting.
- **Overfitting guards held**: dropout 0.2, weight decay 1e-4, grad-clip 1.0, early-stop patience 6 on val metric, all stats fit train-only with `fit_split="train"` assertions.

## See also

- `FEATURES.md` — every input feature explained: what it is, dim, why (current v3 implementation)
- `FEATURES_v2.md` — proposal for the next feature iteration: per-feature audit grounded in v3 ablations, drops ~84 dims (scene, F1–F4, 2h/6h windows), adds ~34 cleaner ones (per-app recency, 24h/7d periodicity priors)
- `REPORT.md` — detailed v1 methodology, dataset exploration, baseline design
- `REPORT_v2.md` — per-task architecture (Local + Global + Profile encoders, gated fusion), v2 ablations
- `REPORT_v3.md` — full v3 writeup: feature schemas, ablation rounds R0–R6, overfit audit, Markov prior design, limitations
- `REPORT_v4.md` — comprehensive comparison: all baselines (MFU / MRU / HourMFU / Markov-1 / v1 GRU / v1 TGT-lite / v2 GRU / v3 R0 / v3 R4 / v3 R6 / v4 / v4 + Markov) for both tasks; tests two new features (per-app recency, periodicity priors) and falsifies them on this dataset; **production picks**: v1 GRU for Task A, v3 R6 for Task B
- `paper_deep_dive.md` — literature context (MISApp, TGT, Appformer, MAPLE, ATPP, and why we chose what we chose)
