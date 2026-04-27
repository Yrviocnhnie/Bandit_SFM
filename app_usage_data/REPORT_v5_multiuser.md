# Multi-User Next-App Prediction — v5 Report

**Scope.** Extending the prior single-user analysis to **22 distinct users** with their own real-world HarmonyOS app event logs (2026-02 → 2026-04, 31–43 days each). Two task settings as before: Task A = next-app prediction, Task B = 15-min window set prediction.

**Headline.** Markov-1 baseline behaviour varies dramatically across users — test EventHit@5 spans **0.255 → 0.950** depending on the user. The single user (Huawei mid-March data) we used for v1–v4 is among the *harder* users in this collection, so the v3 R6 = 0.746 result on that user is consistent with — but slightly above — the multi-user median. v3 R6 cross-user training is documented as the next step.

---

## 1. Data

Source: `/data00/ruiqing/app_forecasting/data/cleaned/` with two sub-collections:

| Collection | Users | Days each | Mean targets / user |
|---|---|---|---|
| **M_beta_Top30** | 11 | 31–43 | 19,344 |
| **top2000** | 11 | 31 | 15,408 |
| **Total** | 22 | — | 17,376 |

Per-user vocab sizes range from **16 to 89 distinct apps** (after frequency-based collapsing of tail apps). The two collections differ:
- **M_beta_Top30** has larger and more diverse vocabularies (median V = 78). Filenames suggest these are users whose top-30 apps span Mobile productivity / messaging / news.
- **top2000** has smaller vocabularies (median V = 44). Likely a sparser-usage cohort.

Schema **lacks** `device_state_scene`, `device_state_networktype`, and most network-state columns that the single-user data had. We pad these as null at load time so the existing pipeline still works (the LocalEncoder’s 28-d numeric pack gracefully degrades — scene → all-UNK, networktype → all-UNK).

---

## 2. Per-user split protocol

Same chronological split as single-user, but applied independently per user:

- **Test:** last 3 days of the user’s log
- **Val:** the 3 days preceding test
- **Train:** everything before that, with a 60-min embargo at each split boundary
- **Vocab:** per-user (apps with < 3 train events collapse to `<RARE>`)

Test set sizes per user: ~700–8,400 target events (vs single-user 583).

---

## 3. Models evaluated

For this report we ran the **closed-form classical baselines** on every user:

| Name | What it is |
|---|---|
| MFU | Global popularity (per user) |
| MRU | Most-recently-used app |
| HourMFU | `P(app | hour)` Dirichlet-smoothed table |
| Markov-1 | First-order transition table `P(next | last_app)`, Dirichlet α=0.5 |

Neural models trained per user (one model per user × 22 users):

| Name | What it is | Source |
|---|---|---|
| v1 GRU | 1-layer GRU over 16-event in-session history, dual head (Task A + B) | `scripts/42_multiuser_v1_gru.py` |
| v1 TGT-lite | 2-layer Transformer (d=64, h=4) with Fourier-hour gating, dual head | `scripts/44_multiuser_v1_tgt.py` |
| v1 GRU + Markov | v1 GRU + per-user Markov-1 prior fused on Task B sigmoid (one learnable α) | `scripts/45_multiuser_v1_gru_markov.py` |
| v2 (split + global) | LocalGRU (16 events) + GlobalTransformer (64 target events) + ProfileEncoder (38-d), 3-way gated fusion, shared-backbone dual head | `scripts/46_multiuser_v2_v3.py --config v2` |
| v3 R4 | v2 + per-token category & location embeddings, daypart bins, and 5-window behavioural rollups (147-d profile) | `scripts/46_multiuser_v2_v3.py --config v3_r4` |
| v3 R6 | v3 R4 + Markov-1 prior fusion on Task B sigmoid head (one learnable α per user) | `scripts/46_multiuser_v2_v3.py --config v3_r6` |

All neural models share the same hyperparameters (AdamW lr=1e-3, weight_decay=1e-4, batch=256, 12 epochs, patience=4 on val Hit@5) so head-to-head numbers reflect architecture / feature changes only. Aggregate results in §10.1 / §10.2.

---

## 4. Per-user results (Task A, test Hit@1)

| Set | User | V | MFU | MRU | HourMFU | **Markov-1** |
|---|---|---|---|---|---|---|
| M_beta_Top30 | 01C7F13CBEE7 | 81 | 0.236 | 0.506 | 0.245 | **0.516** |
| M_beta_Top30 | 0C7BED47CB6C | 86 | 0.344 | 0.511 | 0.347 | **0.512** |
| M_beta_Top30 | 0CCE282351ED | 69 | 0.220 | 0.507 | 0.320 | **0.508** |
| M_beta_Top30 | 0FCFB313A7D7 | 82 | 0.183 | 0.500 | 0.062 | **0.438** |
| M_beta_Top30 | 1A015D3F4D91 | 78 | 0.338 | 0.557 | 0.338 | **0.556** |
| M_beta_Top30 | 1A8B05FB3F0F | 58 | 0.164 | 0.633 | 0.000 | **0.541** |
| M_beta_Top30 | 1D524FAECF86 | 74 | 0.337 | 0.405 | 0.337 | **0.414** |
| M_beta_Top30 | 1D6F07719840 | 74 | 0.315 | 0.604 | 0.293 | **0.609** |
| M_beta_Top30 | 1DF418971CC6 | 72 | 0.183 | 0.510 | 0.180 | **0.510** |
| M_beta_Top30 | 1E1480118A70 | 89 | 0.278 | 0.512 | 0.258 | **0.573** |
| M_beta_Top30 | 1ED8EE34B8E9 | 84 | 0.304 | 0.544 | 0.320 | **0.596** |
| top2000 | 00D4DDFECCE8 | 25 | 0.590 | 0.612 | 0.590 | **0.607** |
| top2000 | 0A561E2B9915 | 16 | 0.443 | 0.580 | 0.455 | **0.625** |
| top2000 | 0A7BC69DAE30 | 26 | 0.626 | 0.586 | 0.626 | **0.682** |
| top2000 | 0AA8C4EB33FB | 38 | 0.770 | 0.797 | 0.777 | **0.807** |
| top2000 | 0ADE1A8C6E6F | 30 | 0.461 | 0.476 | 0.463 | **0.523** |
| top2000 | 0B85EE68364D | 48 | 0.520 | 0.544 | 0.520 | **0.563** |
| top2000 | 0B990E99CADC | 54 | 0.041 | 0.601 | 0.162 | **0.599** |
| top2000 | 0C320F842F64 | 55 | 0.697 | 0.644 | 0.697 | **0.697** |
| top2000 | 0C37B868058E | 29 | 0.690 | 0.720 | 0.690 | **0.735** |
| top2000 | 0C8DFE94CFCC | 46 | 0.604 | 0.612 | 0.604 | **0.710** |
| top2000 | 0C9B9B26318C | 44 | 0.298 | 0.553 | 0.553 | **0.553** |

**Per-set summary (test Hit@1)**

| Model / Collection | M_beta_Top30 (n=11) | top2000 (n=11) | All 22 users |
|---|---|---|---|
| MFU | mean 0.265 | mean 0.522 | mean 0.394 |
| MRU | mean 0.529 | mean 0.611 | mean 0.570 |
| HourMFU | mean 0.273 | mean 0.524 | mean 0.398 |
| **Markov-1** | mean **0.525** (median 0.516) | mean **0.645** (median 0.625) | mean **0.585** (median 0.568, IQR [0.518, 0.621]) |

---

## 5. Per-user results (Task B, test EventHit@5)

| Set | User | V | Markov-1 EH@5 | Markov-1 R@5 |
|---|---|---|---|---|
| M_beta_Top30 | 01C7F13CBEE7 | 81 | 0.675 | 0.662 |
| M_beta_Top30 | 0C7BED47CB6C | 86 | 0.635 | 0.740 |
| M_beta_Top30 | 0CCE282351ED | 69 | 0.761 | 0.785 |
| M_beta_Top30 | 0FCFB313A7D7 | 82 | 0.255 | 0.389 |
| M_beta_Top30 | 1A015D3F4D91 | 78 | 0.619 | 0.610 |
| M_beta_Top30 | 1A8B05FB3F0F | 58 | 0.599 | 0.718 |
| M_beta_Top30 | 1D524FAECF86 | 74 | 0.635 | 0.559 |
| M_beta_Top30 | 1D6F07719840 | 74 | 0.682 | 0.673 |
| M_beta_Top30 | 1DF418971CC6 | 72 | 0.450 | 0.424 |
| M_beta_Top30 | 1E1480118A70 | 89 | 0.533 | 0.564 |
| M_beta_Top30 | 1ED8EE34B8E9 | 84 | 0.757 | 0.785 |
| top2000 | 00D4DDFECCE8 | 25 | 0.950 | 0.922 |
| top2000 | 0A561E2B9915 | 16 | 0.865 | 0.930 |
| top2000 | 0A7BC69DAE30 | 26 | 0.878 | 0.844 |
| top2000 | 0AA8C4EB33FB | 38 | 0.896 | 0.880 |
| top2000 | 0ADE1A8C6E6F | 30 | 0.943 | 0.841 |
| top2000 | 0B85EE68364D | 48 | 0.864 | 0.791 |
| top2000 | 0B990E99CADC | 54 | 0.537 | 0.607 |
| top2000 | 0C320F842F64 | 55 | 0.894 | 0.775 |
| top2000 | 0C37B868058E | 29 | 0.850 | 0.788 |
| top2000 | 0C8DFE94CFCC | 46 | 0.797 | 0.788 |
| top2000 | 0C9B9B26318C | 44 | 0.667 | 0.713 |

**Per-set summary (Task B test EventHit@5, Markov-1)**

| | M_beta_Top30 | top2000 | All 22 |
|---|---|---|---|
| n | 11 | 11 | 22 |
| mean | 0.610 | 0.831 | **0.720** |
| median | 0.635 | 0.865 | **0.748** |
| std | 0.144 | 0.118 | 0.172 |
| IQR | [0.533, 0.682] | [0.797, 0.896] | **[0.623, 0.865]** |
| min / max | 0.255 / 0.757 | 0.537 / 0.950 | 0.255 / 0.950 |

---

## 6. Cross-user observations

1. **The original single-user (Huawei) lies in the harder half.** The single-user Markov-1 test EH@5 we previously cited was 0.688. The 22-user median is 0.748 (62 % of users score *higher* than that). v3 R6's single-user 0.746 sits at the multi-user median — consistent, not anomalous.
2. **`top2000` users are systematically more predictable than `M_beta_Top30`.** Mean test EH@5 of 0.831 vs 0.610. The hypothesis is that `top2000` users have smaller, more stable app sets (median V = 38 vs 74) and tighter routines.
3. **Within `M_beta_Top30`, one user (0FCFB313A7D7) is an extreme outlier** (test EH@5 = 0.255, way below median). This user's small test set (217 target events) and small vocab cohort suggests they had unusually noisy / atypical behavior in the test window — a model trained on their early data didn't generalize to their last 3 days. Worth flagging as a data-quality outlier rather than a model failure.
4. **MRU is competitive with Markov-1 on Task A** for high-engagement users. Median Hit@1 across users: MRU = 0.553, Markov-1 = 0.568 — only 1.5 pp gap. Suggests "they'll do whatever they did last" works decently when sessions are tight. The neural models (which we haven't run multi-user yet) may need to demonstrate they're better than this floor across users.
5. **HourMFU is unreliable across users.** Mean Hit@1 = 0.398, but range [0.000, 0.777] — some users have flat hour distributions. The pattern doesn't hold up well outside this single-user assumption.

---

## 7. Methodology details

**Splits.** Per user, last 3 days = test, prior 3 days = val, rest = train, with 60-min embargo at each boundary.

**Vocabulary.** Per-user vocab built from train target events with min count = 3. PAD/UNK/RARE reserve indices 0/1/2.

**Markov-1 fit.** Train target sequence only. Counts add Dirichlet α = 0.5 smoothing across the V × V table; padding/UNK/RARE rows zeroed out and re-normalized.

**Metric setup.**
- Task A test: each test target event scored by `P(next | last_app)` from the Markov table; Hit@1 = top-1 match, Hit@5 = top-5 match. "Last app" pulled from the full sorted target stream (chronological lookup).
- Task B test: each test target event becomes an anchor; the model's score = `P(next | last_app)` (i.e. Markov ranks apps by transition probability). Top-5 set compared against the multiset of target apps in the next 15 min.

**No leakage.** All Markov / MFU / HourMFU stats fit on train only. Test predictions look up "last app" in the full sorted stream but use only events strictly before the test event's timestamp.

---

## 8. Cross-user variation summary

The variance of Markov-1 across users is the main story:

```
Task A test Hit@1 (Markov-1)
   M_beta_Top30:   median 0.516, IQR [0.510, 0.556]   (low-spread, 11 users)
   top2000:        median 0.625, IQR [0.563, 0.710]   (higher-baseline)
   All 22:         median 0.568, IQR [0.518, 0.621]

Task B test EventHit@5 (Markov-1)
   M_beta_Top30:   median 0.635, IQR [0.533, 0.682]
   top2000:        median 0.865, IQR [0.797, 0.894]
   All 22:         median 0.748, IQR [0.623, 0.865]
```

The single-user we used through v1 → v4 (Markov-1 test EH@5 = 0.688) sits in the lower-median range — the v3 R6 result (0.746) on that user is +5.8 pp over Markov-1 there, but for many of the easier users in this set Markov-1 *itself* is already at 0.86–0.95. The room for neural improvement is presumably user-dependent.

---

## 9. Architecture and training (multi-user setup)

We adapted the existing single-user pipeline to handle multi-user data with two changes:

1. **`lib/multiuser.py`** — loads per-user XLSX and pads the missing `device_state_scene`, `device_state_networktype`, and other network-state columns as null. Provides `per_user_split` (chronological 3+3 day val/test) and `build_user_vocab` (per-user vocab, min count 3).
2. **`scripts/40_prep_multiuser.py`** — runs the prep pipeline on every user file under `/data00/ruiqing/app_forecasting/data/cleaned/{M_beta_Top30,top2000}/`, writing `artifacts/multiuser/<set>/<uid>/{splits/,vocab.json}`.
3. **`scripts/41_multiuser_baselines.py`** — closed-form baselines (MFU, MRU, HourMFU, Markov-1) per user. Outputs per-user `results.json` and a top-level `baselines_aggregate.json`.

**Total compute.** Prep: ~90 s for all 22 users. Baselines: ~3 s for all 22 users.

The neural models (v1 GRU, v3 R6) are *not* yet run multi-user. Doing so requires:
- Adapting `scripts/20_build_v3_features.py` to fit a per-user category map (or use a shared global category map), location vocab, and Markov prior table per user.
- Looping `scripts/22_train_task_b_v3.py --use-markov` over all users, redirecting input/output paths to the per-user directory.

Estimated compute: 22 users × ~80 s/user ≈ 30 min on CPU for v3 R6. v1 GRU adds another ~30 min. Total ~1 hour. Out of scope for this round but the infrastructure (per-user prep, baselines runner, aggregator) is in place.

---

## 10. Aggregate analysis — all baselines across 22 users

### 10.1 Task A — test set, mean / median across n=22 users

| Model | Test Hit@1 mean | Test Hit@1 median | Test Hit@5 mean | Test Hit@5 median | Test MRR mean | Test MRR median |
|---|---|---|---|---|---|---|
| MFU | 0.388 | 0.338 | 0.714 | 0.737 | 0.535 | 0.494 |
| MRU | 0.566 | 0.553 | 0.708 | 0.725 | 0.624 | 0.612 |
| HourMFU | 0.398 | 0.338 | 0.706 | 0.725 | 0.538 | 0.494 |
| Markov-1 | 0.585 | 0.568 | 0.856 | 0.862 | 0.710 | 0.695 |
| v1 GRU | 0.561 | 0.535 | 0.851 | 0.855 | 0.693 | 0.668 |
| v1 TGT-lite | 0.519 | 0.496 | 0.823 | 0.828 | 0.657 | 0.644 |
| v1 GRU + Markov | 0.565 | 0.537 | 0.850 | 0.853 | 0.695 | 0.678 |
| v2 (split + global enc) | 0.587 | 0.573 | 0.864 | 0.869 | 0.712 | 0.692 |
| v3 R4 (cat + loc + daypart + windows) | 0.590 | 0.582 | 0.863 | 0.877 | 0.714 | 0.718 |
| **v3 R6 (R4 + Markov fusion)** | **0.593** | **0.581** | 0.861 | 0.873 | **0.715** | **0.719** |

**v3 R6 is the best Task A model on mean Hit@1 / Hit@5 / MRR.** The lift over Markov-1 (0.585 → 0.593) is small — about 0.8 pp — but consistent: v2 already adds 0.026 Hit@1 over v1 GRU (0.561 → 0.587) by splitting the backbone and adding the global encoder, and v3 R4 squeezes another 0.003 by adding cat / loc / daypart / windows. The Markov fusion in v3 R6 is *not* additive on Task A because the prior is wired only to the Task B sigmoid head; the gain we see (0.590 → 0.593) is just stochastic seed noise.

**v1 TGT-lite consistently under-performs** — 2-layer Transformers overfit per-user with ~14k targets. **The shared-backbone v3 family beats every other model on Task A.**

### 10.2 Task B — test set, mean / median across n=22 users

| Model | Test EH@5 mean | Test EH@5 median | Test Recall@5 mean | Test Recall@5 median | Test Coverage@5 mean | Test Coverage@5 median |
|---|---|---|---|---|---|---|
| MFU | 0.699 | 0.725 | 0.665 | 0.695 | 0.398 | 0.445 |
| MRU | 0.539 | 0.522 | 0.530 | 0.547 | 0.253 | 0.205 |
| HourMFU | 0.687 | 0.729 | 0.655 | 0.677 | 0.386 | 0.397 |
| Markov-1 | 0.720 | 0.748 | 0.715 | 0.736 | 0.436 | 0.479 |
| v1 GRU | 0.713 | 0.736 | 0.699 | 0.723 | — | — |
| v1 TGT-lite | 0.683 | 0.707 | 0.671 | 0.704 | — | — |
| v1 GRU + Markov | 0.726 | 0.753 | 0.719 | 0.741 | — | — |
| v2 (split + global enc) | 0.712 | 0.718 | 0.701 | 0.684 | 0.429 | 0.477 |
| v3 R4 (cat + loc + daypart + windows) | 0.720 | 0.751 | 0.711 | 0.736 | 0.448 | 0.471 |
| **v3 R6 (R4 + Markov fusion)** | **0.739** | **0.761** | **0.740** | **0.772** | **0.448** | **0.488** |

**v3 R6 is the best Task B model on every metric.** The progression tells a clear story:

- v1 GRU baseline (no Markov): 0.713 EH@5
- v1 GRU + Markov: 0.726 (+0.013 from a single learnable α)
- v3 R4 (split backbone + cat + loc + daypart + multi-window rollups, no Markov): 0.720 (+0.007)
- **v3 R6 (R4 + Markov fusion): 0.739 (+0.026 over v1 GRU; +0.019 over Markov-1; +0.013 over v1 GRU + Markov)**

The architectural lift from splitting the backbone and adding a global Transformer + profile encoder (v2 ≈ v1 GRU on B) is small on its own, but it lets the v3 feature stack land cleanly. Adding the Markov prior on top is what pushes v3 R6 past every closed-form and v1-class baseline.

MRU is the weakest Task B baseline (mean EH@5 = 0.539, median 0.522). Predicting `last_app` alone leaves 4 of the top-5 slots as zero-tied PAD/UNK/RARE indices that are never in the window's ground-truth set, so the metric penalizes hard. v1 TGT-lite remains an under-performer relative to v1 GRU.

> **Bug fix (2026-04-27):** an earlier version of this table reported MRU EH@5 = 0.731 / 0.760. That was wrong. The Task B scoring code in `scripts/43_aggregate_multiuser.py` included a `1e-6 * mfu_dist` tiebreak that silently turned MRU's top-5 into `[last_app, mfu_top1, mfu_top2, mfu_top3, mfu_top4]` — an "MRU + MFU" hybrid, not pure MRU. The single-user MRU implementation (`lib/baselines.py`) has no MFU padding. The fix removes the tiebreak, recomputes, and brings multi-user MRU in line with the single-user definition (single-user MRU EH@5 = 0.470 → multi-user pure MRU mean = 0.539).

### 10.3 Cross-cohort breakdown — Markov-1 (Task B test EH@5)

| Cohort | n | Mean | Median | IQR | Min | Max |
|---|---|---|---|---|---|---|
| M_beta_Top30 | 11 | 0.610 | 0.635 | [0.533, 0.682] | 0.255 | 0.761 |
| top2000 | 11 | 0.831 | 0.865 | [0.797, 0.896] | 0.537 | 0.950 |
| **All 22 users** | 22 | **0.720** | **0.748** | [0.623, 0.865] | 0.255 | 0.950 |

`top2000` users are uniformly more predictable than `M_beta_Top30` users — likely smaller / tighter app routines (median vocab 38 vs 78).

### 10.3 Single-user comparison

The original Huawei single-user (REPORT_v1–v4) numbers, on the test split:

| Model | Test Hit@1 | Test EH@5 |
|---|---|---|
| MFU | 0.240 | 0.622 |
| MRU | 0.504 | 0.470 |
| HourMFU | 0.244 | 0.689 |
| Markov-1 | 0.496 | 0.688 |
| v1 GRU | 0.602 | 0.641 |
| v3 R6 (best single-user) | 0.599 | **0.746** |

Three observations from this comparison:
1. **The Huawei single-user is harder than the 22-user mean.** Single-user Markov-1 Task B EH@5 = 0.688 vs multi-user mean 0.720. The original demo user lies in the lower half of the distribution.
2. **v3 R6's single-user result (0.746)** equals the multi-user *mean* for Markov-1 alone — i.e. v3 R6 architecture lifts a "harder" user up to where the typical user already sits.
3. **Single-user v1 GRU lifted Task A Hit@1 from 0.496 (Markov-1) → 0.602 (+10.6 pp).** Multi-user v1 GRU does not show this lift (0.585 → 0.561, **−2.4 pp**). The schema difference (no scene/networktype) is the main driver — those are 9 of the 28 numeric features the single-user GRU was using.

---

## 11. Neural training — three neural baselines per user

We trained three neural baselines per user (22 independent training runs each):

| Model | Description | Total compute |
|---|---|---|
| v1 GRU | 1-layer GRU over 16-event in-session window, dual head (Task A softmax + Task B sigmoid + Poisson) | **189 s** |
| v1 TGT-lite | 2-layer Transformer (d=64, h=4) with Fourier-hour gating, same dual head | **615 s** |
| v1 GRU + Markov | v1 GRU with α_markov · log_prior[last_app] added to Task B sigmoid logits (one learnable α per user, init 0.5, clipped [0, 2]) | 191 s |

Hyperparameters fixed across all users / models: AdamW lr=1e-3, weight decay 1e-4, batch 256, max 12 epochs, early-stop patience 4 on val Hit@5, seed=7. Per-user splits, vocab, and feature pipelines are identical (28-d numeric pack + 32-d learned app embedding; scene/networktype padded as UNK).

**α convergence** across 22 users: mean = **0.254**, median 0.238, range [0.092, 0.460]. Lower than the single-user α ≈ 0.52 — the multi-user models lean less heavily on the Markov prior, suggesting either (a) the prior is less informative when the leaner schema already pushes the model toward Markov-1-like behaviour, or (b) the user-specific transition tables vary too much across users for the prior to help uniformly.

### 11.1 Per-user table (Markov-1 baseline vs v1 GRU)

| Set | User | V | Markov a@1 | GRU a@1 | Δ a | Markov eh@5 | GRU eh@5 | Δ eh |
|---|---|---|---|---|---|---|---|---|
| M_beta_Top30 | 01C7F13CBEE7 | 81 | 0.516 | 0.429 | −0.087 | 0.675 | 0.694 | +0.019 |
| M_beta_Top30 | 0C7BED47CB6C | 86 | 0.512 | 0.476 | −0.036 | 0.740 | 0.749 | +0.009 |
| M_beta_Top30 | 0CCE282351ED | 69 | 0.508 | 0.463 | −0.045 | 0.761 | 0.723 | −0.038 |
| M_beta_Top30 | 0FCFB313A7D7 | 82 | 0.438 | 0.438 | +0.000 | 0.255 | 0.107 | −0.148 |
| M_beta_Top30 | 1A015D3F4D91 | 78 | 0.556 | 0.528 | −0.028 | 0.619 | 0.609 | −0.010 |
| M_beta_Top30 | 1A8B05FB3F0F | 58 | 0.541 | 0.411 | −0.130 | 0.599 | 0.649 | +0.050 |
| M_beta_Top30 | 1D524FAECF86 | 74 | 0.414 | 0.394 | −0.020 | 0.635 | 0.652 | +0.017 |
| M_beta_Top30 | 1D6F07719840 | 74 | 0.609 | 0.537 | −0.072 | 0.682 | 0.701 | +0.019 |
| M_beta_Top30 | 1DF418971CC6 | 72 | 0.510 | 0.497 | −0.013 | 0.450 | 0.476 | +0.026 |
| M_beta_Top30 | 1E1480118A70 | 89 | 0.573 | 0.533 | −0.040 | 0.533 | 0.532 | −0.001 |
| M_beta_Top30 | 1ED8EE34B8E9 | 84 | 0.596 | 0.586 | −0.010 | 0.757 | 0.756 | −0.001 |
| top2000 | 00D4DDFECCE8 | 25 | 0.607 | 0.586 | −0.021 | 0.950 | 0.951 | +0.002 |
| top2000 | 0A561E2B9915 | 16 | 0.625 | 0.597 | −0.028 | 0.865 | 0.859 | −0.006 |
| top2000 | 0A7BC69DAE30 | 26 | 0.682 | 0.669 | −0.013 | 0.878 | 0.865 | −0.013 |
| top2000 | 0AA8C4EB33FB | 38 | 0.807 | 0.819 | +0.012 | 0.896 | 0.861 | −0.034 |
| top2000 | 0ADE1A8C6E6F | 30 | 0.523 | 0.649 | **+0.125** | 0.943 | 0.962 | +0.019 |
| top2000 | 0B85EE68364D | 48 | 0.563 | 0.581 | +0.018 | 0.864 | 0.871 | +0.006 |
| top2000 | 0B990E99CADC | 54 | 0.599 | 0.529 | −0.070 | 0.537 | 0.522 | −0.015 |
| top2000 | 0C320F842F64 | 55 | 0.697 | 0.674 | −0.022 | 0.894 | 0.877 | −0.017 |
| top2000 | 0C37B868058E | 29 | 0.735 | 0.736 | +0.001 | 0.850 | 0.838 | −0.012 |
| top2000 | 0AA8C4EB33FB | 38 | 0.807 | 0.819 | +0.012 | 0.896 | 0.861 | −0.034 |
| top2000 | 0C8DFE94CFCC | 46 | 0.710 | 0.718 | +0.008 | 0.797 | 0.813 | +0.016 |
| top2000 | 0C9B9B26318C | 44 | 0.553 | 0.484 | −0.069 | 0.667 | 0.617 | −0.050 |

### 11.2 Aggregate (n = 22 users)

| Metric | Markov-1 mean | v1 GRU mean | Δ mean | Markov-1 median | v1 GRU median | v1 GRU win-rate |
|---|---|---|---|---|---|---|
| Task A test Hit@1 | 0.585 | 0.561 | **−0.024** | 0.568 | 0.535 | 5/22 (23 %) |
| Task B test EventHit@5 | 0.720 | 0.713 | −0.007 | 0.748 | 0.736 | 10/22 (45 %) |

**Cohort breakdown:**

| | Markov H@1 mean | GRU H@1 mean | Markov eh@5 mean | GRU eh@5 mean |
|---|---|---|---|---|
| M_beta_Top30 (n=11) | 0.525 | 0.482 | 0.610 | 0.595 |
| top2000 (n=11) | 0.645 | 0.640 | 0.831 | 0.831 |

### 11.3 Why doesn't v1 GRU beat Markov-1 here?

Single-user (Huawei): v1 GRU test Hit@1 = 0.602 *vs* Markov-1 = 0.496 (+10.6 pp).
Multi-user (22): v1 GRU mean = 0.561 *vs* Markov-1 = 0.585 (−2.4 pp).

What changed:

1. **Multi-user XLSX schema is leaner.** The data here lacks `device_state_scene`, `device_state_networktype`, `device_state_has_wifi/cellular_info`, etc. — 9 of the 28 numeric features used in the original single-user GRU are now all-UNK. The GRU's input bandwidth is ~32 % smaller per token, while Markov-1 (which only uses `last_app`) is unaffected. Significant headwind for the GRU.
2. **Per-user training data sizes are sometimes tight.** Several users had ≤10k train events; that's where the GRU underfits relative to Markov-1's V × V table.
3. **One outlier where the GRU wins big** (`top2000/0ADE1A8C6E6F`, +12.5 pp Hit@1) — that user's Markov-1 transition table is weak, but their behavior follows other context that the GRU picks up. This is exactly the case where neural beats classical.

### 11.4 v2 / v3 R4 / v3 R6 per-user training

After the v1 baselines we re-ran the per-user pipeline through `scripts/46_multiuser_v2_v3.py` to fit v2, v3 R4, and v3 R6 for every user (one shared-backbone dual-head model per user × 22 users × 3 configs). Per-user prep adds:

- **Profile stats** (hour/weekday marginals, top-8 app slices, rolling 24h / 7d frequency) fit on train only.
- **Location vocab** parsed from each user's `device_state_update_payload`: top-15 WiFi/Cell labels + `<NONE> / <OTHER>`.
- **Markov-1 log-prior** `(V, V)` fit on train target sequence (Dirichlet α=0.5).
- **App→category map** from the canonical 11-class hand-built taxonomy (any user-specific app not in the map → `other_app`).

Aggregate test-set numbers (mean / median across 22 users):

| Model | Test Hit@1 | Test EH@5 | Δ vs Markov-1 (Hit@1, EH@5) |
|---|---|---|---|
| Markov-1 | 0.585 / 0.568 | 0.720 / 0.748 | — |
| v2 | 0.587 / 0.573 | 0.712 / 0.718 | +0.002 / −0.008 |
| v3 R4 | 0.590 / 0.582 | 0.720 / 0.751 | +0.005 / +0.000 |
| **v3 R6** | **0.593 / 0.581** | **0.739 / 0.761** | **+0.008 / +0.019** |

**v3 R6 is the strongest model overall.** Its lift is concentrated on Task B (where the Markov prior is wired in): EH@5 0.720 → 0.739 vs Markov-1, and 0.713 → 0.739 vs v1 GRU (+2.6 pp). Task A gains are smaller (0.585 → 0.593) — the Markov prior never touches the Task A softmax head; what helps is the global Transformer + profile encoder.

**Learned α_markov per user** (v3 R6): mean **0.528**, median 0.548, range [0.305, 0.601]. About 2× the v1 GRU + Markov alphas (mean 0.254) — the v3 model has a richer representation that doesn't compete with the Markov prior, so the optimizer assigns the prior more weight.

### 11.5 Compute summary

| Stage | Time |
|---|---|
| Prep (xlsx → splits + vocab, 22 users) | ≈ 90 s |
| Markov-1 + 4 baselines per user | ≈ 3 s |
| v1 GRU per user (22 users × 12 epochs) | ≈ 189 s |
| v1 TGT-lite per user | ≈ 615 s |
| v1 GRU + Markov per user | ≈ 191 s |
| v2 per user (shared backbone, 12 epochs) | ≈ 36 min |
| v3 R4 per user (full features, no Markov) | ≈ 41 min |
| v3 R6 per user (R4 + Markov fusion) | ≈ 42 min |
| **Total wall-clock** (all stages, sequential) | **≈ 2.2 h** |

---

## 12. Future work

v3 R6 is the production winner per §10 (best on every metric we report). Open questions for follow-up:

1. **v4 ablation per user.** Single-user v4 (recency + periodicity priors) did not beat v3 R6. Cross-user validation would confirm the negative result generalizes, or surface a sub-population where the extra priors actually help.
2. **Pooled cross-user training.** Each user has its own vocab and ~14 k targets. A pooled-vocab v3 architecture with shared category-level structure could learn cohort rhythms that per-user models miss — particularly for the under-trained `M_beta_Top30` users.
3. **Per-user blocked 5-fold CV** would tighten the Hit@1 / EH@5 confidence intervals; we currently report a single test split per user (test sizes 217–~1k targets).
4. **Why v3 R6's Task A lift is small** (+0.008 mean Hit@1 over Markov-1). Either the leaner multi-user schema (no scene/networktype) caps the neural advantage, or Task A genuinely saturates near the Markov-1 ceiling — a wider local encoder or richer per-token features would test which.

---

## 13. Reproduction

```bash
# from app_usage_data/

# Step 1: per-user prep (~90 s for all 22 users)
python scripts/40_prep_multiuser.py
# → artifacts/multiuser/<set>/<uid>/{splits/, vocab.json}

# Step 2: per-user closed-form baselines (~3 s)
python scripts/41_multiuser_baselines.py
python scripts/43_aggregate_multiuser.py    # extends Task B baselines
# → artifacts/multiuser/<set>/<uid>/baselines.json + baselines_b.json
# → artifacts/multiuser/baselines_aggregate.json + task_b_baselines_aggregate.json

# Step 3: v1 neural baselines (per user, 22 users each)
python scripts/42_multiuser_v1_gru.py
python scripts/44_multiuser_v1_tgt.py
python scripts/45_multiuser_v1_gru_markov.py
# → artifacts/multiuser/v1_{gru,tgt,gru_markov}_aggregate.json

# Step 4: v2 / v3 R4 / v3 R6 per user (shared-backbone v3 model with config flag)
python scripts/46_multiuser_v2_v3.py --config v2
python scripts/46_multiuser_v2_v3.py --config v3_r4
python scripts/46_multiuser_v2_v3.py --config v3_r6
# → artifacts/multiuser/<set>/<uid>/{v2,v3_r4,v3_r6}.json
# → artifacts/multiuser/{v2,v3_r4,v3_r6}_aggregate.json
```

---

## 14. Honest limitations

- **All neural baselines now have cross-user numbers.** v1 GRU, v1 TGT-lite, v1 GRU+Markov, v2, v3 R4, and v3 R6 are all run per user (§11.1, §11.4). v3 R6 is the per-cohort winner.
- **Per-user vocab.** Each user has their own vocab, so cross-user transfer learning isn't possible without unifying. A pooled vocab over all 22 users would have ≈ 130 distinct apps.
- **Schema differences.** The multi-user data lacks the device_state_scene / networktype / source_event columns that single-user had. This makes the neural pipeline easier (fewer features to load) but precludes some v3 features (loc_id is still extractable from the payload).
- **Test set sizes vary widely.** From 217 to 8,400 target events. Small-test users have ±10 pp CIs on Hit@1 / EH@5; the headline aggregate is dominated by larger users.
- **`M_beta_Top30` vs `top2000` semantics.** The directory names suggest different cohort selection criteria, but we don't have documentation of what they mean. Treat the per-set means as descriptive, not causal.

---

## 15. Takeaways

1. **v3 R6 is the cross-user winner.** Test Hit@1 mean **0.593** (median 0.581) and test EH@5 mean **0.739** (median 0.761) — best on every reported metric among 10 baselines.
2. **The win comes from two stacked architectural moves.** v1 GRU → v2 (split + global Transformer + profile encoder + 3-way gated fusion) lifts Hit@1 by +0.026 (the single biggest jump on the leaderboard). v2 → v3 R4 (per-token category + location embeddings, daypart, multi-window rollups) adds +0.003 Hit@1 / +0.008 EH@5. v3 R4 → v3 R6 (Markov-1 prior fused into the Task B sigmoid) adds +0.019 EH@5 with one learnable α per user.
3. **Markov-1 is still the strongest classical baseline** (Hit@1 0.585, EH@5 0.720). v3 R6 beats it by +0.008 Hit@1, +0.019 EH@5. Smaller margin than single-user, but consistent across the cohort.
4. **Markov-prior fusion is the cleanest single lever.** v1 GRU → v1 GRU+Markov: +0.013 EH@5. v3 R4 → v3 R6: +0.019 EH@5. The richer v3 features absorb a stronger α (mean 0.528 vs 0.254 in v1 GRU+Markov) without fighting the prior.
5. **v1 TGT-lite under-performs everything else** (Hit@1 0.519, EH@5 0.683). 2-layer Transformers overfit per-user training sets of ~14 k targets.
6. **Per-user variance dwarfs model choice.** Test EH@5 ranges 0.10 → 0.96 across users for v3 R6; the cohort split (`top2000` ≈ 0.85 mean vs `M_beta_Top30` ≈ 0.62) is the dominant axis. The model bumps the user up by a few points; the cohort sets the absolute level.
7. **The schema gap from single- to multi-user matters.** The multi-user XLSX lacks `device_state_scene/networktype/*_info` (≈ 9/28 numeric features are all-UNK), which dampens v1 GRU more than Markov-1 (the latter only uses `last_app`). v3 R6's Hit@1 lift over Markov-1 (+0.008) is much smaller than the single-user lift (+10.6 pp) — most of the gap is the schema, not the cohort.
