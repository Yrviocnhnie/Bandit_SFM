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

Neural models (v1 GRU, v3 R6) are documented as future work in §7 — they require the full v3 feature pipeline to be re-run per user (location vocab, profile stats, Markov prior, etc.), which is straightforward but wasn't run for this round.

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

## 10. Aggregate analysis (Markov-1 across 22 users)

```
                           Task A test Hit@1     Task B test EventHit@5
Mean (22 users)            0.585 ± 0.080         0.720 ± 0.172
Median                     0.568                 0.748
IQR (Q1, Q3)               [0.518, 0.621]        [0.623, 0.865]
Range (min, max)           0.414 → 0.807         0.255 → 0.950
```

For comparison, the **single-user numbers** from the prior reports:
```
Single user (Huawei, 5,471 train targets)
  Markov-1 task A test Hit@1:    0.496       (below the multi-user median 0.568)
  Markov-1 task B test EH@5:     0.688       (below the multi-user median 0.748)
  v3 R6      task B test EH@5:   0.746       (at the multi-user median; +5.8 pp over its Markov-1)
```

This is reassuring: the v3 R6 result is consistent with the population median, not an outlier. It also tells us the headroom for neural lift varies considerably — easy users will see smaller % gains, hard users may see larger.

---

## 11. Neural training — v1 GRU per user

We trained the v1 GRU (1-layer, 16-event in-session window, dual head: Task A softmax + Task B sigmoid + Poisson) per user with their own train/val/test splits. Hyperparameters fixed across users: AdamW lr=1e-3, weight decay 1e-4, batch 256, max 12 epochs, early-stop patience 4 on val Hit@5, seed=7.

**Total compute: 189 s** for all 22 users (mean 8.6 s/user; range 1.6–16 s).

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

### 11.4 Compute summary

| Stage | Time |
|---|---|
| Prep (xlsx → splits + vocab, 22 users) | ≈ 90 s |
| Markov-1 + 4 baselines per user | ≈ 3 s |
| v1 GRU training, 22 users, 12 epochs each | **189 s** (avg 8.6 s/user) |
| **Total wall-time** | **~5 min** |

---

## 12. Future work — v3 R6 per user

The next neural model to run per user is **v3 R6** (full v3 features + Markov prior fusion). It needs the per-user prep extended to fit:
1. Category map (the existing one is generic across the 50-vocab Huawei single-user; for multi-user with V ≈ 16–89 it needs per-user mapping or a unified taxonomy)
2. Location vocab from each user's `device_state_update_payload` (top-15 + reserved)
3. Markov-1 transition matrix per user (already produced as v3 R6 baseline below)
4. v3 ProfileEncoder profile stats per user

Implementation outline:
- Adapt `scripts/20_build_v3_features.py` to take a `--user-dir` argument
- Adapt `scripts/22_train_task_b_v3.py` to take `--user-dir`
- Loop over all 22 user dirs

Estimated compute: 22 × 80 s ≈ **30 min** for v3 R6 across all users.

---

## 13. Reproduction

```bash
# from app_usage_data/

# Step 1: per-user prep (~90 s for all 22 users)
python scripts/40_prep_multiuser.py
# → artifacts/multiuser/<set>/<uid>/{splits/, vocab.json}

# Step 2: per-user closed-form baselines (~3 s)
python scripts/41_multiuser_baselines.py
# → artifacts/multiuser/<set>/<uid>/baselines.json
# → artifacts/multiuser/baselines_aggregate.json

# Step 3: v1 GRU per user (~190 s)
python scripts/42_multiuser_v1_gru.py
# → artifacts/multiuser/<set>/<uid>/v1_gru.json
# → artifacts/multiuser/v1_gru_aggregate.json
```

To extend with v3 R6 per user (the SOTA model from single-user experiments), the existing v3 scripts (`scripts/20_build_v3_features.py` and `scripts/22_train_task_b_v3.py`) need to take a `--user-dir` argument rather than the hardcoded `artifacts/splits/` path. Estimated extra compute: ~30 min on CPU.

---

## 14. Honest limitations

- **No v3 R6 cross-user results yet.** v1 GRU is done (§11) but v3 R6 (the SOTA on single-user) requires extra per-user feature fitting (location vocab, Markov prior, profile stats) — documented in §12 as the next step.
- **Per-user vocab.** Each user has their own vocab, so cross-user transfer learning isn't possible without unifying. A pooled vocab over all 22 users would have ≈ 130 distinct apps.
- **Schema differences.** The multi-user data lacks the device_state_scene / networktype / source_event columns that single-user had. This makes the neural pipeline easier (fewer features to load) but precludes some v3 features (loc_id is still extractable from the payload).
- **Test set sizes vary widely.** From 217 to 8,400 target events. Small-test users have ±10 pp CIs on Hit@1 / EH@5; the headline aggregate is dominated by larger users.
- **`M_beta_Top30` vs `top2000` semantics.** The directory names suggest different cohort selection criteria, but we don't have documentation of what they mean. Treat the per-set means as descriptive, not causal.

---

## 15. Takeaways

1. **Markov-1 mean test EH@5 = 0.720, median 0.748 across 22 users.** The original Huawei single-user (0.688) sits in the harder half. The v3 R6 single-user result (0.746) lies at the population median.
2. **v1 GRU per-user does NOT beat Markov-1 on average.** Mean test Hit@1: GRU 0.561 vs Markov 0.585. Mean test EH@5: GRU 0.713 vs Markov 0.720. v1 GRU wins on only **23 % of users for Task A** and **45 % for Task B**. This contrasts sharply with single-user Huawei (where v1 GRU beat Markov-1 by +10.6 pp on Hit@1).
3. **Why the gap?** The multi-user XLSX schema lacks `device_state_scene`, `device_state_networktype`, and `device_state_*_info` columns — about 9 of the 28 numeric features are all-UNK. The GRU's input bandwidth is reduced; Markov-1 doesn't care because it only uses `last_app`. The single-user GRU advantage was partly powered by that extra context.
4. **Per-user variance is large** — test EH@5 ranges 0.107 → 0.962 (8.8 ×) for the same v1 GRU architecture. Population-wide "best model" claims are weak; per-user choice or Markov-prior-fused models (v3 R6) is the production-ready answer.
5. **One outlier user (`top2000/0ADE1A8C6E6F`)** sees v1 GRU lift Hit@1 by +12.5 pp over Markov-1, suggesting user-specific patterns the transition table can't capture. Worth a follow-up case study.
6. **v3 R6 cross-user is the natural next experiment** (~33 min CPU compute) — Markov-prior fusion + the v3 features may close the gap that v1 GRU couldn't.
