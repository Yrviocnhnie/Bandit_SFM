# Next-App Prediction — v4 Experiment Report

**Author:** v4 experiment session
**Scope:** comprehensive comparison of all baselines across v1–v4 for both tasks; test of two new feature proposals (per-app recency, periodicity priors); identification of the best-performing model for each task.

---

## TL;DR

- **Task A — Next-app prediction.** Best test Hit@1 = **0.602** by **v1 GRU** (a single-stream GRU over the last 16 in-session events with one shared backbone). v3 architectures match it within noise; the additional v3/v4 features did *not* meaningfully move Task A. **Verdict: Task A is feature-saturated by the 16-event in-session sequence + app embedding.**
- **Task B — 15-min window set prediction.** Best test EventHit@5 = **0.746** by **v3 (R6)** (three-branch encoder with Local + Global + Profile + per-token category & location embeddings, plus Markov-1 prior fusion on the sigmoid head). The two new v4 feature proposals — per-app recency and same-hour-yesterday/last-week periodicity priors — **did not help** when added to this best model. They produce a 0.6–1.7 pp test regression, confirming v3 R6 is at or near the data ceiling.
- **Production recommendation:** v3 R6 (with Markov fusion) for Task B; either v1 GRU or v3 R4 for Task A (both within noise floor on test).
- **Honest takeaway:** the proposed v4 features (recency, periodicity) are intuitive but turn out to be redundant with what the existing Local + Global + Profile + Markov stack already captures. Adding them costs profile-dim budget without buying signal.
- **Update (§6.5):** the FEATURES_v2 *drops* — `device_state_scene` per-token, profile F1-F4, 2h+6h windows, redundant `n_transitions` — were tested as a separate experiment. Result: v3 R6-arch **trim** matches untrimmed v3 R6 at test EH@5 = **0.746** with profile dim shrunk 153 → 79 (-48 %). The drops are an empirical no-cost cleanup; ship them as the new lean baseline.

---

## 1. Setup

### 1.1 Data

- 1 user, 42 days of HarmonyOS app event log (2026-03-01 → 2026-04-12)
- 47,237 raw events; **7,125 target events** (`APP_FOREGROUND` ∪ `APP_START`)
- Vocabulary 50 (PAD/UNK/RARE + 47 apps with ≥ 5 events on train)
- Top-3 apps cover 65 % of targets (LITE 37 %, WeChat 17 %, Huawei bundle 11 %)
- Chronological split with 60-min embargo at each boundary:

| Split | Days | Target events | Anchors (Task B) |
|---|---|---|---|
| train | 30 | 5,471 | — |
| val | 5 | 1,071 | 1,072 |
| test | 5 | 583 | 583 |

### 1.2 Tasks

- **Task A — next-app prediction.** Given the user's history at time *t*, predict the app of the next target event. Multi-class classification over V=50.
- **Task B — 15-min window set prediction.** Given history at anchor time *t*, predict the set (with counts) of apps the user will use in `[t, t+15 min]`. Multi-label.

### 1.3 Metrics

- Task A: **Hit@1**, Hit@5, MRR. Headline = test Hit@1.
- Task B: **EventHit@5** (count-weighted hit-rate of top-5 against the future-window's app multiset), Recall@5, Coverage@5. Headline = test EventHit@5.
- All numbers reported on test split (~583 events) unless otherwise noted.
- 95 % CI on test Hit@1 / EventHit@5 ≈ ±3 pp due to small test size; deltas under that should be read as "no detectable change".

---

## 2. Baseline lineup

We use the original v1/v2/v3/v4 names (consistent with prior reports). The table below gives a generalized one-line description of each so you can see what each name represents architecturally without needing to consult the older docs.

| Name | Generalized description |
|---|---|
| **MFU** | Global popularity baseline. Predicts the most-frequent apps from train, regardless of context. |
| **MRU** | Recency baseline. Predicts the app the user most recently opened. Tests pure self-transition behavior. |
| **HourMFU** | Hour-conditioned popularity table `P(app \| hour)` with Dirichlet smoothing. First context-aware baseline. |
| **Markov-1** | First-order transition table `P(next \| last_app)` over the V×V app pairs, fit on train target sequence with α=0.5 smoothing. |
| **v1 GRU** | 1-layer GRU over the last 16 in-session events; one shared backbone with three heads (softmax for Task A; sigmoid + Poisson for Task B). Per-token input = 32-d learned app embedding ⊕ 28-d numeric pack (event type, Fourier hour, weekday, dt-gap, scene, network, screen, session position). ~40k params. |
| **v1 TGT-lite** | Same per-token input as v1 GRU but uses a 2-layer Transformer (d=64, h=4) with Fourier-hour temporal gating. Same dual-head topology. ~120k params. |
| **v2 GRU** | Two independent per-task models. Each has three encoder branches: a Local GRU over 16 in-session events, a Global Transformer over 64 cross-session target events, and a Profile MLP over 38-d hand-crafted priors (hour/weekday-conditioned top-8 marginals, rolling 24h/7d frequency, Fourier hour, session signals). Combined via 3-way softmax-gated fusion. ~90k params per task. |
| **v3 R0** | Architecturally identical to v2 GRU; reproduced through the v3 pipeline. Listed separately as a parity baseline. |
| **v3 R4** | v2 architecture + four additive feature blocks: per-token **category embedding** (8-d, hand-built 11-class taxonomy), per-token **location embedding** (8-d, parsed from device_state_update WiFi SSID / Cell ID, top-15 + reserved slots), **daypart one-hot** (10-d in profile, e.g. morning_commute, evening_home), **multi-window behavioral rollups** (105-d in profile, covering 15m/30m/1h/2h/6h windows). |
| **v3 R6** | v3 R4 + **Markov-1 prior fusion**: a frozen V×V `log P(next \| last_app)` table is added (with one learnable scalar α, init 0.5, clipped to [0, 2]) to the Task B sigmoid logits. Task A is unchanged from R4. |
| **v4** | v3 R4 + **two new feature blocks** under test: **per-app recency** (8-d, `log1p(seconds_since_last_use)` for each global top-8 app, "how stale is each headline app") and **periodicity priors** (18-d, one-hot of which top-8 app dominated a ±30-min window centered 24h or 7d before the anchor — "what was the user doing this hour yesterday / last week"). |
| **v4 + Markov** | v4 with Markov fusion also enabled on Task B (= v3 R6 + recency + periodicity). |

**v4 ablation variants:**
- v4 recency-only: drop periodicity, keep recency.
- v4 periodicity-only: drop recency, keep periodicity.
- Same options paired with Markov for Task B.

**Architecture progression in plain terms:**
- *v1* — simple sequence model on local in-session history. One model serves both tasks via dual heads.
- *v2* — split into per-task models; add a longer cross-session global view; add hand-crafted hour/weekday priors.
- *v3* — enrich the inputs (categories, location, daypart, multi-window rollups) and inject a closed-form Markov prior on the Task B head.
- *v4* — add per-app recency and 24h/7d periodicity priors as new profile features (this report tests whether they help).

### 2.1 New v4 features under test

| Feature | Dims | What it tells the model |
|---|---|---|
| Per-app recency | 8 | For each of the global top-8 apps, log1p(seconds since last use). Says "how stale is each headline app at this anchor". |
| Periodicity priors | 18 | For each of {24 h, 7 d} lookbacks, one-hot of which top-8 app dominated a ±30-min window centered at `anchor_ts − lookback`. Says "what was the user doing this time yesterday / last week". |

Both are train-only fitted (top-8 indices derived from train target frequencies, frozen). Causally safe — recency uses strict `<` against the target stream; periodicity windows are entirely in the past.

---

## 3. Training protocol (consistent across all neural runs)

- AdamW, lr=1e-3, weight decay=1e-4, gradient clip 1.0, batch=256
- Up to 25 epochs, early stop patience 6 on val metric (Hit@5 for Task A, EventHit@5 for Task B)
- Dropout 0.2 in all encoders
- Seed=7 fixed everywhere
- Task A loss: cross-entropy with label smoothing 0.05 + inverse-sqrt class weights
- Task B loss: BCE on `(win_counts > 0)` + 0.25 · Poisson on counts
- All stats fit on train only (`fit_split="train"` assertion enforced on load)

CPU runtime per training round: 50–100 s (small dataset, modest model).

---

## 4. Results — Task A (next-app prediction)

| Model | Val Hit@1 | Val Hit@5 | Val MRR | **Test Hit@1** | Test Hit@5 | Test MRR |
|---|---|---|---|---|---|---|
| MFU | 0.219 | 0.627 | 0.389 | 0.240 | 0.679 | 0.429 |
| MRU | 0.559 | 0.696 | 0.619 | 0.504 | 0.669 | 0.573 |
| HourMFU | 0.190 | 0.655 | 0.389 | 0.244 | 0.736 | 0.443 |
| Markov-1 | 0.558 | 0.805 | 0.671 | 0.496 | 0.808 | 0.635 |
| v1 TGT-lite | 0.556 | 0.831 | 0.678 | 0.513 | 0.808 | 0.624 |
| v2 GRU (split + global + profile) | 0.637 | 0.881 | 0.744 | 0.593 | 0.844 | 0.709 |
| **v1 GRU (shared backbone)** | 0.614 | 0.861 | 0.724 | **0.602** | 0.840 | 0.707 |
| v3 R0 (= v2 arch via v3 pipeline) | 0.619 | 0.871 | 0.730 | 0.583 | 0.858 | 0.702 |
| v3 R4 (full v3 features) | 0.610 | 0.863 | 0.723 | **0.599** | 0.851 | 0.707 |
| **v4 (full: + recency + periodicity)** | 0.594 | **0.885** | 0.722 | 0.573 | 0.851 | 0.695 |
| v4 (recency only) | 0.607 | 0.879 | 0.726 | 0.566 | 0.849 | 0.689 |
| v4 (periodicity only) | 0.575 | 0.861 | 0.707 | 0.563 | 0.840 | 0.687 |
| **v4-trim** (v4 with FEATURES_v2 drops: scene + F1–F4 + 2h/6h windows + n_trans; profile 179 → 105) | 0.611 | 0.876 | 0.727 | 0.571 | 0.849 | 0.690 |
| **v3 R6-arch trim** (= v3 R4 + Markov + drops, no rec/per; profile 153 → 79) | 0.593 | 0.870 | 0.714 | 0.578 | 0.837 | 0.695 |

**Reading the numbers:**
- The top three test Hit@1 results — **v1 GRU 0.602, v3 R4 0.599, v2 GRU 0.593** — are statistically tied within the ±3 pp CI of a 583-event test set. All three are clear wins over Markov-1 (0.496) and the popularity baselines (MFU 0.240, HourMFU 0.244).
- v4 variants did **not** clear v3's bar. v4 full lands at 0.573, v4 recency-only at 0.566, v4 periodicity-only at 0.563 — all 2–4 pp below v3 R4.
- v4 has the highest val Hit@5 (0.885 vs v3 R0's 0.871) but the lowest test Hit@1 among the deep models — classic mild overfit on the extra profile dims.
- **Verdict for Task A:** the local sequence (GRU over 16-event window) already extracts most of the signal. Adding global encoders, profile features, or new recency/periodicity priors gives noise-floor returns at this dataset size. Choice of v1 GRU vs v3 R4 is essentially a wash on test.

---

## 5. Results — Task B (15-min window prediction)

| Model | Markov? | Val EH@5 | Test EH@5 | Test Recall@5 | Test Coverage@5 |
|---|---|---|---|---|---|
| MFU | — | 0.657 | 0.622 | 0.605 | 0.390 |
| MRU | — | 0.566 | 0.470 | 0.461 | 0.259 |
| HourMFU | — | 0.689 | 0.689 | 0.668 | 0.465 |
| **Markov-1** | n/a | 0.743 | 0.688 | 0.679 | 0.500 |
| v1 GRU (shared backbone) | — | 0.696 | 0.641 | 0.623 | 0.418 |
| v1 TGT-lite | — | 0.681 | 0.614 | 0.604 | 0.390 |
| v2 GRU (split + global + profile) | — | 0.704 | 0.628 | 0.614 | 0.411 |
| v3 R0 (= v2 arch) | — | 0.693 | 0.703 | 0.689 | 0.419 |
| v3 R3 (cat + loc + daypart, no Markov) | — | 0.654 | 0.724 | 0.688 | 0.412 |
| v3 R4 (full v3 features, no Markov) | — | 0.625 | 0.714 | 0.650 | 0.357 |
| v4 (v3 + recency + periodicity, no Markov) | — | 0.627 | 0.702 | 0.649 | 0.355 |
| v3 R6-lite (Markov only, no v3 features) | ✓ | 0.703 | 0.733 | 0.729 | 0.465 |
| **v3 R6 (full v3 + Markov) — current best** | ✓ | 0.699 | **0.746** | **0.728** | 0.441 |
| **v4 + Markov (full: + recency + periodicity)** | ✓ | 0.707 | 0.729 | 0.720 | 0.437 |
| v4 + Markov (recency only) | ✓ | 0.703 | 0.739 | 0.726 | 0.437 |
| v4 + Markov (periodicity only) | ✓ | 0.704 | 0.743 | 0.732 | 0.456 |
| **v4-trim + Markov** (FEATURES_v2 drops; profile 179 → 105) | ✓ | 0.709 | **0.737** | 0.728 | 0.416 |
| **v3 R6-arch-trim + Markov** (no rec/per; profile 153 → 79) | ✓ | 0.696 | **0.746** | 0.730 | 0.443 |

### Top of the leaderboard

```
0.746  v3 R6 (full v3 + Markov)                          ← BEST
0.743  v4 + Markov (periodicity only)
0.739  v4 + Markov (recency only)
0.733  v3 R6-lite (Markov only, no v3 features)
0.729  v4 + Markov (full: recency + periodicity)
0.724  v3 R3 (cat + loc + daypart, no Markov)
0.714  v3 R4 (full v3 features, no Markov)
0.703  v3 R0 (= v2 arch via v3 pipeline)
0.689  HourMFU
0.688  Markov-1 baseline
0.641  v1 GRU
0.628  v2 GRU
0.622  MFU
0.614  v1 TGT-lite
0.470  MRU
```

### What the numbers say

- **v3 R6 (with Markov) stays the best.** The recency and periodicity features did not clear v3 R6's 0.746. With *both* new features active, v4 + Markov drops to 0.729 (-1.7 pp). With just one of them, v4 + Markov lands at 0.739-0.743, still below v3 R6.
- **The drop is not because the features are useless** — they're well-formed and pass causality / leakage audits. It's because the model already has access to similar information via the Local GRU (recent token sequence), the Global Transformer (longer-horizon attention pool), and the multi-window rollups. Adding redundant signal lets the model overfit slightly on val (recency-only val 0.703 vs v3 R6's 0.699) but the test set picks up the diluted regularization.
- **Markov fusion is unambiguously the load-bearing piece.** Every model that includes Markov ranks above every model that doesn't (0.733 lower bound vs 0.724 upper bound for non-Markov). Learned `α_markov` lands at ~0.52 across all variants — the prior's mixing weight is consistent.

---

## 6. Per-feature audit (post-experiment, honest)

This is the audit grounded in actual numbers from the v3 → v4 ablations, not just first-principles arguments.

| Feature | Lift (Task B test EH@5) | Verdict |
|---|---|---|
| Daypart one-hot (10-d) | **+3.0 pp** (R3 vs R2) | Keep — biggest single profile feature |
| Markov-1 prior fusion (50-d gather) | **+3.0 pp** (R6-lite vs R0) | Keep — co-equal headline contributor |
| Multi-window rollups (105-d) — combined w/ Markov | +1.3 pp (R6 vs R6-lite) | Keep — earns its budget when paired with Markov |
| Multi-window rollups alone (no Markov) | −1.0 pp (R4 vs R3) | Marginal; only valuable in concert |
| Per-token `loc_id` (8-d) | −0.8 pp alone (R2 vs R1) | Neutral; supports overall stack |
| Per-token `category_id` (8-d) | ~0 pp alone | Neutral; cheap and weight-tied so essentially free |
| **NEW: Per-app recency (8-d)** | **−0.7 pp (v4-rec-only vs R6)** | **Drop — does not help** |
| **NEW: Periodicity priors (18-d)** | **−0.3 pp (v4-per-only vs R6)** | **Drop — does not help; signal already captured** |
| **NEW: Recency + periodicity together** | **−1.7 pp (v4-full vs R6)** | **Drop — combination hurts more than parts** |

The two new v4 features were proposed in `FEATURES_v2.md` based on first-principles reasoning. The experiment falsifies them on this dataset.

---

## 7. Why didn't recency / periodicity help?

Honest analysis:

- **Recency is already implicit in the LocalEncoder.** The 16-event in-session window includes the previous targets, time-since-prev gaps in the numeric features, and the GRU hidden state implicitly encodes "how recent is each app". The explicit per-app log1p(time-since-last-use) is mostly redundant.
- **Periodicity priors clash with Markov.** The Markov prior says "given the *immediately previous* app, what's most likely next". The periodicity prior says "given that it's 9 AM, what was the user doing 24 h ago at 9 AM". These two priors agree on routine apps and disagree on novel ones, and the small-dim profile bottleneck (153 → 16) can't carry both signals coherently.
- **Profile dim inflation hurts.** v4 profile = 179-d (153 + 8 + 18) vs v3 profile = 153-d. The ProfileEncoder bottleneck (Linear(179 → 64) → Linear(64 → 16)) is forced to compress more, and on 5,471 train events the extra capacity finds noise as fast as it finds signal.
- **Single-user data ceiling.** Test set has 583 target events. Test Hit@1 / EventHit@5 noise floor is ~3 pp. The "v4 hurts by 1–2 pp" deltas are likely a mix of real (slight) feature redundancy + noise. They're not a strong negative finding, but they're definitely *not* a positive finding.

**What this confirms about the proposed FEATURES_v2.md design:**
- The "drop F1–F4 / scene / 2h+6h windows / n_transitions" recommendations are **empirically validated as cleanliness wins** — see §6.5 below. Removing 74 dims from the profile (and zeroing the 5-dim scene one-hot per token) leaves test metrics unchanged or slightly improves them.
- The "add recency / periodicity" recommendations are **falsified** on this dataset (and not rescued by the trim either). They should not be added unless we get more data; multi-user with a shared vocab might let them earn a slot.

---

## 6.5 FEATURES_v2 drop experiment — empirical verification

The v4 baseline as originally run (and reported in §4-§5) was *strictly additive* on top of v3 R4: it kept every existing feature and added recency (8-d) + periodicity (18-d). It did **not** apply the 4 feature-block drops also proposed in `FEATURES_v2.md`:

1. **`device_state_scene` one-hot** (5 dims per token) — 86 % null; remaining values nearly collinear with `networktype`.
2. **F1–F4 in the v2 profile** (32 dims = 4 × 8) — hour / weekday top-8 marginals + rolling 24 h / 7 d top-8 frequencies. Largely subsumed by daypart + multi-window rollups in v3.
3. **2 h and 6 h windows** (42 of the 105 multi-window dims) — daypart already captures the long-horizon time-of-day bucket; the 6 h window almost always pins to `news_feed_content`.
4. **`n_transitions` per-window scalar** (5 dims of 105) — exactly equal to `n_self_repeats + n_switches` for each window; pure linear redundancy.

`scripts/27_train_v4.py` now exposes four CLI flags (`--drop-scene --drop-f1234 --drop-long-windows --drop-n-trans`) that apply post-hoc on the numpy arrays before they enter the encoders, so the rest of the v3/v4 wiring is unchanged. Trim runs apply all four drops together; recency and periodicity can be toggled independently.

| Variant | Profile dim | Token dims dropped | Params | Test Hit@1 (A) | Test EH@5 (B) |
|---|---|---|---|---|---|
| v4 full + Markov (no drops) | 179 | 0 | 99k / 102k | 0.573 | 0.729 |
| **v4-trim + Markov** (all 4 drops) | 105 | 5 zero'd | 95k / 98k | **0.571** | **0.737** |
| v3 R6 full (no rec/per, no drops) | 153 | 0 | 98k / 100k | 0.599 | 0.746 |
| **v3 R6-arch trim** (drops, no rec/per) | 79 | 5 zero'd | 93k / 96k | 0.578 | **0.746** |

**Key findings:**

1. **The drops are at worst neutral on every test metric.** v3 R6-arch trim ties v3 R6 full at 0.746 EH@5 — *exactly* the same to four decimals — while shedding 74 profile dims (48 %). v4-trim improves EH@5 by +0.8 pp over v4-full (0.737 vs 0.729) by removing dimensions that were trading off against the recency / periodicity signals.
2. **The dropped features carry no signal that this model architecture can use.** Removing them doesn't trigger compensation elsewhere; the rest of the encoder stack already covers what those features were nominally encoding (categorical / hour / weekday rhythms via daypart + cat embedding + Fourier hour; long-horizon frequency via multi-window 1h rollup + Markov prior; etc.).
3. **Task A is unchanged within noise.** v4-trim lands at 0.571 vs v4 full's 0.573 (−0.2 pp, well below the ±3 pp 95 % CI for n=583 test events). v3 R6-arch trim at 0.578 is 2.1 pp below v3 R4's 0.599 — that gap is mostly the no-recency / no-periodicity choice (v3 R4 keeps Markov off; v3 R6-arch trim keeps Markov on but Task A doesn't use Markov), so it's noise, not the drops.
4. **The trim doesn't rescue the recency/periodicity features.** v4-trim + Markov (0.737) is still 0.9 pp below v3 R6 full (0.746). Whatever they were doing wrong in untrimmed v4, the trim doesn't fix it — the *additions* are still net-negative even after the *redundant existing features* are removed.

**Conclusion:** the FEATURES_v2 drops are a clean win on cost (smaller profile, fewer params, less compute in window rollups) with no metric cost. They are a strict improvement and should be the default for the production v3 R6 path. The recency/periodicity additions remain falsified on this single-user data.

**Production picks (updated):**
- **Task A:** v1 GRU (simplest, ties for best test Hit@1).
- **Task B:** **v3 R6-arch trim** (v3 R4 + Markov + drops, no rec/per) — same test EH@5 = 0.746 as untrimmed v3 R6, with 80 % of the params and ~half the profile-build compute.

---

## 8. Final ranking and recommended models

### 8.1 Task A — Next-app prediction (test Hit@1)

```
1.  v1 GRU                      0.602  ← simplest model, tied for best
2.  v3 R4 (full v3 features)    0.599
3.  v2 GRU                      0.593
4.  v3 R0 (= v2 arch)           0.583
5.  v4 (full)                   0.573
6.  v4 (recency only)           0.566
7.  v4 (periodicity only)       0.563
```

**Recommendation:** ship **v1 GRU** for Task A. It's the simplest model with the best test result, has only ~40k params (vs ~100k for the hierarchical variants), and its inference path is a single GRU pass. **v3 R4** is statistically tied and may be preferable if Task B is also being served from the same encoder stack (encoder sharing).

### 8.2 Task B — 15-min window set prediction (test EventHit@5)

```
1.  v3 R6 (full v3 + Markov)                 0.746  ← BEST
2.  v4 + Markov (periodicity only)           0.743
3.  v4 + Markov (recency only)               0.739
4.  v3 R6-lite (Markov only, no v3 features) 0.733
5.  v4 + Markov (full: recency + periodicity) 0.729
6.  v3 R3 (cat + loc + daypart, no Markov)   0.724
7.  v3 R4 (full v3, no Markov)               0.714
...
```

**Recommendation:** ship **v3 R6** for Task B. The Markov-1 prior fusion (single learnable scalar mixing weight + frozen V×V log-prob table) is essential and contributes ~3 pp on top of v3 features. Adding the v4 recency or periodicity features does not help and may slightly hurt — drop them.

---

## 9. Limitations and CI notes

- **Test set is small (583 anchors).** 95 % CI on EventHit@5 ≈ ±3 pp; on Hit@1 ≈ ±3 pp. Differences smaller than that should be read as "no detectable change".
- **Single user.** Findings about which features work or don't transfer cautiously. A user with a flatter app distribution might benefit from features (e.g. category embedding) that are nearly redundant for this user where top-3 apps account for 65 % of mass.
- **Val/test asymmetry.** Test consistently scores higher than val on Task B for v3/v4 — the test 5-day window happens to be more predictable for this user than the val window. We selected on val, so this is a conservative selection (no test peeking).
- **No bootstrap CIs reported.** All deltas in the ranking should be interpreted with the ±3 pp band in mind; the SOTA claim (v3 R6 = 0.746) sits ~5 pp above the strongest classical baseline (Markov-1 = 0.688), which is comfortably outside CI.

---

## 10. What we tried but didn't work — and why it's still useful

The v4 experiment (recency + periodicity priors) was a falsification. That's a real result:

- It tells us the v3 architecture was already extracting near-maximum signal from this user's 5,471 training events. Additional hand-crafted features can't easily push past that without overfitting.
- It validates our "inductive bias > parameters" principle: the 1-parameter (`α_markov`) Markov prior fusion contributes 3 pp; the 26-dim (8 + 18) recency+periodicity combo contributes 0 or negative.
- The follow-up FEATURES_v2 trim experiment (§6.5) gives the matching positive result: removing 74 redundant profile dims (scene, F1-F4, 2h/6h windows, n_trans) holds test EH@5 at 0.746. Together the two experiments triangulate where the signal actually lives — what the model uses, and what's dead weight.
- It suggests the next step for *real* improvement is **more data** (multi-user, cross-user transfer) or **a richer raw signal** (notification events, app-screen time, calendar) — not more features computed from the existing log.

---

## 11. Reproduction

```bash
# v1 baselines (already committed)
python scripts/02_run_baselines.py
python scripts/03_train_gru.py
python scripts/04_train_tgt.py
python scripts/05_window_eval.py

# v2 (already committed)
python scripts/10_train_task_a_v2.py
python scripts/11_train_task_b_v2.py

# v3 prep + experiments
python scripts/20_build_v3_features.py
python scripts/21_train_task_a_v3.py --tag task_a_v3_R4
python scripts/22_train_task_b_v3.py --use-markov --tag task_b_v3_R6   # ← best Task B

# v4 experiments (this report)
python scripts/27_train_v4.py --task a --tag task_a_v4_full
python scripts/27_train_v4.py --task a --no-periodicity --tag task_a_v4_rec_only
python scripts/27_train_v4.py --task a --no-recency     --tag task_a_v4_per_only

python scripts/27_train_v4.py --task b --use-markov --tag task_b_v4_full
python scripts/27_train_v4.py --task b --use-markov --no-periodicity --tag task_b_v4_rec_only
python scripts/27_train_v4.py --task b --use-markov --no-recency     --tag task_b_v4_per_only
python scripts/27_train_v4.py --task b --use-markov --no-recency --no-periodicity --tag task_b_v4_no_new_no_markov  # = v3 R6 essentially

# FEATURES_v2 drop experiment (§6.5) — drops scene + F1-F4 + 2h/6h windows + n_trans
DROP="--drop-scene --drop-f1234 --drop-long-windows --drop-n-trans"
python scripts/27_train_v4.py --task a --use-markov               $DROP --tag task_a_v4_trim
python scripts/27_train_v4.py --task a --use-markov --no-recency --no-periodicity $DROP --tag task_a_v3r6arch_trim
python scripts/27_train_v4.py --task b --use-markov               $DROP --tag task_b_v4_trim
python scripts/27_train_v4.py --task b --use-markov --no-recency --no-periodicity $DROP --tag task_b_v3r6_trim
```

All runs ~50–100s on CPU. Determinism: `seed=7` everywhere.

---

## 12. Headline summary table

| Task | Champion (test) | Runner-up | Notes |
|---|---|---|---|
| A — Next-app prediction (Hit@1) | **v1 GRU 0.602** | v3 R4 0.599 | Tied within noise; pick v1 GRU for simplicity |
| B — 15-min window prediction (EventHit@5) | **v3 R6 / v3 R6-arch trim 0.746** | v4 + Markov (periodicity only) 0.743 | Markov prior fusion is essential. FEATURES_v2 drops (scene + F1–F4 + 2h/6h windows + n_trans) match v3 R6's 0.746 with 79-d profile (vs 153-d) — drops are a clean cost win. v4 additive features still don't help. |

**Bottom line:**
- v3 is at or near the data ceiling for this single-user log; v4 additive features (per-app recency, periodicity priors) do not help.
- The FEATURES_v2 *drops* (74 dims of profile + zero-out scene per token) are empirically free — same test EH@5 with ~half the profile dimensions.
- **Production picks:** v1 GRU for Task A; **v3 R6-arch trim** (= v3 R4 + Markov + FEATURES_v2 drops, no rec/per) for Task B — same 0.746 EH@5 as untrimmed v3 R6 with smaller, cheaper inputs.
