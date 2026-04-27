# Feature Proposal — v2 (cleaner v3, evidence-grounded)

**Status:** Proposal, not yet implemented. This is a deliberate redesign of the v3
feature set guided by what the v3 ablations actually showed. The current
implementation is documented in `FEATURES.md`; this document is the *target*
state we'd move to next, with motivation for every add and every drop.

---

## 0. TL;DR

- **Drop 4 feature blocks** that ablations showed are neutral, redundant, or noise:
  - `device_state_scene` one-hot (5 dims, per-token) — 86% null and a relabeling of `networktype` we already encode
  - F1–F4 in profile (32 dims): hour-conditional, weekday-conditional, rolling 24h, rolling 7d top-8 marginals — replaced (better) by `daypart` + multi-window rollups
  - 6 h window rollup (21 dims) — almost always reverts to `news_feed_content` and adds noise more than signal
  - Linear-redundant per-window scalars — `n_transitions = n_self_repeats + n_switches` (drop n_transitions); top-5 → top-3 shares (drop bottom 2 since they're <0.05 noise)
- **Add 2 new feature blocks** that fill gaps not covered by v3:
  - Per-app recency (8 dims): "how long since each headline app was last used"
  - Periodicity priors (18 dims): same-hour-yesterday, same-hour-last-week phase-aligned lookups
- **Keep** everything that has empirical evidence: `app_emb`, `category_emb`, `loc_emb`, daypart one-hot, multi-window rollups (trimmed), Markov-1 prior fusion on Task B.
- **Net dim change**: profile 153 → 95 (−38%); token Local 76 → 71 / Global 84 → 79.

| Configuration | v3 (current) | v2 (proposed) |
|---|---|---|
| Per-token dim (Local / Global) | 76 / 84 | 71 / 79 |
| Profile dim | 153 | 95 |
| Total params (Task B + Markov) | ~101k | ~95k |
| Markov prior fusion | yes | yes (unchanged) |

The hypothesis is: same or better metric performance, less overfit risk, more interpretable signal map, ~6k fewer params.

---

## 1. Empirical evidence (the v3 ablations)

`artifacts/results/task_{a,b}_v3_R*.json` gives us round-by-round Δ on the same architecture. Translating to ground truth:

### Task B — test EventHit@5

| Round | Delta from prior | What was added | Test EH@5 |
|---|---|---|---|
| R0 (baseline, no v3 features) | — | — | 0.703 |
| R1 | + `category` per-token | (8-d) | 0.702 |
| R2 | + `loc` per-token | (8-d) | 0.694 |
| R3 | + `daypart` in profile | (10-d) | **0.724** |
| R4 | + multi-window rollups | (105-d) | 0.714 |
| R6-lite (no v3 features, just Markov prior) | + Markov | — | **0.733** |
| R6 (full v3 + Markov prior) | + everything + Markov | — | **0.746** |

| Δ test EH@5 | Block in question |
|---|---|
| **+0.030** | `daypart` one-hot |
| **+0.030** | Markov prior (alone) |
| +0.013 | Multi-window rollups + cat + loc *together*, only when Markov is also on (R6 vs R6-lite) |
| −0.001 | `category` token embedding alone |
| −0.008 | `loc` token embedding alone |
| −0.010 | Multi-window rollups alone (without Markov) |

### Task A — test Hit@1

R0 → R4: 0.583 → 0.599. Range across all 5 rounds is 0.556 to 0.599. With 583 test events the 95% CI is roughly ±3pp, so **none of the v3 feature additions are detectable signal on Task A**. Task A is feature-saturated by the local app sequence; v2/v3 architecture gains never materialized at test scale.

### What the data tells us, plainly

- **Daypart and Markov each carry one big chunk of signal** (~3pp each). They're the two empirical wins and we keep both.
- **Multi-window rollups, category, location** are all neutral-to-slightly-negative *on their own* but combine usefully *with Markov* (full R6 beats R6-lite by 1.3pp). They earn their keep but barely.
- **F1–F4 (the v2 hand-crafted profile features)** were never directly ablated in v3 because v3 inherits them from v2 unchanged. But theoretically they encode the same hour/weekday rhythm signals that daypart now does in 10 dims (vs F1+F2's 16). Two routes to the same signal in the same model = redundancy.
- **Per-token `device_state_scene`** is mostly null (86%) and the present codes (1, 2, 3, 4) almost perfectly correlate with `networktype` (which is already encoded). Verified empirically.

---

## 2. Per-feature audit & verdict table

### 2.1 Per-event token features

| Feature | v3 dim | Verdict | Reasoning |
|---|---|---|---|
| `app_id` → embedding | 32 | **Keep** | Essential. The only way the model carves semantic similarity between apps. |
| `category_id` → embedding | 8 | **Keep** | Net-neutral on its own (R1 vs R0), but cheap and supports tail-app generalization through shared category structure. Helps the *embedding regularization* even if it doesn't move the metric. |
| `loc_id` → embedding | 8 | **Keep** | Slightly hurts alone (R2 vs R1) but contributes when stacked. Captures real "where is the user" context. |
| event-type one-hot | 4 | **Keep** | TARGET vs BACKGROUND vs PAGE_SWITCH vs OTHER is an irreplaceable category — supervised vs context-only events. |
| Fourier hour | 4 | **Keep** | Smooth, cheap, well-justified. |
| weekday one-hot | 7 | **Keep** | No natural ordering; one-hot is right; cheap. |
| `log1p(sec_since_prev)` standardized | 1 | **Keep** | Pacing signal, well-encoded. |
| `session_position / 32` | 1 | **Keep** | Burst-depth signal. |
| **`device_state_scene` one-hot** | **5** | **DROP** | 86% null; the 14% that fires is collinear with `networktype` (1=WiFi, 3=cellular, mixed=2/4=none). Verified by direct crosstab (see §3). |
| `device_state_networktype` one-hot | 4 | **Keep** | Denser than scene; complements `loc_id`. |
| `is_missing_state` | 1 | **Keep** | Lets the model down-weight imputed-state rows. |
| `screen_on` | 1 | **Keep** | Strong context (screen-off events are almost never user-initiated). |
| `dt_bin` (Global only) → embedding | 8 | **Keep** | Essential for the Global encoder; staleness signal. |

**Per-token totals after edits**:
- LocalEncoder: 32 + 8 + 8 + 4 + 4 + 7 + 1 + 1 + 4 + 1 + 1 = **71-d** (was 76)
- GlobalEncoder: 71 + 8 (`dt_bin`) = **79-d** (was 84)

### 2.2 Per-anchor profile features (v3 = 153-d → v2 = 95-d)

| Block | v3 dim | v2 verdict | Reasoning |
|---|---|---|---|
| F1 hour-conditional top-8 | 8 | **Drop** | Fixed top-8 slot identity; loses 15% of probability mass. Daypart (10-d) gives sharper, more discriminative time-of-day partitioning at lower dim. |
| F2 weekday-conditional top-8 | 8 | **Drop** | Same redundancy with weekday one-hot already in token features. |
| F3 rolling 24h top-8 | 8 | **Drop** | Replaced cleanly by per-app recency (8-d) which encodes *when* the app last fired, not just whether it appeared in some 24h window. |
| F4 rolling 7d top-8 | 8 | **Drop** | With only 42 days of data, "rolling 7d" is essentially user-baseline — the mean is dominated by the global app distribution, no marginal signal. |
| F5 Fourier hour at anchor | 4 | **Keep** | Smooth time-of-day. Only 4 dims; complements daypart's discrete bins with a continuous representation. |
| F6 session signals | 2 | **Keep** | `session_position`, `is_session_start`. Cheap, and `session_pos` for an arbitrary anchor is *not* the same as session_pos at the row that triggered the anchor, so it provides distinct info. |
| **Daypart one-hot** | **10** | **Keep — load-bearing** | Empirically the biggest single profile contribution (+3pp test EH@5 in R3 vs R2). |
| `is_weekend` | NEW (1) | **Add** | Cheap; weekend behaviour is markedly different. Daypart bins partially absorb this but a dedicated flag costs nothing and helps the model not have to derive it. |
| `weekday_idx / 6` | NEW (1) | **Add** | Continuous Mon–Sun position in case there's a mid-week gradient (Mon vs Wed differs even with same daypart). |
| `is_session_start_at_anchor` (binary) | NEW (1) | **Add** | At anchor time we may not be inside a session; this captures whether the most-recent target was within the 5-min idle threshold. |
| `log1p(sec_since_session_start)` | NEW (1) | **Add** | Continuous version of session_position based on time, not event count. |
| Multi-window rollups | 105 | **Trim to 51** | Drop 6h, drop 2h windows (mean-revert to daypart's signal); drop `n_transitions` per window (linear-redundant); reduce top-5 → top-3 dwell shares. |
| Per-app recency (top-8 apps) | NEW (8) | **Add** | log1p(sec_since_last_use) for each top-8 app, clipped at 7d. Strong per-app decay signal that no current feature captures. |
| Periodicity priors (24h, 7d) | NEW (18) | **Add** | At anchor time *t*, look up dominant top-8 app at `t − 24h ± 30m` and `t − 7d ± 30m`. Captures explicit daily/weekly routine. |
| Markov gather (Task A only) | NEW (8) | **Add (Task A)** | `log_prior[last_app, top_8_app_indices]` injected as 8 profile dims. Mirrors what Task B gets at the head-fusion level. |

### 2.3 Side inputs

| Side input | Verdict | Reasoning |
|---|---|---|
| Markov-1 (V, V) log-prior fused on Task B sigmoid head with learnable α | **Keep — load-bearing** | R6 vs R6-lite confirms the prior contributes ~+3pp test EH@5; learned α = 0.52 confirms it's in active use. |
| Class weights (inverse-sqrt-freq) for Task A CE | **Keep** | Critical for not collapsing to LITE (37% of training mass). |

---

## 3. Why each new feature is worth adding

### 3.1 Per-app recency (8-d)

For each of the global top-8 apps `a ∈ {LITE, WeChat, Huawei bundle, AWEME, MMS, MALL, GALLERY, HEALTH}` (frozen at train time):

```
recency_a = log1p( min( anchor_ts - last_target_ts(a) , 7 days ) )
```

If app `a` was just used a moment ago, its slot is near 0; if not seen for a week, it's clipped to log1p(7d) ≈ 13.4.

**Why this works for both tasks**:
- **Task A**: complements the LocalEncoder's last-token signal. The encoder sees "the most recent event was WeChat", but it doesn't know whether MALL was used 2 minutes ago vs 6 hours ago. Recency surfaces that.
- **Task B**: directly relevant to "what apps fire in next 15 min". An app touched 30s ago has a very different probability of firing than one untouched in 6 hours, and the relationship is monotonic in recency.

**Why top-8 only**: tail apps have <100 train events; their per-app recency estimates are noisy, and they rarely fire in any window anyway. The model can fall back on the per-token embedding signal for tail apps.

**Causality**: computed by `np.searchsorted` over the sorted target stream with strict `<` (no leakage of future events).

### 3.2 Periodicity priors (18-d): same-hour-yesterday + same-hour-last-week

For each lookback period `L ∈ {24h, 7d}`:
- Find target events in `[anchor_ts − L − 30min, anchor_ts − L + 30min]`.
- Compute multinomial mode (most-frequent app) over top-8 apps + 1 OTHER bucket.
- One-hot encode → 9-d.

Two periods × 9-d = **18 dims**.

**Why this is a real new signal:**
- Daypart bins captures coarse rhythm ("morning_commute" vs "evening_home").
- It does not capture *cycle-locked* patterns: "every Tuesday at 9 AM I open Calendar to check the day's meetings".
- 24h periodicity captures daily routines; 7d captures weekly ones (work-week structure, Sunday family-call patterns).
- Output is one-hot (sparse, low-noise), so it's calibrated against the headline apps where signal is strongest.

**Causality safety**: lookbacks use only target events strictly before the anchor; we don't peek at "real" yesterday-or-last-week if those days fall in val/test. Implementation will assert on this with a planted-future test.

### 3.3 Anchor-side temporal trims

- `is_weekend` (1-d): cheap and orthogonal to weekday one-hot in token features (weekday spans Mon-Sun; is_weekend collapses to a single binary).
- `weekday_idx / 6` (1-d): continuous index — the model can learn smooth weekly drift if it exists.
- `is_session_start_at_anchor` and `log1p(sec_since_session_start)` (2-d): for anchor times that aren't target events (Task B's 5-min anchor grid), tells the model whether we're in the middle of a session or its first moment.

### 3.4 Markov-1 gather as profile (Task A only, 8-d) — optional

In v3, Markov is fused only on Task B's sigmoid logits. For Task A, the LocalEncoder sees the previous app at its last token, but the head has to *re-derive* the transition probabilities through the gradient.

Add a profile feature: `log_prior[last_app, top_8_app_indices] ∈ ℝ^8`. This makes the closed-form transition prior available to the Task A softmax head without forcing it through the gated fusion bottleneck.

Expected lift: 0.5–1pp Task A test Hit@1, modest but worth it given the cost (8 dims, ~70k extra params via the profile head).

---

## 4. Why each removal is justified

### 4.1 `device_state_scene` (5-d one-hot per token)

Direct evidence from the raw data (verified in this session):

```
scene NaN     : 41,838 rows (88%)
scene = 1     :  2,547 rows  (always WiFi   → networktype = 0)
scene = 3     :  2,551 rows  (mostly cellular → networktype = 2)
scene = 2     :    295 rows  (mixed: 141 WiFi, 154 cellular — transition state)
scene = 4     :      6 rows  (network info missing)
```

Scene fires only on `DATA_NETWORK_CHANGED` payloads. Codes 1 and 3 are **direct relabelings** of `networktype = 0` (WiFi) and `networktype = 2` (cellular). Code 2 is a transition state. Code 4 is anomalously rare.

We already encode `networktype` as a 4-d one-hot. The marginal information from `scene` over `networktype` is essentially "the user's network changed in the last few minutes" — which is already implied by `is_missing_state` and `loc_id` reverting to a prior value. Adding 5 more dims for redundancy is wasted capacity.

**Risk:** the 295 transition-state rows might be informative (e.g. "user just left home WiFi"). If we want that signal, a 1-d "network_just_changed" boolean (built from a 60-second lookback) replaces the 5-d one-hot at 1/5 the cost.

### 4.2 F1–F4 (v2 hand-crafted profile, 32 dims)

F1 (hour-conditional top-8) and F2 (weekday-conditional top-8) duplicate signal that daypart's 10-d encoding produces with sharper bin edges. F3 (rolling 24h) and F4 (rolling 7d) duplicate the multi-window rollups (now trimmed to 15m / 30m / 60m) and the new periodicity priors at 24h / 7d.

Together they're 32 dims of redundancy.

### 4.3 6h and 2h window rollups (42 dims)

R3 (just daypart) test EH@5 = 0.724.
R4 (daypart + 5 windows) test EH@5 = 0.714.

The 5-window stack actively *hurts* without Markov. The 15m/30m/60m windows are doing real work; the 2h and 6h windows are just diluting it. After Markov is added (R6), test EH@5 = 0.746 vs R6-lite (no v3 features) = 0.733; the extra 13pp comes from the *useful* parts of the rollups, not the long-horizon ones. Trimming to 3 windows preserves the win without the noise.

### 4.4 `n_transitions` per-window scalar (5 dims)

`n_transitions = n_self_repeats + n_switches`. Linear redundancy. The Linear layer downstream can recover it; carrying it explicitly is wasteful.

---

## 5. Final feature list

### Per-token (Local, GruEncoder input)

| Feature | Dim | Notes |
|---|---|---|
| `app_emb` | 32 | learned, V=50, weight-tied to Global |
| `category_emb` | 8 | learned, 11 classes, weight-tied |
| `loc_emb` | 8 | learned, 18 classes, weight-tied |
| event-type one-hot | 4 | TARGET / BACKGROUND / PAGE_SWITCH / OTHER |
| Fourier hour | 4 | sin/cos at 24h and 12h periods |
| weekday one-hot | 7 | Mon–Sun |
| log1p(sec_since_prev) standardized | 1 | per-event pacing |
| session_position / 32 | 1 | depth in current session |
| networktype one-hot | 4 | WiFi / cellular / unknown |
| is_missing_state | 1 | flag for null/UNK device state |
| screen_on | 1 | derived |
| **Total** | **71** | (was 76 in v3) |

### Per-token (Global, Transformer input)

| Feature | Dim | Notes |
|---|---|---|
| All Local features | 71 | (same as above) |
| `dt_bin_emb` | 8 | learned, 6 buckets, log-spaced staleness |
| **Total** | **79** | (was 84 in v3) |

### Per-anchor profile

| Block | Dim | Description |
|---|---|---|
| Anchor temporal | 16 | Fourier hour (4) + daypart one-hot (10) + is_weekend (1) + weekday/6 (1) |
| Session state | 4 | session_position/32, is_session_start, is_session_start_at_anchor, log1p(sec_since_session_start) |
| Recent rollups (3 windows × 17 dims) | 51 | Per W ∈ {15m, 30m, 60m}: 4 scalars + 11 dominant_cat one-hot + 3 top-app dwell shares |
| Per-app recency | 8 | log1p(time-since-last-use) for top-8 apps, clipped at 7d |
| Periodicity priors | 18 | Same-hour-yesterday + same-hour-last-week → top-8 + OTHER one-hot, 9 each |
| Markov prior gather (Task A only) | 8 | log_prior[last_app, top_8_indices] |
| **Total profile dim (Task A)** | **105** | |
| **Total profile dim (Task B)** | **97** | (no Markov gather; prior is on the head) |

(The slight bump above 95 in §0 reflects the optional Task A Markov gather; treat the spec as 95-d if you skip that.)

### Side inputs (unchanged from v3)

- Markov-1 log-prior `(50, 50)` fit on train, fused on Task B sigmoid head with learnable scalar α (init 0.5, clipped [0, 2])
- Class weights for Task A CE (inverse-sqrt-frequency)

### Targets (unchanged from v3)

- Task A: `target_app` scalar; loss = label-smoothed CE with class weights
- Task B: `win_counts ∈ ℝ^50` (15-min horizon counts); loss = BCE on `(win > 0)` + 0.25 · Poisson(λ, win)

---

## 6. Verification of the `device_state_scene` removal

```
scene × networktype crosstab (verified at proposal time):
                  net=0(WiFi)  net=2(cellular)  net=-1(UNK)
  scene=1            2547             0              0      ← exclusively WiFi
  scene=2             141           154              0      ← transition state
  scene=3              52          2499              0      ← exclusively cellular
  scene=4               0             0              6      ← network unknown
  scene=NaN             —             —              —      (86% of rows)
```

`networktype` (already encoded as 4-d one-hot) carries the same partition into WiFi/cellular/unknown with much better coverage. The 295 "transition state" rows (scene=2) might carry an extra bit, but they're 0.6% of the data and we'd rather drop the 5-d one-hot than carry it for that fraction. We can add an explicit `network_changed_in_last_60s` binary flag if we miss it.

---

## 7. New code paths (proposed)

```
lib/v3/
  recency.py          ← NEW: per-app recency builder (causal lookups via searchsorted)
  periodicity.py      ← NEW: 24h and 7d phase-aligned dominant-app lookups
  features_v3.py      ← MODIFY: assemble profile with new blocks, drop F1–F4
  window_rollups.py   ← MODIFY: drop 6h, 2h windows; reduce per-window dim from 21 to 17
  models_v3.py        ← MODIFY: ConfigV3 numeric_dim 28 → 23, profile_dim 153 → 95
scripts/
  20_build_v3_features.py  ← MODIFY: also fit top-8 app indices for recency / periodicity
  21_train_task_a_v3.py     ← MODIFY: use new profile builder, accept Markov gather flag
  22_train_task_b_v3.py     ← MODIFY: use new profile builder
  27_eval_v2_proposal.py    ← NEW: side-by-side eval of v3 R6 vs proposed v2 stack
```

The v3 stack stays usable; the proposed v2 features live behind a new flag (`--use-v2-features`) so we can A/B them cleanly.

---

## 8. Validation plan

To confirm the proposal moves the needle (or at least doesn't regress):

| Round | Config | Expected Δ vs v3 R6 |
|---|---|---|
| **A. Parity** | All v3 features minus dropped blocks (no scene, no F1–F4, no 2h+6h windows). No new blocks. | ±0.5pp test EH@5; sanity check that drops are harmless |
| **B. + recency** | A + per-app recency (8-d) | +0.5–1.5pp expected (Task B more than A) |
| **C. + periodicity** | B + 18-d periodicity priors | +0.5–1pp |
| **D. + Markov gather (Task A only)** | A–C + Task A markov-prior profile feature (8-d) | +0.5–1pp on Task A Hit@1 |
| **E. Full v2** | All of the above | Should reach or exceed v3 R6 (test EH@5 = 0.746) at lower profile dim |

Stop rule: if any round B/C/D regresses by > 1pp on val, drop that block and re-run. Same overfit guards as v3 (dropout 0.2, early-stop patience 6, train-only fit on every stat).

5-fold blocked temporal CV on the winning config; report mean ± std.

---

## 9. What I'm uncertain about

To be honest about confidence:

- **Per-app recency**: high confidence it adds signal (Task B is fundamentally about "which apps will fire in the next window" and recency is the strongest temporal feature); medium confidence on the magnitude (0.5–1.5pp).
- **Periodicity priors**: medium confidence. 42 days isn't a lot — the 7d lookback effectively only has 6 reference days for any anchor. There's a chance this overfits to early-data idiosyncrasies. Worth testing but I'd expect smaller gains than recency.
- **F1–F4 removal**: high confidence in safety (their information is fully recoverable from daypart + per-app recency + periodicity priors).
- **`device_state_scene` removal**: high confidence (verified redundancy).
- **6h window removal**: high confidence (the 6h dominant category is `news_feed_content` for ~70% of anchors; mostly noise).
- **Markov-prior gather for Task A**: low-stakes — small dim (8), small expected gain (0.5–1pp). Worth trying; if it doesn't help, drop it cleanly.

---

## 10. Implementation pointers

1. Add `lib/v3/recency.py` with `build_recency_for_anchors(anchors, target_stream, top8_apps)` returning `(M, 8)` log1p-clipped recency in seconds.
2. Add `lib/v3/periodicity.py` with `build_periodicity_for_anchors(anchors, target_stream, lookback_seconds, half_window_seconds=1800)` returning `(M, 9)` one-hot (top-8 apps + OTHER).
3. Modify `lib/features.py::NUMERIC_FEAT_DIM` to drop scene one-hot (28 → 23). Update `encode_events` accordingly.
4. Modify `lib/v3/features_v3.py::profile_v3_dim` to compute new dim from blocks, not v2-base.
5. Modify `scripts/20_build_v3_features.py` to also persist `top8_apps` (the indices used by recency and periodicity, frozen at train time).
6. Modify `lib/v3/models_v3.py::ConfigV3` defaults: `numeric_dim=23`, `profile_dim=95`.
7. The two training scripts (21, 22) get a `--use-v2-features` flag that swaps the profile-build call.
8. New eval script `scripts/27_eval_v2_proposal.py` runs the validation table in §5 head-to-head against v3 results.

Order of implementation by expected ROI: drop scene → drop F1–F4 → trim windows → add per-app recency → add periodicity → optional Task A Markov gather.

---

## 11. Headline summary

**Removed from v3 (84 dims):**
- `device_state_scene` (5 dims, per token) — verified redundant with networktype
- F1–F4 in v2 base profile (32 dims) — replaced with sharper signals (daypart + recency + periodicity)
- 6h and 2h windows in multi-window rollups (42 dims) — empirically near-noise without Markov
- `n_transitions` per window (3 dims, linear-redundant with `n_self_repeats + n_switches`)
- Top-5 → top-3 dwell shares per window (2 dims, bottom 2 are mostly < 0.05)

**Added (34 dims):**
- Per-app recency for top-8 apps (8 dims)
- Periodicity priors at 24h + 7d, top-8 + OTHER one-hot (18 dims)
- Anchor-side temporal flags: `is_weekend`, `weekday_idx/6`, `is_session_start_at_anchor`, `log1p(sec_since_session_start)` (4 dims)
- Markov gather for Task A: 8 dims (Task A only)

**Net dim change**:
- Token (Local / Global): 76 / 84 → **71 / 79** (5-dim drop from scene removal)
- Profile: 153 → **95** (Task B) / **103** (Task A with Markov gather)
- Total trainable params (Task B + Markov): ≈ 95k (was 101k)

**Result**: a leaner, better-justified, more interpretable feature set where every dimension can be defended on either empirical or first-principles grounds — and the two confirmed wins (daypart, Markov fusion) are preserved unchanged.
