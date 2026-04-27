# Feature Reference — Next-App Prediction Demo

A complete walkthrough of every input the v3 model sees, what each feature *means*, why we include it, and how its dimensions were chosen. Intended as a single-file reference complementing the architecture docs in `REPORT_v3.md`.

---

## TL;DR — what enters the model

Three feature levels feed the v3 model:

| Level | What | Where it goes | Total dim |
|---|---|---|---|
| **Per-event tokens** (16 short, 64 long) | "what just happened on the phone" | LocalEncoder + GlobalEncoder | 76-d / 84-d per token |
| **Per-anchor profile** (1 vector per prediction) | "what's the user's *current rhythm*" | ProfileEncoder | 153-d |
| **Markov prior** (Task B only, 1 vector per prediction) | "what app usually follows the previous app" | additive bias on Task B head | 50-d (gathered from V×V table) |

The model also receives the target labels — `target_app` (Task A) or `win_counts` (Task B) — as supervision.

---

## 1. Per-event token features

When the model looks at "the last 16 events" (LocalEncoder, in-session) or "the last 64 target events" (GlobalEncoder, cross-session), each event is converted into a **token vector**. Same schema in both encoders, with a small extra in the GlobalEncoder.

### 1.1 `app_id` → 32-d embedding

**What:** integer index into a 50-token vocabulary (3 reserved + 47 real apps after rare-collapsing).

**Dim:** 32-d, learnable.

**Why 32 and not, say, 50 (one-hot):**
- 47 active apps is a small vocab, but apps share semantic structure (WeChat ≈ MMS as messaging apps; MALL ≈ TAOBAO ≈ TAKEAWAY as shopping). A *learned* 32-d embedding lets the model discover those clusters from data; a one-hot can't.
- 32 is roughly `√(V × output_dim) = √(50 × 64) ≈ 56` — we sized down to 32 to keep params small (this user has only 5,471 train events; oversized embeddings memorize).
- Same `nn.Embedding` instance is shared between LocalEncoder and GlobalEncoder ("weight tying"). The same app should mean the same thing in both contexts.

### 1.2 `category_id` → 8-d embedding (v3 addition)

**What:** integer ID into an 11-class hand-built taxonomy:
```
PAD, social_messaging, news_feed_content, shopping_marketplace,
telephony_calling, contacts_identity, media_camera, entertainment_video,
productivity, system_utility, other_app
```

**Dim:** 8 — small enough to discourage overfitting, large enough to express category interactions (4-bit dimensions can get bumpy).

**Why include this when we already have an app embedding?**
- The vocab is heavily imbalanced (LITE alone is 37% of targets). Tail apps may have only 5–20 occurrences in training. The category gives the model another way to generalize: even if MALL and TAOBAO have few examples each, "shopping" has many.
- Categories carry semantic signal that's hard to learn from raw transitions. Two apps from the same category often serve interchangeable user intents (browse XHS or AWEME — both are "scroll content").
- The mapping is **hand-built**, not learned — that's intentional. We're injecting domain knowledge, not asking the network to derive it from a tiny dataset.

**Source:** `lib/v3/categories.py` — fixed dictionary `app_label → category`. Two raw-bundle hash IDs that have <5 events each fall back to `other_app`.

### 1.3 `loc_id` → 8-d embedding (v3 addition)

**What:** integer index into an 18-token location vocab:
```
PAD, NONE, OTHER, + top-15 most-frequent train-time locations
```
Each location label is `wifi:<SSID>` (preferred) or `cell:<CellId>` (fallback), parsed from the `device_state_update_payload` JSON column. Forward-filled within the same calendar day so events with no payload inherit the most recent observed location.

**Dim:** 8. Same sizing logic as categories — small vocab, small dim.

**Why we include it:**
- This user's app usage is highly location-dependent: WeChat at home, more shopping apps on commute (cellular), more browsing at work WiFi.
- The raw `device_state_update_payload` column is the only stable location signal available; we extract it once at preprocessing time, fit a vocab on the train split, and freeze it.
- 97.7% of train rows resolve to a non-`NONE` location after forward-fill, so the signal is dense.

**Why not GPS coordinates:** the dataset doesn't have them; only city-level (Shenzhen) is recorded, and it's the same city for the whole 42 days, so no signal there.

**Note — per-token application:** both `category_id` (1.2) and `loc_id` (1.3) are looked up *per token* in the history sequence, not just at the anchor. So every event in the 16-event short history and 64-event long history carries its own category/loc embedding. That lets the GRU and Transformer notice within-sequence patterns like "after a `news_feed_content` token while on home WiFi, the user usually opens a shopping app" — patterns that wouldn't surface from a single anchor-level snapshot.

### 1.4 Numeric per-event features → 28-d vector

This is the v1 / v2 numeric feature pack, unchanged in v3. Concatenated as a flat 28-d vector:

| Sub-feature | Dim | Encoding | Why we need it |
|---|---|---|---|
| `event_type` one-hot | 4 | TARGET / BACKGROUND / PAGE_SWITCH / OTHER | Distinguishes target events (the ones we predict) from background context. PAGE_SWITCH is informative — same-app intra-page switches often precede a category change. |
| Fourier hour | 4 | sin/cos of `2π·hour/24` and `2π·hour/12` | Smooth time-of-day encoding; 23:55 and 00:05 should be near, not far. Two periods (24h, 12h) capture both day-cycle and AM/PM. |
| weekday one-hot | 7 | Mon=0…Sun=6 | Weekday vs weekend behaviour differs (work apps vs entertainment). One-hot avoids spurious ordering. |
| `log(1 + sec_since_prev)` | 1 | standardized on train | Time gap to previous event — captures whether the user is bursting through apps or coming back after a break. Log + standardize because the raw distribution is heavy-tailed. |
| `session_position / 32` | 1 | 0–1 normalized | Where in the current session is this event? Predictions early in a session look different from late-session noise. |
| `device_state_scene` one-hot | 5 | 4 known states + UNK | "Scene" reflects rough activity context (home/walk/etc.). Heavy nulls (86% missing) — we keep an explicit UNK class. |
| `device_state_networktype` one-hot | 4 | wifi/cellular/UNK + reserved | Network change is a strong context signal — moving from WiFi to cellular usually indicates leaving home/office. |
| `is_missing_state` | 1 | binary | Tells the model when scene/network is unknown (vs deterministic UNK). Lets it down-weight the affected channels. |
| `screen_on` | 1 | binary, derived from SCREENON/OFF events | Screen state — predictions during screen-off shouldn't trust dwell features. |

**Why exactly these 28 numbers:**
- Every dimension is something we can compute *deterministically* from the raw event log without leakage.
- They cover the four context axes that matter for app prediction: temporal (hour, weekday, session position), behavioral (event type, screen state), environmental (scene, network), and pacing (time gap).
- We deliberately avoid raw timestamps, app durations, or any forward-looking signal.

### 1.5 `dt_bin` → 8-d embedding (Global encoder only)

**What:** the time gap between this token and the *anchor moment* (when we're predicting), bucketed log-uniformly into 6 buckets (0–60 s, 1–10 min, 10 min–1 h, 1–6 h, 6–24 h, >24 h).

**Dim:** 8.

**Why only in Global, not Local:** Local sees the last 16 events in the same session, all within minutes of each other — `dt_bin` would be near-constant. Global spans 64 cross-session events that can stretch back hours or days, so encoding "how stale is this token" is critical so the model can discount old observations.

### Final per-token shape

| Encoder | Concatenation | Dim |
|---|---|---|
| LocalEncoder | `app_emb(32) ⊕ cat_emb(8) ⊕ loc_emb(8) ⊕ numeric(28)` | **76** |
| GlobalEncoder | `app_emb(32) ⊕ cat_emb(8) ⊕ loc_emb(8) ⊕ numeric(28) ⊕ dt_emb(8)` | **84** |

Each is then projected by a `Linear` to the encoder's working dim (64 for GRU, 48 for Transformer).

---

## 2. Per-anchor profile features (153-d)

The token sequences capture *what just happened*. The profile vector captures *the user's broader rhythm at this moment* — things that aren't easy to read off a sequence of 16 tokens. ProfileEncoder receives a 153-d vector per prediction, projects to 64-d → ReLU → LN → Dropout → 16-d.

### 2.1 v2 baseline profile (38-d)

| Sub-feature | Dim | What it is | Why |
|---|---|---|---|
| **F1: hour-conditional top-8 app marginal** | 8 | `P(app | hour=h_now)` for the 8 globally-most-frequent apps | "What does this user usually open *at this hour*?" — captures circadian rhythm directly |
| **F2: weekday-conditional top-8** | 8 | Same idea, conditioned on weekday | Captures Monday-vs-Sunday differences |
| **F3: rolling 24h top-8 frequency** | 8 | Histogram of the user's last 24h of target events, sliced to the global top-8 apps, normalized | Captures recent personal trend (independent of static priors) |
| **F4: rolling 7d top-8 frequency** | 8 | Same as F3 but over 7 days | Captures medium-term routine |
| **F5: Fourier hour** | 4 | Same encoding as the per-event Fourier hour, applied to anchor time | Continuous time-of-day available even when no token is informative |
| **F6: session signals** | 2 | `session_position / 32`, `is_session_start (0/1)` | Where are we in the current burst of activity? |

All four count-based slices (F1–F4) are fit on **train data only**, then applied causally at val/test time. The `<PAD>`, `<UNK>`, and `<RARE>` columns are zeroed before normalization so the slices reflect only real apps.

**Why "top-8" rather than full V?** F1–F4 would be 4×50 = 200 dims if we used the full vocab. Most of those would be near-zero on this user. Slicing to the top-8 most-active apps captures 80%+ of probability mass with a tenth of the parameters.

### 2.2 `daypart` one-hot (10-d, v3 addition)

**What:** a coarse 10-class binning of (hour, weekday) → bin:
```
0 early_morning   (5–7)
1 morning_commute (7–9, weekday)
2 morning_work    (9–12, weekday)
3 noon            (12–14)
4 afternoon_work  (14–17, weekday)
5 evening_commute (17–19, weekday)
6 evening_home    (19–22)
7 night           (22–24)
8 late_night      (0–5)
9 weekend_daytime (7–19, weekend)
```

**Dim:** 10 (one-hot).

**Why this on top of Fourier hour:**
- Fourier hour is smooth and continuous, but app behavior often has *sharp* boundaries (commute hours, end of work day). One-hot captures discontinuities Fourier can't.
- 10 bins is just coarse enough to remain dense (every bin has ≥ 200 train events) and fine enough to separate distinct routines.
- This was the **biggest single contributor** in the Task B feature ablation: R3 (with daypart) → R2 (without) = **+3.0 pp** test EventHit@5.

### 2.3 Multi-window behavioral rollups (105-d, v3 addition)

For five windows W ∈ {15 m, 30 m, 1 h, 2 h, 6 h} we summarize the target events that occurred in `[t − W, t)` (strictly before the anchor — causal). Each window gets a 21-d block:

| Sub-feature | Dim | Why |
|---|---|---|
| `n_unique_apps` (log1p) | 1 | How many distinct apps did the user touch? Distinguishes "monogamous" sessions (1 app, deep) from "browsing" sessions (many apps, shallow). |
| `n_unique_categories` (log1p) | 1 | Same idea but at category level — robust to within-category app switching (WeChat → MMS counts the same). |
| `n_transitions` (log1p) | 1 | Total event count − 1. Indirect measure of activity intensity. |
| `n_self_repeats` (log1p) | 1 | Same-app-to-same-app transitions. High → user is sticking on one app (e.g. doomscrolling). |
| `n_switches` (log1p) | 1 | Different-app transitions = `n_transitions − n_self_repeats`. High → high context switching. |
| dominant category one-hot | 11 | Which category had the most dwell time in the window? | Tells the model the "vibe" of this window in one symbol. |
| top-5 app duration shares | 5 | Sorted descending share of total dwell time | Captures concentration: 0.9, 0.05, 0.03, 0.01, 0.01 means one dominant app; 0.3, 0.2, 0.2, 0.15, 0.15 means even spread |

That's **5 + 11 + 5 = 21 dims per window × 5 windows = 105 dims total**.

**Why five windows specifically:**
- 15m / 30m matches the prediction horizon (the next 15 min is *probably* doing more of what the last 15 min did).
- 1h / 2h gives the model the recent session context (is the user mid-task or just-arrived?).
- 6h gives the half-day context — useful for separating "still on the morning routine" from "transitioning to afternoon".

Why log1p these counts: the raw distribution is right-skewed (a few super-active windows blow out the dynamic range of the dense layer). `log1p` smooths it and brings the variance into the same scale as the other features.

**Causality safety:** the rollups are computed by `searchsorted` on the sorted target stream, with strict `< anchor_ts` comparison. We unit-tested this with a planted future event — it never appears in any window count.

### 2.4 Profile total

```
v2 base:       38-d
+ daypart:    +10-d
+ rollups:   +105-d
=             153-d
```

Projected to 16-d by `Linear(153 → 64) → ReLU → LN → Dropout(0.2) → Linear(64 → 16)`. The 64-d hidden layer gives the model room to compose features (e.g., "evening + heavy social_messaging recent → expect more social_messaging") before the 16-d bottleneck forces compression.

---

## 3. Markov prior — V × V table (Task B only)

Not really a "feature" in the traditional sense — it's a frozen lookup that gets *added* to the Task B sigmoid head's logits.

**What:** `(V, V)` = `(50, 50)` matrix of `log P(next_app | last_app)`, fit on train target events only with Dirichlet α=0.5 smoothing.

**How it's used at runtime:**
```
logits_b_sig = head_sig(h_fused) + α_markov · log_prior[last_app_idx]
```
where `α_markov` is a single learnable scalar (init 0.5, clipped to [0, 2], converged at 0.52).

**Why include it as a separate signal rather than learning it from data:**
- The neural model already sees the previous app via the LocalEncoder. But to derive `P(a | a_prev)` from the embedding it would need to *learn* the V×V transition pattern from 5,471 training events — i.e., re-discover the Markov-1 baseline.
- Why force it to do that when we can just hand it the closed-form table? This is an explicit case of **inductive bias > parameters** for small data.
- The single learnable scalar lets the model decide how much to trust the prior. At α=0 the prior is ignored; at α→∞ it dominates. Convergence at 0.52 means the model uses Markov as one source of evidence among several, which is the desired behavior.

**Why only on Task B:**
- Task A is a softmax over 50 classes — Markov-1's prediction is just `transition[last_app, :]`, which the GRU already easily matches/beats (v1 GRU Task A Hit@1 = 0.602 vs Markov-1 0.496).
- Task B is a multi-label set prediction over a 15-min horizon — Markov-1 is ironically *the strongest* baseline there because the rhythm of "what tends to follow what" averages across burst behavior. v3 R6 keeps Markov as a frozen prior and lets the neural model add temporal/contextual nuance on top.

---

## 4. Targets (supervision, not input)

| Task | Target | Dim | Loss |
|---|---|---|---|
| A | `target_app` (next event's app idx) | scalar (long) | Cross-entropy + label smoothing 0.05 + inv-sqrt-frequency class weights |
| B | `win_counts` (per-app launch count in the next 15 min) | (50,) float | BCE on `(win_counts > 0).float()` + 0.25 · Poisson on `win_counts` |

The Task A class weights are critical: without them, LITE (37% of training data) would dominate the gradient, and the model would learn to predict LITE most of the time. Inverse-sqrt weighting evens out the gradient signal from rare apps without going so far that it destabilizes training.

---

## 5. End-to-end picture

For a single prediction at time `t`:

```
Input data assembled per anchor:
  history_app[16]      — last 16 in-session app indices
  history_feat[16,28]  — corresponding numeric features
  history_mask[16]     — which positions are valid
  history_category[16] — derived from history_app
  history_loc[16]      — forward-filled location per row

  long_app[64]         — last 64 cross-session target apps
  long_feat[64,28]     — corresponding numeric features
  long_mask[64]
  long_dt_bin[64]      — log-bucketed time-since-anchor
  long_category[64]
  long_loc[64]

  profile[153]         — anchor-level rhythm vector
  last_app_idx         — scalar, index of immediate previous target event
                          (used for Markov prior gather)

Forward pass:
  h_local  = LocalEncoder(...)        → (64,)
  h_global = GlobalEncoder(...)       → (32,)
  h_profile = ProfileEncoder(...)     → (16,)

  h_fused = GatedFusion(h_local, h_global, h_profile)  → (64,)

  Task A:  logits_a = Linear(64 → 50)(h_fused)
  Task B:  logits_b_sig  = Linear(64 → 50)(h_fused) + α · log_prior[last_app]
           log_rate_b    = Linear(64 → 50)(h_fused)
```

Task A predicts via `argmax(softmax(logits_a))`; Task B ranks apps by `sigmoid(logits_b_sig)` and returns the top-5.

---

## 6. Sizing rationale recap

Every dim was chosen with these constraints in mind:

- **5,471 train target events** is small. Each learnable parameter is a chance to overfit.
- **47 active apps**, heavily skewed (LITE 37%). Most apps have <100 examples. Embedding dim has to be small enough to avoid memorizing per-app.
- **Single user** → no cross-user generalization to bank on.
- **CPU-only training** → wide tensors are slow.

Hence: 32-d app embedding (not 64), 8-d category/location embeddings (not 16), top-8 slices for hour/weekday marginals (not full V=50), 16-d profile bottleneck (not 32).

The total v3 model lands at:
- TaskAModelV3: **97 677** parameters
- TaskBModelV3 + Markov prior: **100 928** parameters

Both safely within range for the dataset size; both train in ~80 s on CPU; both reproduce deterministically with `seed=7`.

---

## 7. Train-only stats and where they live

All statistics that touch the data are fit on train only and persisted to disk with a `fit_split="train"` stamp that is asserted on load. This is the leakage barrier:

| Statistic | File | What it stores |
|---|---|---|
| `dt_scaler` | (in-memory, refit each run) | mean and std for `log(sec_since_prev)` standardization |
| Vocab | `artifacts/vocab.json` | Train-frequency vocab with min-count=5 cutoff |
| v2 profile stats | `artifacts/v2_profile_stats.pkl` | Hour/weekday marginals, top-8 indices for slicing |
| Category map | `artifacts/v3/category_map.json` | Hand-built `app_label → category` dictionary |
| Location vocab | `artifacts/v3/loc_vocab.pkl` | Top-15 location labels + reserved tokens |
| Per-row location IDs | `artifacts/v3/loc_ids_{train,val,test}.npy` | Aligned with parquet row order |
| Markov-1 prior | `artifacts/v3/markov_prior.pkl` | (V, V) log-prob table fit on train targets |

Loading any of these checks `fit_split == "train"`. The location IDs for val/test are produced **using the train-fit vocab** at preprocessing time — out-of-vocab labels fall back to `<OTHER>` cleanly.

---

## 8. Summary table of every dim

| Where | Feature | Dim | Source |
|---|---|---|---|
| Per-token (Local & Global) | `app_emb` | 32 (learned) | `Embedding(50, 32)` |
| | `cat_emb` | 8 (learned) | `Embedding(11, 8)` |
| | `loc_emb` | 8 (learned) | `Embedding(18, 8)` |
| | event_type one-hot | 4 | derived |
| | Fourier hour | 4 | sin/cos of t/24, t/12 |
| | weekday one-hot | 7 | derived |
| | `log1p(sec_since_prev)` standardized | 1 | derived |
| | `session_position/32` | 1 | derived |
| | scene one-hot | 5 | from raw `device_state_scene` |
| | network one-hot | 4 | from raw `device_state_networktype` |
| | `is_missing_state` | 1 | derived |
| | `screen_on` | 1 | derived from SCREENON/OFF |
| Per-token (Global only) | `dt_bin_emb` | 8 (learned) | `Embedding(6, 8)` |
| **Token total** | LocalEncoder / GlobalEncoder | **76 / 84** | |
| | | | |
| Per-anchor (profile) | hour-cond top-8 marginal (F1) | 8 | train stats |
| | weekday-cond top-8 marginal (F2) | 8 | train stats |
| | rolling 24h top-8 freq (F3) | 8 | causal stream stats |
| | rolling 7d top-8 freq (F4) | 8 | causal stream stats |
| | Fourier hour (F5) | 4 | derived |
| | session signals (F6) | 2 | derived |
| | daypart one-hot | 10 | derived |
| | window rollups | 105 | causal stream aggregates |
| **Profile total** | ProfileEncoder input | **153** | |
| | | | |
| Side input (Task B only) | Markov prior log P(next | last) gathered | 50 | train V×V table |

---

That's every dimension the model touches. The pattern across all of v3:

1. **Embeddings stay small** (8-d for categorical features other than apps) because we have ~5k samples.
2. **Hand-crafted statistics** (top-8 slices, daypart bins, Markov table) are used wherever the dataset is too small to learn the same thing from scratch.
3. **Causality is enforced explicitly** — every count, every slice, every prior uses only data with `event_ts < anchor_ts`.
4. **Train-only fitting** is mechanically enforced via `fit_split="train"` assertions on every persisted stats file.

If you want to extend v3 with a new feature, the checklist is: (a) add it to a `lib/v3/<file>.py` builder; (b) make sure it's fit on train only with the `fit_split` stamp; (c) add it to either the per-token concatenation in `LocalEncoderV3`/`GlobalEncoderV3`, or to the profile concat in `prep.py`; (d) re-run `scripts/20_build_v3_features.py` and the desired training round.
