# Next-App Prediction — Demo Report

**Author:** demo sprint, 2026-04
**Data:** 42 days of real app-usage events for a single Huawei-device user (Shenzhen)
**Scope:** two prediction tasks, four baselines, two neural models, end-to-end training and evaluation

---

## Contents
1. [Problem framing](#1-problem-framing)
2. [Dataset](#2-dataset)
3. [Data-processing pipeline](#3-data-processing-pipeline)
4. [Feature vector (per event)](#4-feature-vector-per-event)
5. [Vocabulary](#5-vocabulary)
6. [Train / val / test split](#6-train--val--test-split)
7. [Task A — Next-app prediction](#7-task-a--next-app-prediction)
8. [Task B — 15-minute window prediction](#8-task-b--15-minute-window-prediction)
9. [Model architectures](#9-model-architectures)
10. [Training procedure](#10-training-procedure)
11. [Baselines](#11-baselines)
12. [Metrics](#12-metrics)
13. [Results](#13-results)
14. [Analysis & interpretation](#14-analysis--interpretation)
15. [Reproduction](#15-reproduction)
16. [Limitations & follow-ups](#16-limitations--follow-ups)

---

## 1. Problem framing

We study one Huawei phone user across 42 days of continuous event telemetry. We cast next-app behavior as two prediction tasks:

- **Task A — Next-app.** Given the user's recent event history up to a target event, predict the app associated with that target event. Per-event classification over the 50-way vocabulary.
- **Task B — 15-minute window.** Given history up to a time *t*, predict the *set* of apps the user will launch in `[t, t+15 min]`. Multi-label set prediction.

These are complementary: Task A is the classic sequential problem (MISApp / TGT / Chen et al.) — "what will the user open next"; Task B is the deployment framing — "what will the user need in the near future" (useful for pre-fetching, notification routing, suggestion surfaces).

Both are evaluated on the same user's held-out days with identical splits.

---

## 2. Dataset

Source: `app_usage_cleaned_dictionary_mapped.xlsx`, sheet `final_compact` — a cleaned event log for one Huawei device in Shenzhen, 2026-03-01 → 2026-04-12 (42 days).

| Property | Value |
|---|---|
| Raw rows | 47,237 |
| After dedup | 43,964 (−7 %) |
| Target events (`APP_FOREGROUND` + `APP_START`) | 7,125 |
| App-usage rows | 37,509 |
| Unique apps | 70 → **50 vocab** (47 apps + 3 reserved tokens) after rare-collapse |
| Sessions | 2,557 (300-s idle cutoff, 32-event session cap) |

**Event-type distribution (after enrichment):**

| evt_type | Count | What it is |
|---|---|---|
| TARGET | 7,125 | APP_FOREGROUND / APP_START — the events we predict |
| BACKGROUND | 3,699 | APP_BACKGROUND — app left foreground |
| PAGE_SWITCH | 13,291 | ABILITY_OR_PAGE_SWITCH — intra-app navigation |
| OTHER | 19,849 | SCREENON/OFF, INTO_HOME_KEY, INTOKEYGUARD, DEVICE_STATE_UPDATE, PROCESS_START, … — context |

**Class imbalance (top targets):** LITE 37 %, WeChat 17 %, XXXXXX 11 % — top-3 apps = 65 % of all targets.

**Other signals (embedded in DEVICE_STATE_UPDATE payloads):** screen state, network type (WiFi / cell), `scene` (Huawei OS network-context category, values 1..4), location at base-station resolution, cell ID, signal strength, carrier. Raw JSON is parsed once; we keep `scene`, `networktype`, plus derived booleans.

---

## 3. Data-processing pipeline

Four passes over the raw sheet, all in `lib/data.py`:

### 3.1 Load
Read the xlsx `final_compact` sheet with openpyxl; parse `event_ts` to pandas datetime.

### 3.2 Deduplicate
Collapse `(event_ts, app_label_clean, name_norm)` triples; keep the first occurrence; retain a `dup_count` column so the model can still see bursty duplicates as a feature.
- Raw rows: **47,237**
- After dedup: **43,964** (−3,273; max duplicate run = 6)

### 3.3 Enrich
Derive:
- `sec_since_prev` (re-computed after dedup so it reflects the surviving stream), **clipped at 3,600 s** to avoid outliers from overnight gaps destabilizing the `log1p` standardization.
- `hour`, `minute`, `weekday` (0=Mon … 6=Sun).
- `evt_type` ∈ {TARGET, BACKGROUND, PAGE_SWITCH, OTHER}.
- `session_id`: new session whenever `sec_since_prev > 300 s`; each session is capped at 32 events (overflows start a new session). This gives us a fixed-size history window upper bound.
- `session_position`: 0-indexed rank within the session.
- `scene` (from `DEVICE_STATE_UPDATE` payload), forward-filled within each calendar day; `-1` means "no scene known yet today."
- `net` (network type from the payload), same treatment.
- `is_missing_state`: 1 if scene/net were unresolved at this row (kept as an explicit signal, not imputed).
- `screen_on`: 1 after `SCREENON_EVENT`, 0 after `SCREENOFF_EVENT`, carry-forward. Default 1 if no SCREEN event yet.

### 3.4 Inspect — real row example
A concrete training target from the processed data:

```
event_ts          = 2026-03-01 08:22:31
app_label_clean   = LITE
name_norm         = APP_START
evt_type          = TARGET
session_id        = 9
session_position  = 6
hour              = 8, weekday = 6 (Sunday)
sec_since_prev    = 3.0
scene             = 2, net = 2
is_missing_state  = 0, screen_on = 1
```

And the 16 events immediately preceding it in the same session:

```
08:22:21  SCREENON_EVENT  —              OTHER         +723  screen=1  (session starts; >300s idle before)
08:22:26  INTO_HOME_KEY   —              OTHER           +5
08:22:26  APP_FOREGROUND  LITE           TARGET          +0
08:22:28  APP_BACKGROUND  LITE           BACKGROUND      +2
08:22:28  APP_FOREGROUND  PHONE_CONTACTS TARGET          +0
08:22:28  PROCESS_START   CALLUI         OTHER           +0
08:22:31  APP_START       LITE   ← target to predict    +3
```

The model sees the 7 events before 08:22:31 (with 9 pad slots left-filled) and must predict `LITE`.

---

## 4. Feature vector (per event)

Every event becomes one **token**. Each token contributes:

### 4.1 Learned app embedding (32-dim)
`Embedding(vocab_size=50, dim=32)` lookup. Dense 32-d vector; learned during training. Apps with similar transition patterns drift to nearby points in this space.

### 4.2 Hand-crafted numeric features (28-dim)

| Slice | Dim | Feature | Encoding |
|---|---|---|---|
| [0:4] | 4 | evt_type | one-hot: TARGET / BACKGROUND / PAGE_SWITCH / OTHER |
| [4:8] | 4 | Fourier hour | `[sin(2πh/24), cos(2πh/24), sin(2πh/12), cos(2πh/12)]` |
| [8:15] | 7 | weekday | one-hot Mon..Sun |
| [15] | 1 | `log1p(sec_since_prev)` | standardized (train mean/std) |
| [16] | 1 | `session_position / 32` | 0..1 |
| [17:22] | 5 | scene | one-hot over {UNK, 1, 2, 3, 4} |
| [22:26] | 4 | net | one-hot over {UNK, 0, 1, 2} |
| [26] | 1 | `is_missing_state` | 0/1 |
| [27] | 1 | `screen_on` | carry-forward 0/1 |

### 4.3 Per-token input the model sees
```
token = concat( app_embedding(app_idx),     # 32
                numeric_features )          # 28
      -> Linear(60 → 64) + LayerNorm        # projection to d_model=64
```

So each event becomes a 64-dim vector. A history of 16 events is a `(16, 64)` tensor.

### 4.4 Side input (per sample, not per token)
`side_hour_fourier ∈ ℝ^4` at the *prediction time* (target's hour for Task A, anchor's hour for Task B). This is what drives the TGT-lite temporal gate.

---

## 5. Vocabulary

Built on training-split target events only (no leakage).

- 70 raw `app_label_clean` values observed overall.
- **Rare-app collapse:** any app with < 5 target events in train is mapped to `<RARE>`.
- Result: **50 tokens**
  - index 0 = `<PAD>` (padding for short histories)
  - index 1 = `<UNK>` (unseen at inference)
  - index 2 = `<RARE>` (collapsed rare apps)
  - indices 3..49 = 47 real apps, sorted by frequency

Top apps: `LITE=3, WECHAT=4, XXXXXX=5, CALLUI=6, PHONE_CONTACTS=7, CLOCK_CALENDAR=8, MMS=9, GALLERY=10, ...`

Test-set target coverage (fraction of test target apps whose label exists in train vocab): **99.3 %**.

---

## 6. Train / val / test split

**Strict chronological split** with a 60-minute embargo on the training-side of each boundary so `sec_since_prev` can't leak across splits.

| Split | Days | Rows | Targets |
|---|---|---|---|
| Train | 2026-03-01 → 2026-04-03 (30 days, minus 60-min edge) | 34,691 | 5,471 |
| Val | 2026-04-03 → 2026-04-08 (5 days) | 5,521 | 1,071 |
| Test | 2026-04-08 → 2026-04-12 (5 days, last one partial) | 3,752 | 583 |

---

## 7. Task A — Next-app prediction

### 7.1 Per-sample construction

**One training sample per target event.** For each target at row `i` in the enriched stream:

- **Label `y`** = `app_idx(row i)` — the app to predict (one of 50).
- **History** = the 16 events immediately preceding row `i` *within the same session* (so we never cross a 5-min idle gap). If fewer than 16 events precede the target in its session, we left-pad with `<PAD>` tokens and mark them `history_mask = False`.
- **History includes ALL event types** — TARGET, BACKGROUND, PAGE_SWITCH, OTHER — each wearing its evt_type one-hot. This is deliberate: non-target events carry timing and intent signal (screen-on, intra-app navigation).
- **Side input:** `side_hour_fourier` computed from the target's own hour (but not its app — that would leak the label).

Training-sample totals:

| Split | Samples (= # targets) |
|---|---|
| Train | 5,471 |
| Val | 1,071 |
| Test | 583 |

Per-batch shapes:

```
history_app         (B, 16)          int64
history_feat        (B, 16, 28)      float32
history_mask        (B, 16)          bool        # True = real event, False = pad
side_hour_fourier   (B, 4)           float32
target_app          (B,)             int64       # gold label, 0..V-1
```

### 7.2 Loss
`L_A = CrossEntropy(logits_a, target_app)` with label smoothing ε = 0.05 and inverse-sqrt-frequency class weights (dampens LITE's 37 % dominance).

### 7.3 Inference
`argmax(logits_a)` for top-1, `argsort(-logits_a)[:K]` for top-K.

---

## 8. Task B — 15-minute window

### 8.1 Anchor

An **anchor** is the moment in time where we stand and ask "what apps will the user launch in the next 15 minutes?". We use a **5-minute uniform grid** from 06:00 to 24:00 on each evaluation day — ≈ 216 anchors/day — to avoid biasing the evaluation toward busy periods (which would happen if we only evaluated at target-event timestamps).

Of 1,080 val anchors we keep only those with at least one target event in the next 15 min (**446 non-empty anchors** on val, **282** on test). Empty-horizon anchors are skipped — they don't have a prediction to score.

### 8.2 Ground truth

For anchor *t*, define the multiset:

```
G_w(t, Δ)  = { (app_a, count_a) : app_a appears count_a times
                                  as a target in [t, t + Δ=15min] }
G(t, Δ)    = unique keys of G_w(t, Δ)
```

Real example (anchor `2026-04-03 06:45:00`, 15-min horizon):

```
06:45:00  anchor
06:45:03  CLOCK_CALENDAR  target
06:45:?   LITE            target       × many
...
06:59:xx
```

→ `G_w = {CLOCK_CALENDAR: 4, LITE: 4}`, `|G| = 2`, total events = 8.

Across val, unique apps per anchor: **mean 2.18, max 7**. Events per anchor: **mean 4.3**.

### 8.3 Training supervision

During training we don't use the 5-min grid. We attach a Task-B label to **each training target event** (same 5,471 samples as Task A):

- At each target's timestamp `t`, compute `G_w(t, 15min)` over the rest of the stream.
- Build a `win_counts ∈ ℝ^50` vector of counts per app (zero where app didn't appear).

This gives dense, structurally aligned supervision: the model sees Task A's "predict next app" and Task B's "what will the next 15 min look like" simultaneously on the same input history.

### 8.4 Loss
```
L_B = 0.5 · BCE(logits_b_sig, 1_{win_counts > 0})       # existence head
    + 0.2 · PoissonNLL(log_rate_b, win_counts)          # count head
```

The BCE head learns "which apps appear" (multi-label classification), the Poisson head learns "how often." They share the backbone but have separate output layers.

### 8.5 Joint objective

Every learnable model is trained to minimize:

```
L = L_A + L_B = CE + 0.5·BCE + 0.2·PoissonNLL
```

All three heads gradient-flow through the same GRU / TGT backbone.

### 8.6 Inference — the B1 decoder

At evaluation we switch from training-time anchors (target events) to **evaluation-time anchors (5-min grid)**:

```python
for t in anchor_grid(eval_days, stride=300, hours=[6,24]):
    history     = last_16_events_before(t)   # can cross sessions
    hour_fourier= fourier_hour(t.hour)
    h           = model.backbone(history, hour_fourier)   # (64,)
    scores      = sigmoid(model.heads.b_sig(h))           # (50,)
    # ties broken by log_rate_b
    top_k_apps  = argsort(-scores)[:K]
    compare(top_k_apps, ground_truth_multiset)
```

**Why a different history sampler for eval?** At an arbitrary grid anchor like 14:35:00 we may not be in a session at all; we might be in a post-lunch idle gap. We take the last 16 events with `ts < t` from the whole enriched stream, not session-restricted. `history_mask` stays mostly True because 16 events is small enough to fit within one recent session for any active day.

**B2 — iterative rollout** (scoped but not implemented in this sprint): unroll Task A recursively, pretending each prediction is the next event, until the simulated clock exceeds 15 min. More faithful to the sequence but accumulates error and is ~16× more compute per anchor. Left as follow-up.

### 8.7 Metrics at each anchor

Let `T_K` = the model's top-K set, `G` = the ground-truth set, `G_w` = the frequency-weighted multiset.

| Metric | Formula |
|---|---|
| Precision@K | `|T_K ∩ G| / K` |
| Recall@K    | `|T_K ∩ G| / |G|` |
| F1@K        | harmonic mean of P@K and R@K |
| Jaccard@K   | `|T_K ∩ G| / |T_K ∪ G|` |
| Coverage@K  | 1 if `G ⊆ T_K` else 0 |
| **EventHit@K** (primary) | `Σ count_a · 𝟙[a ∈ T_K] / Σ count_a` over `(a, count_a) ∈ G_w` |

**Why EventHit is our primary metric:** if the user opens LITE 4 times and WeChat 4 times in the window (8 events total), predicting both correctly is worth all 8 events — more representative of the *experienced* quality of the prediction than a flat set-recall.

Reported at K ∈ {1, 3, 5, 10}; headline tables use K=5.

---

## 9. Model architectures

### 9.1 Shared backbone

All learnable models use the same **TokenEncoder + dual-head** setup:

```
Input (per token):
   app_idx           int      ──┐
                                 ├─ concat ─ Linear(60→64) ─ LayerNorm ─► x ∈ ℝ^64
   numeric features  ℝ^28    ──┘

Sequence of 16 tokens: (B, 16, 64)
   │
   ▼
Backbone (GRU or Transformer) ─► pooled ∈ ℝ^64

Heads:
   head_A    : Linear(64 → 50)            logits_a       (Task A softmax)
   head_B_sig: Linear(64 → 50)            logits_b_sig   (Task B multi-label)
   head_Brate: Linear(64 → 50)            log_rate_b     (Task B Poisson rate)
```

### 9.2 GRU-64
- 1 layer, hidden size 64, dropout 0.2.
- Pooling: last non-padded token of the sequence.
- **~40 k parameters total.**

### 9.3 TGT-lite
- 2 Transformer encoder layers, `d_model=64`, 4 heads, 4× FFN expansion, dropout 0.2.
- **Fourier-hour gate:** a learned MLP `Linear(4→64) → SiLU → Linear(64→64) → sigmoid` converts `side_hour_fourier` into a (B, 64) gate, element-wise multiplied with the pooled token representation. This is the "temporal gating" primitive from the TGT paper (arXiv:2502.16957).
- Pooling: last non-padded token.
- **~120 k parameters total.**

### 9.4 Why these sizes?
With only 5,471 training targets, larger models overfit fast. We picked the smallest architectures that can represent meaningful interactions:
- GRU-64 has enough capacity to capture short-term transitions + context.
- TGT-lite has 3× more parameters; its extra capacity is bet on the hope that the temporal gate will transfer daily rhythm signal. Empirically it didn't pay off at this scale (see §11).

---

## 10. Training procedure

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Max epochs | 25 |
| Early-stop patience | 5 epochs without val-Hit@5 improvement |
| Gradient clipping | max-norm 1.0 |
| CE label smoothing (Task A) | ε = 0.05 |
| Class weights | inverse-sqrt frequency over train target apps |
| Task A weight in loss | 1.0 |
| Task B BCE weight | 0.5 |
| Task B Poisson weight | 0.2 |
| Device | CPU (no GPU needed; training completes in minutes) |
| Seed | 7 |

Combined training loss per batch:

```
L = CE(logits_a, target_app, smoothing=0.05, class_weight=w)
  + 0.5 * BCE_with_logits(logits_b_sig, (win_counts > 0).float())
  + 0.2 * Poisson(exp(log_rate_b), win_counts)
```

**Convergence:** GRU-64 reached its peak val Hit@5 around epoch 17 and then slowly plateaued, early-stopping at epoch 22. TGT-lite showed similar convergence but with higher variance across epochs — consistent with its 3× larger parameter count at the same data scale.

**Training cost (CPU, single threaded):**
- GRU-64: ~0.5 s/epoch × 22 epochs ≈ **11 s total**
- TGT-lite: ~1.2 s/epoch × 11 epochs ≈ **13 s total**

---

## 11. Baselines

Four non-learning baselines, all fit in milliseconds on the train target stream.

### 11.1 MFU (Most Frequently Used)
Global frequency count of apps in train targets. Every anchor gets the same ranked list.
- Task A: argmax is always LITE, the top-1 app by frequency.
- Task B: top-K is always `[LITE, WeChat, XXXXXX, XHS_HOS, CALLUI]`.
- Defines the "null model" floor — beating MFU means the model uses the input at all.

### 11.2 MRU (Most Recently Used)
The last non-reserved app in the history window. Only one app gets score 1, all others 0.
- Task A: predicts self-repetition. Surprisingly strong because 56 % of consecutive targets repeat the same app.
- Task B: same single app, so top-K is effectively K=1 padded with zeros — hurts recall.

### 11.3 HourMFU (hour-conditioned MFU)
Per-hour frequency table `count[hour][app]`, smoothed + normalized into 24 distributions. Prediction picks the hour bucket of the target/anchor.
- Task A: often slightly worse than MFU because HourMFU can be tricked by rare apps that happen to be frequent at one hour.
- Task B: slightly better than MFU because the top-5 sets are hour-tailored.

### 11.4 Markov-1 (first-order Markov chain)
Count-based transition matrix over train target events. `P(next_target | previous_target)` with Dirichlet α=0.5 smoothing. Previous target is carried through the session.
- Captures self-transitions AND legitimate app-switches (PHONE_CONTACTS → CALLUI is high-probability even if WhatsApp-to-something is rare).
- Surprisingly strong on Task B because the Markov row for a "sticky" app is precisely "repeat + a few likely alternatives" — a perfect 15-min prior.

---

## 12. Metrics

### 12.1 Task A
- **Hit@1** – argmax equals label.
- **Hit@5 / Hit@10** – label in top-5 / top-10 predictions.
- **MRR** – mean reciprocal rank of the true app in the full sorted list.
- **macro-F1** – per-class F1 averaged over V classes (exposes class-imbalance cheating).

### 12.2 Task B (per anchor, then mean-aggregated across anchors)
- **Precision@K** = `|T_K ∩ G| / K`
- **Recall@K**    = `|T_K ∩ G| / |G|`
- **F1@K**        = `harmonic_mean(P@K, R@K)`
- **Jaccard@K**   = `|T_K ∩ G| / |T_K ∪ G|`
- **Coverage@K**  = `𝟙[G ⊆ T_K]`
- **EventHit@K**  = `(Σ_{a ∈ G_w ∩ T_K} c_a) / (Σ c_a)` *(frequency-weighted; primary metric)*
- **EventHit_micro@K** — same numerator/denominator pooled across anchors.

All Task B metrics reported at K ∈ {1, 3, 5, 10}; the headline table uses K = 5.

---

## 13. Results

### 13.1 Task A — next-app prediction

| Model | Split | Hit@1 | Hit@5 | MRR | macro-F1 |
|---|---|---|---|---|---|
| MFU | val | 0.219 | 0.627 | 0.389 | 0.012 |
| MFU | test | 0.240 | 0.679 | 0.429 | 0.017 |
| MRU | val | 0.559 | 0.696 | 0.619 | 0.493 |
| MRU | test | 0.504 | 0.669 | 0.573 | 0.498 |
| HourMFU | val | 0.190 | 0.655 | 0.389 | 0.026 |
| HourMFU | test | 0.244 | 0.736 | 0.443 | 0.046 |
| Markov-1 | val | 0.558 | 0.805 | 0.671 | 0.414 |
| Markov-1 | test | 0.496 | 0.808 | 0.635 | 0.343 |
| **GRU-64** | val | **0.614** | **0.861** | **0.724** | **0.501** |
| **GRU-64** | test | **0.602** | 0.840 | **0.707** | 0.542 |
| TGT-lite | val | 0.556 | 0.831 | 0.678 | 0.406 |
| TGT-lite | test | 0.513 | 0.808 | 0.642 | 0.485 |

### 13.2 Task B — 15-min window (B1 decoder, primary: EventHit@5)

| Model | Split | P@5 | R@5 | EventHit@5 | Coverage@5 | Jaccard@5 |
|---|---|---|---|---|---|---|
| MFU | val | 0.262 | 0.631 | 0.657 | 0.433 | 0.270 |
| MFU | test | 0.277 | 0.605 | 0.622 | 0.390 | 0.271 |
| MRU | val | 0.204 | 0.555 | 0.566 | 0.350 | 0.217 |
| MRU | test | 0.189 | 0.461 | 0.465 | 0.259 | 0.200 |
| HourMFU | val | 0.268 | 0.659 | 0.689 | 0.444 | 0.278 |
| HourMFU | test | 0.296 | 0.668 | 0.689 | 0.465 | 0.298 |
| **Markov-1** | **val** | 0.284 | **0.721** | 0.743 | **0.502** | **0.301** |
| **Markov-1** | **test** | 0.289 | **0.679** | **0.688** | **0.500** | **0.300** |
| GRU-64 | val | 0.277 | 0.672 | 0.696 | 0.457 | 0.288 |
| GRU-64 | test | 0.272 | 0.623 | 0.641 | 0.418 | 0.279 |
| TGT-lite | val | 0.265 | 0.652 | 0.681 | 0.442 | 0.273 |
| TGT-lite | test | 0.264 | 0.604 | 0.614 | 0.390 | 0.271 |

---

## 14. Analysis & interpretation

### Self-repetition is the dominant signal
MRU's Hit@1 of 0.559 on val means **>55 % of consecutive target events repeat the same app** — scrolling LITE, messaging WeChat, etc. No model can look non-trivial unless it clears this bar.

### GRU-64 is the best for next-app
GRU wins Task A by +5 pp Hit@1 over the strongest baseline (Markov-1), +17 pp over MFU. This is the value of sequence modeling with explicit context: hour, weekday, screen_on, scene, and event-type tokens let the model refine transition patterns that Markov-1 alone can't see.

Interestingly, macro-F1 tracks the same ordering (GRU 0.501 > Markov-1 0.414). Models aren't just cheating on the LITE majority; they are correctly discriminating across the 47-app vocabulary.

### Markov-1 wins Task B
This is the most interesting finding. On the 15-minute window:

- **Markov-1 EventHit@5 = 0.743 val / 0.688 test** — the best number in the table.
- GRU EventHit@5 = 0.696 / 0.641 — close, but consistently behind.

Why? The 15-min horizon for this user is dominated by self-repetition plus a small set of co-used apps. Markov-1's top-5 is essentially `[self, most-likely-switch-1, most-likely-switch-2, MFU-filler-1, MFU-filler-2]` — precisely the structure of a 15-min window. The neural models have more expressive ranking but there isn't enough *variance* in the ground truth for that expressiveness to pay off at this horizon.

### TGT-lite underperforms GRU
TGT-lite has 3× the parameters but loses by ~6 pp on Task A Hit@1. At 5,471 training examples the Transformer's capacity is wasted — consistent with the TGT paper itself noting that temporal gating benefits really accrue at ≥30k targets or multi-user settings. At single-user scale the GRU's smaller capacity plus stronger inductive bias for sequence modeling wins.

### Self-consistency check
The baselines respect the expected ordering:
- MFU < Markov-1 in Task A (static frequency loses to transition modeling).
- MRU ≫ MFU in Task A Hit@1 (0.559 vs 0.219) — self-repetition is the single biggest signal.
- MRU ≪ MFU in Task B (0.566 vs 0.657 EventHit@5) — a one-app prediction can't cover a diverse 15-min window.
- HourMFU > MFU in Task B but not always Task A — hour helps more when you're predicting a *set* than a single next event.

All consistent with the data's structural properties; the pipeline looks sound.

---

## 15. Training sanity checks

- GRU train loss decreased monotonically for 17 epochs (from 4.47 → 2.31), val Hit@5 plateaued around epoch 17–22.
- TGT-lite train loss plateaued faster (11 epochs), consistent with mild overfitting at this data scale.
- All metric unit tests in `lib/metrics.py` pass (perfect precision = perfect recall when `|G| = K`; Jaccard = 1 when T = G; MAP = 1 for perfect ranking).
- Task B ground-truth construction validated against two hand-coded anchors.

---

## 16. Reproduction

```bash
cd app_usage_data
python scripts/01_prep_data.py        # builds splits + vocab (artifacts/)
python scripts/02_run_baselines.py    # MFU / MRU / HourMFU / Markov1
python scripts/03_train_gru.py        # trains GRU-64 checkpoint
python scripts/04_train_tgt.py        # trains TGT-lite checkpoint
python scripts/05_window_eval.py      # neural Task A + B (B1 decoder)
python scripts/07_report.py           # assembles tables + figures
```

Runtime on a laptop CPU: ≈ 5 minutes end-to-end.

Artifacts land in:
- `artifacts/splits/{train,val,test}.parquet` — processed data splits
- `artifacts/vocab.json` — 50-token vocabulary
- `artifacts/checkpoints/{gru.pt, tgt.pt}` — trained model weights
- `artifacts/results/{baselines,task_a,task_b}.json` — per-model metrics
- `artifacts/results/task_{a,b}_table.csv` — flat tables
- `figures/{task_a_hit1.png, task_b_eventhit5.png}` — headline bar charts

---

## 17. Limitations and caveats

- **Single user, 42 days.** No cross-user generalization demonstrated.
- **Test window is 5 days (583 targets).** Wilson 95 % CIs on Hit@1 are ~±4 pp; differences below that threshold are not significant.
- **Task B used only the B1 decoder** (direct sigmoid head). The B2 rollout decoder (iterate the Task A model forward 15 min) is scoped in the plan but not implemented in this sprint.
- **No ablations run** (temporal gate on/off, context tokens on/off, missingness flag on/off, vocabulary size).
- **Hour features are at hour granularity.** Continuous time-of-day (seconds within day, Fourier) would be a cheap change; left as a future ablation.
- **Class imbalance is severe.** LITE alone is 37 % of training targets. The `macro_f1` column in Task A corrects for this; Hit@1 does not.

---

## 18. Follow-up work

1. Implement the B2 (iterative rollout) decoder for Task B; compare head-to-head with B1.
2. Run the four mandatory ablations on TGT-lite:
   - no temporal gate (constant gate = 1).
   - no context tokens (mask out BACKGROUND / PAGE_SWITCH / OTHER).
   - no `is_missing_state` flag.
   - full 70-class vocab vs rare-collapsed 47-class.
3. Replace integer-hour Fourier with continuous time-of-day Fourier (`seconds_since_midnight / 86400`).
4. Blocked 5-fold temporal CV on the top-2 models; report mean ± std Hit@1 / EventHit@5.
5. Extend horizon to 30 min and 60 min; the Markov-beats-neural pattern may or may not hold.

---

## Appendix A — file layout

```
app_usage_data/
├── app_usage_cleaned_dictionary_mapped.xlsx      source data
├── paper_deep_dive.md                            literature notes (MISApp, TGT, …)
├── REPORT.md                                     this document
├── README.md                                     reproduction instructions
├── lib/
│   ├── data.py        load / dedup / enrich / split / vocab / anchor grid
│   ├── features.py    per-event encoder + history builders
│   ├── baselines.py   MFU / MRU / HourMFU / Markov-1
│   ├── models.py      GRU-64 and TGT-lite with dual heads
│   ├── train.py       class weights, TargetWindowDataset, window-count targets
│   └── metrics.py     hit@k, mrr, macro-F1, P/R/F1, EventHit, coverage, CIs
├── scripts/
│   ├── 01_prep_data.py
│   ├── 02_run_baselines.py
│   ├── 03_train_gru.py
│   ├── 04_train_tgt.py
│   ├── 05_window_eval.py
│   └── 07_report.py
├── artifacts/
│   ├── splits/*.parquet
│   ├── vocab.json
│   ├── session_stats.json
│   ├── checkpoints/*.pt
│   └── results/*.{json,csv}
└── figures/*.png
```

## Appendix B — hyperparameter quick-ref

| Setting | Value |
|---|---|
| Vocab size V | 50 (47 apps + PAD/UNK/RARE) |
| App embedding dim | 32 |
| Numeric feature dim | 28 |
| Per-token input dim | 60 → Linear → 64 |
| d_model | 64 |
| History length k | 16 |
| Task B horizon | 15 min |
| Task B anchor stride (eval) | 5 min |
| Task B daily window | 06:00 – 24:00 |
| Session idle cutoff | 300 s |
| Session max length | 32 events |
| Rare-app threshold | < 5 train targets |
| Train / Val / Test | 30 / 5 / 5 days, 60-min embargo |
| Batch size | 256 |
| Learning rate | 1e-3 (AdamW) |
| Weight decay | 1e-4 |
| Label smoothing | 0.05 |
| Loss weights | `L_A + 0.5·BCE + 0.2·Poisson` |
| Class weights | inverse-sqrt frequency |
| Max epochs | 25 |
| Early-stop patience | 5 |


---

> **Follow-up work** — two successive experiments live alongside this report:
> - `REPORT_v2.md`: per-task models with a 64-event global-history Transformer + 38-d hand-crafted profile vector. Headline: +2.3 pp Task A Hit@1 on val over v1 GRU.
> - `REPORT_v3.md`: v2 + per-token category + WiFi/Cell-ID location embeddings, daypart one-hot, multi-window rollups (15 m – 6 h), and Markov-1 prior fusion on the Task B head. Headline: **test EventHit@5 = 0.746**, +5.8 pp over the strongest v1 baseline (Markov-1) and +11.8 pp over v2.
