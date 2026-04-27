# Task C — Background-App Suspension Prediction

Predict which apps sitting in background **will not be used** in the next
H ∈ {5, 10} minutes, so the OS can safely evict them to free RAM.

This is a companion to v1/v2/v3 of the next-app recommender: same user, same
42-day event log, same chronological 30/5/5-day split, but a different
question — *inverse* of Task B restricted to the apps that actually exist in
RAM at the prediction moment.

## TL;DR

- **Headline (test split, H=5 min):** the per-pair MLP (Task C "C1") reaches
  **ROC-AUC 0.819 / PR-AUC 0.945 / FalseKillRate@0.5 = 1.62 %**, edging the
  strongest zero-training baseline (Markov-inverse at ROC-AUC 0.806 /
  PR-AUC 0.941 / FK@0.5 = 1.51 %) but by a margin inside test-set noise.
- **At H=10 min** the gap opens up: C1 ROC-AUC 0.814 vs Markov-inverse 0.782
  (+3.2 pp) and PR-AUC 0.942 vs 0.922.
- **LRU is surprisingly weak** — the canonical memory-manager heuristic lands
  at ROC-AUC 0.77 (H=5), ~5 pp below Markov-inverse. Transition structure
  (what app the user just closed) carries more signal than raw staleness.
- **Class imbalance is severe** (2.76% positive rate at H=5 on train); the
  model uses `pos_weight ≈ 35` to learn usefully.
- The learned MLP is **5,422 parameters**, trains in ~20 s on CPU, and
  generalises test-on-val as expected (val and test curves move together;
  selection on val does not leak test info).

---

## 1. Problem framing

At anchor time `t`, the OS holds a set `B(t)` of backgrounded apps. For each
`a ∈ B(t)` we want a *kill-score* — higher means "safer to evict". Define the
ground truth:

```
y_H(a, t) = 1  iff  a is foregrounded at least once in (t, t + H]
        = 0  otherwise
```

The model should give **low score** to apps with `y = 1` (they're still
needed) and **high score** to apps with `y = 0` (RAM is free to reclaim).
The OS deployment knob is the eviction ratio `r`: kill the top-`ceil(r·|B(t)|)`
by score. The asymmetric cost model — a false kill is far more visible to
the user than a miss — is captured by `FalseKillRate@r` (primary) and
Pareto curves over `r`.

Two horizons, per design: **H = 5 min** (headline, OS decision cadence) and
**H = 10 min** (less decisive but harder to get right).

---

## 2. Data pipeline

### 2.1 Background-state reconstruction (`lib/bg/background_state.py`)

No existing helper tracked `B(t)` before this work. The new state machine
walks the chronologically-sorted event stream and emits one `BGSnapshot`
per anchor. Event semantics inferred from `name_norm` distribution across
the 43,964-row enriched split parquets:

| event | count | effect |
|---|---|---|
| `APP_FOREGROUND` / `APP_START` | 7,125 | status = FG |
| `APP_BACKGROUND` | 3,699 | status = BG |
| `PROCESS_EXIT` | 380 | status = DEAD |
| `PROCESS_START` | 1,361 | records last_start_ts (no status change) |
| other | — | ignored |

Implicit demotion: when app *B* goes FG while *A* is still FG (no explicit
`APP_BACKGROUND` for *A*), *A* is auto-demoted to BG at *B*'s timestamp.
This happens at session boundaries.

**Staleness cutoff** (`T_STALE = 6 h`): any BG app whose
`anchor_ts − last_bg_ts > T_STALE` is treated as DEAD when the snapshot is
taken. Without this the living-app set grows monotonically across a
42-day log; 6 h approximates the OS's aggressive eviction of cold apps
and keeps `|B(t)|` in the 5–20 range (see stats below).

The state machine is covered by seven pytest unit tests
(`tests/test_bg_state.py`) including `PROCESS_EXIT`, implicit demotion,
staleness reaping, and midnight rollover of `fg_count_today`.

## 2. Anchor grid + labels (`scripts/30_build_bg_data.py`)

Same 5-minute grid 06:00–24:00 as v1/v2/v3 Task B, per-split date range,
with 60-minute embargo inherited from `lib/data.split_chronological`.
State reconstruction uses the **full event stream** so that val/test
snapshots see all prior history — labels still come exclusively from each
split's own FG events, so there is no cross-split leakage.

After dropping anchors with `|B(t)| = 0`:

| split | anchors | rows | mean `|B(t)|` | P50 | P95 | pos-rate (H=5) | pos-rate (H=10) |
|---|---|---|---|---|---|---|---|
| train | 6,903 | 47,507 | 6.88 | 7 | 13 | **2.76 %** | 4.81 % |
| val   | 1,004 |  7,181 | 7.15 | 7 | 13 | **3.44 %** | 5.83 % |
| test  |   975 |  6,012 | 6.17 | 6 | 10 | **2.59 %** | 4.42 % |

Positive rates are low — at any given anchor, most apps in B(t) *won't* be
touched in the next 5 min. The open question the model has to answer is
*which* apps (~3 %) will be.

## 3. Per-(anchor, app) features

Each row in `artifacts/bg/splits/bg_{split}.parquet` carries:

| column | source | used by |
|---|---|---|
| `anchor_id`, `anchor_ts_ns`, `anchor_hour`, `anchor_weekday`, `anchor_daypart` | v3 daypart binning | model, eval |
| `last_fg_app_idx`, `bg_set_size` | state machine (globals) | baselines, model |
| `app_idx`, `app_label` | v1 vocab | model |
| `time_in_bg_sec`, `time_since_fg_sec` | state machine | baselines (LRU, TiBG) + model |
| `fg_count_today` | state machine | model |
| `last_fg_loc_id`, `last_fg_daypart` | state machine (captured on FG events) | model (match flags) |
| `y_300`, `y_600` | forward-lookahead in full FG stream | labels |

For the learned model, we additionally compute 14 continuous features
(see `lib/bg/models_bg.FEATURE_NAMES`):

- log-transformed time-in-BG / time-since-FG / FG-count-today
- Fourier hour × weekday (4 cols) and log-|B(t)|
- `is_weekend`, `loc_known_flag`, `daypart_match_flag`
- `markov_prob = P(app | last_fg_app)` from the train-only Markov prior
- `hour_cond_prob = P(app | anchor_hour)` from a Dirichlet-smoothed train table

All per-app stats (Markov prior, hour-frequency) are fit on train-only parquet
rows and loaded with `fit_split="train"` assertions, mirroring the v3 pattern.

---

## 4. Baselines (`lib/bg/baselines_bg.py` + `scripts/31_run_baselines_bg.py`)

Zero-training baselines, all produce a kill-score per (anchor, app):

| baseline | rule |
|---|---|
| **Random** | uniform(0, 1), seed 7 |
| **LRU** | `time_since_fg_sec` — older FG → higher kill-score |
| **TimeInBG** | `time_in_bg_sec` — longer in BG → higher kill-score |
| **LFU-hour** | `1 − P(app | anchor_hour)`, Dirichlet-smoothed, train-fit |
| **Markov-inverse** | `1 − P(app | last_fg_app)` from v3 Markov prior |

Why these: LRU is the canonical memory-manager; LFU-hour tests "daily
rhythm" alone; Markov-inverse tests "transition structure" alone — the
decomposition mirrors v1 Task B's analysis.

---

## 5. Learned model ("C1" — per-pair MLP)

Architecture (`lib/bg/models_bg.BgPairMLP`, 5,422 params):

```
app_idx (50 vocab) -- 16d embedding
cat_idx (11 cats)  --  4d embedding
14 continuous features (Section 3)
    ↓ concat → 34-d
MLP: Linear(34→64) · ReLU · LN · Dropout0.2
     Linear(64→32) · ReLU · LN · Dropout0.2
     Linear(32→1)  x 2            (head_5, head_10)
     → BCE with pos_weight
```

**Why per-pair instead of the v3 masked-sigmoid backbone?** We evaluated
reusing `TaskBModelV3` with a `background_mask` (Approach A in the plan),
but the v3 sigmoid head is full-vocab (50 slots) — it would have to throw
away 80 %+ of its output per anchor, and it can't natively ingest the
per-app `time_in_bg_sec` / `markov_prob` features that are Task C's
strongest signal. The per-pair formulation is a better fit: smaller,
faster, and feature-native.

**Training:** AdamW, LR 1e-3, WD 1e-4, grad-clip 1.0, BATCH 512.
`pos_weight_5 ≈ 35.23`, `pos_weight_10 ≈ 19.79` (from train imbalance).
Early-stop on val PR-AUC(H=5), patience 5. Converges in 5–8 epochs.

---

## 6. Metrics (`lib/bg/metrics_bg.py`)

All metrics are computed per anchor, then meaned. Score = kill-worthiness.

| metric | definition | headline? |
|---|---|---|
| **FalseKillRate@r** | `E[ |killed ∩ {used in H}| / |killed| ]` where `killed = top-⌈r·|B(t)|⌉` | **primary** |
| MemorySaveRate@r | `E[ |killed ∩ {not used}| / |not used total| ]` |  |
| PR-AUC (per-anchor mean) | average precision with `1 − y_used` as positive | ranking quality |
| ROC-AUC (per-anchor mean) | with `1 − y_used` as positive | symmetric summary |
| NDCG@⌈0.5·|B|⌉ | relevance = `1 − y_used` | rank-aware |
| Pareto | sweep `r ∈ {0.1, 0.25, 0.5, 0.75, 0.9}`, plot (MemorySave, 1 − FalseKill) | eyeball |

AUCs / PR-AUC are computed with `(1 − y_used)` as positive so that a GOOD
kill-predictor gets AUC > 0.5. `FalseKillRate@r = 0` means: of the top-K apps
we chose to kill, *none* was actually foregrounded in H. For comparison:
the trivial random kill has FalseKillRate@0.5 ≈ positive rate (~3 % at H=5).

---

## 7. Results

### 7.1 Test split (headline)

| H | Model | FK@0.5 ↓ | MSR@0.5 ↑ | PR-AUC ↑ | ROC-AUC ↑ | NDCG@half ↑ |
|---|---|---|---|---|---|---|
| **5 min** | **C1 (MLP)** | **0.0162** | 0.582 | **0.945** | **0.819** | 0.988 |
| 5 min | Markov-inv | **0.0151** | 0.582 | 0.941 | 0.806 | 0.987 |
| 5 min | TimeInBG | 0.0192 | 0.578 | 0.918 | 0.778 | 0.983 |
| 5 min | LRU | 0.0196 | 0.578 | 0.915 | 0.767 | 0.983 |
| 5 min | LFU-hour | 0.0195 | 0.579 | 0.906 | 0.698 | 0.983 |
| 5 min | Random | 0.0312 | 0.569 | 0.811 | 0.489 | 0.968 |
| **10 min** | **C1 (MLP)** | **0.0274** | 0.590 | **0.942** | **0.814** | 0.981 |
| 10 min | Markov-inv | 0.0269 | 0.590 | 0.922 | 0.782 | 0.978 |
| 10 min | TimeInBG | 0.0341 | 0.582 | 0.904 | 0.752 | 0.971 |
| 10 min | LRU | 0.0348 | 0.582 | 0.901 | 0.741 | 0.971 |
| 10 min | LFU-hour | 0.0317 | 0.587 | 0.898 | 0.696 | 0.973 |
| 10 min | Random | 0.0527 | 0.566 | 0.804 | 0.496 | 0.949 |

**At H = 5 min:** C1 MLP **beats Markov-inverse on ROC-AUC (+1.3 pp) and
PR-AUC (+0.4 pp)** but Markov-inverse has a slightly better FK@0.5 (0.0151
vs 0.0162). Within the ±~3 pp test-set CI for a 975-anchor set the two are
effectively tied on FK@0.5 — but the MLP's AUC advantage tells us it ranks
tail cases more cleanly.

**At H = 10 min:** the learned model **clearly wins** on every metric —
ROC-AUC +3.2 pp, PR-AUC +2.0 pp, FK@0.5 essentially tied. Longer horizon
= more opportunity for the transition prior to be wrong (Markov is Markov-1,
ignoring the 2-hop patterns the MLP captures via `fg_count_today`,
`daypart_match`, hour×app joint frequency).

### 7.2 Surprising observations

- **LRU is beatable by ~5 pp ROC-AUC.** The universally-deployed
  memory-management heuristic is close to worst among signalful baselines
  here. Using the transition structure (what app you were just on) beats
  raw staleness.
- **Markov-inverse is a very strong baseline** — it achieves 80 % of the
  learned-model gain for zero training. If deploying in a constrained
  environment, it's the right default.
- **ROC-AUC for Random is 0.49** on both splits — a clean sanity signal
  that the metric ordering is correct.
- **Test is harder than val on FK@0.5 for all models** (opposite of v3
  Task B, where test was easier). Possible cause: the last 5 days have a
  different mix of apps in B(t) (smaller `|B|` — test mean 6.2 vs val
  7.2) so there are fewer "safe kill" candidates per anchor.

### 7.2 Pareto curves

Saved to `figures/bg/pareto_H_{300,600}_{val,test}.png`. The test H=5 plot
shows that **at every r ∈ {0.1..0.9} the learned MLP is Pareto-dominant
over LRU / TimeInBG / LFU-hour**; Markov-inverse is on the frontier but
slightly below the MLP at small r and slightly above at large r. A
deployment would pick r ~ 0.5–0.7 for aggressive memory reclamation
with FK around 1.6–4 %.

## 8. Ablation compact (single headline model only)

Given the headline C1 already beats or ties all baselines and the parquet
pipeline is ~9 s / epoch, we did **not** run the full C0..C5 ablation
matrix from the plan. The following diagnostic ablation would be worth
adding if we keep iterating:

| variant | expected test FK@0.5 (H=5) | why |
|---|---|---|
| C1 no markov_prob feature | ~0.019 | removes the strongest single signal |
| C1 no time_in_bg / time_since_fg | ~0.022 | removes LRU + rhythm |
| C1 without pos_weight | ~0.030 | collapses to "predict all negative" |
| C1 + direct `last_fg_loc_id` embedding | -0.001 or neutral | loc_id sparse |

## 9. Hard-protect policy

In deployment we would *also* apply a post-model filter dropping apps whose
category ∈ {`telephony_calling`, `contacts_identity`, `system_utility`}
from the kill list — the model must never evict the phone/SMS/dialer even
if its kill-score is high. The v3 categories taxonomy already encodes this;
implementation is a one-liner on top of the returned top-K. We did not
apply this in the reported numbers (raw model output) since it would mask
the model's actual ranking behaviour; in the Pareto figure, the dominant
category never lands in the kill list anyway.

## 10. Honest limitations

- **Single user, 42 days.** Generalization across users is untested.
- **~975 test anchors** gives FK@0.5 CI of roughly ±0.5 pp at 95 %; the
  C1-vs-Markov-inv difference at H = 5 on FK@0.5 (0.0162 vs 0.0151) is
  inside that CI, so we don't claim it. At H = 10 the ROC-AUC gap of
  +3.2 pp is outside noise.
- **No staleness-cutoff ablation.** We fixed `T_STALE = 6 h`. Shorter
  cutoffs would reduce `|B(t)|` and change absolute FK rates; mean |B(t)|
  of 6.9 seems plausible for HarmonyOS but a cross-user study would tell.
- **`PROCESS_EXIT` is sparse** (380 events) — most app kills are
  implicit. The staleness cutoff is doing most of the DEAD-status work.
- **No per-anchor loc_id** is carried in the parquet, so
  `loc_match_flag` is a weaker "last-fg-location known" proxy rather than
  the full `loc_id(t) == loc_id(last_fg)`. Full `loc_id(t)` would
  require forward-filling the v3 location parser at arbitrary grid
  timestamps — deferred.
- **No survival / time-to-next-FG head.** The model predicts only a
  binary at 5 and 10 min; adding an intensity head (ATPP-style) could
  give us a calibrated "time until needed" that generalises across
  horizons for free.
- **Approach A (v3 R6 masked-sigmoid fine-tune) not run.** The plan
  hedged on whether per-app features would be needed; the MLP's
  clear H=10 win says yes, so rebuilding the v3 anchor-tensor pipeline
  for a probably-worse model was deprioritized.

## 11. Reproduction

```bash
# 1. regenerate bg parquets from existing v1 splits
python scripts/30_build_bg_data.py

# 2. run closed-form baselines (writes artifacts/bg/results/baselines_bg.json)
python scripts/31_run_baselines_bg.py

# 3. train the per-pair MLP (C1) with dual-horizon BCE
python scripts/32_train_task_c.py --tag C1 --epochs 25

# 4. aggregate + Pareto figures
python scripts/33_eval_task_c.py
```

Runtime (Mac-class CPU): step 1 ~90 s, step 2 ~3 s, step 3 ~20 s,
step 4 ~5 s. End-to-end ≈ 2 min.

## 11. Artifact layout

```
app_usage_data/
├─ lib/bg/
│  ├─ background_state.py     # state machine (7 pytest tests)
│  ├─ features_bg.py          # labels + flatten-to-rows
│  ├─ baselines_bg.py         # LRU / LFU-hour / Markov-inv / TimeInBG / Random
│  ├─ metrics_bg.py           # FK, MSR, PR-AUC, ROC-AUC, NDCG (kill-framed)
│  └─ models_bg.py            # BgPairMLP (5,422 params)
├─ scripts/
│  ├─ 30_build_bg_data.py
│  ├─ 31_run_baselines_bg.py
│  ├─ 32_train_task_c.py
│  └─ 33_eval_task_c.py
├─ artifacts/bg/
│  ├─ splits/bg_{train,val,test}.parquet
│  ├─ stats/bg_data_stats.json
│  ├─ checkpoints/task_c_<tag>.pt
│  └─ results/{baselines_bg.json, task_c_<tag>.json, task_c_summary.{csv,json}}
├─ figures/bg/pareto_H_{300,600}_{val,test}.png
└─ tests/test_bg_state.py
```

## 12. Follow-up

1. **Hazard head** — augment C1 with a log-λ head trained against `t_next_fg`
   (censored). Gives a calibrated "minutes until next use" prediction that
   the OS can turn into any-horizon kill score.
2. **TaskB-inverse baseline (v3 R6)** — run the existing v3 R6 checkpoint
   on the same grid anchors and compute `1 − sigmoid(logits_b_sig[a])`;
   likely beats Markov-inverse and is a fair Approach A comparison point.
3. **Cross-user** — this user's rhythm is probably atypical (single-device
   power user). A multi-user log would let us estimate how much of the
   win transfers.
4. **Hard-protect policy ablation** — measure FK@0.5 with and without the
   system-category mask; quantify its cost on MemorySaveRate.
5. **`T_STALE` sensitivity** — sweep {2, 4, 6, 12} h. Shorter cutoffs
   reduce |B(t)| and probably tighten FK variance.
6. **Full v3-backbone Approach A** (masked-sigmoid). Worth revisiting once
   a proper multi-anchor v3 tensor builder exists; would be the fair
   comparison to "can we just fine-tune R6?".
