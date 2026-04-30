# Task C — Evaluation Metrics

The full set of metrics used to evaluate the background-suspension model and its baselines, with formulas, intuitions, and notes specific to the H = 60 min single-horizon setting.

Companion to:
- `REPORT_bgkill_data.md` — label construction (`y_3600`, etc.)
- `REPORT_bgkill_features_review.md` — feature schema
- `REPORT_bgkill_model_plan.md` — model variants and leaderboard layout

This report covers **only the metrics**: how each is computed, what it intuitively measures, and which to lead with at H=60.

---

## 1. The unit of evaluation

Every metric is computed **per anchor**, then **mean-aggregated** across anchors with `|B(t)| ≥ 1`.

This means an anchor with `|B(t)| = 2` and one with `|B(t)| = 15` contribute equally — we never pool rows globally. The unit is *one OS decision moment*, regardless of how crowded the BG set happens to be.

`anchor_id` in the parquet is the grouping key.

---

## 1.5 A worked example anchor (used throughout the rest of this report)

To make every metric concrete, here's a single toy anchor we'll trace through every calculation. Five apps in `B(t)` at one anchor `t`:

| Rank by `kill_score` | App | `kill_score` | `y` (1 = foregrounded in (t, t+60min]) |
|---|---|---|---|
| 1 | A | 0.95 | 0 (safe to kill) |
| 2 | B | 0.80 | 0 (safe to kill) |
| 3 | C | 0.60 | **1 (user actually used it!)** |
| 4 | D | 0.30 | 0 (safe to kill) |
| 5 | E | 0.10 | 1 (correctly kept) |

Counts:
- 3 safe-to-kill apps (`y = 0`): A, B, D
- 2 will-be-used apps (`y = 1`): C, E

A perfect ranking would have A, B, D scored highest (in any order among themselves) and C, E lowest. The model gets it *almost* right — but ranks C above D, which is its one mistake.

We'll trace this single anchor through every metric below.

---

## 2. The deployment frame

At anchor `t`, the OS:

1. Looks at `B(t)` — apps currently in background.
2. Chooses an **eviction ratio** `r ∈ [0, 1]`.
3. Kills the top `K = ⌈r · |B(t)|⌉` apps by `kill_score`.

Outcome of each killed app:

- **False kill** — killed, but the user *did* foreground it in `(t, t+H]`. **Bad** (user feels lag, lost state).
- **Correct kill** — killed, user *didn't* foreground it. **Good** (RAM reclaimed).

Metrics quantify this trade-off at fixed `r` (operating-point view) and threshold-free (ranking quality).

Convention used below:

```
score   = kill_score   (higher → more kill-worthy)
y       = 1 if app was foregrounded in (t, t+H]   (i.e. user came back to it)
killed  = top-⌈r · |B(t)|⌉ rows of an anchor by score
```

---

## 3. FalseKillRate@r  *(primary headline)*

> *"Of the apps we killed, what fraction did the user need?"*

```
FK@r (per anchor)  =  |killed ∩ {y = 1}|  /  |killed|
FK@r (overall)     =  mean over anchors
```

**Lower is better.** 0 = no false kills; ~0.25 at H=60 = random scoring.

| `r` | If FK@r is high… | If FK@r is low… |
|---|---|---|
| 0.5 | half the kills are wrong → user loses state often | half the kills are good → safe at this aggression level |

This is the metric the OS engineer cares about most: *"how often will my user be annoyed?"*

### Worked example (toy anchor from §1.5)

```
B(t) sorted by kill_score:
  Rank 1  A  score=0.95  y=0
  Rank 2  B  score=0.80  y=0
  Rank 3  C  score=0.60  y=1   ← false kill if killed
  Rank 4  D  score=0.30  y=0
  Rank 5  E  score=0.10  y=1
```

At `r = 0.5`: K = ⌈0.5 × 5⌉ = **3**, so we kill the top-3 by score: **{A, B, C}**.

```
killed ∩ {y = 1}  =  {C}                        (one false kill)
FK@0.5            =  |{C}| / |{A,B,C}|  =  1/3 ≈ 0.333
```

So at `r = 0.5` the model's false-kill rate is **33% on this anchor** — one of the three apps it chose to evict was actually about to be used.

At other `r` values for the same anchor:

| r | K | killed | false kills | FK@r |
|---|---|---|---|---|
| 0.1 | 1 | {A} | 0 | 0/1 = 0.00 |
| 0.25 | 2 | {A, B} | 0 | 0/2 = 0.00 |
| 0.5 | 3 | {A, B, C} | 1 | 1/3 ≈ 0.33 |
| 0.75 | 4 | {A, B, C, D} | 1 | 1/4 = 0.25 |
| 0.9 | 5 | all | 2 | 2/5 = 0.40 |

The reported **FK@0.5** in the leaderboard is the mean of this per-anchor FK over all anchors in the split.

---

## 3. SafeKillRecall@r  *(formerly "MemorySaveRate@r")*

> *"Of the apps it was safe to kill, how many did I actually kill?"*

```
SafeKillRecall@r = E_anchor [ |killed ∩ {y = 0}|  /  |{y = 0} in B(t)| ]
```

**Higher is better.** This is the *recall* of the kill decision — the fraction of reclaimable RAM successfully reclaimed.

### Pairs with FK@r

These two metrics jointly describe the **operating point** at ratio `r`:

- low `r` → low FK and low recall (kill few, miss few but also miss correct kills).
- high `r` → high FK and high recall (kill many, catch most safe but also catch some wrong).

Never improvable in isolation — different `r` slides up the model's own curve. To compare *models*, sweep `r` and look at the Pareto curve.

### Worked example (same anchor)

```
At r=0.5: killed = {A, B, C}
killed ∩ {y=0}     = {A, B}                  (2 correct kills)
{y=0} in B(t)      = {A, B, D}                (3 apps it would have been safe to kill)

SafeKillRecall@0.5 = 2 / 3  ≈ 0.667
```

So we successfully reclaimed 2 of the 3 reclaimable apps → **67% memory recall** for this anchor.

Across all `r` values:

| r | K | killed | correct kills (y=0 in killed) | recall denom (total y=0) | SafeKillRecall@r |
|---|---|---|---|---|---|
| 0.1 | 1 | {A} | 1 | 3 | 0.333 |
| 0.5 | 3 | {A, B, C} | 2 | 3 | 0.667 |
| 0.9 | 5 | all | 3 | 3 | 1.000 |

Pairs with FK: at `r = 0.5` we get FK = 0.33 *and* SafeKillRecall = 0.67 — both useful, both needed to describe the operating point.

---

## 4. Pareto curve

For each model, plot

- **x-axis:** SafeKillRecall@r
- **y-axis:** 1 − FalseKillRate@r

over `r ∈ {0.1, 0.25, 0.5, 0.75, 0.9}` (or finer). Each model produces a curve.

**Upper-right corner is best** — high recall AND low false kills.

A model **Pareto-dominates** another if its curve is at-or-above at every `r`. If curves cross, the comparison depends on which deployment region matters (low `r` for conservative eviction, high `r` for aggressive).

The report figure is one Pareto plot per (split, horizon).

---

## 5. ROC-AUC  (anchor-mean)  *(threshold-free headline)*

```
ROC-AUC = P( kill_score(safe-to-kill app) > kill_score(used app) )
        = "rank-correlation between score and (1 − y) within an anchor"
```

Computed per anchor (only on anchors with both classes present), mean across anchors.

**Range:** 0.5 (random) → 1.0 (perfect).
**Higher is better.**

### Why this is the single best summary number

- Threshold-free: doesn't depend on choosing `r`.
- Symmetric: a model that always inverts the right answer scores 0.0; perfect = 1.0.
- Robust to imbalance: doesn't degenerate when positive rate shifts (H=5 → H=60).
- One number per model → makes leaderboards readable.

### Worked example (same anchor)

```
B(t):    A     B     C     D     E
score:   0.95  0.80  0.60  0.30  0.10
y:       0     0     1     0     1     ← 3 safe-to-kill, 2 used
```

There are `n_safe × n_used = 3 × 2 = 6` (safe, used) pairs. ROC-AUC = fraction of those pairs where the safe-to-kill app got the *higher* score.

| Safe app (kill_score) | Used app (kill_score) | Did safe > used? |
|---|---|---|
| A (0.95) | C (0.60) | ✓ |
| A (0.95) | E (0.10) | ✓ |
| B (0.80) | C (0.60) | ✓ |
| B (0.80) | E (0.10) | ✓ |
| D (0.30) | C (0.60) | ✗ |
| D (0.30) | E (0.10) | ✓ |

5 / 6 pairs correctly ordered →

```
ROC-AUC (this anchor) = 5 / 6 ≈ 0.833
```

The one wrong pair is `D vs C` — the model ranked C (used) above D (safe). Mean of this per-anchor value across all anchors is the reported ROC-AUC.

In our convention, the metric returns ROC-AUC for the kill direction (positive class = `y = 0`, "safe to kill"), so values > 0.5 always mean "good kill-predictor."

---

## 6. PR-AUC  (anchor-mean)

Per-anchor area under the precision–recall curve, computed with `y = 0` (safe-to-kill) as the positive class. Mean across anchors.

```
PR-AUC ≈ "average precision when ranking by kill_score"
```

**Higher is better.** Range depends on class balance — at H=60 (~75 % "safe to kill"), random scoring gives PR-AUC ≈ 0.75; signal lifts above that floor.

### Why have it alongside ROC-AUC

PR-AUC is more sensitive to changes in the *rare* class (false kills) and to the highest-confidence portion of the ranking. ROC-AUC weights all rank pairs equally; PR-AUC effectively weights the top of the ranking more. The two together cross-check each other.

### Worked example (same anchor)

Walk down the ranking by descending `kill_score`, tracking precision and recall after each step. Positive class = "safe to kill" (`y = 0`). 3 total positives in this anchor.

| step | app | y | y=0 (positive)? | cumulative TP | cumulative FP | precision = TP/(TP+FP) | recall = TP/3 |
|---|---|---|---|---|---|---|---|
| 1 | A (0.95) | 0 | ✓ | 1 | 0 | 1.000 | 0.333 |
| 2 | B (0.80) | 0 | ✓ | 2 | 0 | 1.000 | 0.667 |
| 3 | C (0.60) | 1 |  | 2 | 1 | 0.667 | 0.667 |
| 4 | D (0.30) | 0 | ✓ | 3 | 1 | 0.750 | 1.000 |
| 5 | E (0.10) | 1 |  | 3 | 2 | 0.600 | 1.000 |

PR-AUC = sum of `(recall_i − recall_{i−1}) × precision_i` across steps where recall increases (i.e., when we hit a positive):

```
A: (0.333 − 0)     × 1.000 = 0.333
B: (0.667 − 0.333) × 1.000 = 0.333
D: (1.000 − 0.667) × 1.000 / 3 → re-derive carefully:
   precision_at_D = 0.750
   recall jump from 0.667 to 1.000  = 0.333
   contribution = 0.333 × 0.750 = 0.250

PR-AUC ≈ 0.333 + 0.333 + 0.250 = 0.917
```

For this anchor the model's PR-AUC is **0.917** — the top of the ranking is very clean (A and B are perfect calls), and the only blemish is C inserting between B and D. PR-AUC penalises that blemish heavily because it's near the top of the ranking.

ROC-AUC for the same anchor was 0.833. PR-AUC of 0.917 is *higher* here because the failure (D ranked below C) happens at a recall level where precision was already high — the model nailed the easy positives at the top.

---

## 7. NDCG@half  *(rank-aware sanity)*

Normalised Discounted Cumulative Gain at `k = ⌈0.5 · |B(t)|⌉`, with relevance = `1 − y`:

```
DCG@k  = Σ_{i=1..k}  rel_{ranked-i} / log₂(i + 1)
NDCG@k = DCG@k / IDCG@k    (IDCG = DCG of the optimal ranking)
```

**Higher is better.** Range [0, 1].

Rank-aware: getting the most-safe-to-kill app at position 1 contributes more than at position 5 (logarithmic discount).

In practice clusters at 0.96–0.99 because the H=60 task is "easy at the top of the ranking" — the most stale apps in B(t) are obviously safe. NDCG mostly moves on the *close calls* in the top half. Useful as a **sanity** signal more than a primary discriminator.

### Worked example (same anchor)

For our 5-app anchor, `k = ⌈0.5 × 5⌉ = 3`.

```
Discount weights:
  position 1:  1 / log2(2) = 1.000
  position 2:  1 / log2(3) ≈ 0.631
  position 3:  1 / log2(4) = 0.500
```

Model's top-3 ranked by score: A (rel = 1), B (rel = 1), C (rel = 0).

```
DCG@3  = 1 × 1.000  +  1 × 0.631  +  0 × 0.500  =  1.631
```

Ideal top-3 (the 3 highest-relevance apps in B(t) in order): A, B, D — all with rel = 1.

```
IDCG@3 = 1 × 1.000 + 1 × 0.631 + 1 × 0.500  =  2.131
```

```
NDCG@3 = 1.631 / 2.131 ≈ 0.766
```

The model loses ~23 % of the ideal score because position 3 of its top-3 was C (used, relevance 0) instead of D (safe, relevance 1). The discount weight at position 3 (= 0.5) determines how badly that swap is penalised.

NDCG is then averaged across anchors that have at least one positive in their top half.

---

## 8. Random-baseline values at H = 60

Because positive rate at H=60 (~25 %) is much higher than at H=5 (~3 %), the random-scoring "floor" for each metric shifts:

| Metric | Random floor at H=5 | Random floor at H=60 |
|---|---|---|
| FK@0.5 | ≈ 0.03 | **≈ 0.25** |
| SafeKillRecall@0.5 | ≈ 0.5 | ≈ 0.5 |
| ROC-AUC | 0.50 | 0.50 |
| PR-AUC | ≈ 0.10–0.15 | **≈ 0.5–0.6** |
| NDCG@half | ≈ 0.97 | ≈ 0.96 |

Practical implication: **at H=60, FK@0.5 of 0.05 is already excellent** (5× better than random's 0.25). Don't compare H=60 absolute numbers directly to H=5 numbers — the meaningful comparison is "how much better than that horizon's random floor."

---

## 9. Operating-point summary at `r = 0.5`

The headline `r = 0.5` operating point captures "kill half of B(t)". For C3-Pro at H=60, the target numbers are roughly:

| Metric | Random | Markov-inv | C3-Pro target |
|---|---|---|---|
| FK@0.5 | 0.25 | ≈ 0.18 | **< 0.10** |
| SafeKillRecall@0.5 | 0.50 | ≈ 0.55 | **> 0.70** |
| ROC-AUC | 0.50 | ≈ 0.78 | **> 0.90** |
| PR-AUC | ≈ 0.55 | ≈ 0.85 | **> 0.92** |

(Numbers are estimates pending re-runs on the new 2 h-window data.)

---

## 10. The standard leaderboard row

Every model variant emits the following row:

```
ROC-AUC@H60   PR-AUC@H60   FK@0.5   SafeKillRecall@0.5   NDCG@half
```

Plus per-`r` columns for `FK@0.25`, `FK@0.75` etc. that go into the Pareto figure.

Layout:

| Model | n_anchors | ROC-AUC | PR-AUC | FK@0.5 | SKR@0.5 | NDCG@half |
|---|---|---|---|---|---|---|
| Random | 975 | 0.50 | 0.55 | 0.25 | 0.50 | 0.96 |
| LRU | 975 | 0.74 | 0.84 | 0.18 | 0.58 | 0.97 |
| Markov-inverse | 975 | 0.78 | 0.86 | 0.16 | 0.60 | 0.97 |
| C2 (re-run) | 975 | 0.84 | 0.90 | 0.12 | 0.65 | 0.98 |
| C3.1 | 975 | … | … | … | … | … |
| ... | ... | ... | ... | ... | ... | ... |
| **C3-Pro** | 975 | **0.91** | **0.94** | **0.06** | **0.74** | **0.99** |

---

## 11. Pareto figures (output)

For every (split ∈ {val, test}, horizon = H=60):

- `pareto_<split>_H60.png` — full leaderboard, all models on one panel.
- `pareto_focused_<split>_H60.png` — only the four key rows: Random, LRU, Markov-inverse, C3-Pro. For the report.

`r`-value labels annotated on the C3-Pro curve.

---

## 12. Implementation pointer

All metrics live in **`lib/bg/metrics_bg.py`**:

| Function | What it does |
|---|---|
| `compute_metrics(df, score_col, y_col, r_values)` | The main entry — returns a dict with FK@r, SafeKillRecall@r (under legacy key `memory_save_rate`), `roc_auc_mean`, `pr_auc_mean`, `ndcg_half`. |
| `roc_auc(y, score)` | Rank-sum ROC AUC, tie-averaged. |
| `pr_auc(y, score)` | AP-style precision-recall AUC. |
| `ndcg_at_k(y, score, k)` | NDCG with relevance = `1 − y`. |
| `_topk_kill_mask(scores, r)` | Mask of top-`⌈r·|B(t)|⌉` rows by score. |

The function `compute_metrics` already supports any `y_col` so it works with `y_300 / y_600 / y_1800 / y_3600` — switch to `y_3600` for the H=60 single-horizon plan.

---

## 13. What I deliberately don't add

| Skipped | Reason |
|---|---|
| **Calibration** (Brier score, expected calibration error) | We use top-K ranking, not absolute probability — calibration matters only if a downstream system trusts the score as a probability. |
| **Inference-latency / memory footprint** | Single-user CPU prototype; deployment-stage concerns. |
| **Per-app metrics** (per-app FK rate) | Not actionable for an OS-level decision; plus very few positives per app at this user. |
| **Per-time-of-day metrics** | Useful for diagnostics, not for the headline leaderboard. |
| **Bootstrap confidence intervals** | Test set is 975 anchors; CIs are roughly ± 3–4 pp on ROC-AUC. Worth doing once for the final report figure but not per training run. |

---

## 14. TL;DR

| What | Use |
|---|---|
| One-number headline | **ROC-AUC@H60** |
| Deployment-realistic operating point | **FK@0.5 + SafeKillRecall@0.5** at r = 0.5 |
| Full tradeoff curve | **Pareto figure** sweeping r ∈ {0.1, 0.25, 0.5, 0.75, 0.9} |
| Imbalance-aware secondary | **PR-AUC** |
| Sanity / top-of-ranking | **NDCG@half** |
| Always cross-check | **vs Random** (any model with ROC-AUC < 0.55 has a bug) |

Single-line summary of the metric stack:

> **ROC-AUC** for ranking, **FK@0.5** for the deployment number, **Pareto** for the figure. Everything else is a sanity check.
