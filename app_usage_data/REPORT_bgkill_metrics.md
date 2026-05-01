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

## 2. Two deployment frames

The metrics in this report split into **two parallel families** corresponding to two different deployment scenarios:

### 2.A  Rank-based, fixed-budget eviction  *(see §§3–9)*

At anchor `t`, the OS:

1. Looks at `B(t)` — apps currently in background.
2. Chooses an **eviction ratio** `r ∈ [0, 1]` driven by RAM pressure.
3. Kills the top `K = ⌈r · |B(t)|⌉` apps by `kill_score`.

The OS supplies *how many* to kill; the model only supplies the ranking. This is the Android / HarmonyOS LowMemoryKiller-style "free this much RAM right now" pattern.

Metrics: **FK@r, SafeKillRecall@r, ROC-AUC, PR-AUC, NDCG, Pareto** (§§3–9).

### 2.B  Threshold-based, "yes/no per app"  *(see §10)*

At anchor `t`, the model decides per-app:

```
predicted_kill(a) = 1  if  score(a) > τ
                    0  otherwise
```

τ is a fixed score threshold tuned on val. Number of apps killed varies per anchor — could be 0, could be all of `B(t)`. This matches a "background-suspend if confident" policy where the model owns the kill decision rather than the OS.

Metrics: **KillPrecision@τ, KillRecall@τ, F1@τ, MCC@τ** (§10).

### Outcome categories  *(shared across both frames)*

For any app with `predicted_kill = 1`:

- **False kill** (FP) — user *did* foreground it in `(t, t+H]`. **Bad** (user pain).
- **Correct kill** (TP) — user *didn't* foreground it. **Good** (RAM saved).

For any app with `predicted_kill = 0`:

- **Missed save** (FN) — user *didn't* foreground; could have killed safely but didn't.
- **Correct keep** (TN) — user *did* foreground; we correctly kept it alive.

Convention:

```
score   = kill_score                (higher → more killable)
y       = 1 if app was foregrounded in (t, t+H]
killed  (rank-based, §§3–9)  = top-⌈r · |B(t)|⌉ rows of an anchor by score
killed  (threshold-based, §10) = rows with score > τ
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

## 10. Threshold-based metrics (yes/no per-app deployment)

The metrics in §§3–9 evaluate **rank-based** policies (§2.A) — sort `B(t)` by kill-score, kill the top-K. This section evaluates the **threshold-based** alternative (§2.B): for each app independently, kill iff `score(a) > τ`.

The user picks τ once (e.g. tuned on val for max-F1), freezes it, and the model produces per-app binary kill decisions. This matches the "yes/no per app" deployment that maps directly to the binary `y_3600` ground truth.

### 10.1  Confusion matrix at threshold τ

|                         | GT: y = 0 (safe to kill)            | GT: y = 1 (user wanted)            |
|-------------------------|--------------------------------------|-------------------------------------|
| **Predicted: kill (s > τ)** | **TP** — correct kill (RAM saved)   | **FP** — false kill (user pain)    |
| **Predicted: keep (s ≤ τ)** | **FN** — missed save                | **TN** — correct keep              |

Asymmetric cost: an FP (killing a wanted app) costs more than an FN (failing to save RAM that was safe). This forces τ to lean conservative.

### 10.2  KillPrecision @ τ  *(headline — answers "of kills, how many were correct?")*

```
KillPrecision = TP / (TP + FP)
```

**Higher is better.** Equivalent to `1 − (false-kill rate at τ)`. The user-proposed metric "of all the apps the model decides to kill, how many were mistakes" is exactly `1 − KillPrecision = FP / (TP+FP)`.

A KillPrecision of 0.85 means: across all apps the model decided to kill, 85 % were genuinely safe to kill and 15 % were apps the user actually wanted.

### 10.3  KillRecall @ τ  *(answers "of safe-to-kill apps, how many got killed?")*

```
KillRecall = TP / (TP + FN)
```

**Higher is better.** Equivalent to *SafeKillRecall* at threshold τ. The user-proposed metric "of all the apps GT says we should kill, how many did we actually kill" is exactly KillRecall.

A KillRecall of 0.65 means: of the apps that were safe to kill, the model captured 65 % of them — the other 35 % were left in RAM.

KillPrecision and KillRecall trade off as τ moves: lowering τ lets more apps qualify for killing → KillRecall ↑ but KillPrecision ↓ (more false kills mixed in).

### 10.4  F1 @ τ  *(combined precision/recall summary)*

```
F1 = 2 · KillPrecision · KillRecall / (KillPrecision + KillRecall)
```

Harmonic mean — penalizes models that are good at one of P/R but bad at the other. Useful as a single-number target when picking τ on val.

### 10.5  MCC @ τ — Matthews Correlation Coefficient

```
MCC = (TP·TN − FP·FN) / sqrt((TP+FP) (TP+FN) (TN+FP) (TN+FN))
```

Range **[−1, +1]**:  −1 = perfectly inverted, 0 = random, +1 = perfect. Robust to class imbalance — a predict-all-zero baseline gets accuracy = 0.78 on this task (since 22 % are positive) but MCC = 0. **MCC = 0** is the right reference for "this model isn't doing anything."

For headline reporting under class imbalance, **MCC is preferred over Accuracy.**

### 10.6  How τ is chosen

Three reasonable rules, all evaluated on val and frozen for test:

| Rule | Formula | When to use |
|---|---|---|
| τ = 0.5 (Bayes default) | — | Sigmoid output, well-calibrated. With `pos_weight ≈ 3` in BCE training, this is sub-optimal — kills too eagerly. |
| τ* = arg max F1 on val | scan τ ∈ [0, 1] | The default for cross-model comparison. Each model gets its own τ*, but applied frozen to test. |
| τ = at-most-X% false-kill | `min τ s.t. FP/(FP+TP) ≤ X` on val | When there's a hard product constraint on user pain. |

The standard leaderboard reports τ\* (max-F1 on val) for every model.

### 10.7  Worked example (toy anchor from §1.5)

Same 5-app anchor as before. Pick τ = 0.5:

| App | Score | y | Predicted kill (s > 0.5) | Outcome |
|---|---|---|---|---|
| A | 0.95 | 0 | 1 | **TP** |
| B | 0.80 | 0 | 1 | **TP** |
| C | 0.60 | 1 | 1 | **FP** ✗ |
| D | 0.30 | 0 | 0 | **FN** |
| E | 0.10 | 1 | 0 | **TN** |

Counts: TP = 2, FP = 1, FN = 1, TN = 1.

| Metric | Value | Reading |
|---|---:|---|
| KillPrecision | 2 / 3 = **0.667** | 2 of the 3 kills were correct; 1 was a mistake |
| KillRecall | 2 / 3 = **0.667** | 2 of the 3 safe-to-kill apps were killed; 1 was missed |
| F1 | 2·0.667·0.667/(0.667+0.667) = **0.667** | balanced number |
| Accuracy | (2+1)/5 = **0.60** | 3 of 5 decisions correct |
| MCC | (2·1 − 1·1) / √(3·3·2·2) = 1/6 ≈ **0.167** | weakly positive — model is doing slightly better than random on this anchor |

Compare to the rank-based view at r = 0.5 from §3 (same anchor): FK@0.5 = 1/3 ≈ 0.333, SafeKillRecall@0.5 = 2/3 ≈ 0.667. The recall agrees (we caught 2 of 3 safe apps either way). KillPrecision (0.667) = 1 − FK@0.5 (0.333) ✓. The two metric families converge here only because at this anchor the threshold τ = 0.5 happens to kill the same 3 apps as the top-r=0.5 rank cut would; in general they pick different kill sets.

### 10.8  Threshold selection — practical recipe

```python
from sklearn.metrics import f1_score
def pick_tau(val_scores, val_y_kill_label):
    taus = np.linspace(0.02, 0.98, 49)
    f1s = [f1_score(val_y_kill_label, val_scores > t) for t in taus]
    return taus[int(np.argmax(f1s))]
```

For models trained with `pos_weight = 3.0`, expect τ* in the [0.6, 0.8] range — calibration is biased toward predicting kill more eagerly than 0.5 implies.

### 10.9  User-proposed metrics → standard equivalents

| User's name | Standard equivalent | Formula |
|---|---|---|
| "of all GT-killed apps at an anchor, how many killed by mistake" | **1 − KillPrecision @ τ** | `FP / (TP + FP)` |
| "of all GT-killed apps, how many we actually killed" | **KillRecall @ τ** | `TP / (TP + FN)` |
| "Mean kill-score on positives" | (already in `REPORT_bgkill_v3.md` §5.5 as **PosScoreNorm**) | per-anchor min-max scale, then mean over y=1 rows |

The first two map cleanly onto KillPrecision/Recall — exactly the standard binary-classification metrics. The third (mean kill-score on positives) is rank-aware rather than threshold-based; it lives in §5.5 of `REPORT_bgkill_v3.md` rather than here.

### 10.10  Threshold-based vs rank-based — which to lead with?

| Deployment | Use |
|---|---|
| RAM-pressure event with fixed kill-budget | Rank-based (§§3–9): FK@0.5, SafeKillRecall@0.5, Pareto |
| Continuous "suspend if confident" policy | Threshold-based (§10): KillPrecision@τ\*, KillRecall@τ\*, F1@τ\* |
| You want one robust number under class imbalance | MCC @ τ\* (this section) |
| You don't want to commit to any τ | ROC-AUC, PR-AUC (§§5–6) — they integrate over all thresholds |

Both families are valid. They answer different questions. The leaderboard (§11 below) lists both side-by-side.

---

## 11. The standard leaderboard row

Every model variant emits **both** metric families:

```
Rank-based (§§3–9):       ROC-AUC@H60   PR-AUC@H60   FK@0.5   SafeKillRecall@0.5   NDCG@half
Threshold-based (§10):    KillPrecision@τ*   KillRecall@τ*   F1@τ*   MCC@τ*
```

`τ*` = threshold tuned on val to maximise F1 (one τ per model, frozen for test). Plus per-`r` columns for `FK@{0.25, 0.75}` etc. that go into the Pareto figure.

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

## 12. Pareto figures (output)

For every (split ∈ {val, test}, horizon = H=60):

- `pareto_<split>_H60.png` — full leaderboard, all models on one panel.
- `pareto_focused_<split>_H60.png` — only the four key rows: Random, LRU, Markov-inverse, C3-Pro. For the report.

`r`-value labels annotated on the C3-Pro curve.

---

## 13. Implementation pointer

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

## 14. What I deliberately don't add

| Skipped | Reason |
|---|---|
| **Calibration** (Brier score, ECE) | Track-A (rank-based) doesn't depend on calibration. Track-B (threshold-based, §10) does — we side-step it by tuning τ\* per model on val rather than using τ = 0.5. |
| **Inference-latency / memory footprint** | Single-user CPU prototype; deployment-stage concerns. |
| **Per-app metrics** (per-app FK rate) | Not actionable for an OS-level decision; plus very few positives per app at this user. |
| **Per-time-of-day metrics** | Useful for diagnostics, not for the headline leaderboard. |
| **Bootstrap confidence intervals on every run** | Test set is 817 anchors; CIs are roughly ± 3–4 pp on ROC-AUC. We do compute them once via `scripts/35_eval_h60.py` for the final report figures (REPORT_bgkill_v3.md §5.6); per training run is wasteful. |

---

## 15. TL;DR

### Track A — rank-based (RAM-pressure / fixed-budget eviction)

| What | Use |
|---|---|
| One-number headline | **ROC-AUC@H60** |
| Deployment-realistic operating point | **FK@0.5 + SafeKillRecall@0.5** at r = 0.5 |
| Full tradeoff curve | **Pareto figure** sweeping r ∈ {0.1, 0.25, 0.5, 0.75, 0.9} |
| Imbalance-aware secondary | **PR-AUC** |
| Sanity / top-of-ranking | **NDCG@half** |

### Track B — threshold-based (yes/no per app at fixed τ)

| What | Use |
|---|---|
| Headline harmful-decision rate | **1 − KillPrecision@τ*** = false-kill rate |
| Headline coverage of safe apps | **KillRecall@τ*** |
| Combined single number | **F1@τ*** for picking τ; **MCC@τ*** for cross-baseline comparability |
| Threshold | **τ\*** = arg-max F1 on val, frozen on test |

### Always cross-check

| Check | Reason |
|---|---|
| ROC-AUC ≥ 0.55 | Below = model has a bug. |
| KillRecall@τ\* > Random's | Confirms the threshold isn't degenerate. |
| FK@0.5 ≤ Markov-inverse's | The strongest closed-form baseline. |

Single-line summary of the full metric stack:

> **ROC-AUC** for ranking quality (track A), **MCC@τ\*** for classification quality (track B), **FK@0.5 / KillRecall@τ\*** for the two deployment numbers, **Pareto** for the rank-based tradeoff curve. Everything else is a sanity check.
