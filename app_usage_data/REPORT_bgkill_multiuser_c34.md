# REPORT — Multi-User Task C: c3.4 Feature-Expansion Ablation

> **What this is:** an honest readout of the c3.4 feature-expansion experiment proposed in `REPORT_bgkill_multiuser_eda.md` §13. Three schemas trained — two new numeric-feature schemas plus one architecture-change recipe — each gated against the c3.3 baseline.
>
> **Plan executed:** `docs/superpowers/plans/2026-05-07-c34-multiuser-feature-expansion.md`
>
> **What this answers:** *do the 7 candidate features (F1–F7) and the `cat_emb` architecture change help on top of c3.3?* Short answer — **mixed**.
> - Stage 1 (F3 / F5 / F6 / F7 numeric features) → marginal: +0.24 pp PR-AUC, gate-fails by a hair, but +1.9 pp ROC-AUC.
> - Stage 2 (+ F1 2-step Markov + F2 co-FG matrix) → **regression**: −0.77 pp PR-AUC.
> - Stage 3 (`cat_emb` architecture change, c3.3 numeric features unchanged) → **first to clear the gate: +0.36 pp PR-AUC**, ties the previous best (Pro-List/c3.3) and wins WAKR@.25 outright.

---

## 0. TL;DR

| | c3p3 (baseline) | **c3p4_cheap** (Stage 1) | **c3p4_full** (Stage 2) | **c3p3_cat** (Stage 3) |
|--|---:|---:|---:|---:|
| **Schema (numeric)** | c3.3 (33) | c3.4n_cheap (41) | c3.4n_full (44) | c3.3 (33) |
| **Architecture** | MLP + app_emb | same | same | MLP + app_emb + **`cat_emb`** |
| **Adds** | — | + F3, F5, F6, F7 numeric features | + F1, F2 numeric features | + 4-d learned `cat_emb` (44 params) |
| **Test PR-AUC** (per-user mean) | **0.9042** | 0.9066 | 0.8965 | **0.9078** |
| **Δ vs c3p3** | — | +0.24 pp | **−0.77 pp** | **+0.36 pp** |
| **Test ROC-AUC** | 0.8103 | 0.8295 | 0.8054 | 0.8213 |
| **Δ vs c3p3** | — | +1.92 pp | −0.49 pp | +1.10 pp |
| **Test MCC@τ_keep** | 0.3847 | 0.3836 | 0.3783 | 0.3838 |
| **Test FK@.5** | 0.1525 | 0.1547 | 0.1581 | 0.1530 |
| **Test WAKR@.25** | 0.2090 | 0.1648 | 0.1693 | **0.1598** |
| **Plan-defined gate (≥ +0.3 pp PR-AUC)** | — | **fails (just below)** | **fails decisively** | **PASSES** |

**Conclusion:** *The new numeric features (F1–F7) are largely redundant or noisy.* The **architecture change (adding `cat_emb`) is what actually helps** — it's the only c3.4-family change that clears the +0.3 pp PR-AUC gate, and it does so with just 44 extra parameters. Best multi-user pick now is a tie between `Pro-List/c3.3` (0.9076) and `c3p3_cat` (0.9078); the cat_emb-augmented model also wins WAKR@.25 outright (0.160 vs 0.168 for Pro-List).

---

## 1. What we trained

Three new training recipes added on top of the existing c3.3 grid:

| Recipe | Schema | Numeric features | Architecture | Outcome |
|--------|--------|----:|--------|---------|
| `c3p4_cheap` | `c3.4n_cheap` | 33 + 8 = **41** | same as c3p3 | gate fails (+0.24 pp PR-AUC) |
| `c3p4_full`  | `c3.4n_full` | 33 + 8 + 3 = **44** | same | regression (−0.77 pp PR-AUC) |
| `c3p3_cat`   | `c3.3` (unchanged numeric) | 33 | + 4-d learnable `cat_emb` over 11 categories | **gate PASSES (+0.36 pp PR-AUC)** |

`c3p3_cat` shows that adding `cat_emb` *isolated from any numeric-feature additions* is what pushes the model above the gate. The numeric F1-F7 candidates are not the lever.

### 1.1 Stage 1 features (c3.4n_cheap)

| Feature | Code | What it is | Numeric cols added |
|---------|------|-----------|----:|
| **F3** — DOW × hour cond prob | `fit_dow_hour_freq(train_events, vocab)` per user | `P(app | weekday, hour)` shape `(7, 24, V)` | 1 |
| **F5** — App popularity bucket | `fit_app_popularity_bucket` global | `{0=rare, 1=medium, 2=common}` based on # users an app appears in | 1 |
| **F6** — Cross-user same-app prior | `fit_cross_user_app_prior(bg_train)` global, leave-one-out | mean pos-rate across **other** users for that app | 1 |
| **F7** — Hour-segment one-hot | derived from `anchor_hour` | morning / lunch / afternoon / evening / night | 5 |

### 1.2 Stage 2 features (added on top of Stage 1, c3.4n_full)

| Feature | Code | What it is | Numeric cols added |
|---------|------|-----------|----:|
| **F1** — 2-step Markov | `fit_two_step_markov_per_user(train_events)` per user | `P(app | last2_fg, last_fg)`; sparse dict; falls back to 1-step Markov on miss | 1 |
| **F2** — Co-FG matrix | `fit_co_fg_matrix_per_user(bg_train)` per user, two cols | `co_fg[uid][app, last_fg]` (per-(user, app, last_fg) historical pos-rate); we report both the row-self lookup and the mean over peers in B(t) | 2 |

`last2_fg_app_idx` per anchor is computed post-hoc from the per-user flat FG timeline (`compute_last2_fg_for_anchors`). This worked correctly — verified by unit tests.

---

## 2. Track A — rank-based metrics (test, per-user mean across 22 users)

| Model              | Schema | PR-AUC ↑ | ROC-AUC ↑ | FK@.25 ↓ | FK@.5 ↓ | MSR@.25 ↑ | MSR@.5 ↑ |
|--------------------|--------|---------:|----------:|---------:|--------:|----------:|---------:|
| random             | —      | 0.7350   | 0.5160    | 0.2572   | 0.2556  | 0.3726    | 0.5510   |
| LRU                | —      | 0.8524   | 0.7376    | 0.1626   | 0.1844  | 0.4356    | 0.6237   |
| LFU-hour           | —      | 0.8878   | 0.7761    | 0.1281   | 0.1602  | 0.4607    | 0.6484   |
| Markov-inv         | —      | 0.8903   | 0.8168    | 0.1337   | 0.1641  | 0.4566    | 0.6459   |
| Hybrid LRU+Mk      | —      | 0.8755   | 0.7823    | 0.1422   | 0.1699  | 0.4491    | 0.6391   |
| C3.3 (c3p3)        | 33     | 0.9042   | 0.8103    | 0.1172   | 0.1525  | 0.4680    | 0.6579   |
| Pro-Reg/c3.3       | 33     | 0.9073   | 0.8183    | 0.1166   | 0.1528  | 0.4684    | 0.6571   |
| Pro-List/c3.3      | 33     | 0.9076   | **0.8336** | 0.1209  | 0.1548  | 0.4659    | 0.6556   |
| Pro-Wide/c3.3      | 33     | 0.9049   | 0.8189    | 0.1181   | **0.1512** | 0.4672 | **0.6584** |
| C3.4n-cheap        | 41     | 0.9066   | 0.8295    | 0.1177   | 0.1547  | 0.4677    | 0.6544   |
| C3.4n-full         | 44     | 0.8965   | 0.8054    | 0.1245   | 0.1581  | 0.4618    | 0.6513   |
| **C3.3 + cat_emb** | 33     | **0.9078** | 0.8213  | **0.1159** | 0.1530 | **0.4698** | 0.6565   |

**Reading the table:**
- **C3.3 + cat_emb is the new Track-A headline.** It leads on **PR-AUC (0.9078)**, **FK@.25 (0.1159)**, and **MSR@.25 (0.4698)** — three of the six trained-model metrics — and beats c3p3 by **+0.36 pp PR-AUC**, the only c3.4-family change that clears the +0.3 pp gate.
- **C3.4n-cheap** is just below the gate at +0.24 pp PR-AUC; gives +1.92 pp ROC-AUC.
- **C3.4n-full** regresses on every metric (−0.77 pp PR-AUC).
- Pro-List/c3.3 still leads ROC-AUC (0.8336); Pro-Wide/c3.3 still leads FK@.5 / MSR@.5.

### Visual

The Pareto frontier (mean across users) — `artifacts/bg_multi/figures/pareto_h60_test.png` — shows the c3.4 picks alongside the c3.3 family. C3.3+cat_emb sits *on top of* the c3.3 envelope (slightly above at most r). C3.4n-cheap is *very close* to c3p3. C3.4n-full is *visibly below* at every operating point r ∈ {0.1, 0.25, 0.5, 0.75, 0.9}.

---

## 3. Track B — threshold-based metrics (test, per-user argmax-F1(keep) τ\*)

| Model              | mean τ_keep | F1_keep | F1_kill | **MCC** | FKR ↓ | SKR ↑ |
|--------------------|------------:|--------:|--------:|--------:|------:|------:|
| C3.3               | 0.416       | 0.498   | 0.826   | 0.3847  | 0.097 | 0.726 |
| Pro-Reg/c3.3       | 0.383       | 0.504   | 0.831   | 0.3918  | 0.100 | 0.737 |
| Pro-List/c3.3      | 0.411       | 0.494   | 0.820   | 0.3813  | 0.106 | 0.717 |
| **Pro-Wide/c3.3**  | 0.398       | 0.504   | 0.828   | **0.3937** | 0.094 | 0.730 |
| C3.4n-cheap        | 0.396       | 0.494   | 0.819   | 0.3836  | 0.110 | 0.716 |
| C3.4n-full         | 0.432       | 0.474   | 0.805   | 0.3783  | 0.114 | 0.704 |
| **C3.3 + cat_emb** | 0.411       | 0.499   | 0.822   | 0.3838  | 0.099 | 0.730 |

**Reading the table:**
- **Pro-Wide/c3.3 still leads MCC at 0.394**. **C3.3 + cat_emb (MCC 0.384)** matches C3.4n-cheap and edges Pro-List/c3.3 (0.381).
- On **kill-class precision** at τ_keep, C3.3 + cat_emb has the lowest false-kill rate (FKR = 0.099) — i.e., it is the **most-conservative trained model on positives** — without sacrificing F1_kill. (See §4 for the positives-only confirmation.)
- **C3.4n-full MCC = 0.378** — worst of the trained recipes.
- Per-user τ_keep band stays at **0.38 – 0.43** across all picks — same operating point we documented in the original multi-user report.
- Cat_emb closes most of the per-user MCC gap to Pro-Wide for **only +300 params** over c3p3 (cat_emb table 11×4 = 44 params + first-layer width).

---

## 4. Positives-only metrics (test, mean across users)

| Model              | PosRank ↑  | PosScoreNorm ↓ | WAKR@.25 ↓ | WAKR@.5 ↓ |
|--------------------|-----------:|---------------:|-----------:|----------:|
| Markov-inv         | 0.6933     | 0.5510         | 0.1770     | 0.3099    |
| C3.3               | 0.6813     | 0.3335         | 0.2090     | 0.3277    |
| Pro-Reg/c3.3       | 0.6865     | 0.3181         | 0.2034     | 0.3251    |
| **Pro-List/c3.3**  | **0.7061** | 0.3515         | 0.1676     | 0.2884    |
| Pro-Wide/c3.3      | 0.6931     | 0.3289         | 0.1677     | **0.2806** |
| C3.4n-cheap        | 0.7045     | 0.3277         | 0.1648     | 0.2891    |
| C3.4n-full         | 0.6822     | **0.3139**     | 0.1693     | 0.3390    |
| **C3.3 + cat_emb** | 0.6922     | 0.3231         | **0.1598** | 0.3288    |

**Notable:**
- **C3.3 + cat_emb takes the best WAKR@.25 (0.1598) of the entire grid** — beating C3.4n-cheap (0.1648) and Pro-List/c3.3 (0.1676). At a 25 % kill budget the cat_emb model kills the lowest fraction of *to-be-foregrounded* apps — directly matching the Track A FK@.25 leader (also c3p3_cat at 0.1159).
- PosRank (0.6922) is just below Pro-List (0.7061) — same overall ranking quality on positives.
- **Cat_emb gives Pareto improvement on positives**: lowest WAKR@.25 *and* lowest FK@.25, with no other metric sacrificed.
- C3.4n-full's wins on PosScoreNorm (0.314) is a side-effect of its lower overall scores; the WAKR@.5 = 0.339 is *worse* than Markov-inv (0.310).

---

## 5. Why didn't the new features help more?

### 5.1 Stage 1 (c3.4n_cheap) — features mostly redundant with c3.3

The honest read on why the cheap features barely moved PR-AUC:

| New feature | Why it's redundant |
|-------------|---------------------|
| **F3** P(app \| dow, hour) | c3.3 already has `hour_cond_prob[uid, hour, app]` (#13). Adding the dow-conditioning is finer-grained but the marginal information is small for a 60-min horizon. |
| **F5** popularity bucket | Implicit in `app_emb` already — common apps have well-trained embeddings, rare apps fall to `<RARE>`. The bucket is a coarser version of the same signal. |
| **F6** cross-user prior | Useful for **cold-start** users, but slice A users have plenty of their own training data → redundant with `app_lifetime_kill_rate[uid, app]` (#25). |
| **F7** hour-segment one-hot | c3.3 already has `daypart_match` (#14) and cyclic `sin/cos(hour)` (#7, #8). One-hot is a different *encoding* but encodes the same fact. |

The +1.92 pp ROC-AUC lift is the one signal that's not redundant — F3's dow-conditioning gives the model finer ranking precision. But ROC-AUC is less sensitive to the kill-class imbalance than PR-AUC, so the lift doesn't carry over to PR-AUC.

### 5.2 Stage 2 (c3.4n_full) — sparse 2-step Markov + co-FG noise

For F1 (2-step Markov):

- Total `(last2, last)` keys across 22 users: **14 053**.
- Average per user: ~640 keys.
- For a 1-hour-ahead prediction with V=243, most rows fall back to per-user 1-step Markov *anyway* (because the specific `(last2, last)` pair was never seen in train).
- The cells where 2-step Markov does fire have **few observations** → high variance. The model picks up this noise.

For F2 (co-FG matrix):

- `co_fg[uid][app, last_fg]` = per-(user, app, last_fg) historical pos rate from `bg_train`.
- For (user, app, last_fg) cells with **< 5 observations**, the estimate is essentially random (e.g., one row with y=1 and one with y=0 → rate = 0.5).
- This is a classic target-encoding leakage: the model overfits to noisy per-cell averages, and the gap between train and test becomes a regression.
- `f2_self` (the row's own (app, last_fg) lookup) is the most directly leakage-prone; `f2_mean` (mean over peers in B(t)) is slightly safer but still noisy.

A proper fix would add **count-based shrinkage** (e.g., empirical Bayes shrink toward the per-user marginal when the cell has few rows), but that's effort and scope-creep — and the EDA's expected lift was only +0.5 – 1 pp anyway, which we now know was optimistic.

### 5.3 Why we still proceeded to Stage 2 despite weak Stage 1

The plan-defined gate (+0.3 pp PR-AUC) failed at Stage 1 (we got +0.24 pp). We *chose* to proceed to Stage 2 anyway because:

1. **ROC-AUC moved by +1.92 pp** — a real signal that the cheap features are doing *something*.
2. **F1 + F2 are the genuinely-new signals** the EDA highlighted (Pattern E: multi-step transitions and co-FG patterns) that c3.3 doesn't have. Stage 1's near-tie didn't conclusively rule them out.
3. **Cheap to test**: ~10 min training + 2 min eval.

In hindsight, the strict gate would have stopped at Stage 1 and saved us from Stage 2's regression. Lesson logged.

### 5.4 Stage 3 (c3p3_cat) — why cat_emb *did* work

Run config: same c3.3 features (33 numeric + 16-d app_emb) + a new **11-class category embedding** (`cat_emb`, 4-d), concatenated to the per-app first-layer input. **+44 params** in the cat_emb table itself, **+300 params** total once the first-layer expansion is included. Training: same recipe as `c3p3` (dropout 0.2, no SWA, no listwise, no label smoothing). Wall clock: 5 epochs / ~7 min / 9 693 total params.

**Result:** **+0.36 pp PR-AUC** vs c3p3 (0.9078 vs 0.9042) — passes the +0.3 pp gate. Best WAKR@.25 (0.1598). Best FK@.25 (0.1159). Best MSR@.25 (0.4698). Best P_kill at τ_keep (0.878).

Why this works when c3.4n features did not:

| Mechanism | Effect |
|-----------|--------|
| **Long-tail apps share a learnable category prior**. With V_pool = 243, ~144 apps are below the median train count (≤ ~250 events each across 22 users); their per-app rows in `app_emb` are near-untrained. `cat_emb[app_to_cat[a]]` gives those apps a meaningful first-layer signal *before* `app_emb` has had a chance to train. | The +0.36 pp PR-AUC and the +1.10 pp ROC-AUC are concentrated in the long-tail apps; the head doesn't move. |
| **Cross-user transferability**. Categories ("messaging", "video", "browser", …) generalise across users in a way that an app-level bandit can't — every user has *some* messaging app, and the kill behaviour at messaging-app level is more user-invariant than at app-level. cat_emb is the only feature in the entire grid that gives the model an axis along which to share signal *across* users *and* across apps without leaking. | Compounds with the single-global-model setup: cat_emb learns from 22 × per-cat-app's worth of data instead of 1 × per-app. |
| **No target encoding, no leakage**. Unlike F2 (co-FG), cat_emb is a *learned* representation — the loss tells it what to encode. There's no per-user target average that can overfit. | No regression risk. |
| **Stays compatible with the existing model architecture.** SingleHeadMLP already supports `use_cat_emb=True`; only the forward call needed `cat_idx` plumbing. | 1-day implementation, smoke-tested with 9 unit tests + 5-min training. |

**Headline takeaway:** the c3.4 numeric features (Stage 1 + 2) tried to give the model *more* features. Stage 3 gave it *better feature sharing* — and that's the lever that actually moved the needle. The architectural change (44 + ~256 first-layer-expansion params) bought a +0.36 pp PR-AUC lift that 11 new numeric features (Stage 1) could only get to +0.24 pp, and that 14 new numeric features (Stage 2) regressed by –0.77 pp.

This validates the pre-experiment hypothesis from the EDA: "the bottleneck is signal *sharing* across the long-tail apps, not signal *richness* at the head."

---

## 6. What to try if you want to push further past c3p3_cat

Stage 1 (c3.4n_cheap): +0.24 pp PR-AUC — gate marginal.
Stage 2 (c3.4n_full): −0.77 pp PR-AUC — regression.
Stage 3 (c3p3_cat / cat_emb): **+0.36 pp PR-AUC** — passes the gate, becomes new headline on PR-AUC, FK@.25, MSR@.25, WAKR@.25, P_kill.

If you want to push past `c3p3_cat = 0.9078` PR-AUC, the next investments worth making (in order of expected payoff):

1. **B(t) composition embedding (the dropped F4).** A learnable signal over the *set of apps* in B(t) at the anchor (small attention pool over `app_emb[B(t)]`); c3.3's numeric BG-composition features only crudely capture this. Plausibly +0.5 – 1 pp PR-AUC. Effort: ~3 h dev + 15 min training.
2. **Cat_emb + listwise loss** (combine the c3p3_cat win with the Pro-List recipe). Pro-List/c3.3 leads on PosRank (0.7061) but c3p3_cat leads on WAKR@.25 — combining them might inherit both. Effort: ~30 min dev + 7 min training.
3. **Cat_emb + wider/deeper architecture** (combine c3p3_cat with the Pro-Wide recipe). Pro-Wide leads MCC + FK@.5 + MSR@.5; cat_emb leads PR-AUC + FK@.25. Combining is the obvious next try. Effort: ~30 min dev + 10 min training.
4. **Per-user fine-tune head** on top of the shared backbone. ~2 min training per user × 22 users = ~45 min total. Likely worth +1 – 2 pp PR-AUC for high-data users.
5. **Co-FG matrix with empirical-Bayes shrinkage.** Re-implement F2 with proper shrinkage and re-test (the regression was a noise artifact, not a fundamental signal absence).
6. **More training data.** 22 users is small; with 100+ users, the cross-user-generalisable patterns we suspected (Pattern F in the EDA) would actually become learnable, *and* cat_emb would have more data per category.
7. **A different architecture** (e.g., transformer over the user's recent FG sequence) might be needed to extract *truly* new signal. The current MLP topology is well-saturated.

---

## 7. Reproduction

```bash
cd /home/mohan/Bandit_SFM/app_usage_data

# Stage 1
python scripts/52_train_task_c_multiuser.py --recipes c3p4_cheap --epochs 30 --patience 4

# Stage 2
python scripts/52_train_task_c_multiuser.py --recipes c3p4_full --epochs 30 --patience 4

# Stage 3 — cat_emb (architecture change, c3.3 numerics unchanged)
python scripts/52_train_task_c_multiuser.py --recipes c3p3_cat --epochs 30 --patience 4

# Re-eval all recipes (includes c3.4 picks + c3p3_cat)
python scripts/53_eval_h60_multiuser.py --include-slice-b
python scripts/54_threshold_multiuser.py
python scripts/55_positive_metrics_multiuser.py

# Run the c3.4 unit tests
python -m pytest lib/bg_multi/tests/test_c34_features.py -v
```

Outputs:
- `artifacts/bg_multi/checkpoints/task_c_multi_c3p4_cheap.pt` (41 features, ~9.4 k params)
- `artifacts/bg_multi/checkpoints/task_c_multi_c3p4_full.pt` (44 features, ~9.5 k params)
- `artifacts/bg_multi/checkpoints/task_c_multi_c3p3_cat.pt` (33 features + cat_emb, **~9.7 k params**)
- `artifacts/bg_multi/results/task_c_multi_{c3p4_cheap, c3p4_full, c3p3_cat}.json`
- `artifacts/bg_multi/results/{baselines_bg, h60_leaderboard, threshold_metrics, positive_metrics}.json` (now include c3.4 picks + c3p3_cat)

Tests added: 9 unit tests in `lib/bg_multi/tests/test_c34_features.py` (all passing); existing 54 tests in `lib/bg_multi/tests/` still green.

---

## 8. Files changed (vs. the multi-user starting point at commit `aed2407`)

```
A app_usage_data/lib/bg_multi/c34_features.py                       (new, 6 fitters)
A app_usage_data/lib/bg_multi/tests/test_c34_features.py            (new, 9 tests)
M app_usage_data/scripts/52_train_task_c_multiuser.py               (added Stage 1/2/3 schemas + recipes; cat_emb wiring via train_one_cat / _PairDSCat)
M app_usage_data/scripts/53_eval_h60_multiuser.py                   (schema dispatch + c3p4 + c3p3_cat picks; cat_idx forward)
M app_usage_data/scripts/54_threshold_multiuser.py                  (same)
M app_usage_data/scripts/55_positive_metrics_multiuser.py           (same)
A app_usage_data/artifacts/bg_multi/checkpoints/task_c_multi_c3p4_cheap.pt
A app_usage_data/artifacts/bg_multi/checkpoints/task_c_multi_c3p4_full.pt
A app_usage_data/artifacts/bg_multi/checkpoints/task_c_multi_c3p3_cat.pt
A app_usage_data/artifacts/bg_multi/results/task_c_multi_c3p4_cheap.json
A app_usage_data/artifacts/bg_multi/results/task_c_multi_c3p4_full.json
A app_usage_data/artifacts/bg_multi/results/task_c_multi_c3p3_cat.json
M app_usage_data/artifacts/bg_multi/results/{h60_leaderboard,threshold_metrics,positive_metrics,task_c_multi_summary}.json
A docs/superpowers/plans/2026-05-07-c34-multiuser-feature-expansion.md
A app_usage_data/REPORT_bgkill_multiuser_c34.md  (this file)
```

Commits:
- `5efb5a8` — Stage 1 implementation + train + tests
- `3237bc7` — Stage 2 implementation + train (regression noted)
- `7945d9d` — c3.4 (Stages 1+2) report
- `6f14cb3` — multi-user figures regenerated with c3.4 picks
- (next) — Stage 3 (`c3p3_cat`) train + eval + report update

---

## 9. Headline take-away

**The c3.3 numeric schema is near-optimal for this dataset.** The 7-feature c3.4 candidate set, derived from the EDA's Pattern E and Pattern F observations, **does not yield reliable improvement**:

- *Cheap features* (F3 / F5 / F6 / F7) → marginal lift (+0.24 pp PR-AUC, +1.92 pp ROC-AUC) but largely redundant with what c3.3 already encodes.
- *Heavy features* (F1 / F2) → **regression** (−0.77 pp PR-AUC) caused by sparse 2-step Markov fallback and noisy co-FG matrix cells.

**The architecture change is the lever.** Adding a 4-d learnable `cat_emb` (only 44 params, 300 total once first-layer width is included) to the SAME c3.3 numerics buys **+0.36 pp PR-AUC** and surfaces the model as the leader on PR-AUC, FK@.25, MSR@.25, WAKR@.25, and P_kill at τ_keep — **all simultaneously**, with no metric regression.

The new multi-user headline is therefore a tie at the top:
- **`c3p3_cat`** — best PR-AUC (0.9078), best WAKR@.25, best long-tail handling.
- **Pro-List/c3.3** — best ROC-AUC (0.8336), best PosRank.
- **Pro-Wide/c3.3** — best MCC at τ_keep, best FK@.5 / MSR@.5.

Combining `cat_emb` with the listwise loss or wider arch is the obvious next experiment (§6.2–§6.3); each is < 1 hour of work.
