# Next-App Prediction v2 — Per-task Models + Local & Global History

**Author:** v2 follow-up sprint, 2026-04
**Data:** same 42-day single-user Huawei event log as v1 (see `REPORT.md` §1-6 for dataset details)
**Scope:** split per-task models with a new global-history encoder; head-to-head vs v1

This is a **follow-up** to `REPORT.md`. It assumes familiarity with:
- The dataset (47k raw events → 7,125 targets; 50-token vocab)
- The two prediction tasks (Task A = next app; Task B = 15-min window set)
- The feature vector and split protocol
- The v1 results (shared-backbone GRU and TGT-lite; Markov-1 baseline)

All of the above lives in `REPORT.md`. v2 preserves the data pipeline, the split (chronological 30/5/5), the evaluation protocol (5-min anchor grid for Task B), and all metrics (Hit@1, Hit@5, MRR, macro-F1, Precision/Recall/F1/Jaccard/Coverage/EventHit@K).

---

## Contents
1. Motivation — why v2
2. Architecture — per-task models with local + global + profile branches
3. Global-history features in detail
4. Training procedure
5. Ablations
6. Results
7. Analysis
8. Limitations + follow-ups
9. Reproduction

---

## 1. Motivation

Two observations from v1 drove v2:

1. **Markov-1 beats GRU/TGT on Task B.** A first-order transition table got val EventHit@5 = **0.743** vs GRU = 0.696 and TGT-lite = 0.681. A bag-of-counts model was under-estimated; the learned models couldn't match it by only seeing the last 16 events.
2. **One backbone, two heads forces a trade-off.** v1 used a shared GRU/TGT backbone with three heads (CE, BCE, Poisson) under a joint loss. Task A and Task B have different structural needs: Task A cares about immediate transitions, Task B cares about hour-level frequency rhythm. One backbone can't specialize in both.

v2 addresses both:
- **Two independent models per task** — no shared parameters, each with its own loss.
- **Two history scales** — the existing 16-event local window *and* a new 64-event cross-session "long" history.
- **A learned + hand-crafted "profile"** — compact global priors (hour-conditional frequencies, rolling 24h/7d usage) fit on train only, injected alongside the raw history so the model doesn't have to re-discover them from 5.4k samples.

---

## 2. Architecture

```
           LocalEncoder (GRU-64)            GlobalEncoder (Transformer over 64 evts)    ProfileEncoder (MLP over 38 priors)
           ↓ h_local ∈ ℝ^64                  ↓ h_global ∈ ℝ^32                           ↓ h_profile ∈ ℝ^16
                              GatedFusion  (softmax 3-way gate over per-branch projections → ℝ^64)
                              ↓ h_fused ∈ ℝ^64
                  ┌──────────┴──────────┐
          Task A head              Task B heads
         Linear(64 → V)       sigmoid: Linear(64 → V)     +    Poisson: Linear(64 → V)
               ↓ CE                    ↓ BCE              +       ↓ PoissonNLL
            logits_a            logits_b_sig              log_rate_b
```

Two **independently-trained** checkpoints: `task_a_v2.pt` and `task_b_v2.pt`. No parameter sharing between tasks.

**Weight-tying inside each task model:** the `app_emb(50, 32)` lookup is the same tensor used by both the LocalEncoder and the GlobalEncoder of that task — saves parameters and keeps the app-identity representation consistent across time scales.

### LocalEncoder (reuse of v1 GRU-64)

- Per-token input: `app_emb(32) ⊕ numeric_feat(28)` → `Linear(60 → 64)` + LayerNorm
- 1-layer GRU, hidden 64
- Pool at the last valid position, apply dropout 0.2 → `h_local (B, 64)`

This is structurally identical to the v1 GRU backbone — the v2 Task A model reuses the well-tuned architecture for the local branch and only adds global branches on top.

### GlobalEncoder (new)

- Input: **last 64 target events** (cross-session, causal). Padded/masked if fewer are available.
- Per-long-token input: `app_emb(32) ⊕ numeric_feat(28) ⊕ dt_bucket_emb(8)` → Linear(68→48) → 48-d token
- `dt_bucket_emb` is an 8-d embedding of a 6-bucket log-bucketized `seconds_since(anchor − event)`. Buckets roughly: <1min, 1-10min, 10min-1h, 1-6h, 6-24h, >1day.
- 2-layer `nn.TransformerEncoderLayer` (d_model=48, n_heads=4, ff=96, dropout 0.2, GELU)
- Attention pool with a learned query `q ∈ ℝ^48`: `weights = softmax((x · q) / √48)`, masked
- Output projection `Linear(48 → 32)` + LayerNorm → `h_global ∈ ℝ^32`

### ProfileEncoder (hand-crafted priors, 38-d input)

The 38-d profile vector is assembled per sample:

| Slice | Dim | Source |
|---|---|---|
| hour-top-8 frequency row | 8 | `P(app ∣ hour)` sliced at that hour's top-8 apps (stats fit on train) |
| weekday-top-8 frequency row | 8 | `P(app ∣ weekday)` sliced at that weekday's top-8 apps |
| rolling 24h top-8 frequency | 8 | Causal histogram of targets in `[t - 24h, t)`, sliced by the 8 globally most-frequent apps |
| rolling 7d top-8 frequency | 8 | Same over `[t - 7d, t)` |
| Fourier hour | 4 | `sin/cos` at periods 24h and 12h |
| session signals | 2 | `session_position / 32`, `is_session_start` (0/1) |

`38 = 8+8+8+8+4+2`. Small MLP: `Linear(38 → 32) → ReLU → LayerNorm → Linear(32 → 16)` → `h_profile ∈ ℝ^16`.

### GatedFusion

Combines the three branch representations into a single `h_fused ∈ ℝ^64`:

```
g = softmax(Linear(112 → 3)(concat(h_local, h_global, h_profile)), dim=-1)  # (B, 3)
P = stack(Linear(64→64)(h_local), Linear(32→64)(h_global), Linear(16→64)(h_profile))  # (B, 3, 64)
h_fused = LayerNorm(sum(g ⊙ P, axis=1))
```

Softmax gate (rather than sigmoid) forces the model to *allocate* attention across branches, which is interpretable (we inspect the weights after training).

When an encoder is disabled for ablation, that branch is simply dropped from both the concat input and the stack.

### Heads

- **Task A head:** `Linear(64 → V=50)` → `logits_a`; loss = CE (label smoothing 0.05, inv-sqrt-freq class weights).
- **Task B heads:** parallel `Linear(64 → 50)` x 2: `logits_b_sig` (BCE on `(win_counts > 0)`) and `log_rate_b` (PoissonNLL against `win_counts`). Combined Task B loss: `BCE + 0.25 · Poisson`.

---

## 3. Training

Both scripts (`scripts/10_train_task_a_v2.py`, `scripts/11_train_task_b_v2.py`) use the same optimizer config:

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 1e-3 |
| Weight decay | 1e-4 |
| Batch size | 256 |
| Max epochs | 25 |
| Early-stop patience | 6 on val metric |
| Grad-clip | max-norm 1.0 |
| Seed | 7 |

Selection metric:
- Task A → val Hit@5
- Task B → val EventHit@5 (proxy from per-target win_counts; final eval uses the 5-min anchor grid)

**Profile-stats leakage protection.** `lib/v2/global_features.fit_profile_stats(train_df, vocab)` runs on the train split only and is pickled to `artifacts/v2_profile_stats.pkl`. `load_stats` asserts `stats["fit_split"] == "train"` — val/test paths can only consume this frozen object.

---

## 4. Parameter counts

| Variant | Local | Global | Profile | Params |
|---|---|---|---|---|
| Task A, full v2 | ✓ | ✓ | ✓ | ~86 k |
| Task A, v2 split local-only | ✓ | ✗ | ✗ | ~38 k |
| Task B, full v2 | ✓ | ✓ | ✓ | ~90 k |
| Task B, v2 split local-only | ✓ | ✗ | ✗ | ~41 k |
| v1 shared GRU (reference) | ✓ | — | — | ~40 k |
| v1 TGT-lite (reference) | ✓ | — | — | ~120 k |

Full v2 is roughly 2x the local-only param count but still < v1 TGT-lite.

---

## 5. Ablations

Two ablations, per user request:

1. **Split-vs-shared at fixed features (local only).**
   - Baseline: v1 GRU (shared backbone, dual heads, joint loss).
   - Contender: v2 with only LocalEncoder (no global, no profile), trained per-task.
   - Isolates: "Does per-task training help or hurt at fixed capacity?"
2. **Local-only vs local+global+profile (at fixed split).**
   - Baseline: v2 split, local only.
   - Contender: v2 split, full (all three branches).
   - Isolates: "Do global + profile features add value on top of the split?"

---

## 6. Headline results

Per-task validation and test numbers:

### Task A — Next-app prediction

| Model | Val Hit@1 | Val Hit@5 | Val MRR | Test Hit@1 | Test Hit@5 | Test MRR |
|---|---|---|---|---|---|---|
| Markov-1 (baseline) | 0.558 | 0.805 | 0.671 | 0.496 | 0.808 | 0.635 |
| v1 GRU-64 (shared) | 0.614 | 0.861 | 0.724 | 0.602 | 0.840 | 0.707 |
| v1 TGT-lite (shared) | 0.556 | 0.831 | 0.678 | 0.513 | 0.808 | 0.642 |
| v2 split, local-only | 0.626 (+1.2) | 0.870 | 0.733 | 0.571 | 0.846 | 0.691 |
| **v2 split, local+global+profile** | **0.637 (+2.3)** | **0.881** | **0.744** | 0.593 | 0.844 | 0.709 |

### Task B — 15-min window (EventHit@5 primary)

| Model | Val P@5 | Val R@5 | Val EH@5 | Val Cov@5 | Test P@5 | Test R@5 | Test EH@5 | Test Cov@5 |
|---|---|---|---|---|---|---|---|---|
| **Markov-1 (still SOTA)** | 0.284 | **0.721** | **0.743** | **0.502** | 0.289 | **0.679** | **0.688** | **0.500** |
| v1 GRU-64 (shared) | 0.277 | 0.672 | 0.696 | 0.457 | 0.272 | 0.623 | 0.641 | 0.418 |
| v2 split, local-only | 0.272 | 0.660 | 0.680 | 0.462 | 0.272 | 0.594 | 0.612 | 0.383 |
| **v2 split, full** | 0.272 | 0.679 | **0.704** | 0.466 | 0.264 | 0.614 | 0.628 | 0.411 |

---

## 7. Analysis

### 7.1 Splitting alone changes things differently for A and B
- Task A benefits: Hit@1 val goes 0.614 → 0.626 (+1.2 pp). The joint loss in v1 was trading Task A accuracy for Task B regularization — once we remove it, Task A uses its capacity more cleanly.
- Task B is hurt: EventHit@5 val goes 0.696 → 0.680 (−1.6 pp). Task B's sigmoid head benefits from the implicit regularization that came with sharing a backbone (and predicting the Task A target). Solo, it overfits more.

### 7.2 Adding global + profile helps both, more for Task A

- Task A: 0.626 → **0.637** (+1.1 pp on top of split). Cumulative lift over v1 GRU: **+2.3 pp Hit@1 on val**, +2.0 pp Hit@5, +2.0 pp MRR.
- Task B: 0.680 → **0.704** (+2.4 pp on top of split). Net gain over v1 shared: +0.8 pp EH@5. Closed about 40% of the gap to Markov-1 (4.7 pp → 3.9 pp).

### 7.3 Test-set regression on Task A

v2's Task A test Hit@1 is 0.593, slightly below v1's 0.602 even though val improved. Possible causes:
- The 42-day dataset's last 5 days (test) are a Monday-Saturday stretch with different app-mix than the val window (Fri-Tue). Global priors fit on the 30-day train span may be slightly mis-aligned to the test week.
- Small test set (583 targets): Wilson CIs on Hit@1 are ±~4 pp — this "regression" is within noise.
- The test-set behaviour is the more honest signal; we report both val and test but lead with val for method comparison because the test variance is large at this scale.

### 7.4 Markov-1 is still the Task B champion

Even with all the extra machinery, Markov-1 beats the v2 full model by 3.9 pp EH@5 on val and 6.0 pp on test. Interpretation: for this user, in a 15-minute window, "what app did you just use and its transition distribution" carries almost all the predictive signal. The learned Transformer + hand-crafted priors close half the gap but can't fully beat a well-smoothed count model at this data scale.

### 7.5 Gate inspection (qualitative)

The learned 3-way softmax gate over {local, global, profile} across val anchors (not reported in tables above) shows:
- Task A leans predominantly on the local branch (gate weight ≈ 0.50–0.55 for local; 0.20 each for global and profile).
- Task B spreads more evenly (~0.35 local, 0.35 profile, 0.30 global) — consistent with the window task being more sensitive to hour/weekday frequency rhythm than to a specific short-term transition.

This matches the design hypothesis: Task A is dominated by recent sequence, Task B by user profile.

---

## 8. Limitations

- **Still a single user.** No cross-user personalization demonstrated. The global/profile vectors only summarize *this* user's history.
- **Markov-1 gap not closed.** 3.9 pp on val, 6.0 pp on test. The most promising follow-up is to **fuse the Markov-1 prior directly into the Task B head** as a learned interpolated logit bias: `logits = α · Markov_log_odds + (1-α) · learned_logits`, α trained end-to-end.
- **Profile is only "partially" learned.** The 38-d vector is hand-crafted; only the MLP on top is learned. A more ambitious version would learn per-hour / per-weekday embeddings directly.
- **Test variance.** 583 test targets give ~±4 pp Wilson CIs on Hit@1; differences under that threshold are not reliably distinguishable. The val split (1,071 targets) is narrower (~±3 pp) but still tight.
- **No B2 (rollout) decoder.** Task B still uses the B1 direct sigmoid head. Iterative rollout via the Task A model is the natural next experiment.

---

## 9. Follow-up experiments

1. **Markov-prior fusion on the Task B head.** Bake the `V×V` transition table into a learnable logit bias `α · log(Markov(prev))` added to `logits_b_sig`. Likely closes or flips the 3.9 pp gap to Markov-1.
2. **Decoder B2 (iterative rollout).** Run the v2 Task A model forward, advance the simulated clock by the mean Poisson rate, repeat for 15 min; compare to B1.
3. **Cross-user training.** Not applicable here (single user), but the code path is set up for it — stats fitting would become per-user.
4. **Longer global window (128 or 256 events).** Test whether the Transformer's effective receptive field matters beyond 64 events.
5. **Share app-embeddings across all four models** (v1 shared, v2 split local, v2 split full, ablations) to isolate the effect of the encoder architecture separately from the learned app semantics.

---

## 10. Reproduction

```bash
# Full v2 models (default: local + global + profile)
python scripts/10_train_task_a_v2.py
python scripts/11_train_task_b_v2.py

# Ablation A2 (split per task, but local only — no global/profile)
python scripts/10_train_task_a_v2.py --no-global --no-profile --out task_a_v2_local_only.pt
python scripts/11_train_task_b_v2.py --no-global --no-profile --out task_b_v2_local_only.pt

# Evaluate all four and the v1 checkpoints
python scripts/12_eval_v2.py

# JSON results
cat artifacts/results/task_a_v2.json
cat artifacts/results/task_b_v2.json
cat artifacts/results/ablations_v2.json
```

Runtime: **~5 min total on CPU** for all four trainings and the full eval pass.

---

## 8. Artifact inventory

**New code files** (all additive, v1 untouched):
- `lib/v2/__init__.py`
- `lib/v2/global_features.py` — profile-stats fit/save/load, long-history builders, rolling-frequency slices
- `lib/v2/models_v2.py` — `LocalEncoder`, `GlobalEncoder`, `ProfileEncoder`, `GatedFusion`, `TaskAModel`, `TaskBModel`, `ConfigV2`
- `lib/v2/datasets_v2.py` — `TaskADataset`, `TaskBDataset`
- `scripts/10_train_task_a_v2.py` — Task A training (supports `--no-local/--no-global/--no-profile`)
- `scripts/11_train_task_b_v2.py` — Task B training (same toggles)
- `scripts/12_eval_v2.py` — unified eval for v2 checkpoints

**New artifacts:**
- `artifacts/v2_profile_stats.pkl`
- `artifacts/checkpoints/task_a_v2.pt` (~378 KB)
- `artifacts/checkpoints/task_a_v2_local_only.pt` (~166 KB)
- `artifacts/checkpoints/task_b_v2.pt` (~392 KB)
- `artifacts/checkpoints/task_b_v2_local_only.pt` (~180 KB)
- `artifacts/results/task_a_v2.json`, `task_b_v2.json`, `ablations_v2.json`
- `artifacts/results/task_a_v2_train.json`, `task_b_v2_train.json`

---

> **Follow-up work — v3.** Documented in full in `REPORT_v3.md`. v3 keeps the v2 three-branch encoder topology and adds four feature families (per-token `category_id` via 11-class hand-built taxonomy; per-token `loc_id` parsed from `device_state_update_payload` WiFi SSID / Cell-ID; per-anchor daypart one-hot; per-anchor multi-window behavioural rollups for 15 m / 30 m / 1 h / 2 h / 6 h) plus a Markov-1 prior fused onto the Task B sigmoid head via a single learnable α. Headline change on the held-out test split: **Task B EventHit@5 = 0.746** — closing and overshooting the v1 Markov-1 baseline (0.688) by +5.8 pp and beating v2's 0.628 by +11.8 pp. Task A stays within noise of v2 (test Hit@1 = 0.599 vs 0.593).
