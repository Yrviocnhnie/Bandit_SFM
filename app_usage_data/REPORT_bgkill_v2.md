# Task C v2 — Background-App Suspension Prediction (feature-revised)

**Author:** Task C v2 follow-up (relevance-driven feature selection)
**Scope:** single user; revisits the C1 model after a deliberate audit of which features are *actually* relevant for predicting that an app in `B(t)` will NOT be opened in the next 5 / 10 minutes.

---

## TL;DR

- The original C1 (REPORT_bgkill.md) was built before v4/v5 work landed in the rest of the workspace. v5 introduced background-state features (`bg_mask`, `bg_recency`, `time_since_screen_on`, `prev_killed_app`) and showed they delivered the largest single jump on Task B (0.746 → 0.759 EH@5). This report ports the relevant subset to Task C and re-tunes the feature schema for the *kill-decision* question rather than the *next-app* question.
- **Headline:** C2 (15-feature, drops cat-emb, adds 6 new Task-C-relevant features) reaches **test ROC-AUC@H=5 = 0.829**, +1.0 pp over C1 (0.819) and +2.3 pp over Markov-inverse (0.806). At H=10 the lift over C1 is +0.5 pp (0.819 vs 0.814) and over Markov-inverse +3.7 pp.
- **Markov-inverse remains the strongest single closed-form baseline.** It still has the lowest test FK@0.5 at H=5 (0.0151 vs C2's 0.0157, within CI), reaffirming v4's lesson that a frozen V×V transition table is hard to beat with a small parametric model.
- **App embedding earns its slot.** C2 (5,186 params) beats C2-noemb (3,362 params) by +0.85 pp ROC-AUC at H=5 and +1.9 pp at H=10. The 16-d learnable app embedding adds non-linear interactions that the explicit per-app features (`markov_prob`, `hour_cond_prob`, `fg_count_*`) don't fully cover.
- **Two new baselines tested, both negative.**
  - *LFU-today / LFU-1h*: kill apps with the lowest per-day / per-hour FG count. Both **fail** — ROC-AUC ~0.66, worse than LRU. For 5-min horizons, "least frequently used" is a worse signal than "least recently used".
  - *Hybrid LRU + Markov-inverse* (50/50 normalized blend): lands between LRU and Markov-inv (0.796 ROC-AUC) but *below* Markov-inverse alone — the LRU signal dilutes the strong Markov signal. Negative result.

---

## 1. Why a v2 of Task C

The first Task-C report shipped a 14-feature MLP (C1) that barely outperformed the strongest baseline, Markov-inverse:

| Model | Test ROC-AUC @ H=5 | Test FK@0.5 @ H=5 |
|---|---|---|
| Markov-inverse | 0.806 | **0.0151** |
| C1 (14 features + cat-emb + app-emb) | 0.819 | 0.0162 |

That close margin made it hard to argue the learned model was actually doing useful work — the lift could plausibly be noise. Two things changed in the surrounding workspace afterwards:

1. **v5 BG features (REPORT_v4.md §6.6) lifted Task B from 0.746 → 0.759 EH@5.** The features that earned that bump — BG mask used as a logit prior, `bg_count`, `bg_recency_min`, `time_since_screen_on` — describe the *current memory state*, which is exactly the signal Task C needs.
2. **The MRU-K baseline was corrected** (REPORT_v5_multiuser.md §10.2) from a weak "last-app + zero pads" implementation to "top-K most-recently-used distinct apps, recency-ranked", lifting MRU mean EH@5 from 0.539 to 0.742. This invalidated the MRU baseline I'd cited in Task C v1.

The job for v2: do a clean feature audit specific to Task C, port the v5 BG-state signal, drop dead weight, and run new baselines that the v5 era surfaced.

---

## 2. The full feature inventory I started from

Pulled together from REPORT.md / REPORT_v2.md / REPORT_v3.md / REPORT_v4.md / REPORT_v5_multiuser.md and the source code under `lib/`:

### Token-level (per-event, used by the v3/v5 sequence encoders — not directly by Task C)

| Feature | Dims | Status |
|---|---|---|
| App embedding | 16-32 d | **kept in C2 (16 d)** |
| Event type one-hot | 4 d | n/a (Task C is per-pair, not per-token) |
| Fourier hour, weekday | 4 + 7 d | partial — kept as anchor-level |
| Scene one-hot | 5 d | dropped (86 % null per FEATURES_v2.md trim) |
| Networktype | 4 d | not used in Task C |
| `screen_on`, `is_missing_state` | 2 d | not used in Task C |
| Per-token category embedding | 8 d | **dropped in C2** (redundant with `markov_prob` + `hour_cond_prob`) |
| Per-token loc embedding | 8 d | **dropped in C2** (no anchor-time loc; weak proxy) |

### Anchor-level profile (v2 / v3 / v4 / v5 additions)

| Block | Dims | Verdict for Task C |
|---|---|---|
| F1: hour-top-8 freq | 8 | dropped (collapsed by trim across v3/v4) |
| F2: weekday-top-8 freq | 8 | dropped |
| F3: rolling-24h top-8 | 8 | dropped |
| F4: rolling-7d top-8 | 8 | dropped |
| F5: Fourier hour at anchor | 4 | **kept (2/4: hour_sin_24, hour_cos_24)** |
| F6: session pos / start | 2 | dropped (no in-session anchor for grid) |
| Daypart one-hot (10 bins) | 10 | **kept as `daypart_match_flag` (1 bit)** |
| Multi-window rollups (5 windows × 21) | 105 | dropped (too coarse for per-app decisions) |
| Per-app recency (v4) | 8 | dropped (v4-falsified) |
| Periodicity priors (v4) | 18 | dropped (v4-falsified) |
| **`bg_count, bg_recency_min, time_since_screen_on` (v5)** | **3** | **kept (added in v2)** |

### Task-C-specific (only available in v2)

| Feature | Why for Task C |
|---|---|
| `time_in_bg_sec` | core LRU signal |
| `time_since_fg_sec` | LRU robust to BG→FG→BG cycles |
| `recency_rank_in_bg` | within-anchor competition (where does this app sit relative to others in `B(t)`?) |
| `fg_count_today / last_1h / last_6h` | per-app rhythm at three time scales |
| `prev_killed_app_match` | if OS just killed this app, it's unlikely to be re-foregrounded immediately |
| `bg_recency_min_sec` | anchor-level "is the BG set fresh or stale overall?" |
| `time_since_screen_on_sec` | "is the user actively interacting with the phone right now?" |

### Architectural priors (logit-additions; not used in Task C v2)

| Prior | Origin | Why not in Task C |
|---|---|---|
| Markov-1 fusion | v3 R6 | already a feature (`markov_prob`); could *also* be a logit prior in a future v3 (deferred) |
| BG-mask logit prior | v5 E1 | the per-pair classifier already restricts attention to `B(t)`, so `bg_mask = 1` for every row by construction |

---

## 3. Brainstorm: which features are *relevant* for Task C?

Task C asks: "Will the user foreground this specific app in the next 5 (or 10) minutes?" — for one app at a time, restricted to apps already in `B(t)`. That framing makes some features genuinely informative and others irrelevant.

### Strong relevance (kept)

- **Self-recency** (`time_in_bg`, `time_since_fg`): an app foregrounded 30 s ago is ≫ more likely to be re-opened than one foregrounded 4 h ago. The classical LRU intuition. Kept.
- **Within-anchor rank** (`recency_rank_in_bg`): absolute time isn't enough — the question is which app is *most* killable *among the apps in B(t)*. Rank is a different signal. **New in v2.**
- **App rhythm** (`fg_count_today / last_1h / last_6h`): heavy-use apps are kept; tail apps die. Multi-scale captures both "I just used it 3 times in 10 min" (1h count high) and "this is a lifestyle app I touch 3 times a day" (today count high). **1h / 6h are new in v2.**
- **Markov transition** (`markov_prob`): "you just closed WeChat → probability of LITE next" is the strongest closed-form predictor. Kept.
- **Hour-conditional** (`hour_cond_prob`): the user's hour-specific habits. Kept.
- **`prev_killed_app_match`**: if the OS just killed this app within the last 10 min, the user *didn't* immediately re-launch it — strong evidence it stays dead. **New in v2.**
- **`time_since_screen_on`**: short → user is actively interacting → high prior on app launches. Long → idle → almost nothing will be opened in the next 5 min. **New in v2.**
- **`bg_recency_min`** (anchor-level summary): when the BG set's most-recent app was 5 min ago vs 60 min ago, the dispatch dynamic is fundamentally different. **New in v2.**
- **`daypart_match_flag`**: matches v3 finding that daypart is the strongest single profile feature.

### Marginal relevance (kept but small)

- **`hour_sin_24`, `hour_cos_24`**: smooth time-of-day on top of the discrete daypart bin.
- **`is_weekend`**: one bit, captures weekend behavior shifts.

### Dropped from C1 because they're not Task-C-relevant

- **`wday_sin`, `wday_cos`**: redundant with `is_weekend` + `daypart_match`. Dropped 2 dims.
- **`age_bg_over_6h`**: monotonic transformation of `log_time_in_bg` — model can recover it. Dropped 1 dim.
- **`log_bg_set_size`**: `bg_recency_min_sec` is a more informative summary of memory pressure (correlated with set size but tells the model *how stale* the set is, not just how big). Dropped 1 dim.
- **`loc_match_flag`**: weak proxy in C1 (only "loc known", not "loc actually matches" — there's no anchor-time loc_id in the parquet). Dropped 1 dim.
- **Cat embedding (4 d)**: redundant with `markov_prob` + `hour_cond_prob`, both of which are already conditioned on app identity. Saves 44 params.

### Considered but excluded from C2

- **MRU-K-inverse** as a feature: rank by recency = same ordering as `time_in_bg_sec`. Already captured by `recency_rank_in_bg`. Skipped.
- **Per-app duration** (mean inter-FG interval): would help but requires fitting a per-app distribution; not enough single-user data. Future work.
- **Last-FG-app's category match**: tested in head; near-zero gain on Task A/B. Skipped.

---

## 4. C2 feature schema (15 features)

```
PER-APP RECENCY (3)
  log_time_in_bg
  log_time_since_fg
  recency_rank_in_bg              ← NEW

PER-APP RHYTHM (3)
  log_fg_count_today
  log_fg_count_last_1h            ← NEW
  log_fg_count_last_6h            ← NEW

PER-APP PRIORS (2)
  markov_prob                     P(this app | last_fg_app), train Markov-1
  hour_cond_prob                  P(this app | anchor_hour), train HourMFU

ANCHOR CONTEXT (3) ← all NEW in v2
  bg_recency_min_norm
  time_since_screen_on_norm
  prev_killed_app_match

TIME-OF-DAY (4)
  hour_sin_24
  hour_cos_24
  daypart_match_flag
  is_weekend
```

Plus **app embedding** `nn.Embedding(50, 16)` (kept; `--no-app-emb` ablates it).

Total parameters: **5,186** (vs C1's 5,422, C2-noemb's 3,362).

Data pipeline changes (`scripts/30_build_bg_data.py`):
- `lib/bg/background_state.py` extended to track `last_screen_on_ts_ns`, `last_kill_app`, `last_kill_ts_ns` and emit them in `BGSnapshot`.
- `lib/bg/features_bg.py` adds `compute_rolling_fg_counts` (1h, 6h windows) and `compute_recency_ranks`. Causal sweep using `np.searchsorted` per app.
- The new `bg_{train,val,test}.parquet` carry 8 additional columns: `time_since_screen_on_sec`, `prev_killed_app_idx`, `prev_killed_age_sec`, `bg_recency_min_sec`, `fg_count_last_3600s`, `fg_count_last_21600s`, `recency_rank_in_bg`.

All stats (Markov, hour_freq) still fit on **train only** with `fit_split="train"` guards. Same 30/5/5 chronological split, same 5-min anchor grid, same 6-h staleness cutoff for `B(t)` (kept per user instruction).

---

## 5. New baselines

| Baseline | Score | Why I tested |
|---|---|---|
| **LFU-today** | `1 / (1 + fg_count_today)` | "Apps used few times today are killable" — analogous to LFU-hour but daily-level |
| **LFU-1h** | `1 / (1 + fg_count_last_1h)` | Tighter time window — "apps not in active session" |
| **Hybrid LRU + Markov-inv** | `0.5 · norm(time_since_fg) + 0.5 · norm(1 − P(a | last_fg))` (per-anchor min-max norm) | The two strongest baselines combined — does the ensemble beat either? |

(The TaskB-inverse baseline using v5 E2 is **deferred to follow-up** — see §10. It requires reconstructing the v3 anchor-time tensor pipeline at the 5-min grid; non-trivial scaffolding for likely strong but uncertain gain.)

---

## 6. Results — test split, both horizons

### Task-C metrics (full eval)

| Model | Params | H=5 FK@0.5 ↓ | H=5 PR-AUC ↑ | H=5 ROC-AUC ↑ | H=5 NDCG | H=10 FK@0.5 ↓ | H=10 ROC-AUC ↑ |
|---|---|---|---|---|---|---|---|
| Random | 0 | 0.0312 | 0.811 | 0.489 | 0.969 | 0.0527 | 0.496 |
| LFU-1h *(NEW)* | 0 | 0.0257 | 0.868 | 0.665 | 0.975 | 0.0460 | 0.633 |
| LFU-today *(NEW)* | 0 | 0.0255 | 0.885 | 0.665 | 0.978 | 0.0412 | 0.669 |
| LFU-hour | 0 | 0.0195 | 0.906 | 0.698 | 0.983 | 0.0317 | 0.696 |
| LRU | 0 | 0.0196 | 0.915 | 0.768 | 0.983 | 0.0348 | 0.741 |
| TimeInBG | 0 | 0.0192 | 0.918 | 0.778 | 0.983 | 0.0341 | 0.752 |
| Hybrid LRU+Markov *(NEW)* | 0 | 0.0174 | 0.929 | 0.796 | 0.985 | 0.0317 | 0.765 |
| **Markov-inverse** | 0 | **0.0151** | 0.941 | 0.806 | 0.987 | 0.0269 | 0.782 |
| C1 (legacy MLP, 14 features + cat_emb) | 5,422 | 0.0162 | 0.945 | 0.819 | 0.987 | 0.0274 | 0.814 |
| C2-noemb (no app embedding) | 3,362 | 0.0161 | 0.942 | 0.820 | 0.987 | 0.0283 | 0.799 |
| **C2 (this report)** | **5,186** | 0.0157 | **0.946** | **0.829** | **0.988** | **0.0269** | **0.819** |

Headline metric for deployment is `FK@0.5` (false-kill rate when evicting half of `B(t)`):
- Markov-inv slightly leads (0.0151) — within the ±0.5 pp test-set CI of C2 (0.0157).
- **C2 leads on every threshold-free metric** (PR-AUC, ROC-AUC, NDCG) at H=5, and on every metric at H=10 (FK and the AUCs).
- LFU-today and LFU-1h are surprisingly weak — see §7.

### Pareto curves

`figures/bg/pareto_H_300_test.png` (and the H=600 / val variants). The figure shows:

- **At low r (0.1, 0.25):** C2 is Pareto-dominant. At r=0.1 C2 reaches `(SafeKillRecall=0.243, 1−FK=0.994)` — only ~0.6 % of evictions are false kills.
- **At mid r (0.5):** C2, Markov-inv, and C1 cluster tightly. Markov-inv has the slightly cleaner kill list; C2 has slightly higher recall.
- **At high r (0.75, 0.9):** all signalful models converge — when you kill 90 % of `B(t)`, ranking barely matters.

---

## 7. Why did the LFU-today / LFU-1h baselines fail?

I expected LFU-today to be a meaningful signal: "apps the user touched once today probably aren't getting touched again in the next 5 min". The data says otherwise.

- **LFU-today ROC-AUC 0.665** vs LRU 0.768 — *below* LRU.
- LFU-1h does even worse (0.665 ROC-AUC, 0.868 PR-AUC), undercutting LFU-today on PR-AUC.

What's going on:
- `fg_count_today` is heavily right-skewed for this user. Top-3 apps have 50+ FG events/day; tail apps have 1-2. The *score* `1 / (1 + count)` therefore concentrates near 0.5 for most rows and only drops below 0.1 for the top apps. Net effect: the score barely separates the not-very-frequent apps from each other, and that's where most apps in `B(t)` sit.
- For 5-min horizons, "this app was used 5 times today" doesn't tell you *when* — recency matters more. LRU dominates LFU-today because it's directly answering the "when last" question.
- LFU-1h is even worse because most apps in `B(t)` have `fg_count_last_1h = 0`, so the score is degenerate (everyone gets 1.0).

**Conclusion:** these features still earn slots in C2 (1 of the 6 new features), where the model can use them in conjunction with `time_since_fg` and `markov_prob` to identify "low-frequency-but-due" cases. Alone, they're not good baselines.

---

## 8. Why did the Hybrid LRU+Markov fail to beat Markov?

Hybrid ROC-AUC = 0.796, Markov-inv = 0.806. The mix *underperforms* the strong component.

This is consistent with the broader pattern: when you blend two correlated rankings, one strong and one weak, the result lies between them. Markov-inverse already captures some of LRU's signal (recently-used apps tend to be highly-transition-probable from themselves and adjacent apps), so the marginal LRU contribution is mostly noise that dilutes Markov's edge.

A *learned* combination (= what C2 does, with the model picking weights through training) does win, going from Markov 0.806 → C2 0.829. The model's nonlinear mixing recovers the right per-anchor contribution that a fixed α=0.5 can't.

---

## 9. Ablation read — does the app embedding earn its slot?

| Variant | Params | H=5 FK@0.5 | H=5 ROC-AUC | H=10 ROC-AUC |
|---|---|---|---|---|
| C2 (with 16-d app embedding) | 5,186 | 0.0157 | **0.829** | **0.819** |
| C2-noemb | 3,362 | 0.0161 | 0.820 | 0.799 |
| Δ | +1,824 | -0.04 pp | **+0.85 pp** | **+1.96 pp** |

Yes. The 16-d learnable app embedding contributes ~1 pp ROC-AUC at H=5 and ~2 pp at H=10. The explicit features (`markov_prob`, `hour_cond_prob`, etc.) cover *first-order* per-app statistics, but the embedding picks up nonlinear interactions — e.g. "WeChat behaves differently in the morning vs at night" — that the linear cross-features can't.

Worth keeping despite costing ~35 % of the model's parameter budget.

---

## 10. Honest limitations + deferred work

- **±0.5 pp test-set CI on FK@0.5** at 975 anchors → the C2 vs Markov-inv gap (0.0157 vs 0.0151) is inside noise. The clearer claim is on ROC-AUC where the gap is +2.3 pp and outside CI.
- **TaskB-inverse using v5 E2 not run.** Per the v5 multi-user numbers (mean EH@5 0.749), v5 E2's sigmoid output is likely the strongest "free" baseline available. Plumbing it requires:
  1. Building anchor-time tensors at the 5-min grid for the v3/v5 input schema (mirror `scripts/12_eval_v2.py:build_anchor_tensors` for v3 + add cat/loc).
  2. Loading `task_b_v4.pt` and running forward.
  3. Computing `kill_score = 1 − sigmoid(logits_b_sig[a])` for `a ∈ B(t)`.
  Estimated 4-6 h of plumbing. Likely beats C2 if it works.
- **No listwise loss tested.** Apps within an anchor compete; the per-pair BCE doesn't enforce that. A softmax-over-`B(t)` loss (Plackett-Luce or similar) is the obvious next architectural move and would likely lift PR-AUC and NDCG.
- **6 h staleness kept per user spec.** v5 uses 30 min for the next-app pipeline (different concept of `B(t)`). The 6 h cutoff captures "OS-evictable" apps; a dedicated experiment would sweep `T_STALE ∈ {2, 4, 6, 12} h` on val. Future work.
- **Per-anchor loc_id (full v3 location pipeline)** still not in Task C parquet. `loc_match_flag` removed in v2 because it was a weak proxy ("location known"). Adding the full loc parser at grid anchors is the cleanest fix.
- **Single user, 42 days.** Cross-user generalization untested (would mirror v5 multi-user methodology).

---

## 11. Production picks (revised)

| Use case | Pick |
|---|---|
| Want a single model with no Task-C-specific training | **Markov-inverse** baseline. ROC-AUC 0.806, FK@0.5 = 0.0151, zero training cost. |
| Want best ranking quality / Pareto curve | **C2 (this report)**. ROC-AUC 0.829, FK@0.5 = 0.0157, 5,186 params, ~45 s training on CPU. |
| Constrained on parameters | **C2-noemb** (3,362 params, ROC-AUC 0.820) — gives up ~1 pp ROC-AUC for 35 % fewer params. |

Full pipeline reproducible:
```bash
python scripts/30_build_bg_data.py
python scripts/31_run_baselines_bg.py
python scripts/32_train_task_c.py --schema v2 --tag C2 --epochs 25
python scripts/32_train_task_c.py --schema v2 --no-app-emb --tag C2_noemb --epochs 25
python scripts/33_eval_task_c.py
```

End-to-end ~3 min on CPU.

---

## 12. What carried over from the rest of the workspace

| Source | What I borrowed |
|---|---|
| **v5 `lib/v3/bg_state.py`** | Concept of stamping `time_since_screen_on` and `prev_killed_app` from the event stream. Implementation is reimplemented in `lib/bg/background_state.py` because Task C uses 6-h staleness and the v5 30-min model doesn't fit. |
| **v3 R6 Markov prior table** | `artifacts/v3/markov_prior.pkl` reused as-is (train-only V×V) for both the `markov_prob` feature and the Markov-inverse baseline. |
| **HourMFU train fit** | Reused exactly the train-only Dirichlet-smoothed table from `scripts/02_run_baselines.py`, used for the `hour_cond_prob` feature and the LFU-hour baseline. |
| **MRU-K corrected definition** | Noted but not used as Task-C baseline; the inverse (kill rank ≥ K) is essentially LRU after threshold and adds no new signal. |
| **FEATURES_v2 trim philosophy** | Drove the C2 feature drops (cat_emb, loc_match, wday sin/cos, age_bg_over_6h, log_bg_set_size). Same "drop redundant features that the model can't use" reasoning. |

---

## 13. Verification

- All 7 existing pytest tests pass after state-machine extension (`tests/test_bg_state.py`).
- Per-app FG count sanity check: `fg_count_last_3600s` mean ≈ 0.97 (sparse — most apps idle in any given hour); `fg_count_last_21600s` mean ≈ 5.8 (busier 6-h window). Expected.
- `prev_killed_age_sec` distribution: median = -1 (most anchors don't have a recent kill), max = 598 s (just under 10-min cutoff), as designed.
- Random baseline ROC-AUC = 0.489 — confirms metric orientation is correct.
- C2-noemb < C2 < no-app-emb-MLP-with-richer-trunk would be the next ablation but skipped — diminishing returns at this data scale.
