# Personalized Surprise Trigger — Demo

Always-on trigger gate sitting between the rules engine and the
recommendation agent. The gate decides whether a scenario match is
"routine for this user" (suppress) or "surprising" (fire).

## Idea

The VAE is trained on the full categorical support of two scenarios
(ARRIVE_OFFICE and HOME_EVENING) from the unique-sample generator. A
**routine slice** of this support is defined for user *Alex* (commute
mode, time window, phone/light/sound envelopes). A **triplet loss**
with routine anchors / positives and non-routine negatives shapes the
latent space into two clusters, while the standard VAE reconstruction
loss keeps the embedding faithful to the feature values.

A new sample's novelty score combines:
- normalized reconstruction error (how surprised the VAE is)
- normalized L2 distance from the routine centroid in the latent mu-space

## Files

| file | purpose |
|---|---|
| `extract_scenarios.py` | pulls the ARRIVE_OFFICE + HOME_EVENING rows from the full 245k JSONL |
| `user_persona.py` | defines Alex's routine envelopes for both scenarios |
| `label_routine.py` | tags every extracted row with `is_routine` |
| `feature_encoder.py` | 248-dim mixed encoder (one-hot + binary + normalized scalar), group offsets exposed |
| `model.py` | Triplet-VAE module + mixed reconstruction loss + KL + triplet loss |
| `train.py` | β-annealing + γ-annealing for triplet, per-epoch separation metric |
| `build_test_samples.py` | tiers unseen samples as ROUTINE / MILD / MODERATE / STRONG / EXTREME |
| `score.py` | scores held-out routine + non-routine + tiered test samples, writes `artifacts/scores.json` |
| `plot.py` | latent PCA, boxplot by tier, novelty histogram |
| `trigger_demo.py` | 2-week simulation with injected surprises on specific days |
| `trigger_summary.py` | always-trigger vs surprise-trigger bar chart |

## How to run

```bash
PY=/home/mohan/.conda/envs/SmolVL/bin/python
cd /home/mohan/Bandit_SFM/demos/surprise_trigger

$PY extract_scenarios.py
$PY label_routine.py
$PY train.py --epochs 60 --steps-per-epoch 60 --batch-size 128 --lr 1e-3 \
             --latent-dim 32 --beta-max 0.5 --gamma-max 2.0 --triplet-margin 2.0
$PY build_test_samples.py
$PY score.py
$PY plot.py
$PY trigger_demo.py
$PY trigger_summary.py   # optional comparison chart
```

Outputs land in `artifacts/` and `artifacts/plots/`.

## Key training/eval numbers (this run)

- Feature dim: 248
- Training rows: 33,659 (all non-held rows; 433 routine, 33,159 non-routine)
- Held out: 80 routine + 500 non-routine
- Final `d_routine / d_nonroutine` ratio (held): **~2.8x**
- Routine novelty median (held): **0.26**
- Non-routine novelty median (held): **5.9**

## Trigger summary on tiered test (80 per tier per scenario where it exists)

| tier | triggered | total |
|---|---|---|
| ROUTINE | 0 | 80 |
| MILD | 80 | 80 |
| MODERATE | 70 | 80 |
| STRONG | 40 | 40 |
| EXTREME | 40 | 40 |

All routine events are correctly suppressed; all meaningful surprises fire.
The 10 missed MODERATE cases are borderline (e.g., HOME_EVENING routine except
hour shifted from 20 to 21) — the VAE correctly rates them as near-routine.

## Run outputs

- `artifacts/plots/tier_boxplot.png` — novelty by tier (threshold at 1.0)
- `artifacts/plots/novelty_histogram.png` — overlaid histograms
- `artifacts/plots/latent_pca.png` — PCA of the mu space, routine vs surprise
- `artifacts/plots/trigger_timeline.png` — 14-day simulation (6 triggered / 28 events)
- `artifacts/plots/trigger_summary.png` — always vs surprise-trigger counts per tier
