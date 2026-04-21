"""Simulate 2 weeks of Alex's scenario-triggers and compare
'always trigger' vs 'surprise-only trigger' using the trained novelty score."""
from __future__ import annotations

import json
import random
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F

from feature_encoder import ENCODING, encode_features
from model import TripletVAE, TripletVAEConfig
from score import novelty, recon_and_mu, encode_batch

DEMO_DIR = Path(__file__).parent
ART_DIR = DEMO_DIR / "artifacts"
PLOTS = ART_DIR / "plots"

SIM_DAYS = 14
SEED = 42
NOVELTY_THRESHOLD = 1.0
COOLDOWN_HOURS = 6


def pick(rows, pred, rng, n):
    pool = [r for r in rows if pred(r)]
    rng.shuffle(pool)
    return pool[:n]


def main():
    random.seed(SEED)
    rng = random.Random(SEED)

    # Build a fake 14-day schedule: each day has 1 ARRIVE_OFFICE (morning) and 1 HOME_EVENING (evening).
    # Occasionally inject a surprise.
    tests = [json.loads(l) for l in (DEMO_DIR / "data" / "test_samples.jsonl").open()]

    by_cat = {}
    for r in tests:
        by_cat.setdefault((r["scenario_id"], r["tier"]), []).append(r)
    labeled = [json.loads(l) for l in (DEMO_DIR / "data" / "demo_two_scenarios_labeled.jsonl").open()]
    routine_rows_ao = [r for r in labeled if r["scenario_id"] == "ARRIVE_OFFICE" and r.get("is_routine")]
    routine_rows_he = [r for r in labeled if r["scenario_id"] == "HOME_EVENING" and r.get("is_routine")]

    schedule = []
    for day in range(SIM_DAYS):
        # ARRIVE_OFFICE event
        if day == 3:       # day 3: arrived on Saturday (EXTREME)
            r = rng.choice(by_cat.get(("ARRIVE_OFFICE", "EXTREME"), [])) or rng.choice(routine_rows_ao)
            tag = "ARRIVE_EXTREME (weekend at office)"
        elif day == 7:     # day 7: drove instead of transit (MODERATE)
            r = rng.choice(by_cat.get(("ARRIVE_OFFICE", "MODERATE"), [])) or rng.choice(routine_rows_ao)
            tag = "ARRIVE_MODERATE (unusual commute)"
        elif day == 11:    # day 11: MILD phone placement off
            r = rng.choice(by_cat.get(("ARRIVE_OFFICE", "MILD"), [])) or rng.choice(routine_rows_ao)
            tag = "ARRIVE_MILD (phone off-routine)"
        else:
            r = rng.choice(routine_rows_ao)
            tag = "ARRIVE_ROUTINE"
        schedule.append({"day": day, "hour": 8, "row": r, "tag": tag, "scenario_id": "ARRIVE_OFFICE"})

        # HOME_EVENING event
        if day == 5:       # day 5: home_evening_noisy (STRONG)
            r = rng.choice(by_cat.get(("HOME_EVENING", "STRONG"), [])) or rng.choice(routine_rows_he)
            tag = "HOME_STRONG (noisy/dark/lying)"
        elif day == 9:     # day 9: MODERATE
            r_pool = by_cat.get(("HOME_EVENING", "MODERATE"), [])
            r = rng.choice(r_pool) if r_pool else rng.choice(routine_rows_he)
            tag = "HOME_MODERATE"
        elif day == 12:    # day 12: MILD
            r_pool = by_cat.get(("HOME_EVENING", "MILD"), [])
            r = rng.choice(r_pool) if r_pool else rng.choice(routine_rows_he)
            tag = "HOME_MILD"
        else:
            r = rng.choice(routine_rows_he)
            tag = "HOME_ROUTINE"
        schedule.append({"day": day, "hour": 19, "row": r, "tag": tag, "scenario_id": "HOME_EVENING"})

    # Load model
    import torch
    device = torch.device("cpu")
    ck = torch.load(ART_DIR / "triplet_vae.pt", map_location=device, weights_only=False)
    cfg = TripletVAEConfig(**ck["config"])
    model = TripletVAE(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    # centroid + thresholds using saved scores.json
    s = json.loads((ART_DIR / "scores.json").read_text())
    rc50, rc95 = s["routine_train_recon_p50"], s["routine_train_recon_p95"]
    dc50, dc95 = s["routine_train_dist_p50"], s["routine_train_dist_p95"]

    # also need centroid - we'll just re-load labeled routine_train and average
    splits = json.loads((ART_DIR / "splits.json").read_text())
    labeled_rows = labeled
    x_lab = encode_batch(labeled_rows).to(device)
    loss_lab, mu_lab = recon_and_mu(model, x_lab)
    rt = torch.tensor(splits["routine_train"], dtype=torch.long)
    centroid = mu_lab[rt].mean(dim=0, keepdim=True)

    times = []
    novelties = []
    triggered_surprise = []
    triggered_always = [True] * len(schedule)
    last_trigger_idx = -999
    for idx, event in enumerate(schedule):
        r = event["row"]
        x = encode_batch([r]).to(device)
        rec, mu = recon_and_mu(model, x)
        dist = torch.norm(mu - centroid, dim=1).item()
        nov = novelty(float(rec.item()), float(dist), rc50, rc95, dc50, dc95)
        novelties.append(nov)
        surp = nov > NOVELTY_THRESHOLD and (idx - last_trigger_idx) * COOLDOWN_STEP >= COOLDOWN_HOURS
        if surp:
            last_trigger_idx = idx
        triggered_surprise.append(bool(surp))
        times.append(event["day"] + (event["hour"] / 24.0))

    # Plot
    fig, ax = plt.subplots(figsize=(14, 5))
    colors = ["green" if t else "gray" for t in triggered_surprise]
    ax.scatter(times, novelties, c=colors, s=80, zorder=3)
    ax.axhline(NOVELTY_THRESHOLD, linestyle="--", color="red", alpha=0.5, label=f"threshold={NOVELTY_THRESHOLD}")
    ax.set_xlabel("day of 14-day simulation")
    ax.set_ylabel("novelty score")
    ax.set_title("Surprise-trigger on simulated 2 weeks of Alex's data")
    ax.legend()
    PLOTS = ART_DIR / "plots"
    PLOTS.mkdir(exist_ok=True)
    fig.tight_layout()
    fig.savefig(PLOTS / "trigger_timeline.png", dpi=120)
    plt.close(fig)


COOLDOWN_HOURS = 4
COOLDOWN_STEP = 12  # hours between schedule events


if __name__ == "__main__":
    main()
