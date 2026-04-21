"""Produce plots from artifacts/scores.json and the trained model."""
from __future__ import annotations

import json
import random
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.decomposition import PCA

from feature_encoder import ENCODING, encode_features
from model import TripletVAE, TripletVAEConfig


DEMO = Path(__file__).parent
ART = DEMO / "artifacts"
PLOTS = ART / "plots"

TIER_ORDER = ["ROUTINE_HELD", "ROUTINE", "MILD", "MODERATE", "STRONG", "EXTREME", "NONROUTINE_HELD"]
TIER_COLORS = {
    "ROUTINE_HELD": "green",
    "ROUTINE":      "limegreen",
    "MILD":         "gold",
    "MODERATE":     "orange",
    "STRONG":       "red",
    "EXTREME":      "darkred",
    "NONROUTINE_HELD": "grey",
}


def main():
    PLOTS.mkdir(parents=True, exist_ok=True)
    s = json.loads((ART / "scores.json").read_text())
    records = s["records"]

    plot_boxplot(records)
    plot_histograms(records)
    plot_pca(records)


def plot_boxplot(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["category"]].append(r["novelty"])
    cats = [c for c in TIER_ORDER if c in groups]
    data = [groups[c] for c in cats]
    fig, ax = plt.subplots(figsize=(10, 5))
    bp = ax.boxplot(data, tick_labels=cats, showfliers=True, patch_artist=True)
    for patch, cat in zip(bp["boxes"], cats):
        patch.set_facecolor(TIER_COLORS[cat])
        patch.set_alpha(0.6)
    ax.axhline(1.0, color="k", linestyle="--", alpha=0.5, label="threshold=1.0")
    ax.set_ylabel("novelty score")
    ax.set_title("Novelty score by tier (Alex, ARRIVE_OFFICE + HOME_EVENING)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "tier_boxplot.png", dpi=140)
    plt.close(fig)
    print("wrote", PLOTS / "tier_boxplot.png")


def plot_histograms(records):
    groups = defaultdict(list)
    for r in records:
        groups[r["category"]].append(r["novelty"])
    fig, ax = plt.subplots(figsize=(10, 5))
    for c in TIER_ORDER:
        vs = groups.get(c, [])
        if not vs:
            continue
        ax.hist(vs, bins=25, alpha=0.55, color=TIER_COLORS.get(c, "blue"), label=c)
    ax.set_xlabel("novelty score")
    ax.set_ylabel("count")
    ax.set_title("Novelty score distribution per tier")
    ax.axvline(1.0, color="k", linestyle="--", alpha=0.5)
    ax.legend()
    fig.tight_layout()
    fig.savefig(PLOTS / "novelty_histogram.png", dpi=140)
    plt.close(fig)
    print("wrote", PLOTS / "novelty_histogram.png")


def plot_pca(records):
    # Reload model + encode for latent
    device = torch.device("cpu")
    ck = torch.load(ART / "triplet_vae.pt", map_location=device, weights_only=False)
    cfg = TripletVAEConfig(**ck["config"])
    model = TripletVAE(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    from feature_encoder import encode_features as _enc  # local alias
    vecs, cats = [], []
    for r in records:
        if "features" not in r:  # NONROUTINE_HELD rows lack features
            continue
        vecs.append(_enc(r["features"]))
        cats.append(r["category"])
    x = torch.from_numpy(np.stack(vecs, axis=0)).float().to(device)
    with torch.no_grad():
        mu, _ = model.encode(x)
    mu_np = mu.cpu().numpy()
    pca = PCA(n_components=2)
    xy = pca.fit_transform(mu_np)

    fig, ax = plt.subplots(figsize=(8, 7))
    for c in set(cats):
        pts = np.array([xy[i] for i in range(len(cats)) if cats[i] == c])
        if pts.size == 0:
            continue
        ax.scatter(pts[:, 0], pts[:, 1], s=36, color=TIER_COLORS.get(c, "blue"),
                   label=c, alpha=0.75, edgecolors="white", linewidths=0.4)
    ax.set_xlabel("PC 1")
    ax.set_ylabel("PC 2")
    ax.set_title("Latent space (PCA of mu) colored by novelty tier")
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(PLOTS / "latent_pca.png", dpi=140)
    plt.close(fig)
    print("wrote", PLOTS / "latent_pca.png")


def plot_histograms_and_box():
    pass


def plot_trigger_comparison(records):
    pass


if __name__ == "__main__":
    main()
