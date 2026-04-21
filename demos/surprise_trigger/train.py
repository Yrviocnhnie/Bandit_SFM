"""Train a Triplet-VAE on labeled demo data (ARRIVE_OFFICE + HOME_EVENING).

Inputs:  data/demo_two_scenarios_labeled.jsonl
Outputs: artifacts/triplet_vae.pt
         artifacts/training_log.json
         artifacts/splits.json
"""
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch

from feature_encoder import ENCODING, encode_features
from model import (
    TripletVAE,
    TripletVAEConfig,
    kl_divergence,
    mixed_recon_loss,
    triplet_loss,
)


DEMO_DIR = Path(__file__).parent
INPUT_JSONL = DEMO_DIR / "data" / "demo_two_scenarios_labeled.jsonl"
ART_DIR = DEMO_DIR / "artifacts"


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def load_rows():
    with INPUT_JSONL.open() as f:
        return [json.loads(line) for line in f]


def encode_rows(rows):
    arr = np.stack([encode_features(r["features"]) for r in rows], axis=0)
    return torch.from_numpy(arr).float()


def split_indices(rows, seed, n_routine_held, n_nonroutine_held):
    rng = random.Random(seed)
    routine = [i for i, r in enumerate(rows) if r.get("is_routine")]
    nonroutine = [i for i, r in enumerate(rows) if not r.get("is_routine")]
    rng.shuffle(routine)
    rng.shuffle(nonroutine)
    return {
        "routine_held": routine[:n_routine_held],
        "routine_train": routine[n_routine_held:],
        "nonroutine_held": nonroutine[:n_nonroutine_held],
        "nonroutine_train": nonroutine[n_nonroutine_held:],
    }


def sample_triplet_idx(routine_pool, nonroutine_pool, batch_size, rng):
    a = [rng.choice(routine_pool) for _ in range(batch_size)]
    p = []
    for ai in a:
        while True:
            pi = rng.choice(routine_pool)
            if pi != ai:
                p.append(pi)
                break
    n = [rng.choice(nonroutine_pool) for _ in range(batch_size)]
    return a, p, n


def anneal(start, end, step, total):
    if total <= 0:
        return end
    if step >= total:
        return end
    t = step / total
    return start + (end - start) * t


@torch.no_grad()
def eval_separation(model, x_all, routine_held, nonroutine_held, routine_train, device):
    model.eval()
    xr_h = x_all[routine_held].to(device)
    xn_h = x_all[nonroutine_held].to(device)
    xr_t = x_all[routine_train].to(device)
    mu_r_h, _ = model.encode(xr_h)
    mu_n_h, _ = model.encode(xn_h)
    mu_r_t, _ = model.encode(xr_t)
    centroid = mu_r_t.mean(dim=0, keepdim=True)
    d_rh = torch.norm(mu_r_h - centroid, dim=1).mean().item()
    d_nh = torch.norm(mu_n_h - centroid, dim=1).mean().item()
    ratio = d_nh / max(1e-6, d_rh)
    return {
        "d_routine_held_to_centroid": d_rh,
        "d_nonroutine_held_to_centroid": d_nh,
        "ratio_non_over_rout": ratio,
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=60)
    p.add_argument("--steps-per-epoch", type=int, default=80)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--latent-dim", type=int, default=32)
    p.add_argument("--beta-max", type=float, default=1.0)
    p.add_argument("--gamma-max", type=float, default=2.0)
    p.add_argument("--triplet-margin", type=float, default=1.0)
    p.add_argument("--n-routine-held", type=int, default=80)
    p.add_argument("--n-nonroutine-held", type=int, default=500)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--device", type=str, default="cpu")
    args = p.parse_args()

    set_seed(args.seed)
    ART_DIR = Path(__file__).parent / "artifacts"
    ART_DIR.mkdir(parents=True, exist_ok=True)

    rows = load_rows()
    x = encode_rows(rows)
    print(f"Loaded {len(rows)} rows. Feature dim = {x.shape[1]}")

    splits = _split(rows, args.seed, args.n_routine_held, args.n_nonroutine_held)
    print(
        "splits:",
        {k: len(v) for k, v in splits.items()},
    )
    if len(splits["routine_train"]) < 2:
        raise RuntimeError("not enough routine samples for training")

    device = torch.device(args.device)
    cfg = TripletVAEConfig(
        input_dim=ENCODING.total_dim,
        latent_dim=args.latent_dim,
        triplet_margin=args.triplet_margin,
    )
    model = TripletVAE(cfg).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr)

    rng = random.Random(args.seed + 17)
    log = []

    beta_warm = max(1, int(args.epochs * 0.2))
    gamma_warm_start = max(1, int(args.epochs * 0.3))
    gamma_warm_end = max(1, int(args.epochs * 0.7))

    routine_train = splits["routine_train"]
    nonroutine_train = splits["nonroutine_train"]

    for epoch in range(args.epochs):
        beta = anneal(0.0, args.beta_max, epoch, beta_warm)
        if epoch < gamma_warm_start:
            gamma = 0.0
        else:
            gamma = anneal(
                0.0,
                args.gamma_max,
                epoch - gamma_warm_start,
                gamma_warm_end - gamma_warm_start,
            )

        model.train()
        epoch_losses = {"recon": 0.0, "kl": 0.0, "triplet": 0.0, "total": 0.0}

        for step in range(args.steps_per_epoch):
            a_i, p_i, n_i = sample_triplet_idx(
                routine_train, nonroutine_train, args.batch_size, rng
            )
            x_batch = torch.cat(
                [x[a_i], x[p_i], x[n_i]], dim=0
            ).to(device)
            logits, mu, logvar, _ = model(x_batch)
            recon = mixed_recon_loss(logits, x_batch)
            kl = kl_divergence(mu, logvar)
            B = args.batch_size
            mu_a = mu[:B]
            mu_p = mu[B : 2 * B]
            mu_n = mu[2 * B : 3 * B]
            trip = triplet_loss(mu_a, mu_p, mu_n, args.triplet_margin)
            loss = recon + beta * kl + gamma * trip

            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            opt.step()

            epoch_loss_recon = recon.item()
            epoch_loss_kl = kl.item()
            epoch_loss_trip = trip.item()
            epoch_losses["recon"] += epoch_loss_recon
            epoch_losses["kl"] += epoch_loss_kl
            epoch_losses["triplet"] += epoch_loss_trip
            epoch_losses["total"] = (
                epoch_losses.get("total", 0.0) + loss.item()
            )

        # evaluate
        sep = eval_separation(
            model,
            x,
            splits["routine_held"],
            splits["nonroutine_held"],
            splits["routine_train"],
            device,
        )
        row = {
            "epoch": epoch,
            "beta": beta,
            "gamma": gamma,
            "recon": epoch_losses["recon"] / args.steps_per_epoch,
            "kl": epoch_losses["kl"] / args.steps_per_epoch,
            "triplet": epoch_losses["triplet"] / args.steps_per_epoch,
            **separation_to_dict(sep),
        }
        log.append(row)
        print(
            f"ep {epoch:3d} beta={beta:.3f} gamma={gamma:.3f} "
            f"recon={row['recon']:.3f} kl={row['kl']:.3f} trip={row['triplet']:.3f} "
            f"d_R={row['d_routine_held_to_centroid']:.3f} "
            f"d_N={row['d_nonroutine_held_to_centroid']:.3f} "
            f"ratio={row['ratio_non_over_rout']:.2f}"
        )

    ART_DIR.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), "config": vars(cfg)},
               ART_DIR / "triplet_vae.pt")
    (ART_DIR / "training_log.json").write_text(
        json.dumps({"args": vars(args), "per_epoch": log}, indent=2)
    )
    (ART_DIR / "splits.json").write_text(json.dumps(splits))
    print("saved artifacts to", ART_DIR)


def _split(rows, seed, n_routine_held, n_nonroutine_held):
    return split_indices(rows, seed, n_routine_held, n_nonroutine_held)


def separation_to_dict(s):
    return s


if __name__ == "__main__":
    main()
