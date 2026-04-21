"""Score labeled + tiered test samples with the trained Triplet-VAE.

Saves:
  artifacts/scores.json
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from feature_encoder import ENCODING, encode_features
from model import TripletVAE, TripletVAEConfig

DEMO = Path(__file__).parent
ART = DEMO / "artifacts"


def encode_batch(rows):
    arr = np.stack([encode_features(r["features"]) for r in rows], axis=0)
    return torch.from_numpy(arr).float()


@torch.no_grad()
def recon_and_mu(model, x):
    mu, _ = model.encode(x)
    logits = model.decode(mu)
    loss = torch.zeros(x.shape[0], device=x.device)
    for g in ENCODING.groups:
        lb = logits[:, g.offset : g.offset + g.size]
        tb = x[:, g.offset : g.offset + g.size]
        if g.kind == "onehot":
            loss = loss + F.cross_entropy(lb, tb.argmax(dim=1), reduction="none")
        elif g.kind == "binary":
            loss = loss + F.binary_cross_entropy_with_logits(
                lb.squeeze(-1), tb.squeeze(-1), reduction="none"
            )
        elif g.kind == "scalar":
            loss = loss + ((torch.sigmoid(lb) - tb) ** 2).sum(dim=1)
    return loss, mu


def novelty(rec, dist, rc50, rc95, dc50, dc95):
    rn = max(0.0, (rec - rc50) / (rc95 - rc50 + 1e-6))
    dn = max(0.0, (dist - dc50) / (dc95 - dc50 + 1e-6))
    return 0.5 * rn + 0.5 * dn


def main():
    device = torch.device("cpu")

    ck = torch.load(ART / "triplet_vae.pt", map_location=device, weights_only=False)
    cfg = TripletVAEConfig(**ck["config"])
    model = TripletVAE(cfg).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()

    labeled = [json.loads(l) for l in (ART.parent / "data" / "demo_two_scenarios_labeled.jsonl").open()]
    splits = json.loads((ART / "splits.json").read_text())
    tests = [json.loads(l) for l in (ART.parent / "data" / "test_samples.jsonl").open()]

    x_lab = encode_batch(labeled).to(device)
    x_test = encode_batch(tests).to(device)

    recon_lab, mu_lab = recon_and_mu(model, x_lab)
    recon_test, mu_test = recon_and_mu(model, x_test)

    rt = torch.tensor(splits["routine_train"], dtype=torch.long)
    rh = torch.tensor(splits["routine_held"], dtype=torch.long)
    nh = torch.tensor(splits["nonroutine_held"], dtype=torch.long)

    centroid = mu_lab[rt].mean(dim=0, keepdim=True)
    dist_lab = torch.norm(mu_lab - centroid, dim=1)
    dist_test = torch.norm(mu_test - centroid, dim=1)

    rc = recon_lab_cpu = recon_lab[rt].cpu().numpy()
    dc = dist_lab[rt].cpu().numpy()
    rc50 = float(np.percentile(rc, 50))
    rc95 = float(np.percentile(rc, 95))
    dc50 = float(np.percentile(dc, 50))
    dc95 = float(np.percentile(dc, 95))

    records = []
    for i in splits["routine_held"]:
        r = labeled[i]
        rec = float(recon_lab[i].item())
        dis = float(dist_lab[i].item())
        records.append({
            "category": "ROUTINE_HELD",
            "scenario_id": r["scenario_id"],
            "recon_loss": rec,
            "latent_dist": dis,
            "novelty": novelty(rec, dis, rc50, rc95, dc50, dc95),
            "features": r["features"],
        })
    for i in splits["nonroutine_held"]:
        r = labeled[i]
        rec = float(recon_lab[i].item())
        dis = float(dist_lab[i].item())
        records.append({
            "category": "NONROUTINE_HELD",
            "scenario_id": r["scenario_id"],
            "recon_loss": rec,
            "latent_dist": float(dis),
            "novelty": novelty(rec, dis, rc50, rc95, dc50, dc95),
        })
    for i, r in enumerate(tests):
        records_rec = float(recon_test[i].item())
        records_dist = float(dist_test[i].item())
        records.append({
            "category": r["tier"],
            "scenario_id": r["scenario_id"],
            "recon_loss": records_rec,
            "latent_dist": records_dist,
            "novelty": novelty(records_rec, records_dist, rc50, rc95, dc50, dc95),
            "features": r["features"],
        })

    ART.mkdir(parents=True, exist_ok=True)
    with (ART / "scores.json").open("w") as f:
        json.dump({
            "routine_train_recon_p50": rc50, "routine_train_recon_p95": rc95,
            "routine_train_dist_p50": dc50, "routine_train_dist_p95": dc95,
            "records": records,
        }, f, indent=2)
    print("wrote scores.json")


if __name__ == "__main__":
    main()
