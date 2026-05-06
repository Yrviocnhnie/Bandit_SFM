"""Layer 5 — model + checkpoint round-trip + listwise-loss-grouping tests."""
from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[3]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from lib.bg_multi.helpers import GLOBAL_ANCHOR_OFFSET


def _load_grid():
    spec = importlib.util.spec_from_file_location(
        "grid_for_test_model_io", ROOT / "scripts" / "36_train_c3_grid.py",
    )
    m = importlib.util.module_from_spec(spec)
    sys.modules["grid_for_test_model_io"] = m
    spec.loader.exec_module(m)
    return m


def test_make_baseline_model_vocab_size():
    """app_emb table should have V_pool rows (large enough for the multi-user vocab)."""
    grid = _load_grid()
    model = grid.make_baseline_model(num_features=33, vocab_size=243)
    state = model.state_dict()
    # Find any embedding param shape
    emb_keys = [k for k in state if "app_emb" in k.lower() or "embedding" in k.lower()]
    if not emb_keys:
        pytest.skip("no app_emb key found in state_dict")
    found_v = False
    for k in emb_keys:
        if state[k].shape[0] == 243:
            found_v = True
    assert found_v, f"no embedding has shape[0] == 243; got: {[(k, state[k].shape) for k in emb_keys]}"


def test_checkpoint_roundtrip(tmp_path):
    """Saving + loading a tiny model produces identical outputs."""
    grid = _load_grid()
    model_a = grid.make_baseline_model(num_features=33, vocab_size=243)
    model_a.eval()
    # Random input
    n = 16
    a = torch.randint(0, 243, (n,))
    f = torch.randn(n, 33)
    out_before = model_a(a, f).detach().cpu().numpy()
    p = tmp_path / "model.pt"
    torch.save({"state_dict": model_a.state_dict()}, p)
    # Reload into a fresh model
    ckpt = torch.load(p, map_location="cpu", weights_only=False)
    model_b = grid.make_baseline_model(num_features=33, vocab_size=243)
    model_b.load_state_dict(ckpt["state_dict"], strict=False)
    model_b.eval()
    out_after = model_b(a, f).detach().cpu().numpy()
    np.testing.assert_allclose(out_after, out_before, rtol=1e-6, atol=1e-6)


def test_listwise_loss_groups_by_global_anchor():
    """Two users with the same per-user anchor_id=0 must produce different
    listwise-loss groups when global anchor IDs are used.
    """
    grid = _load_grid()
    # 4 rows from 2 users, both with per-user anchor_id=0
    logits = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    y = torch.tensor([0, 1, 0, 1], dtype=torch.float32)
    # Per-user anchor_id (as if not made global) — collision
    per_user_aid = torch.tensor([0, 0, 0, 0], dtype=torch.long)
    # Global anchor_id (collision-free)
    uid_idx = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    global_aid = (uid_idx * GLOBAL_ANCHOR_OFFSET + per_user_aid).long()

    loss_collide = grid.listwise_softmax_nll(logits, y, per_user_aid)
    loss_global = grid.listwise_softmax_nll(logits, y, global_aid)
    # The losses should be different — collision merges 2 users' apps into one softmax,
    # global splits them into 2 separate groups.
    assert float(loss_collide.item()) != float(loss_global.item()), (
        f"loss_collide={float(loss_collide):.4f} == loss_global={float(loss_global):.4f}; "
        "expected different values"
    )


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
