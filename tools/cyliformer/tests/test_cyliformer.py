"""Tests for the Cyliformer derived tool."""

from __future__ import annotations

import math
import pathlib
import sys

import pytest
import torch

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from cyliformer import (
    CylinderBlock,
    Cyliformer,
    LearnedAddressCatcher,
    SelectiveKVCache,
    TorsionalResonanceLoss,
    cyliformer_loss,
)
from cyliformer.block import CylinderBlockConfig
from cyliformer.model import CyliformerConfig


# ---------------------------------------------------------------------------
# LearnedAddressCatcher
# ---------------------------------------------------------------------------


def test_catcher_returns_scalar_tensor():
    torch.manual_seed(0)
    catcher = LearnedAddressCatcher(d_model=32, n_bits=24, max_nodes=64)
    x = torch.randn(2, 8, 32)
    val = catcher(x)
    assert val.dim() == 0
    assert float(val.item()) >= 0.0


def test_catcher_is_differentiable():
    """Catcher contributes to gradients on its projection weights."""
    torch.manual_seed(0)
    catcher = LearnedAddressCatcher(d_model=16, n_bits=16, max_nodes=32)
    x = torch.randn(2, 8, 16, requires_grad=True)
    val = catcher(x)
    val.backward()
    assert catcher.proj.weight.grad is not None
    assert float(catcher.proj.weight.grad.abs().sum().item()) > 0.0


def test_catcher_handles_tiny_input():
    """1 token total -> lambda_2 = 0 (graph has fewer than 2 nodes)."""
    catcher = LearnedAddressCatcher(d_model=8, n_bits=8, max_nodes=8)
    x = torch.zeros(1, 1, 8)
    val = catcher(x)
    assert float(val.item()) == 0.0


# ---------------------------------------------------------------------------
# CylinderBlock
# ---------------------------------------------------------------------------


def test_cylinder_block_preserves_shape():
    torch.manual_seed(0)
    block = CylinderBlock(d_model=32, n_heads=2, n_cylinders=3,
                              catcher_max_nodes=48)
    x = torch.randn(2, 8, 32)
    y = block(x)
    assert y.shape == x.shape


def test_cylinder_block_records_diagnostics():
    torch.manual_seed(0)
    block = CylinderBlock(d_model=32, n_heads=2, n_cylinders=3,
                              catcher_max_nodes=48)
    x = torch.randn(2, 8, 32)
    _ = block(x)
    assert len(block.last_lambda2_per_cylinder) == 3
    assert len(block.last_backreaction_per_cylinder) == 3
    for v in block.last_lambda2_per_cylinder:
        assert v >= 0.0
    for v in block.last_backreaction_per_cylinder:
        assert 0.0 <= v <= 1.0


def test_cylinder_block_phasor_rotation_preserves_norm():
    """The 2D rotation per channel-pair preserves vector norm."""
    torch.manual_seed(0)
    x = torch.randn(2, 8, 16)
    delta = torch.tensor(0.7)
    rotated = CylinderBlock._rotate_channel_pairs(x, delta)
    # Norm of each (real, imag) pair is preserved
    re = x[..., :8]
    im = x[..., 8:]
    re_r = rotated[..., :8]
    im_r = rotated[..., 8:]
    norm_in = (re ** 2 + im ** 2).sum().item()
    norm_out = (re_r ** 2 + im_r ** 2).sum().item()
    assert math.isclose(norm_in, norm_out, rel_tol=1e-5)


def test_cylinder_block_d_model_must_be_even():
    with pytest.raises(ValueError):
        CylinderBlock(d_model=17, n_heads=1, n_cylinders=2)


def test_cylinder_block_inference_prune_can_skip_all():
    """At inference with infer_prune=True, all cylinders may be pruned."""
    torch.manual_seed(0)
    block = CylinderBlock(
        d_model=32, n_heads=2, n_cylinders=3,
        catcher_max_nodes=48, prune_threshold=0.0,  # prune everything
    )
    block.eval()
    x = torch.randn(2, 8, 32)
    y = block(x, infer_prune=True)
    # Output should be the residual (attn out + zero combined)
    assert y.shape == x.shape
    assert any(block.last_pruned_mask)


def test_beam_gain_in_unit_range():
    block = CylinderBlock(d_model=32, n_heads=2, n_cylinders=4)
    gain = block.beam_gain()
    assert 0.0 <= gain <= 1.0 + 1e-9


# ---------------------------------------------------------------------------
# Cyliformer model
# ---------------------------------------------------------------------------


def test_cyliformer_logits_shape():
    torch.manual_seed(0)
    model = Cyliformer(vocab_size=12, d_model=32, n_layers=2,
                         n_heads=2, n_cylinders=3, max_seq_len=8,
                         catcher_max_nodes=48)
    x = torch.randint(0, 12, (3, 8))
    logits = model(x)
    assert logits.shape == (3, 8, 12)


def test_cyliformer_trains_one_step():
    """One backward pass updates LM head weights without crashing."""
    torch.manual_seed(0)
    model = Cyliformer(vocab_size=10, d_model=16, n_layers=2,
                         n_heads=2, n_cylinders=2, max_seq_len=8,
                         catcher_max_nodes=32)
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, 10, (2, 8))
    logits = model(x)
    loss = cyliformer_loss(logits, x, model.all_lambdas())
    loss.backward()
    optim.step()
    # LM head weight (tied to embedding) must have a gradient
    assert model.embed.weight.grad is not None
    assert float(model.embed.weight.grad.abs().sum().item()) > 0.0


def test_cyliformer_param_breakdown_keys():
    model = Cyliformer(vocab_size=10, d_model=16, n_layers=2,
                         n_heads=2, n_cylinders=2, max_seq_len=8)
    b = model.param_count_breakdown()
    for k in ("total", "shared_ffn", "attention", "catcher", "phasor",
                "embedding_and_lm_head"):
        assert k in b
    assert b["total"] > 0


def test_cyliformer_all_lambdas_shape():
    torch.manual_seed(0)
    model = Cyliformer(vocab_size=10, d_model=16, n_layers=3,
                         n_heads=2, n_cylinders=4, max_seq_len=8,
                         catcher_max_nodes=32)
    x = torch.randint(0, 10, (1, 8))
    _ = model(x)
    lams = model.all_lambdas()
    assert len(lams) == 3       # n_layers
    for layer_lams in lams:
        assert len(layer_lams) == 4  # n_cylinders


# ---------------------------------------------------------------------------
# Loss
# ---------------------------------------------------------------------------


def test_cyliformer_loss_returns_scalar():
    logits = torch.randn(2, 8, 10)
    labels = torch.randint(0, 10, (2, 8))
    lams = [[0.1, 0.2, 0.15], [0.3, 0.05, 0.25]]
    loss = cyliformer_loss(logits, labels, lams, lambda_target=0.20)
    assert loss.dim() == 0
    assert float(loss.item()) > 0.0


def test_torsional_resonance_loss_runs():
    torch.manual_seed(0)
    loss_fn = TorsionalResonanceLoss()
    logits = torch.randn(2, 8, 10)
    labels = torch.randint(0, 10, (2, 8))
    lams = [[0.1, 0.2], [0.3, 0.05]]
    beam_gains = [0.7, 0.9]
    l1 = loss_fn(logits, labels, lams, beam_gains)
    l2 = loss_fn(logits, labels, lams, beam_gains)  # second step exercises history
    assert float(l1.item()) > 0
    assert float(l2.item()) > 0


# ---------------------------------------------------------------------------
# SelectiveKVCache
# ---------------------------------------------------------------------------


def test_kvcache_update_and_get():
    cache = SelectiveKVCache(n_cylinders=2, backreaction_threshold=0.5)
    k0 = torch.randn(1, 4, 2, 8)
    v0 = torch.randn(1, 4, 2, 8)
    cache.update(0, k0, v0, backreaction=0.1)
    k_back, v_back = cache.get(0)
    assert torch.allclose(k_back, k0)
    assert torch.allclose(v_back, v0)


def test_kvcache_drops_high_backreaction():
    cache = SelectiveKVCache(n_cylinders=2, backreaction_threshold=0.5)
    k0 = torch.randn(1, 4, 2, 8)
    v0 = torch.randn(1, 4, 2, 8)
    cache.update(0, k0, v0, backreaction=0.9)  # above threshold -> dropped
    k_back, v_back = cache.get(0)
    assert k_back is None
    assert v_back is None


def test_kvcache_memory_footprint_grows():
    cache = SelectiveKVCache(n_cylinders=1)
    cache.update(0, torch.zeros(1, 4, 2, 8), torch.zeros(1, 4, 2, 8), backreaction=0.0)
    m0 = cache.memory_footprint_bytes()
    cache.update(0, torch.zeros(1, 4, 2, 8), torch.zeros(1, 4, 2, 8), backreaction=0.0)
    m1 = cache.memory_footprint_bytes()
    assert m1 > m0
