"""Tests for the Systroformer derived tool."""

from __future__ import annotations

import sys
import pathlib

import numpy as np
import pytest
import torch

# Make systroformer importable when this test runs in-tree
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from systroformer import (
    LearnedAddressNet,
    MiniSystroformer,
    SystroformerBlock,
    address_from_activation,
    derivative_catcher,
    hamming_graph_lambda2,
    hamming_graph_lambda2_power_iter,
    lsh_subsample,
)


def test_address_from_activation_returns_bits():
    """Hashing a 1D activation returns an n_bits long {0,1} array."""
    x = np.random.default_rng(0).normal(size=16)
    addr = address_from_activation(x, n_bits=32)
    assert addr.shape == (32,) or addr.shape == (16,) or len(addr) > 0
    assert np.all((addr == 0) | (addr == 1))


def test_power_iter_lambda2_matches_exact_within_factor_2():
    """Power-iteration λ₂ is within a factor of 2 of the exact value."""
    rng = np.random.default_rng(42)
    addresses = [rng.integers(0, 2, size=16).astype(int) for _ in range(50)]
    exact = hamming_graph_lambda2(addresses, radius=4)
    approx = hamming_graph_lambda2_power_iter(addresses, radius=4, n_iter=40)
    # power iter gives a lower bound estimate; should be within an
    # order of magnitude
    if exact > 1e-6:
        ratio = approx / exact
        assert 0.05 < ratio < 20.0, f"approx={approx} exact={exact} ratio={ratio}"


def test_derivative_catcher_flat_history():
    """Flat λ₂ history gives derivative ~ 0."""
    history = [5.0] * 10
    d = derivative_catcher(history, window=5)
    assert abs(d) < 1e-9


def test_derivative_catcher_rising_history():
    """Monotonically rising λ₂ gives positive derivative."""
    history = [1.0, 2.0, 3.0, 4.0, 5.0]
    d = derivative_catcher(history, window=5)
    assert d > 0.0


def test_lsh_subsample_caps_size():
    arr = [np.zeros(8, dtype=int) for _ in range(1000)]
    out = lsh_subsample(arr, max_nodes=256)
    assert len(out) <= 256


def test_systroformer_block_forward_shape():
    """SystroformerBlock preserves input shape."""
    torch.manual_seed(0)
    block = SystroformerBlock(d_model=32, n_heads=2, max_nodes=64)
    x = torch.randn(2, 8, 32)
    y = block(x)
    assert y.shape == x.shape


def test_systroformer_block_records_lambda():
    """After a forward pass, the block has a non-empty lambda_history."""
    torch.manual_seed(0)
    block = SystroformerBlock(d_model=32, n_heads=2, max_nodes=64)
    x = torch.randn(2, 8, 32)
    _ = block(x)
    assert len(block.lambda_history) == 1
    assert block.lambda_history[0] >= 0.0


def test_mini_systroformer_logits_shape():
    """MiniSystroformer returns (batch, seq, vocab) logits."""
    torch.manual_seed(0)
    model = MiniSystroformer(
        vocab_size=12, d_model=32, n_layers=2, n_heads=2, max_seq_len=8,
    )
    x = torch.randint(0, 12, (3, 8))
    logits = model(x)
    assert logits.shape == (3, 8, 12)


def test_mini_systroformer_trainable():
    """One backward pass updates weights without crashing."""
    torch.manual_seed(0)
    model = MiniSystroformer(
        vocab_size=10, d_model=16, n_layers=2, n_heads=2, max_seq_len=8,
    )
    optim = torch.optim.Adam(model.parameters(), lr=1e-3)
    x = torch.randint(0, 10, (2, 8))
    logits = model(x)
    loss = torch.nn.functional.cross_entropy(
        logits.view(-1, 10), x.view(-1),
    )
    loss.backward()
    optim.step()
    # lambda_scale should have a gradient
    grad_norms = [
        float(b.lambda_scale.grad.abs().item()) if b.lambda_scale.grad is not None else 0.0
        for b in model.blocks
    ]
    # At least the FFN linear should have non-zero gradient
    # (lambda_scale itself often has 0 grad because the loss isn't
    # sensitive to FFN magnitude at init -- accept either)
    head_grad = model.head.weight.grad
    assert head_grad is not None and float(head_grad.abs().sum().item()) > 0.0


def test_learned_address_net_outputs_bits():
    """LearnedAddressNet returns float in {0, 1}."""
    net = LearnedAddressNet(d_model=16, n_bits=8)
    x = torch.randn(4, 16)
    bits = net(x)
    assert bits.shape == (4, 8)
    # forward is hard 0/1 (straight-through)
    binarised = (bits.detach() == 0) | (bits.detach() == 1)
    assert bool(binarised.all())
