"""adaptive-qec smoke tests (scaffold only)."""

from __future__ import annotations

import pathlib
import sys

import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from adaptive_qec import gating_threshold_default
from adaptive_qec.gating import AnomalyGatedDecoder


def test_package_imports():
    assert callable(gating_threshold_default)


def test_default_threshold():
    assert gating_threshold_default() == 0.5


def test_decoder_constructs():
    d = AnomalyGatedDecoder(catcher_window=32, gate_threshold=0.4)
    assert d.catcher_window == 32
    assert d.gate_threshold == 0.4


def test_decoder_not_implemented():
    """Scaffold: decode() must raise NotImplementedError."""
    d = AnomalyGatedDecoder()
    with pytest.raises(NotImplementedError):
        d.decode(syndrome=None)
