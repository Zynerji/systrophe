"""Tests for the morphic-resonance falsification harness.

The harness is only trustworthy if it has both POWER (fires on a true count
effect / true acausal effect) and SPECIFICITY (stays silent on independent
learners and on pure time-trends, and honestly reports unidentifiability).
These tests pin all of that.
"""

import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import morphic_catcher as mc
from morphic_catcher.ctc import solve_ctc_field, ctc_iteration_count


# --------------------------------------------------------------------------- #
# Single-panel harness: specificity
# --------------------------------------------------------------------------- #

def test_independent_learners_no_structure():
    panel = mc.independent_learners(seed=0)
    assert mc.falsify(panel, n_surrogates=120, seed=1).verdict == "no_structure"


def test_secular_trend_is_conventional_not_morphic():
    panel = mc.secular_trend(seed=0)
    v = mc.falsify(panel, n_surrogates=120, seed=1)
    assert v.verdict == "conventional_trend"
    # a pure time-trend must NOT be misread as a count effect
    assert abs(v.identifiability["time_t"]) > abs(v.identifiability["count_t"])


# --------------------------------------------------------------------------- #
# Single-panel harness: power
# --------------------------------------------------------------------------- #

def test_morphic_field_fires():
    panel = mc.morphic_field(seed=0)
    v = mc.falsify(panel, n_surrogates=120, seed=1)
    assert v.verdict == "morphic_signature"
    assert v.count_null["p_value"] < 0.05


def test_morphic_power_majority_over_seeds():
    hits = sum(mc.falsify(mc.morphic_field(seed=s), n_surrogates=120,
                          seed=100 + s).verdict == "morphic_signature"
               for s in range(8))
    assert hits >= 6  # robust majority


# --------------------------------------------------------------------------- #
# Identifiability boundary (the rat-maze confound)
# --------------------------------------------------------------------------- #

def test_uniform_schedule_collapses_to_unidentifiable():
    """The SAME true count effect becomes unidentifiable under uniform
    instantiation (count collinear with time)."""
    panel = mc.morphic_field(schedule="uniform", seed=0)
    v = mc.falsify(panel, n_surrogates=120, seed=1)
    assert v.verdict == "unidentifiable"
    assert v.identifiability["count_time_vif"] >= 10.0


def test_bursty_schedule_is_identifiable():
    panel = mc.morphic_field(schedule="bursty", seed=0)
    v = mc.falsify(panel, n_surrogates=120, seed=1)
    assert v.identifiability["count_time_vif"] < 10.0


# --------------------------------------------------------------------------- #
# Non-locality negative (Concept C)
# --------------------------------------------------------------------------- #

def test_morphic_indistinguishable_from_diffusion():
    """Honest negative: local diffusion and morphic coupling land on the same
    verdict -- counts alone cannot establish non-locality."""
    m = mc.falsify(mc.morphic_field(seed=0), n_surrogates=120, seed=1).verdict
    d = mc.falsify(mc.local_diffusion(seed=0), n_surrogates=120, seed=1).verdict
    assert m == d == "morphic_signature"


# --------------------------------------------------------------------------- #
# CTC-resonance model + across-forms acausality (Concept B)
# --------------------------------------------------------------------------- #

def test_ctc_field_reduces_to_causal_at_zero_acausality():
    f = solve_ctc_field(n=80, acausal_fraction=0.0)
    assert np.all(np.diff(f) >= -1e-9)          # monotone (purely causal)
    assert f[0] == pytest.approx(0.0, abs=1e-9)  # no early lift


def test_ctc_iteration_count_matches_spectral_oracle():
    """The fixed-point iteration count should track the spectral-gap
    prediction (Systrophe D-CTC oracle), within a small factor."""
    d = ctc_iteration_count(n=80, acausal_fraction=0.5, gamma=0.25)
    assert d["iterations"] <= d["predicted_iterations"] + 5
    assert d["contraction_rate"] < 1.0


def test_causal_forms_show_no_acausality():
    forms = mc.multiform_forms(mechanism="causal", seed=0)
    av = mc.falsify_acausal(forms, n_surrogates=120, seed=1)
    assert av.verdict == "no_acausality"


def test_ctc_forms_show_acausal_signature():
    forms = mc.multiform_forms(mechanism="ctc", acausal_fraction=0.6, seed=0)
    av = mc.falsify_acausal(forms, n_surrogates=120, seed=1)
    assert av.verdict == "acausal_signature"
    assert av.null["p_value"] < 0.05
    assert av.across_forms["eventual_total_t"] < -3.0


def test_acausal_strength_increases_with_acausal_fraction():
    ts = []
    for a in (0.2, 0.5, 0.9):
        forms = mc.multiform_forms(mechanism="ctc", acausal_fraction=a, seed=0)
        ts.append(abs(mc.acausal_across_forms(forms)["eventual_total_t"]))
    assert ts[0] < ts[1] < ts[2]


# --------------------------------------------------------------------------- #
# Single-panel acausality is honestly disabled
# --------------------------------------------------------------------------- #

def test_single_panel_acausal_not_used_in_verdict():
    """falsify() must not emit an acausal verdict from a single panel."""
    for gen in (mc.independent_learners, mc.secular_trend, mc.morphic_field):
        v = mc.falsify(gen(seed=0), n_surrogates=80, seed=1)
        assert v.verdict in {"no_structure", "conventional_trend",
                             "morphic_signature", "unidentifiable"}
        assert v.acausal_null is None


# --------------------------------------------------------------------------- #
# Catcher integration (Systrophe always-on rule)
# --------------------------------------------------------------------------- #

def test_catcher_runs_and_reports_verdict():
    res = mc.catcher_verdict(mc.morphic_field(seed=0))
    assert res["verdict"] in {"novel_structure", "smooth", "uniform"}
    assert "lambda_2_at_radius" in res
