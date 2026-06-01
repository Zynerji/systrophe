"""Tests for the Quantum-Realizability Scorer (Ford-Roman QI budget axis).

Acceptance gates (regression anchors), all reported with ACTUAL numbers:

1. van Stockum dust scores < 1 (realizable; energy_conditions proves all
   four pointwise ECs hold -> positive-energy source -> T_kk_min = 0).
2. Alcubierre scores > 1 (QI-forbidden) at the macroscopic sampling time,
   and its geometrized exotic-energy principal reproduces the ~Jupiter-mass
   floor cited in the repo to within an order of magnitude.
3. The novelty-catcher / surrogate lambda_2 separation splits the
   classically-sourceable members (cylinder, Kerr, Goedel, Gott) from the
   exotic-requiring ones (Alcubierre, wormhole).
"""

from __future__ import annotations

import math
import pathlib
import sys

import numpy as np
import pytest

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from warp_drive_zoo import (
    AlcubierreDrive,
    BobrickMartireDrive,
    LentzDrive,
    QILambda2Separation,
    QIScore,
    WarpDriveQIComparison,
    alcubierre_jupiter_floor,
    compare_drives_with_qi,
    ford_roman_qi_bound,
    qi_lambda2_separation,
    qi_normalized_score,
    score_alcubierre,
    score_full_registry,
    score_godel,
    score_gott,
    score_kerr,
    score_van_stockum,
    score_wormhole_throat,
)
from warp_drive_zoo.qi_scorer import C_FORD_ROMAN

# Repo-cited Jupiter-mass floor anchors (knopp_ratchet / warp_geometry):
#   - per-bubble Alcubierre principal (1 m, luminal, kappa=1/12) ~0.059 M_J
#   - multi-stroke ledger floor (feasibility_report)             ~3.78 M_J
JUPITER_FLOOR_LOW = 0.059123240318761365
JUPITER_FLOOR_HIGH = 3.7838873804007274


# ---------------------------------------------------------------------------
# Ford-Roman QI bound: generalisation off the van-Stockum hardcoding
# ---------------------------------------------------------------------------


def test_ford_roman_bound_is_negative_and_scales_as_tau_minus4():
    """rho_sampled >= -3/(32 pi^2 tau^4); halving tau multiplies |bound| by 16."""
    b1 = ford_roman_qi_bound(1.0)
    b_half = ford_roman_qi_bound(0.5)
    assert b1 < 0
    assert b_half < 0
    assert b_half / b1 == pytest.approx(16.0, rel=1e-9)


def test_ford_roman_recovers_vanstockum_module_constant():
    """ford_roman_qi_bound(tau=window) matches anec_quantum_inequality_bound.

    The hardcoded van-Stockum QI bound is -C/(r_max - r_min)^4 with the same
    constant C = 3/(32 pi^2); passing tau = (r_max - r_min) reproduces it.
    """
    from systrophe.geometry.vanstockum import VanStockumInterior
    from systrophe.qftcs.anec_bound import anec_quantum_inequality_bound

    vs = VanStockumInterior(omega=0.8, R=1.0)
    r_min, r_max = 1.01, 10.0
    hardcoded = anec_quantum_inequality_bound(vs, r_min, r_max)
    generalized = ford_roman_qi_bound(r_max - r_min)
    assert generalized == pytest.approx(hardcoded, rel=1e-12)
    assert C_FORD_ROMAN == pytest.approx(3.0 / (32.0 * math.pi ** 2), rel=1e-12)


def test_qi_score_zero_for_nonnegative_source():
    """A non-negative T_kk source scores exactly 0 (realizable)."""
    assert qi_normalized_score(0.0, tau=1.0) == 0.0
    assert qi_normalized_score(+5.0, tau=1.0) == 0.0


# ---------------------------------------------------------------------------
# GATE 1: van Stockum dust is realizable (score < 1)
# ---------------------------------------------------------------------------


def test_gate_van_stockum_realizable():
    """van Stockum dust scores < 1 — its source satisfies all four ECs."""
    s = score_van_stockum(omega=0.8, R=1.0)
    assert s.t_kk_min == 0.0
    assert s.qi_normalized_score < 1.0
    assert s.qi_forbidden is False
    assert s.source_kind == "dust"


def test_van_stockum_score_matches_energy_conditions():
    """Cross-check: the dust density is non-negative everywhere (NEC holds)."""
    from systrophe.geometry.vanstockum import VanStockumInterior
    from systrophe.qftcs.energy_conditions import energy_condition_report

    vs = VanStockumInterior(omega=0.8, R=1.0)
    rep = energy_condition_report(vs)
    assert rep.nec_holds and rep.wec_holds and rep.sec_holds and rep.dec_holds
    s = score_van_stockum(vs=vs)
    assert s.qi_normalized_score == 0.0


# ---------------------------------------------------------------------------
# GATE 2: Alcubierre is QI-forbidden (score > 1) + Jupiter-mass floor
# ---------------------------------------------------------------------------


def test_gate_alcubierre_qi_forbidden():
    """Alcubierre scores > 1 at the macroscopic sampling time (QI-forbidden)."""
    s = score_alcubierre(v_s=1.0, R=1.0, sigma=8.0)
    assert s.t_kk_min < 0.0
    assert s.qi_normalized_score > 1.0
    assert s.qi_forbidden is True
    assert s.source_kind == "exotic"


def test_gate_alcubierre_reproduces_jupiter_mass_floor():
    """Alcubierre exotic energy reproduces the repo Jupiter-mass floor (1 OOM).

    The geometrized principal routed through c^4/G must land within an order
    of magnitude of the repo-cited floor (per-bubble ~0.06 M_J, multi-stroke
    ledger ~3.8 M_J).
    """
    fl = alcubierre_jupiter_floor(R_m=1.0, sigma_m=1.0, v_s_over_c=1.0,
                                  kappa=1.0 / 12.0)
    # Reproduces the per-bubble principal anchor exactly.
    assert fl.jupiter_masses == pytest.approx(JUPITER_FLOOR_LOW, rel=1e-6)
    # Within one order of magnitude of "a Jupiter mass".
    assert 0.1 <= JUPITER_FLOOR_HIGH / fl.jupiter_masses <= 100.0
    assert 1e-2 <= fl.jupiter_masses <= 1e2
    # And within an order of magnitude of the multi-stroke ledger floor.
    assert abs(math.log10(fl.jupiter_masses) - math.log10(JUPITER_FLOOR_HIGH)) <= 2.0


def test_alcubierre_floor_matches_feasibility_report():
    """The scorer's floor equals knopp_ratchet.feasibility_report per-stroke."""
    from systrophe.knopp.knopp_ratchet import feasibility_report

    fr = feasibility_report()
    fl = alcubierre_jupiter_floor(R_m=1.0, sigma_m=1.0, v_s_over_c=1.0,
                                  kappa=1.0 / 12.0)
    # Per-stroke (per-bubble) principal in joules must match.
    assert fl.principal_J == pytest.approx(fr.e_neg_bare_per_stroke_J, rel=1e-9)


def test_alcubierre_score_grows_with_sampling_time():
    """Pfenning-Ford: the longer the wall must be held open, the worse the QI
    violation. Score is monotone increasing in tau (over fixed wall density).
    """
    s_short = score_alcubierre(v_s=1.0, R=1.0, sigma=8.0, tau=0.1)
    s_long = score_alcubierre(v_s=1.0, R=1.0, sigma=8.0, tau=2.0)
    assert s_long.qi_normalized_score > s_short.qi_normalized_score


# ---------------------------------------------------------------------------
# Registry: Goedel / Gott / Kerr realizable; wormhole forbidden
# ---------------------------------------------------------------------------


def test_godel_realizable():
    s = score_godel(a=1.0)
    assert s.qi_normalized_score < 1.0 and not s.qi_forbidden
    assert s.source_kind == "dust"


def test_gott_realizable_vacuum_exterior():
    s = score_gott(mu=0.1, v=0.95)
    assert s.qi_normalized_score < 1.0 and not s.qi_forbidden
    assert s.source_kind == "vacuum"


def test_kerr_realizable_vacuum():
    s = score_kerr(M=1.0, a=1.5)
    assert s.qi_normalized_score < 1.0 and not s.qi_forbidden
    assert s.source_kind == "vacuum"


def test_wormhole_throat_qi_forbidden():
    """Morris-Thorne throat needs local exotic matter (score > 1)."""
    s = score_wormhole_throat(r_throat=1.0, db_dr_at_throat=0.5)
    assert s.t_kk_min < 0.0
    assert s.qi_normalized_score > 1.0 and s.qi_forbidden
    assert s.source_kind == "exotic"


def test_full_registry_classifies_all_six():
    reg = {s.name: s for s in score_full_registry()}
    assert set(reg) == {
        "van_stockum", "godel", "gott", "kerr",
        "alcubierre", "wormhole_throat",
    }
    classical = {"van_stockum", "godel", "gott", "kerr"}
    exotic = {"alcubierre", "wormhole_throat"}
    for name in classical:
        assert reg[name].qi_normalized_score < 1.0, name
    for name in exotic:
        assert reg[name].qi_normalized_score > 1.0, name


# ---------------------------------------------------------------------------
# GATE 3: novelty-catcher / surrogate lambda_2 separation
# ---------------------------------------------------------------------------


def test_gate_lambda2_separates_classical_from_exotic():
    """Address-space catcher + surrogate null separates the two physics classes.

    True within-group Hamming compactness must be tighter than the
    shuffled-label surrogate by > 2 sigma, and the classical members must be
    exactly the four non-forbidden ones.
    """
    reg = score_full_registry()
    sep = qi_lambda2_separation(reg, n_surrogate=500, seed=0)
    assert isinstance(sep, QILambda2Separation)
    # The mask must match the physics: 4 classical, 2 exotic.
    assert sep.classical_mask == [True, True, True, True, False, False]
    # Surrogate test: true clustering tighter than random labelling.
    assert sep.within_mean_hamming_true < sep.within_mean_hamming_surrogate_mean
    assert sep.separation_z > 2.0
    assert sep.separates is True
    assert sep.verdict == "separated"


def test_lambda2_full_graph_disconnected_at_small_radius():
    """The full Hamming graph is disconnected (lambda_2 = 0) at small radius,
    the address-space signature of two well-separated clusters."""
    reg = score_full_registry()
    sep = qi_lambda2_separation(reg, n_surrogate=50, radius=6)
    assert sep.lambda_2_full == pytest.approx(0.0, abs=1e-9)


def test_lambda2_surrogate_has_nonzero_spread():
    """A meaningful surrogate null must have nonzero variance (not z=inf)."""
    reg = score_full_registry()
    sep = qi_lambda2_separation(reg, n_surrogate=500)
    assert sep.within_mean_hamming_surrogate_std > 0.0
    assert math.isfinite(sep.separation_z)


# ---------------------------------------------------------------------------
# Warp-drive-zoo comparison extension
# ---------------------------------------------------------------------------


def test_compare_drives_with_qi_attaches_scores():
    drives = [
        AlcubierreDrive(v_s=1.0, R=1.0, sigma=8.0),
        BobrickMartireDrive(m_ADM=1.0),
        LentzDrive(),
    ]
    cmp_ = compare_drives_with_qi(drives, box_half_size=2.5, n_grid=24)
    assert isinstance(cmp_, WarpDriveQIComparison)
    # Base comparison is preserved.
    assert set(cmp_.base.drives) == {"alcubierre", "bobrick_martire", "lentz"}
    # Alcubierre is QI-forbidden; positive-energy drives are not.
    assert cmp_.qi_forbidden["alcubierre"] is True
    assert cmp_.qi_normalized_score["alcubierre"] > 1.0
    assert cmp_.qi_forbidden["bobrick_martire"] is False
    assert cmp_.qi_normalized_score["bobrick_martire"] == pytest.approx(0.0)
    assert cmp_.qi_forbidden["lentz"] is False


def test_compare_qi_ranking_orders_realizable_first():
    drives = [
        AlcubierreDrive(v_s=1.0, R=1.0, sigma=8.0),
        BobrickMartireDrive(m_ADM=1.0),
    ]
    cmp_ = compare_drives_with_qi(drives, box_half_size=2.0, n_grid=16)
    ranking = cmp_.ranking_by_qi
    vals = [v for (_, v) in ranking]
    assert vals == sorted(vals)
    # The least-exotic (BM positive mass) ranks first.
    assert ranking[0][0] == "bobrick_martire"
    assert ranking[-1][0] == "alcubierre"


def test_bobrick_martire_negative_mass_is_forbidden():
    """Negative-ADM-mass Bobrick-Martire is exotic-requiring (score > 0)."""
    drives = [BobrickMartireDrive(m_ADM=-1.0)]
    cmp_ = compare_drives_with_qi(drives, box_half_size=2.0, n_grid=24)
    assert cmp_.qi_scores["bobrick_martire"].t_kk_min < 0.0


# ---------------------------------------------------------------------------
# QIScore dataclass sanity
# ---------------------------------------------------------------------------


def test_qiscore_fields_consistent():
    s = score_alcubierre()
    assert isinstance(s, QIScore)
    expected = abs(s.t_kk_min) / abs(s.qi_bound)
    assert s.qi_normalized_score == pytest.approx(expected, rel=1e-9)
    assert s.qi_forbidden == (s.qi_normalized_score > 1.0)
