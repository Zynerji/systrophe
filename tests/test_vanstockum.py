"""Van Stockum interior metric tests."""

import numpy as np
import pytest

from systrophe.vanstockum import VanStockumInterior, vanstockum_interior_metric


def test_minkowski_limit_omega_zero():
    """omega = 0 -> Minkowski metric in cylindrical coords."""
    vs = VanStockumInterior(omega=0.0, R=1.0)
    g = vs.metric(0.5)
    assert g[0, 0] == pytest.approx(-1.0)
    assert g[0, 2] == pytest.approx(0.0)
    assert g[2, 2] == pytest.approx(0.5**2)
    assert g[1, 1] == pytest.approx(1.0)
    assert g[3, 3] == pytest.approx(1.0)


def test_tphi_block_determinant_invariant():
    """g_tt g_phiphi - g_tphi^2 = -r^2 throughout the interior."""
    vs = VanStockumInterior(omega=0.7, R=2.0)
    for r in [0.1, 0.5, 1.0, 1.5, 1.9]:
        det = vs.determinant_tphi_block(r)
        assert det == pytest.approx(-r * r, rel=1e-12)


def test_alpha_real_iff_supercritical():
    """alpha = sqrt(4a^2 - 1) is real iff a > 1/2."""
    vs_super = VanStockumInterior(omega=1.0, R=1.0)  # a = 1
    assert vs_super.is_supercritical()
    assert vs_super.alpha == pytest.approx(np.sqrt(3.0))

    vs_sub = VanStockumInterior(omega=0.4, R=1.0)  # a = 0.4
    assert not vs_sub.is_supercritical()
    with pytest.raises(ValueError):
        _ = vs_sub.alpha


def test_interior_ctc_threshold():
    """CTC inside the dust iff omega * R > 1."""
    vs_no = VanStockumInterior(omega=0.9, R=1.0)
    assert not vs_no.has_interior_ctc()
    assert vs_no.gphiphi(1.0) > 0  # at boundary, still positive

    vs_yes = VanStockumInterior(omega=2.0, R=1.0)
    assert vs_yes.has_interior_ctc()
    assert vs_yes.gphiphi(0.9) < 0  # inside, g_phiphi negative


def test_proper_circumference_zero_at_threshold():
    """Proper phi-circumference vanishes at r = 1/omega."""
    omega = 1.5
    vs = VanStockumInterior(omega=omega, R=2.0)
    r_zero = 1.0 / omega
    c = vs.proper_circumference(r_zero)
    assert c == pytest.approx(0.0, abs=1e-12)


def test_metric_rejects_exterior_radius():
    vs = VanStockumInterior(omega=1.0, R=1.0)
    with pytest.raises(ValueError):
        vs.metric(2.0)


def test_interior_regime_labels():
    """interior_regime classifies based on a vs 1.0 (dust CTC threshold)."""
    assert VanStockumInterior(omega=0.3, R=1.0).interior_regime == "subcritical"
    assert VanStockumInterior(omega=0.9, R=1.0).interior_regime == "subcritical"
    assert VanStockumInterior(omega=1.0, R=1.0).interior_regime == "critical"
    assert VanStockumInterior(omega=1.5, R=1.0).interior_regime == "supercritical"


def test_interior_ctc_shell_inner_radius():
    """Inner radius of the interior CTC shell is 1/omega for a > 1."""
    vs_super = VanStockumInterior(omega=2.0, R=1.0)  # a = 2
    assert vs_super.interior_ctc_shell_inner_radius == pytest.approx(0.5)
    vs_sub = VanStockumInterior(omega=0.5, R=1.0)
    assert vs_sub.interior_ctc_shell_inner_radius is None
    vs_crit = VanStockumInterior(omega=1.0, R=1.0)
    assert vs_crit.interior_ctc_shell_inner_radius is None


def test_exterior_regime_labels():
    """regime classifies based on a vs 1/2 (Bonnor exterior threshold)."""
    assert VanStockumInterior(omega=0.3, R=1.0).regime == "subcritical"
    assert VanStockumInterior(omega=0.5, R=1.0).regime == "critical"
    assert VanStockumInterior(omega=0.7, R=1.0).regime == "supercritical"


def test_functional_form_matches_class():
    """Functional helper agrees with class metric on interior."""
    omega = 0.6
    vs = VanStockumInterior(omega=omega, R=2.0)
    rs = np.linspace(0.1, 1.9, 7)
    d = vanstockum_interior_metric(omega, rs)
    for i, r in enumerate(rs):
        g = vs.metric(float(r))
        assert d["g_tt"][i] == pytest.approx(g[0, 0])
        assert d["g_tphi"][i] == pytest.approx(g[0, 2])
        assert d["g_phiphi"][i] == pytest.approx(g[2, 2])
        assert d["g_rr"][i] == pytest.approx(g[1, 1])
        assert d["g_zz"][i] == pytest.approx(g[3, 3])
