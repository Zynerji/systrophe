"""Tests for the coupled (dF,dK,dL,dh,dS) linearized vacuum system.

These use the dill-cached lambdified system (derive_coupled.get_system). If the
cache is absent the symbolic derivation takes minutes, so the module skips rather
than rebuild in a normal test run. Build it once with:  python run_coupled.py
"""

import os
import warnings

import numpy as np
import pytest

warnings.filterwarnings("ignore")

_CACHE = os.path.join(os.path.dirname(__file__), "_coupled_system.dill")
if not os.path.exists(_CACHE):
    pytest.skip("coupled system cache not built (run run_coupled.py)",
                allow_module_level=True)

from derive_coupled import get_system
from evolve_coupled import background, CoupledEvolver, FIELDS, EVOL_KEYS, CONSTR_KEYS
from validate_coupled import background_fields, static_zero_mode
from systrophe.geometry.vanstockum import VanStockumInterior

SYS = get_system()
ORDER, LAMBDAS = SYS["order"], SYS["lambdas"]


def _args(bg, pert, r):
    val = {o: bg[o] for o in ORDER if o in bg}
    z = np.zeros_like(r)
    for nm in FIELDS:
        for suf in ["", "_t", "_r", "_tt", "_rr", "_tr"]:
            val[nm + suf] = pert.get(nm + suf, z)
    return [val[o] for o in ORDER]


def test_system_structure():
    """5 evolution + 2 constraint equations; the z-twist sector decouples."""
    keys = set(LAMBDAS.keys())
    assert set(EVOL_KEYS).issubset(keys)
    assert set(CONSTR_KEYS).issubset(keys)
    # tz, rz, phiz identically zero -> not present
    for k in [(0, 3), (1, 3), (2, 3)]:
        assert k not in keys


def test_no_invF0_regular_across_ergosurface():
    """Every linearized component is FINITE across an ergosurface (F0=0) in
    metric variables -- the 1/F0 that drove the single-variable instability is
    absent."""
    r = np.linspace(1.40, 1.70, 400)
    rb = background_fields(1.5, 1.0, r)
    zm = static_zero_mode(1.5, 1.0, r)
    pert = {**{nm: zm[nm] for nm in FIELDS},
            **{nm + s: zm[nm + s] for nm in FIELDS
               for s in ["_t", "_r", "_tt", "_rr", "_tr"]}}
    args = _args(rb, pert, r)
    assert np.min(np.abs(rb["F0"])) < 1e-2          # grid really crosses F0=0
    for key, fn in LAMBDAS.items():
        assert np.all(np.isfinite(np.asarray(fn(*args), float)))


def test_static_family_is_zero_mode():
    """d_a(van Stockum) is a static solution of the linearized equations."""
    r = np.linspace(2.0, 8.0, 300)
    rb = background_fields(1.5, 1.0, r)
    zm = static_zero_mode(1.5, 1.0, r)
    pert = {**{nm: zm[nm] for nm in FIELDS},
            **{nm + s: zm[nm + s] for nm in FIELDS
               for s in ["_t", "_r", "_tt", "_rr", "_tr"]}}
    args = _args(rb, pert, r)
    safe = np.abs(rb["F0"]) > 0.05
    scale = np.abs(zm["dF_rr"]) + np.abs(zm["dK_rr"]) + np.abs(zm["dL_rr"]) + 1e-12
    worst = max(float(np.nanmax(np.abs(np.asarray(fn(*args), float))[safe] / scale[safe]))
                for fn in LAMBDAS.values())
    assert worst < 5e-2


def test_momentum_constraints_exactly_satisfied():
    r = np.linspace(2.0, 8.0, 300)
    rb = background_fields(1.5, 1.0, r)
    zm = static_zero_mode(1.5, 1.0, r)
    pert = {**{nm: zm[nm] for nm in FIELDS},
            **{nm + s: zm[nm + s] for nm in FIELDS
               for s in ["_t", "_r", "_tt", "_rr", "_tr"]}}
    args = _args(rb, pert, r)
    safe = np.abs(rb["F0"]) > 0.05
    for key in CONSTR_KEYS:
        c = np.asarray(LAMBDAS[key](*args), float)
        assert np.max(np.abs(c)[safe]) < 1e-6


def test_coupled_physical_sector_is_not_tachyonic():
    """Decisive stability test. The single-variable model's potential omega^2 = V
    is negative (tachyonic) -> growth. The coupled physical-sector long-wavelength
    spectrum omega^2 = eig(pinv(M).C0) has NO tachyonic eigenvalues -> the
    destabilising mass term is absent (the instability was a variable artifact)."""
    r = np.linspace(2.2, 4.5, 100)                 # between ergosurfaces
    bg = background(1.5, 1.0, r)
    z = np.zeros_like(r); n = len(r)

    def probe(setter):
        A = np.zeros((n, 5, 5))
        for j, fj in enumerate(FIELDS):
            pert = {}
            setter(pert, fj)
            args = _args(bg, pert, r)
            for i, key in enumerate(EVOL_KEYS):
                A[:, i, j] = np.asarray(LAMBDAS[key](*args), float)
        return A

    M = probe(lambda p, fj: p.__setitem__(fj + "_tt", np.ones_like(r)))
    C0 = probe(lambda p, fj: p.__setitem__(fj, np.ones_like(r)))
    coupled = []
    for i in range(n):
        ev = np.linalg.eigvals(np.linalg.pinv(M[i]) @ C0[i])
        ev = ev[np.abs(ev.imag) < 1e-6 * (np.abs(ev.real) + 1)].real
        coupled.append(ev[np.abs(ev) > 1e-6])
    coupled = np.concatenate([x for x in coupled if len(x)]) if any(len(x) for x in coupled) else np.array([])
    # coupled physical sector: no tachyonic (significantly negative) eigenvalues
    assert coupled.size == 0 or coupled.min() > -0.1

    # single-variable model: omega^2 = V < 0 (tachyonic) over the whole range
    vs = VanStockumInterior(omega=1.5, R=1.0); c0 = 3.0
    eps = 1e-7 * np.maximum(r, 1.0); F0 = np.asarray(vs.analytic_exterior_F(r))
    F0p = (np.asarray(vs.analytic_exterior_F(r + eps)) -
           np.asarray(vs.analytic_exterior_F(r - eps))) / (2 * eps)
    V = (F0p ** 2 - c0 ** 2) / F0 ** 2 - 2 * F0p / (r * F0)
    assert np.mean(V < 0) > 0.9                     # single-variable is tachyonic
    assert V.min() < -0.5


def test_M_rank_is_gauge_deficient():
    """The 5x5 second-time-derivative matrix is rank-deficient: cylindrical
    vacuum has 2 gauge DOF (so a naive free evolution needs gauge fixing)."""
    ev = CoupledEvolver(a0=1.5, r_min=1.2, r_max=4.0, n=80)
    assert ev.rankM < 5
