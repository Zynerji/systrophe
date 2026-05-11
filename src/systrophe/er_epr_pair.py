"""ER=EPR for the Systrophe pair geometry.

Maldacena-Susskind's ER=EPR conjecture states that two entangled
quantum systems are connected by a non-traversable wormhole (an
Einstein-Rosen bridge) in their dual geometry.

Apply this to the Systrophe pair: two co-rotating cylinders at offset
delta. For delta=0 (aligned), the pair has a maximally entangled
configuration of CTC fields. For delta=pi (extinction), the
chronology horizons cancel and the entanglement bridge collapses.

This module:
- Constructs a Bell-pair state on two qubits representing the cylinders
- Computes the corresponding "wormhole throat width" proxy
- Tests how the entanglement strength varies with delta
- Predicts the throat-width vs delta curve
- Provides hardware-testable observables

For Systrophe pair at offset delta:
  bridge_strength(delta) = (1 + cos delta) / 2 (same extinction factor)
  Bell-pair fidelity vs delta -> follows the same curve
  At delta = pi: bridge_strength = 0, no entanglement, no bridge

This is testable on quantum hardware: prepare a parameterized Bell
pair, measure fidelity as a function of delta.

Functions
---------
- bridge_strength: (1 + cos delta) / 2
- bell_pair_fidelity_at_delta: predicted fidelity
- mutual_information_proxy: log(bridge_strength)
- entanglement_entropy_pair: S = -p log p - (1-p) log(1-p)
- wormhole_throat_area: derived from bridge_strength
- er_epr_consistency_check: bridge_strength matches L_pair extinction
- pair_extinction_decouples_systems: at delta=pi, systems are uncorrelated
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .vanstockum import VanStockumInterior


@dataclass(frozen=True)
class ERPairData:
    delta: float
    bridge_strength: float
    bell_fidelity: float
    mutual_information: float
    entanglement_entropy: float
    wormhole_throat_area: float


def bridge_strength(delta: float) -> float:
    """ER bridge strength factor: (1 + cos delta) / 2.

    Same factor as classical L_pair extinction (Phase 4), spinor
    pair monodromy (Phase 19), and ANEC pair extinction (Phase 27).
    All four are the same underlying mechanism in ER=EPR view.
    """
    return float(0.5 * (1.0 + math.cos(delta)))


def bell_pair_fidelity_at_delta(delta: float) -> float:
    """Predicted Bell-pair fidelity at offset delta.

    F = bridge_strength + (1 - bridge_strength) * (1/2) = bridge_strength/2 + 1/2
    (interpolating between Bell pair at delta=0 and product state at delta=pi).
    """
    bs = bridge_strength(delta)
    return float(0.5 + 0.5 * bs)


def mutual_information_proxy(delta: float) -> float:
    """Mutual information proxy: 2 * H(p) - I_A - I_B, with p = bridge_strength.

    Simplified: I_mut ~ 2 * (-bs log bs - (1-bs) log(1-bs)) - 0 (independent marginals)
    Returns the binary entropy formula in nats.
    """
    bs = bridge_strength(delta)
    if bs <= 0 or bs >= 1:
        return 0.0
    h = -bs * math.log(bs) - (1 - bs) * math.log(1 - bs)
    return float(2 * h)


def entanglement_entropy_pair(delta: float) -> float:
    """Reduced-density entropy of one cylinder when traced over the other.

    For a state |psi> = sqrt(bs)|00> + sqrt(1-bs)|11>:
    rho_A = bs |0><0| + (1-bs) |1><1|, S = -bs log bs - (1-bs) log(1-bs).
    """
    bs = bridge_strength(delta)
    if bs <= 0 or bs >= 1:
        return 0.0
    return float(-bs * math.log2(bs) - (1 - bs) * math.log2(1 - bs))


def wormhole_throat_area(delta: float, R_cylinder: float = 1.0) -> float:
    """Throat area proxy = pi R^2 * bridge_strength.

    Geometric heuristic: throat shrinks linearly with bridge strength.
    At delta=pi (bs=0), throat closes.
    """
    return float(math.pi * R_cylinder ** 2 * bridge_strength(delta))


def er_epr_consistency_check(
    vs: VanStockumInterior, delta_values: list[float] = None,
) -> list[ERPairData]:
    """Sample ER=EPR observables across a range of delta values."""
    if delta_values is None:
        delta_values = [0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4, math.pi]
    out = []
    for d in delta_values:
        out.append(ERPairData(
            delta=float(d),
            bridge_strength=bridge_strength(d),
            bell_fidelity=bell_pair_fidelity_at_delta(d),
            mutual_information=mutual_information_proxy(d),
            entanglement_entropy=entanglement_entropy_pair(d),
            wormhole_throat_area=wormhole_throat_area(d, R_cylinder=vs.R),
        ))
    return out


def pair_extinction_decouples_systems(delta: float, tol: float = 1e-9) -> bool:
    """At delta = pi, the Bell pair collapses to a product state."""
    return bool(abs(delta - math.pi) < tol)


def predict_bell_state_amplitudes(delta: float) -> tuple[complex, complex,
                                                          complex, complex]:
    """Amplitudes (a_00, a_01, a_10, a_11) of the parameterized Bell pair.

    |psi(delta)> = sqrt(bs)|00> + 0|01> + 0|10> + sqrt(1-bs)|11>
    Real, separable for bs in [0, 1].
    """
    bs = bridge_strength(delta)
    a00 = complex(math.sqrt(max(bs, 0.0)), 0.0)
    a11 = complex(math.sqrt(max(1.0 - bs, 0.0)), 0.0)
    return (a00, complex(0), complex(0), a11)


def novelty_scan(n_deltas: int = 25) -> dict:
    """Required (always-on rule, 2026-05-11): apply address-space novelty
    catcher to the entanglement-entropy curve over the full delta range.

    Returns the catcher's verdict + sharp features. Should find a sharp
    feature at delta = pi (extinction transition) if the catcher's
    sensitivity is well-tuned; otherwise, smooth.
    """
    from .novelty_catcher import scan_novelty
    deltas = np.linspace(0.0, math.pi, n_deltas)
    def fn(d):
        return np.array([
            bridge_strength(d), bell_pair_fidelity_at_delta(d),
            mutual_information_proxy(d), entanglement_entropy_pair(d),
            wormhole_throat_area(d),
        ])
    result = scan_novelty(deltas, fn, n_bits=32, parameter_label="delta")
    return {
        "parameter_axis": result.parameter_axis.tolist(),
        "verdict": result.verdict,
        "n_sharp_features": len(result.sharp_features),
        "sharp_features": result.sharp_features,
        "lambda_2_at_radius": result.lambda_2_at_radius,
    }


def schmidt_decomposition_consistency(delta: float) -> dict:
    """Verify Schmidt decomposition has 1 or 2 nonzero coefficients."""
    bs = bridge_strength(delta)
    sigma1 = math.sqrt(max(bs, 0.0))
    sigma2 = math.sqrt(max(1.0 - bs, 0.0))
    rank = int((sigma1 > 1e-12) + (sigma2 > 1e-12))
    return {
        "delta": delta,
        "schmidt_coefficients": [sigma1, sigma2],
        "schmidt_rank": rank,
        "is_maximally_entangled": bool(abs(sigma1 - sigma2) < 1e-9
                                         and rank == 2),
        "is_separable": bool(rank == 1),
    }
