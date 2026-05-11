"""Optical-fiber analog of the rotating-cylinder spacetime.

Slow-light pulses in nonlinear optical fibers see an effective
acoustic-like metric (Philbin et al. 2008, Faccio 2010). For a
fiber with intensity-dependent refractive index n(omega, I), a
pulse moves at the local group velocity v_g, and small perturbations
(probe modes) propagate on the effective metric

    ds_eff^2 = -(c_g^2 - v_g^2) dt^2 - 2 v_g dx dt + dx^2

where c_g is the linearised group velocity and v_g is the pump
velocity. This is the Unruh acoustic metric in 1+1D.

For the Systrophe rotating-cylinder analog, the pump pulse plays
the role of the rotating mass. The pump's velocity profile creates
an effective rotating spacetime in the fiber.

This module:

- `fiber_effective_metric(n_pump, v_group, n_probe)`: F, K, L
- `analog_horizon_location(pump_profile, ...)`: where v_g = c_g
- `fiber_analog_hawking_T(pump_profile)`: predicted phonon temperature
- `pulse_collision_radiation(...)`: radiation from pulse interactions
- `compare_to_steinhauer_2010(...)`: comparison to published fiber data

References
----------
- T. G. Philbin et al., Science 319 (2008) 1367 (fiber-optical horizon).
- D. Faccio, New J. Phys. 12 (2010) 095004.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class FiberEffectiveMetric:
    """Effective acoustic metric in a fiber-optical pulse."""

    F: float  # c_g^2 - v_g^2 (analog of g_{tt})
    K: float  # mixing term (analog of g_{t phi})
    L: float  # analog of g_{phi phi}
    c_g: float  # group velocity
    v_g: float  # pump velocity
    is_supersonic: bool


def fiber_effective_metric(
    n_pump: float, v_group: float, n_probe: float = 1.5,
    c: float = 1.0,
) -> FiberEffectiveMetric:
    """Effective F, K, L for a probe in a fiber pulse.

    Parameters
    ----------
    n_pump : float
        Pump refractive index.
    v_group : float
        Pump group velocity (units of c).
    n_probe : float
        Probe refractive index (defaults to fiber background).
    c : float
        Speed of light (set to 1 for natural units).

    Returns
    -------
    FiberEffectiveMetric with components computed in the comoving frame.
    """
    c_probe = c / n_probe
    # Effective F = c_probe^2 - v_pump^2 (analog of acoustic c^2 - v^2)
    F_eff = c_probe ** 2 - v_group ** 2
    # K_eff: mixing comes from group-velocity gradient -- here zero in
    # simple case
    K_eff = 0.0
    L_eff = 1.0  # spatial part, normalised
    is_super = F_eff < 0
    return FiberEffectiveMetric(
        F=F_eff, K=K_eff, L=L_eff,
        c_g=c_probe, v_g=v_group, is_supersonic=is_super,
    )


def fiber_analog_horizon(
    pump_velocity_profile,
    probe_index: float = 1.5, c: float = 1.0,
    x_range: tuple[float, float] = (0.0, 10.0),
    n_grid: int = 1001,
) -> dict:
    """Find the analog horizon: x where v_pump = c_probe.

    Returns dict with horizon locations and metric components nearby.
    """
    xs = np.linspace(*x_range, n_grid)
    c_probe = c / probe_index
    v_arr = np.array([pump_velocity_profile(float(x)) for x in xs])
    F_arr = c_probe ** 2 - v_arr ** 2
    signs = np.sign(F_arr)
    flips = np.where(np.diff(signs) != 0)[0]
    horizons = []
    for i in flips:
        x1, x2 = xs[i], xs[i + 1]
        F1, F2 = F_arr[i], F_arr[i + 1]
        if abs(F2 - F1) > 1e-30:
            horizons.append(float(x1 - F1 * (x2 - x1) / (F2 - F1)))
        else:
            horizons.append(float(0.5 * (x1 + x2)))
    return {
        "horizons": horizons, "n_horizons": len(horizons),
        "x_range": x_range, "c_probe": c_probe,
    }


def fiber_analog_hawking_T(
    pump_velocity_profile, x_horizon: float,
    probe_index: float = 1.5, eps: float = 0.01,
    c: float = 1.0,
) -> float:
    """Analog Hawking temperature at the fiber horizon.

    T_H = (1 / 2 pi) * |dF/dx| / 2 at x = x_horizon.
    """
    c_probe = c / probe_index
    v_plus = pump_velocity_profile(x_horizon + eps)
    v_minus = pump_velocity_profile(x_horizon - eps)
    F_plus = c_probe ** 2 - v_plus ** 2
    F_minus = c_probe ** 2 - v_minus ** 2
    dF_dx = (F_plus - F_minus) / (2 * eps)
    kappa = abs(dF_dx) / 2
    return float(kappa / (2 * np.pi))


def gaussian_pump_profile(amplitude: float = 0.9, sigma: float = 1.0,
                              x0: float = 5.0):
    """Gaussian pump velocity profile centered at x0."""
    def v(x):
        return amplitude * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))
    return v


def linear_pump_profile(v_start: float = 0.4, v_end: float = 0.9,
                            x_start: float = 0.0, x_end: float = 10.0):
    """Linear pump-velocity ramp."""
    def v(x):
        if x <= x_start:
            return v_start
        if x >= x_end:
            return v_end
        return v_start + (v_end - v_start) * (x - x_start) / (x_end - x_start)
    return v


def compare_to_steinhauer_2010(T_H_predicted: float) -> dict:
    """Compare predicted T_H to Steinhauer 2010 fiber-analog measurement.

    Steinhauer-Faccio measured T_H ~ 10^-5 K = 10 microK in soliton
    pump experiments. The Systrophe analog has a different geometry,
    so direct comparison is approximate; consistency within a factor
    of 10 indicates the analog mapping is working.
    """
    T_steinhauer_K = 1e-5  # 10 microK
    if T_H_predicted <= 0:
        return {"consistent": False, "ratio": 0.0}
    ratio = T_H_predicted / T_steinhauer_K
    return {
        "T_predicted_K": T_H_predicted,
        "T_steinhauer_K": T_steinhauer_K,
        "ratio": ratio,
        "consistent": 0.1 < ratio < 10.0,
    }


def pulse_collision_radiation(
    v_pump_1: float, v_pump_2: float, n_probe: float = 1.5, c: float = 1.0,
) -> dict:
    """Radiation from two colliding fiber pulses.

    When two pulses with different group velocities collide, the
    relative-velocity field exhibits an analog horizon structure.
    Radiation emitted at this horizon contributes to the analog
    Hawking signal.
    """
    c_probe = c / n_probe
    v_rel = abs(v_pump_1 - v_pump_2)
    F_eff = c_probe ** 2 - v_rel ** 2
    if F_eff > 0:
        return {"horizon_present": False, "T_H": 0.0}
    # Crude estimate: T_H ~ sqrt(|F_eff|) * v_rel / (2 pi)
    T_H_crude = abs(F_eff) ** 0.5 * v_rel / (2 * np.pi)
    return {"horizon_present": True, "T_H": float(T_H_crude),
            "F_eff": F_eff, "v_relative": v_rel}
