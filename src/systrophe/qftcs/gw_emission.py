"""Gravitational wave emission from cylinder dynamics.

A rotating cylinder with constant omega has no time-varying mass
quadrupole moment, so emits *no* GW in the rigid steady state. GW
emission arises only when:

(a) the cylinder spins up or spins down (dot omega != 0), inducing a
    time-varying angular acceleration analog of the quadrupole;
(b) the cylinder oscillates radially (R(t) varying), modulating the
    moment of inertia;
(c) two cylinders inspiral (Systrophe pair with shrinking separation).

This module computes the leading-order quadrupole strain for each
case, using the standard formula
    h_TT ~ (2 G/c^4 D) Iddot_TT
where D is observer distance and Iddot is the second time-derivative
of the traceless quadrupole.

Functions
---------
- moment_of_inertia: I_zz for the cylinder
- spin_up_strain: GW strain from constant alpha_spin (rad/s^2)
- radial_oscillation_strain: from R(t) = R_0 + dR cos(omega_R t)
- pair_inspiral_strain: from binary cylinder inspiral
- gw_luminosity: total radiated power
- detectable_at_distance: SNR vs interferometer noise

References
----------
- Misner, Thorne, Wheeler (1973), Section 36
- Maggiore (2008), Gravitational Waves Vol. I
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.geometry.vanstockum import VanStockumInterior


def moment_of_inertia(vs: VanStockumInterior,
                       cylinder_length: float = 1.0) -> float:
    """I_zz = (1/2) M R^2 for a solid cylinder, where M is total dust mass."""
    rho_dust = vs.omega ** 2 / (2 * math.pi)  # rough
    M = rho_dust * math.pi * vs.R ** 2 * cylinder_length
    return float(0.5 * M * vs.R ** 2)


def spin_up_strain(vs: VanStockumInterior,
                    alpha_spin: float = 1.0,
                    observer_distance: float = 1.0,
                    cylinder_length: float = 1.0) -> dict:
    """GW strain amplitude from constant angular acceleration alpha_spin.

    Iddot = alpha_spin * I; h ~ (2/D) * Iddot (G=c=1).
    """
    I = moment_of_inertia(vs, cylinder_length)
    I_ddot = alpha_spin * I
    h = 2.0 * I_ddot / max(observer_distance, 1e-30)
    return {
        "alpha_spin": alpha_spin,
        "I_zz": I,
        "I_ddot": float(I_ddot),
        "h_strain": float(h),
        "observer_distance": observer_distance,
    }


def radial_oscillation_strain(vs: VanStockumInterior,
                                dR_amplitude: float = 0.01,
                                omega_R: float = 1.0,
                                observer_distance: float = 1.0,
                                cylinder_length: float = 1.0) -> dict:
    """Strain from radial oscillation R(t) = R_0 + dR cos(omega_R t).

    I(t) ~ I_0 (1 + 2 dR/R_0 cos(omega_R t))
    Iddot peak ~ 2 I_0 (dR/R_0) omega_R^2.
    """
    I0 = moment_of_inertia(vs, cylinder_length)
    rel_amp = dR_amplitude / max(vs.R, 1e-30)
    I_ddot_peak = 2 * I0 * rel_amp * omega_R ** 2
    h_peak = 2 * I_ddot_peak / max(observer_distance, 1e-30)
    f_GW = omega_R / (2 * math.pi)
    return {
        "dR_amplitude": dR_amplitude,
        "omega_R": omega_R,
        "I_ddot_peak": float(I_ddot_peak),
        "h_peak": float(h_peak),
        "f_GW": float(f_GW),
        "observer_distance": observer_distance,
    }


def pair_inspiral_strain(
    vs: VanStockumInterior,
    separation: float = 10.0,
    d_separation_dt: float = -0.01,
    observer_distance: float = 1.0,
    cylinder_length: float = 1.0,
) -> dict:
    """Strain from a Systrophe pair inspiraling.

    For two equal-mass cylinders at separation a, the binary quadrupole
    Iddot ~ M_red * a^2 * Omega_orb^2; using Kepler Omega_orb^2 = M/a^3:
       Iddot ~ M^2 / a.
    """
    M = moment_of_inertia(vs, cylinder_length) / (0.5 * vs.R ** 2)  # mass per cylinder
    a = separation
    Omega_orb_sq = 2 * M / max(a ** 3, 1e-30)
    Iddot = M * a ** 2 * Omega_orb_sq
    h = 2 * Iddot / max(observer_distance, 1e-30)
    # Chirp rate
    df_dt = (3.0 / 4.0) * (2 * M / a ** 4) * d_separation_dt
    return {
        "separation": a,
        "M_per_cylinder": float(M),
        "Omega_orbital": float(math.sqrt(Omega_orb_sq)),
        "h_strain": float(h),
        "df_dt": float(df_dt),
        "observer_distance": observer_distance,
    }


def gw_luminosity(I_ddot_amplitude: float, omega_GW: float) -> float:
    """Total radiated GW power: P ~ (1/5) (Iddot)^2 omega_GW^4 (rough)."""
    return float((1.0 / 5.0) * I_ddot_amplitude ** 2 * omega_GW ** 4)


def detectable_at_distance(
    h_strain: float, f_GW: float,
    detector_noise_strain: float = 1e-23,
    integration_time: float = 3600.0,
) -> dict:
    """SNR estimate: SNR = h * sqrt(f * T) / S_n."""
    snr = h_strain * math.sqrt(max(f_GW * integration_time, 0.0)) / max(detector_noise_strain, 1e-30)
    return {
        "h_strain": h_strain,
        "f_GW": f_GW,
        "S_n": detector_noise_strain,
        "T_int_s": integration_time,
        "snr_estimate": float(snr),
        "is_detectable": bool(snr > 5.0),
    }


def cylindrical_resonant_frequencies(
    vs: VanStockumInterior, n_modes: int = 5,
) -> list[float]:
    """Eigenfrequencies of radial oscillation modes of the cylinder.

    For a uniform rotating cylinder, radial modes have approximately
    omega_n = n * sqrt(rho) where rho is the dust density.
    """
    rho = vs.omega ** 2  # rough
    return [float(n * math.sqrt(rho) / vs.R) for n in range(1, n_modes + 1)]


def total_gw_energy_lost_in_band(
    vs: VanStockumInterior, omega_R_band: tuple[float, float] = (1.0, 10.0),
    dR_amplitude: float = 0.01,
    cylinder_length: float = 1.0,
    n_freq_samples: int = 50,
) -> float:
    """Integrated GW energy radiated over a frequency band."""
    omegas = np.linspace(omega_R_band[0], omega_R_band[1], n_freq_samples)
    total = 0.0
    domega = (omega_R_band[1] - omega_R_band[0]) / max(n_freq_samples - 1, 1)
    I0 = moment_of_inertia(vs, cylinder_length)
    rel_amp = dR_amplitude / max(vs.R, 1e-30)
    for omega_R in omegas:
        I_ddot_amp = 2 * I0 * rel_amp * omega_R ** 2
        P = gw_luminosity(I_ddot_amp, omega_R)
        total += P * domega / max(omega_R, 1e-30)  # rough energy per unit log freq
    return float(total)
