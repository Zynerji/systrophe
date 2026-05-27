"""Floquet engineering of CTC stability.

Map the (drive amplitude, drive frequency) plane and identify regions
where a periodic drive *stabilises* or *destabilises* CTC bands. The
underlying machinery is `floquet_mobius.py` (joint Floquet on the Z_3
branch space) combined with the static Dirac branch energies from
`dirac_spectrum.py` and the Z_3 twist structure of `casimir.py`.

The central diagnostic is the *CTC stability gap*:

    Delta_stable(drive_amp, omega_drive) = min |eps_F - eps_F_mirror|

where eps_F are the joint Floquet quasi-energies and eps_F_mirror is
their Brillouin-zone-mirrored set. A *small* gap means the drive has
pushed the system to a resonance that hybridises across a band edge ---
typically reducing the effective CTC content. A *large* gap means
the drive stabilises the band.

For a matched-pair with branch energies (e_0, e_1, e_2), the resonance
condition is

    omega_drive = e_b - e_b'

for some pair (b, b'). At resonance, the Floquet spectrum exhibits
avoided crossings of width ~ drive_amp.

The output is suitable for stability-map plots like (drive_amp,
omega_drive, ctc_stability_gap).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.lp.floquet_mobius import analyze_floquet_mobius


@dataclass(frozen=True)
class StabilityMapResult:
    """2D stability map: gap as a function of (drive_amp, omega_drive)."""

    drive_amps: np.ndarray
    omega_drives: np.ndarray
    gap_map: np.ndarray  # shape (len(drive_amps), len(omega_drives))
    branch_energies: np.ndarray
    resonance_omega: float  # the (e_1 - e_0) resonance
    max_destabilising_amp: float | None
    max_destabilising_omega: float | None


def ctc_stability_gap(quasi_energies: np.ndarray, omega_drive: float) -> float:
    """Single-number stability diagnostic from a Floquet spectrum.

    Computes min |eps_i - eps_j| over distinct pairs, divided by the
    Brillouin-zone width omega_drive. A small ratio means the
    quasi-energies are bunched (drive-induced hybridisation; CTC
    destabilisation). A large ratio means well-separated (stable).
    """
    if len(quasi_energies) < 2:
        return float("inf")
    qs = np.sort(quasi_energies)
    diffs = np.diff(qs)
    # Plus the wrap-around gap
    wrap = (qs[0] + omega_drive) - qs[-1]
    gaps = np.concatenate([diffs, [wrap]])
    return float(np.min(np.abs(gaps)) / omega_drive)


def floquet_engineering_map(
    branch_energies: np.ndarray,
    drive_amps: np.ndarray,
    omega_drives: np.ndarray,
    hopping: float = 0.0,
    n_steps: int = 200,
) -> StabilityMapResult:
    """Sweep (drive_amp, omega_drive) and compute the CTC stability gap.

    Parameters
    ----------
    branch_energies : (3,) array
        Static energies of the three Z_3 branches (from
        `dirac_spectrum.find_bound_states` or analytic LP eigenvalues).
    drive_amps : 1D array
        Drive amplitudes to sweep.
    omega_drives : 1D array
        Drive frequencies to sweep.
    hopping : float
        Z_3 cyclic-hopping strength (default 0).
    n_steps : int
        Floquet time-evolution discretisation.

    Returns
    -------
    StabilityMapResult with the 2D gap map and identified resonances.
    """
    drive_amps = np.asarray(drive_amps, dtype=float)
    omega_drives = np.asarray(omega_drives, dtype=float)
    n_a = len(drive_amps)
    n_w = len(omega_drives)
    gap_map = np.zeros((n_a, n_w))
    for i, amp in enumerate(drive_amps):
        for j, om in enumerate(omega_drives):
            result = analyze_floquet_mobius(
                branch_energies, hopping=hopping, drive_amp=float(amp),
                omega_drive=float(om), n_steps=n_steps,
            )
            gap_map[i, j] = ctc_stability_gap(result.quasi_energies, float(om))
    # Identify destabilising point: min of gap_map among amps > 0
    if n_a > 1 and drive_amps[0] == 0:
        # Skip amp = 0 (always trivially uniform)
        idx_flat = np.argmin(gap_map[1:].ravel())
        i_idx, j_idx = np.unravel_index(idx_flat, gap_map[1:].shape)
        i_idx += 1
    else:
        idx_flat = int(np.argmin(gap_map))
        i_idx, j_idx = np.unravel_index(idx_flat, gap_map.shape)
    resonance_om = float(abs(branch_energies[1] - branch_energies[0]))
    return StabilityMapResult(
        drive_amps=drive_amps,
        omega_drives=omega_drives,
        gap_map=gap_map,
        branch_energies=np.asarray(branch_energies),
        resonance_omega=resonance_om,
        max_destabilising_amp=float(drive_amps[i_idx]),
        max_destabilising_omega=float(omega_drives[j_idx]),
    )


def identify_floquet_resonances(
    branch_energies: np.ndarray, omega_drives: np.ndarray, tol: float = 0.1
) -> list[dict]:
    """Identify omega_drive grid points near a resonance e_b - e_b'.

    Returns a list of dicts each with
      - omega_drive : the resonance frequency in the grid
      - branch_pair : (b, b')
      - delta_e     : e_b - e_b'
    """
    omega_drives = np.asarray(omega_drives, dtype=float)
    res = []
    for b in range(3):
        for bp in range(3):
            if b == bp:
                continue
            de = abs(branch_energies[b] - branch_energies[bp])
            if de < 1e-9:
                continue
            mask = np.abs(omega_drives - de) < tol
            for om in omega_drives[mask]:
                res.append({"omega_drive": float(om), "branch_pair": (b, bp),
                            "delta_e": float(de)})
    return res


def stabilisation_efficacy(
    map_result: StabilityMapResult, baseline_amp_idx: int = 0
) -> dict:
    """Quantify how much each drive amplitude affects the gap.

    Compares each row of the gap_map to a baseline row (default = the
    no-drive row, drive_amps[0]).
    """
    baseline = map_result.gap_map[baseline_amp_idx]
    differences = map_result.gap_map - baseline[None, :]
    # max amplification (gap larger than baseline)
    max_amp = float(np.max(differences))
    # max suppression (gap smaller than baseline)
    max_supp = float(np.min(differences))
    return {
        "max_gap_amplification": max_amp,
        "max_gap_suppression": max_supp,
        "mean_change": float(np.mean(differences)),
    }
