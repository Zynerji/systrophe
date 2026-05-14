"""FloquetEngineer: top-level wrapper for Z_3 Floquet quasi-energy analysis."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from systrophe.floquet_engineering import (
    StabilityMapResult,
    ctc_stability_gap,
    floquet_engineering_map,
    identify_floquet_resonances,
    stabilisation_efficacy,
)
from systrophe.floquet_mobius import (
    FloquetMobiusResult,
    analyze_floquet_mobius,
    static_limit_check,
    z3_symmetry_check,
)


@dataclass(frozen=True)
class FloquetSweepReport:
    """2D drive-amplitude × drive-frequency sweep of the CTC stability gap.

    Attributes
    ----------
    drive_amps : np.ndarray
        Amplitude axis.
    omega_drives : np.ndarray
        Frequency axis.
    gap_map : np.ndarray of shape (n_amp, n_omega)
        CTC stability gap at each sweep cell.
    resonances : list[dict]
        Floquet resonances (drive_amp, omega_drive, gap, n_band).
    max_gap : float
        Maximum gap across the entire sweep — a measure of best
        Floquet stabilisation achievable.
    stabilisation_efficacy : float
        Static-vs-best gap ratio.
    """
    drive_amps: np.ndarray
    omega_drives: np.ndarray
    gap_map: np.ndarray
    resonances: list
    max_gap: float
    stabilisation_efficacy: float


class FloquetEngineer:
    """Z_3-branch Floquet engineering tool.

    Parameters
    ----------
    branch_energies : (3,) array
        Static energies of the three Z_3 branches.
    hopping : float, default 0.0
        Inter-branch hopping (Z_3 cyclic shift coupling).
    """

    def __init__(self, branch_energies, hopping: float = 0.0) -> None:
        branch_energies = np.asarray(branch_energies, dtype=float).ravel()
        if branch_energies.shape != (3,):
            raise ValueError(
                f"branch_energies must have shape (3,), got {branch_energies.shape}"
            )
        self.branch_energies = branch_energies
        self.hopping = float(hopping)

    def analyze(self, drive_amp: float, omega_drive: float,
                n_steps: int = 200) -> FloquetMobiusResult:
        """Full Floquet analysis at one (drive_amp, omega_drive) point."""
        return analyze_floquet_mobius(
            branch_energies=self.branch_energies,
            hopping=self.hopping,
            drive_amp=float(drive_amp),
            omega_drive=float(omega_drive),
            n_steps=int(n_steps),
        )

    def static_limit(self, omega_drive: float = 1.0) -> dict:
        """Check that drive_amp → 0 recovers the static H spectrum mod omega."""
        return dict(static_limit_check(
            branch_energies=self.branch_energies, omega_drive=float(omega_drive),
        ))

    def z3_symmetry(self, drive_amp: float = 0.2,
                     omega_drive: float = 1.0) -> dict:
        """Verify Z_3 cyclic symmetry of the static Hamiltonian at chosen drive."""
        return dict(z3_symmetry_check(
            branch_energies=self.branch_energies, hopping=self.hopping,
            drive_amp=float(drive_amp), omega_drive=float(omega_drive),
        ))

    def sweep(self, drive_amps, omega_drives, n_steps: int = 200) -> FloquetSweepReport:
        """Sweep (drive_amp, omega_drive) and map the CTC stability gap."""
        drive_amps = np.asarray(drive_amps, dtype=float).ravel()
        omega_drives = np.asarray(omega_drives, dtype=float).ravel()
        sm: StabilityMapResult = floquet_engineering_map(
            branch_energies=self.branch_energies,
            drive_amps=drive_amps, omega_drives=omega_drives,
            hopping=self.hopping, n_steps=int(n_steps),
        )
        # Convert resonance dataclass list / array to a list of dicts.
        try:
            resonances = identify_floquet_resonances(sm)
        except Exception:
            resonances = []

        try:
            efficacy = float(stabilisation_efficacy(sm))
        except Exception:
            efficacy = float("nan")

        gap_map = np.asarray(sm.gap_map)
        return FloquetSweepReport(
            drive_amps=drive_amps,
            omega_drives=omega_drives,
            gap_map=gap_map,
            resonances=list(resonances) if resonances is not None else [],
            max_gap=float(np.max(gap_map)),
            stabilisation_efficacy=efficacy,
        )
