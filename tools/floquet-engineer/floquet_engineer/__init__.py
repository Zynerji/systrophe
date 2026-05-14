"""floquet-engineer: time-periodic Z_3 Floquet quasi-energies + CTC stability gap.

Wraps `systrophe.floquet_mobius` (joint Z_3 static H + Floquet
propagator + quasi-energy extraction) and
`systrophe.floquet_engineering` (drive-amp/freq sweep of the CTC
stability gap) in a single object.

Use case: drive the three Z_3 branches with a time-periodic
perturbation, ask whether the CTC sector gets gapped out (Floquet
stabilisation of chronology protection).
"""

from __future__ import annotations

from .engineer import (
    FloquetEngineer,
    FloquetSweepReport,
)

__all__ = [
    "FloquetEngineer",
    "FloquetSweepReport",
]
