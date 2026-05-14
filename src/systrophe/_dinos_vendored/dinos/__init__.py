"""Vendored subset of the Dinos (Δῖνος) Dirac-Kerr-Newman framework.

Provenance: copied verbatim from `C:/Users/cknop/.local/bin/Dinos/src/dinos/`
on 2026-05-13. Only the modules required by `systrophe.dinos_bridge`
are vendored here:

  * `kerr_corrections`  -- functions used by `kerr_correction_at_tipler_threshold`
    and `CylindricalKerrMapping`.
  * `mobius_z3_cover`   -- functions used by the Z_3 branch-match
    helpers in the bridge.

The full Dinos package is much larger; the bridge module also exposes
a `temporal_loop`-based helper (`evolve_mobius_temporal_loop`) that is
NOT vendored because none of the bridge unit tests exercise it and it
chains imports across `closure`, `casimir`, `geodesic`, `constants`
which would balloon this subdirectory.

Users who need the full Dinos surface should still install / set
SYSTROPHE_DINOS_PATH to the upstream repo at github.com/Zynerji/dinos-DKN;
`dinos_bridge._ensure_dinos_import` prefers an externally-installed
`dinos` package over this vendored fallback.

Note on the Dinos-DKN predictivity status: an independent audit
(`C:/Users/cknop/.local/bin/dinos-infer/FINDINGS.md`, 2026-05-03)
falsified the predictivity claim of the full DKN framework via
cross-validation (lowest-loss Foot branch correct on 42% of 57 LOO
holdouts). The math vendored here -- the Möbius Z_3 cover eigenvalue
closed forms and the Kerr correction kinematics -- is *separate* from
that predictivity claim and is used by the bridge as a mathematical
correspondence with the cylindrical Tipler exterior, not as a
physical-prediction tool. See `systrophe.dinos_bridge` docstring for
the integration's scope.
"""

# Eagerly import the two submodules so `import dinos` then
# `from dinos.kerr_corrections import propose_mapping` works just as
# it does with the upstream package.
from . import kerr_corrections, mobius_z3_cover

__all__ = ["kerr_corrections", "mobius_z3_cover"]
