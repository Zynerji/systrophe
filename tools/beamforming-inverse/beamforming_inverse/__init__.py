"""beamforming-inverse: solve for cylinder complex amplitudes that synthesise a target phasor field.

Priority-2 sibling of implosion-carving. Given a SystropheArray's
fixed cylinder parameters (R_i, a_i, p_i), prescribe a target complex
phasor profile at M sample radii and solve for the N complex
amplitudes c_i = A_i exp(i δ_i) that best synthesise it.

Forward model (from systrophe.geometry.array.SystropheArray.phasor_field):

    z(r) = sum_i  c_i * exp(i α_i ln(r / R_i))

where c_i = A_i e^{i δ_i}. Linear in c, so the inverse is a complex
least-squares (or min-norm) problem.
"""

from __future__ import annotations

from .inverse import (
    BeamformingDesign,
    BeamformingInverseResult,
    build_forward_matrix,
    design_from_array,
    solve_beamforming_inverse,
    synthesised_array,
)

__all__ = [
    "BeamformingDesign",
    "BeamformingInverseResult",
    "build_forward_matrix",
    "design_from_array",
    "solve_beamforming_inverse",
    "synthesised_array",
]
