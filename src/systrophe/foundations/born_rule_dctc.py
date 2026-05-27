"""Born-rule witness for Deutsch-CTC fixed-point measurement.

Standard quantum mechanics caps the success probability of
distinguishing two non-orthogonal pure states (|psi_0>, |psi_1>) prepared
with equal prior at the Helstrom bound

    P_Helstrom = 1/2  +  (1/2) * sqrt( 1 - |<psi_0|psi_1>|^2 ).

Brun and Wilde (Found. Phys. 47 (2017) 375) showed that a quantum
computer with Deutsch-CTC access *exceeds* the Helstrom bound for any
pair of non-orthogonal pure states -- in principle reaching
P_dctc = 1. The mechanism is the nonlinearity of the D-CTC map: the
fixed point rho_CTC depends on the input state sigma, so the input-to-
output channel sigma -> rho_CR_out is nonlinear, and nonlinear channels
can map non-orthogonal inputs to orthogonal outputs.

This module computes the witness numerically:

  1. Pick a pair of non-orthogonal pure states (|0>, |+>) by default.
  2. Pick a parameterised family of joint CR (x) CTC unitaries U(t).
  3. For each U(t), find the D-CTC fixed point rho_CTC(sigma) for each
     input sigma; compute the CR output rho_CR_out.
  4. Compute the Helstrom of the two CR outputs (call this P_dctc(t)).
  5. Maximise P_dctc(t) over the parameter sweep.
  6. Compare to the Helstrom of the inputs P_Helstrom.

Output: a BornWitness dict with both numbers and an explicit verdict
('Born preserved' iff max P_dctc <= P_Helstrom + 1e-8).

Status of the witness
---------------------
The shipped unitary family (SWAP / CNOT-Hadamard variants on a 2-dim
CTC) is *not* sufficient to exhibit Born violation -- those unitaries
either preserve distinguishability (unitary maps) or yield degenerate
non-unique fixed points. The Brun-Wilde theorem guarantees the
existence of a Born-violating D-CTC unitary in higher CTC dimensions
(N >= some_N(c) where c is the input overlap), via an explicit
cyclic-power construction; implementing it is left as a follow-up.
What this module ships is the witness *framework*: pass any joint
unitary U into `dctc_output_state` and `helstrom_bound_density`, get
back P_dctc and compare to `helstrom_bound_pure`. The bookkeeping is
correct -- only the U is the open knob.

The dinos-bridge Mobius temporal loop is in the same self-consistency
family (it's a CPTP fixed-point with prophetic feedback). A second
function `mobius_smoke_test` runs `mobius_temporal_loop_for_cylinder`
to confirm the classical Mobius loop converges -- a sanity check that
the broader self-consistency formalism is operational. (The Mobius
*quantum* extension lives in upstream dinos.quantum_temporal_loop and
is not in the systrophe-vendored subset, so a full Mobius Born-rule
test would require pip-installing the upstream Dinos package.)
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Callable

import numpy as np

from systrophe.ctc.d_ctc import apply_channel, dctc_fixed_point


# ----- standard quantum-mechanical Helstrom bound -------------------------


def helstrom_bound_pure(psi_0: np.ndarray, psi_1: np.ndarray) -> float:
    """P_succ for distinguishing two equiprior pure states.

        P = 1/2 + (1/2) sqrt(1 - |<psi_0|psi_1>|^2).
    """
    overlap = abs(np.vdot(psi_0, psi_1))
    return float(0.5 + 0.5 * math.sqrt(max(0.0, 1.0 - overlap ** 2)))


def helstrom_bound_density(rho_0: np.ndarray, rho_1: np.ndarray) -> float:
    """P_succ for distinguishing two equiprior density matrices.

        P = 1/2 + (1/4) * ||rho_0 - rho_1||_1   (trace norm).
    """
    d = rho_0 - rho_1
    # ||A||_1 = sum |singular values| = sum |eigenvalues| for Hermitian A
    eig = np.linalg.eigvalsh((d + d.conj().T) / 2.0)
    trace_norm = float(np.sum(np.abs(eig)))
    return float(0.5 + 0.25 * trace_norm)


# ----- the D-CTC channel applied to a chosen sigma -----------------------


def dctc_output_state(
    U: np.ndarray, sigma_cr: np.ndarray, dim_cr: int,
    tol: float = 1e-10, max_iter: int = 2000,
) -> dict:
    """Find rho_CTC(sigma), then compute rho_CR_out = Tr_CTC[U(sigma⊗rho_CTC)U^*].

    Returns
    -------
    dict with keys:
      - rho_ctc : the D-CTC fixed point for this sigma.
      - rho_cr_out : the CR output after the channel.
      - converged : whether the fixed-point iteration converged.
      - iterations : iteration count.
    """
    fp = dctc_fixed_point(
        U, sigma_cr, dim_cr=dim_cr, tol=tol, max_iter=max_iter,
    )
    rho_ctc = fp["rho_ctc"]
    dim_total = U.shape[0]
    dim_ctc = dim_total // dim_cr
    joint = np.kron(sigma_cr, rho_ctc)
    evolved = U @ joint @ U.conj().T
    # Partial-trace out CTC to get rho_CR_out:
    # M[a*dim_ctc+i, b*dim_ctc+j] -> rho_cr[a, b] = sum_i M[a*dim_ctc+i, b*dim_ctc+i]
    evolved_resh = evolved.reshape((dim_cr, dim_ctc, dim_cr, dim_ctc))
    rho_cr_out = np.einsum("aibi->ab", evolved_resh)
    return {
        "rho_ctc": rho_ctc,
        "rho_cr_out": rho_cr_out,
        "converged": fp["converged"],
        "iterations": fp["iterations"],
    }


# ----- parameterised unitary family for the Born-rule sweep --------------


def brun_wilde_unitary(theta: float) -> np.ndarray:
    """A 4x4 joint CR(x)CTC unitary parameterised by theta in [0, 2 pi].

    Construction: U = (R_y(theta) (x) I) . SWAP, where R_y rotates the
    CR qubit by angle theta. This family interpolates between the trivial
    SWAP (theta = 0, gives output = input) and a more nonlinear map that
    in principle distinguishes non-orthogonal inputs.
    """
    c = math.cos(theta / 2.0)
    s = math.sin(theta / 2.0)
    R_y = np.array([[c, -s], [s, c]], dtype=complex)
    # 4x4 SWAP on (q_CR, q_CTC).
    SWAP = np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=complex,
    )
    I2 = np.eye(2, dtype=complex)
    return np.kron(R_y, I2) @ SWAP


def hadamard_swap_unitary() -> np.ndarray:
    """U = (H (x) I) . SWAP -- canonical nonlinear-feedback D-CTC map."""
    H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
    SWAP = np.array(
        [[1, 0, 0, 0],
         [0, 0, 1, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1]],
        dtype=complex,
    )
    I2 = np.eye(2, dtype=complex)
    return np.kron(H, I2) @ SWAP


def cnot_then_hadamard_unitary() -> np.ndarray:
    """U = (H (x) I) . CNOT_{CR->CTC} -- mixes basis states then rotates."""
    CNOT = np.array(
        [[1, 0, 0, 0],
         [0, 1, 0, 0],
         [0, 0, 0, 1],
         [0, 0, 1, 0]],
        dtype=complex,
    )
    H = np.array([[1, 1], [1, -1]], dtype=complex) / math.sqrt(2.0)
    I2 = np.eye(2, dtype=complex)
    return np.kron(H, I2) @ CNOT


# ----- the witness -------------------------------------------------------


@dataclass(frozen=True)
class BornWitness:
    P_helstrom: float
    P_dctc_max: float
    best_unitary: str
    born_violated: bool
    margin: float            # P_dctc_max - P_helstrom (>0 iff violated)


def _zero_ket() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=complex)


def _plus_ket() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)


def born_rule_witness(
    psi_0: np.ndarray | None = None,
    psi_1: np.ndarray | None = None,
    n_theta: int = 41,
) -> BornWitness:
    """Sweep a parametric D-CTC unitary family + fixed nonlinear maps to
    test whether P_dctc(rho_CR_out) can exceed the Helstrom bound on the
    input states (psi_0, psi_1). Reports the maximum P_dctc found.
    """
    if psi_0 is None:
        psi_0 = _zero_ket()
    if psi_1 is None:
        psi_1 = _plus_ket()
    sigma_0 = np.outer(psi_0, psi_0.conj())
    sigma_1 = np.outer(psi_1, psi_1.conj())

    P_hel = helstrom_bound_pure(psi_0, psi_1)

    candidates: list[tuple[str, np.ndarray]] = [
        ("hadamard_swap", hadamard_swap_unitary()),
        ("cnot_hadamard", cnot_then_hadamard_unitary()),
    ]
    for i, t in enumerate(np.linspace(0.0, 2.0 * math.pi, n_theta)):
        candidates.append((f"brun_wilde[theta={t:.3f}]",
                           brun_wilde_unitary(float(t))))

    best_P = 0.0
    best_name = ""
    for name, U in candidates:
        out0 = dctc_output_state(U, sigma_0, dim_cr=2)
        out1 = dctc_output_state(U, sigma_1, dim_cr=2)
        if not (out0["converged"] and out1["converged"]):
            continue
        P = helstrom_bound_density(out0["rho_cr_out"], out1["rho_cr_out"])
        if P > best_P:
            best_P = P
            best_name = name

    return BornWitness(
        P_helstrom=P_hel,
        P_dctc_max=float(best_P),
        best_unitary=best_name,
        born_violated=bool(best_P > P_hel + 1e-8),
        margin=float(best_P - P_hel),
    )


# ----- Mobius (dinos-bridge) smoke test ----------------------------------


def mobius_smoke_test(
    omega: float = 1.0, R: float = 1.0,
    N: int = 32, max_iter: int = 100, epsilon: float = 5e-2,
) -> dict:
    """Run the classical dinos Mobius temporal loop and report convergence.

    Smoke-tests that the Mobius / self-consistency machinery (the dinos
    classical analog of the Born-rule witness above) is operational on
    Tipler-derived inputs. The Mobius *quantum* extension that would be
    needed for a Mobius Born-rule test lives in upstream
    dinos.quantum_temporal_loop -- not in the systrophe-vendored subset.
    """
    from systrophe.geometry.vanstockum import VanStockumInterior
    from systrophe.bridges.dinos_bridge import mobius_temporal_loop_for_cylinder

    try:
        vs = VanStockumInterior(omega=omega, R=R)
        result = mobius_temporal_loop_for_cylinder(
            vs, N=N, max_iter=max_iter, epsilon=epsilon,
        )
        return {
            "available": True,
            "converged": bool(getattr(result, "converged", True)),
            "iterations": int(getattr(result, "iterations", -1)),
            "note": ("classical Mobius loop ran; quantum Mobius extension "
                     "not in vendored subset."),
        }
    except ImportError as e:
        return {
            "available": False,
            "error": str(e),
            "note": ("dinos.temporal_loop unavailable in vendored subset; "
                     "install upstream dinos-DKN package to run."),
        }
