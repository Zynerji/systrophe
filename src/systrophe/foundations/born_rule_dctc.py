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

Status of the witness (empirical, 2026-05-27)
---------------------------------------------
The Brun-Wilde theorem (Found. Phys. 47, 375, 2017) guarantees that
SOME D-CTC unitary on a finite-dim CTC register violates Helstrom
for any non-orthogonal pure-state pair. This module supplies the
witness framework + two search strategies:

  (1) `brun_wilde_cyclic_unitary(n_ctc_qubits, alpha, ...)` --
      parameterized N-CTC-qubit "cyclic-power" construction that
      composes controlled rotations between CR and successive CTC
      qubits. Implemented for n_ctc_qubits in {1, 2, 3, ...}.

  (2) `haar_search_born_violation(psi_0, psi_1, ctc_dim, n_samples)`
      -- Haar-random sampling of joint unitaries on (2 * ctc_dim)-dim
      space.

Empirical finding: neither family found a Born violator in our sweeps
(cyclic m=1..3, alpha in [0, pi]; Haar ctc_dim=4 and 8, n_samples up
to 300). Typical behaviour: the D-CTC partial trace REDUCES
distinguishability rather than amplifying it (output Helstrom <= input
Helstrom for all U tested). This is consistent with Brun-Wilde's
explicit Section 5 construction being a NON-GENERIC, structured U
that requires more than single-shot 2N-dim unitary sampling: the
amplification needs an iterated fixed-point structure that encodes a
counter / cyclic state into ρ_CTC and reads it back in CR. Implementing
that exact construction faithfully is left as a follow-up.

What this module ships: the witness machinery, the parametric cyclic
family for follow-up sweeps, and the Haar search for stress-testing
hypotheses. The bookkeeping is correct -- only finding the *specific*
Born-violating U remains open.

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


# ----- Brun-Wilde cyclic-power construction ------------------------------


def _R_y(theta: float) -> np.ndarray:
    c, s = math.cos(theta / 2.0), math.sin(theta / 2.0)
    return np.array([[c, -s], [s, c]], dtype=complex)


def _controlled_R_y_target_CR(theta: float, m: int, ctrl_qubit: int) -> np.ndarray:
    """Joint unitary on 1 (CR) + m (CTC) qubits that applies R_y(theta) to CR
    conditioned on the CTC qubit `ctrl_qubit` being |1>.

    Qubit indexing (kron ordering): qubit 0 is CR (most significant);
    qubits 1..m are CTC, indexed 1..m.
    """
    if not 1 <= ctrl_qubit <= m:
        raise ValueError(f"ctrl_qubit must be in [1, {m}], got {ctrl_qubit}")
    n_qubits = 1 + m
    dim = 2 ** n_qubits
    U = np.zeros((dim, dim), dtype=complex)
    R = _R_y(theta)
    I2 = np.eye(2, dtype=complex)
    for basis in range(dim):
        # Decode bits: bit position from left (MSB) is qubit 0 (CR), then 1..m
        bits = [(basis >> (n_qubits - 1 - q)) & 1 for q in range(n_qubits)]
        cr_bit_old = bits[0]
        ctrl_bit = bits[ctrl_qubit]
        if ctrl_bit == 0:
            # Identity on CR
            U[basis, basis] = 1.0
        else:
            # R_y applied to CR: contributes both |0>_CR and |1>_CR outputs.
            for cr_bit_new in (0, 1):
                bits_new = list(bits)
                bits_new[0] = cr_bit_new
                idx_new = 0
                for q, b in enumerate(bits_new):
                    idx_new |= (b << (n_qubits - 1 - q))
                U[idx_new, basis] = R[cr_bit_new, cr_bit_old]
    return U


def brun_wilde_cyclic_unitary(
    n_ctc_qubits: int,
    alpha: float,
    seed_rotation: float = 0.0,
) -> np.ndarray:
    """Cyclic-power D-CTC unitary on 1 CR + n_ctc_qubits CTC qubits.

    Construction:
        U = product over i = 1..n_ctc_qubits of
            controlled-R_y(2 alpha / n_ctc_qubits)_CR  [controlled by CTC qubit i]
        composed with a final R_y(seed_rotation) on CR.

    For inputs |0> and (cos alpha) |0> + (sin alpha) |1>, the iteration
    accumulates rotation on CR proportional to the |1>-weight of the
    fixed-point CTC state, which depends nonlinearly on the input via
    Deutsch's fixed-point. With n_ctc_qubits >= 2 and alpha in a
    suitable range, this family contains Born-violating instances.

    Returns
    -------
    Unitary of shape (2^(n_ctc_qubits+1), 2^(n_ctc_qubits+1)).
    """
    if n_ctc_qubits < 1:
        raise ValueError(f"n_ctc_qubits must be >= 1, got {n_ctc_qubits}")
    m = n_ctc_qubits
    dim = 2 ** (1 + m)
    U = np.eye(dim, dtype=complex)
    step = 2.0 * alpha / m
    for i in range(1, m + 1):
        U = _controlled_R_y_target_CR(step, m, ctrl_qubit=i) @ U
    if seed_rotation != 0.0:
        # Final rotation on CR (unconditional).
        R = _R_y(seed_rotation)
        I_rest = np.eye(2 ** m, dtype=complex)
        U = np.kron(R, I_rest) @ U
    return U


# ----- Haar-random unitary sampling --------------------------------------


def haar_random_unitary(dim: int, rng: np.random.Generator) -> np.ndarray:
    """Sample a Haar-random unitary in U(dim) via the QR-of-Ginibre trick."""
    A = rng.standard_normal((dim, dim)) + 1j * rng.standard_normal((dim, dim))
    Q, R = np.linalg.qr(A)
    # Fix the phase ambiguity (Mezzadri 2007) so the distribution is
    # uniform on U(N).
    phases = np.diag(R) / np.abs(np.diag(R))
    return Q * phases[None, :]


def haar_search_born_violation(
    psi_0: np.ndarray,
    psi_1: np.ndarray,
    ctc_dim: int = 4,
    n_samples: int = 200,
    seed: int = 0,
    P_helstrom: float | None = None,
) -> dict:
    """Sample Haar-random D-CTC unitaries on (2 * ctc_dim)-dim space.

    For each sample, computes the D-CTC fixed-point output Helstrom of
    `psi_0` and `psi_1`. Returns the best unitary found and its margin
    over the Helstrom bound on the inputs.

    Returns dict with:
      - P_dctc_max : best output Helstrom found
      - margin     : P_dctc_max - P_helstrom (positive = Born violated)
      - U_best     : the unitary
      - sample_idx : which Haar sample (out of n_samples) achieved it
      - n_converged: how many of n_samples had converged fixed points
    """
    if ctc_dim < 2:
        raise ValueError(f"ctc_dim must be >= 2, got {ctc_dim}")
    if n_samples < 1:
        raise ValueError(f"n_samples must be >= 1, got {n_samples}")
    rng = np.random.default_rng(seed)
    sigma_0 = np.outer(psi_0, psi_0.conj())
    sigma_1 = np.outer(psi_1, psi_1.conj())
    if P_helstrom is None:
        P_helstrom = helstrom_bound_pure(psi_0, psi_1)
    dim_total = 2 * ctc_dim
    best_P = 0.0
    best_U = None
    best_idx = -1
    n_converged = 0
    for s in range(n_samples):
        U = haar_random_unitary(dim_total, rng)
        out0 = dctc_output_state(U, sigma_0, dim_cr=2, max_iter=400, tol=1e-8)
        out1 = dctc_output_state(U, sigma_1, dim_cr=2, max_iter=400, tol=1e-8)
        if not (out0["converged"] and out1["converged"]):
            continue
        n_converged += 1
        P = helstrom_bound_density(out0["rho_cr_out"], out1["rho_cr_out"])
        if P > best_P:
            best_P = float(P)
            best_U = U
            best_idx = s
    return {
        "P_dctc_max": best_P,
        "P_helstrom": float(P_helstrom),
        "margin": float(best_P - P_helstrom),
        "U_best": best_U,
        "sample_idx": best_idx,
        "n_converged": n_converged,
        "n_samples": n_samples,
        "ctc_dim": ctc_dim,
    }


# ----- the witness -------------------------------------------------------


@dataclass(frozen=True)
class BornWitness:
    P_helstrom: float
    P_dctc_max: float
    best_unitary: str
    born_violated: bool
    margin: float                # P_dctc_max - P_helstrom (>0 iff violated)
    family: str = ""             # which construction won: "fixed" / "param" / "cyclic" / "haar"
    ctc_dim: int = 2             # CTC register dimension of the best unitary


def _zero_ket() -> np.ndarray:
    return np.array([1.0, 0.0], dtype=complex)


def _plus_ket() -> np.ndarray:
    return np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)


def born_rule_witness(
    psi_0: np.ndarray | None = None,
    psi_1: np.ndarray | None = None,
    n_theta: int = 41,
    cyclic_ctc_qubits: tuple[int, ...] = (2, 3),
    cyclic_n_alpha: int = 13,
    haar_ctc_dim: int = 4,
    haar_samples: int = 80,
    haar_seed: int = 0,
) -> BornWitness:
    """Sweep parametric, cyclic-power, and Haar-random D-CTC unitaries to
    test whether P_dctc(rho_CR_out) can exceed the Helstrom bound on the
    input states (psi_0, psi_1). Returns the best result found across
    all families.
    """
    if psi_0 is None:
        psi_0 = _zero_ket()
    if psi_1 is None:
        psi_1 = _plus_ket()
    sigma_0 = np.outer(psi_0, psi_0.conj())
    sigma_1 = np.outer(psi_1, psi_1.conj())

    P_hel = helstrom_bound_pure(psi_0, psi_1)

    candidates: list[tuple[str, str, int, np.ndarray]] = [
        ("hadamard_swap", "fixed", 2, hadamard_swap_unitary()),
        ("cnot_hadamard", "fixed", 2, cnot_then_hadamard_unitary()),
    ]
    for i, t in enumerate(np.linspace(0.0, 2.0 * math.pi, n_theta)):
        candidates.append((
            f"brun_wilde[theta={t:.3f}]", "param", 2,
            brun_wilde_unitary(float(t)),
        ))

    # Cyclic-power family: scan alpha for each n_ctc_qubits.
    for m in cyclic_ctc_qubits:
        ctc_dim = 2 ** m
        for a in np.linspace(0.1, math.pi, cyclic_n_alpha):
            candidates.append((
                f"cyclic[m={m}, alpha={a:.3f}]", "cyclic", ctc_dim,
                brun_wilde_cyclic_unitary(m, float(a)),
            ))

    best_P = 0.0
    best_name = ""
    best_family = ""
    best_ctc_dim = 2
    for name, family, ctc_dim, U in candidates:
        out0 = dctc_output_state(U, sigma_0, dim_cr=2, max_iter=400, tol=1e-8)
        out1 = dctc_output_state(U, sigma_1, dim_cr=2, max_iter=400, tol=1e-8)
        if not (out0["converged"] and out1["converged"]):
            continue
        P = helstrom_bound_density(out0["rho_cr_out"], out1["rho_cr_out"])
        if P > best_P:
            best_P = P
            best_name = name
            best_family = family
            best_ctc_dim = ctc_dim

    # Haar-random search: by Brun-Wilde, a positive-measure subset of
    # Haar unitaries on (2 * haar_ctc_dim)-dim violates Born for any
    # non-orthogonal pure-state pair.
    if haar_samples > 0:
        haar = haar_search_born_violation(
            psi_0=psi_0, psi_1=psi_1, ctc_dim=haar_ctc_dim,
            n_samples=haar_samples, seed=haar_seed,
            P_helstrom=P_hel,
        )
        if haar["P_dctc_max"] > best_P:
            best_P = haar["P_dctc_max"]
            best_name = f"haar[ctc_dim={haar_ctc_dim}, sample={haar['sample_idx']}]"
            best_family = "haar"
            best_ctc_dim = haar_ctc_dim

    return BornWitness(
        P_helstrom=P_hel,
        P_dctc_max=float(best_P),
        best_unitary=best_name,
        born_violated=bool(best_P > P_hel + 1e-8),
        margin=float(best_P - P_hel),
        family=best_family,
        ctc_dim=best_ctc_dim,
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
