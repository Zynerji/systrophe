"""Demo: Born-rule witness for Deutsch-CTC fixed-point measurement.

Sweeps a parameterised joint CR (x) CTC unitary family, finds the D-CTC
fixed point for two equiprior non-orthogonal pure states |0> and |+>,
and asks whether the resulting CR-output Helstrom distinguishability
exceeds the input Helstrom bound.

Run:
    python examples/foundations/born_rule_dctc_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from systrophe.foundations.born_rule_dctc import (
    born_rule_witness,
    brun_wilde_unitary,
    cnot_then_hadamard_unitary,
    dctc_output_state,
    hadamard_swap_unitary,
    helstrom_bound_density,
    helstrom_bound_pure,
    mobius_smoke_test,
)


def main() -> None:
    psi_0 = np.array([1.0, 0.0], dtype=complex)
    psi_1 = np.array([1.0, 1.0], dtype=complex) / math.sqrt(2.0)

    P_hel = helstrom_bound_pure(psi_0, psi_1)
    print("Two-state discrimination of |0> vs |+>")
    print(f"  Helstrom bound P_succ = {P_hel:.6f}")
    print()

    print("Sweep parametric + cyclic-power + Haar D-CTC unitary families:")
    w = born_rule_witness(
        psi_0=psi_0, psi_1=psi_1,
        cyclic_ctc_qubits=(2, 3),
        haar_ctc_dim=4, haar_samples=80,
    )
    print(f"  max P_dctc(input -> CR output) = {w.P_dctc_max:.6f}")
    print(f"  family that won                = {w.family}")
    print(f"  ctc_dim of winning U           = {w.ctc_dim}")
    print(f"  best unitary label             = {w.best_unitary}")
    print(f"  margin (P_dctc - P_helstrom)   = {w.margin:+.6f}")
    print(f"  Born violated?                 {w.born_violated}")
    print()
    print("Empirical finding (2026-05-27): neither the parametric, cyclic-power,")
    print("nor Haar-random families found a Born-violator. Typical D-CTC")
    print("output Helstrom <= input Helstrom (the partial trace decoheres).")
    print("Brun-Wilde Theorem 1 still guarantees a violator exists; finding it")
    print("requires the specific Section 5 cyclic-counter U, not a generic")
    print("single-shot 2N-dim unitary. Open follow-up.")
    print()

    # Mobius classical smoke test
    m = mobius_smoke_test()
    print(f"Mobius temporal loop smoke test: available={m['available']}")
    if not m["available"]:
        print(f"  reason: {m.get('error', m.get('note', ''))}")
    else:
        print(f"  converged={m['converged']}, iterations={m['iterations']}")
    print()

    save_path = _make_plot(psi_0, psi_1, P_hel)
    if save_path is not None:
        print(f"plot saved to {save_path}")


def _make_plot(psi_0, psi_1, P_hel) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    thetas = np.linspace(0.0, 2.0 * math.pi, 61)
    p_dctc = []
    sigma_0 = np.outer(psi_0, psi_0.conj())
    sigma_1 = np.outer(psi_1, psi_1.conj())
    for t in thetas:
        U = brun_wilde_unitary(float(t))
        out0 = dctc_output_state(U, sigma_0, dim_cr=2)
        out1 = dctc_output_state(U, sigma_1, dim_cr=2)
        if out0["converged"] and out1["converged"]:
            p_dctc.append(helstrom_bound_density(
                out0["rho_cr_out"], out1["rho_cr_out"],
            ))
        else:
            p_dctc.append(float("nan"))
    p_dctc = np.array(p_dctc)

    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.plot(thetas, p_dctc, lw=1.8, label="P_dctc(theta) (CR-output Helstrom)")
    ax.axhline(P_hel, color="tab:red", ls="--", lw=1.4,
               label=f"Helstrom bound = {P_hel:.4f}")
    ax.axhline(1.0, color="tab:gray", ls=":", lw=1.0,
               label="Brun-Wilde D-CTC ceiling (P = 1)")
    ax.set_xlabel("Brun-Wilde unitary parameter theta")
    ax.set_ylabel("P_succ")
    ax.set_title("Born-rule witness: D-CTC SWAP-family sweep")
    ax.set_ylim(0.4, 1.05)
    ax.legend(fontsize=9, loc="lower right")
    out = Path(__file__).with_suffix(".png")
    fig.tight_layout()
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
