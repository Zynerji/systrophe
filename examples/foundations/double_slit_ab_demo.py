"""Double-slit demo driven by the Aharonov-Bohm phase of a van Stockum cylinder.

Geometry (z-axis = cylinder axis):

       source S = (-D, 0)                       detector P = (+D, y)
                                                       *
       . . . . . . . . . . . . . . . . . . . . . . . . .
                       slit U at (0, +d)
                       slit L at (0, -d)
                       cylinder (frame-dragging K(r)) on the z-axis

The detector amplitude is the sum of two paths,
    psi(y) = exp(i k L_U(y))  +  exp(i k L_L(y) + i phi_AB),
with phi_AB = 2 pi K(r_loop), the Aharonov-Bohm phase enclosed by the
closed loop "upper path minus lower path." The detector intensity is
    I(y) = 2 + 2 cos(k * dL(y) + phi_AB).

Three things are shown:
  1. K = 0 (no cylinder): standard double-slit fringes, centred at y = 0.
  2. K != 0 (cylinder on): fringes rigid-shift by -phi_AB / (k * 2d/D).
  3. Pair extinction at delta = pi: the AB phase of an offset
     SystrophePair vanishes and the fringes snap back to the K=0 pattern.

Run:
    python examples/double_slit_ab_demo.py
"""

from __future__ import annotations

import math

import numpy as np

from systrophe.foundations.aharonov_bohm_ctc import (
    aharonov_bohm_phase,
    enclosed_flux,
    pair_extinction_AB,
)
from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.foundations.which_way_ancilla import (
    englert_relation,
    path_coherence,
    sweep_coupling,
)


def fringe_intensity(
    y: np.ndarray, k: float, d: float, D: float, phi_ab: float,
    gamma: float = 1.0,
) -> np.ndarray:
    """|psi_U + psi_L|^2 for a two-slit geometry with AB phase phi_ab.

    `gamma` in [0, 1] is a phenomenological coherence factor: gamma = 1 is
    full visibility (no which-way info), gamma = 0 is full distinguishability
    (no fringes). It is *not* derived from the AB module; it parametrizes the
    observer-effect collapse rather than explaining it.
    """
    L_U = np.sqrt(D * D + d * d) + np.sqrt(D * D + (y - d) ** 2)
    L_L = np.sqrt(D * D + d * d) + np.sqrt(D * D + (y + d) ** 2)
    delta_phase = k * (L_U - L_L) + phi_ab
    return 2.0 + 2.0 * gamma * np.cos(delta_phase)


def ascii_fringes(y: np.ndarray, I: np.ndarray, width: int = 56) -> str:
    """Render |psi|^2(y) as a vertical ASCII histogram."""
    rows = []
    I_max = 4.0  # exact max of 2 + 2 cos(.)
    for yi, Ii in zip(y, I):
        bar_len = int(round(width * Ii / I_max))
        bar = "#" * bar_len + "." * (width - bar_len)
        rows.append(f"  y={yi:+6.3f}  |{bar}| I={Ii:5.3f}")
    return "\n".join(rows)


def main() -> None:
    # --- Cylinder & slit geometry --------------------------------------
    omega, R = 1.0, 1.0          # supercritical: a = omega R = 1 > 1/2
    vs = VanStockumInterior(omega=omega, R=R)
    r_loop = 2.5 * R             # enclosure radius (outside the cylinder)
    k = 60.0                     # detector "wavenumber" (sets fringe spacing)
    d = 0.30                     # slit half-separation
    D = 5.0                      # source-screen and slit-screen distance

    y = np.linspace(-0.4, 0.4, 25)

    K = enclosed_flux(vs, r_loop)
    phi_ab = aharonov_bohm_phase(vs, r_loop)
    print("Van Stockum cylinder")
    print(f"  omega = {omega}, R = {R}, a = {vs.a}, alpha = {vs.alpha:.4f}")
    print(f"  loop radius r = {r_loop}, K(r) = {K:.6f}")
    print(f"  Aharonov-Bohm phase phi_AB = 2 pi K = {phi_ab:.6f} rad "
          f"(= {phi_ab / math.pi:.4f} pi)")
    print()

    # --- 1. K = 0: cylinder absent -------------------------------------
    print("[1] K = 0 (cylinder off / phi_AB = 0): symmetric fringes about y=0")
    I_off = fringe_intensity(y, k, d, D, phi_ab=0.0)
    print(ascii_fringes(y, I_off))
    peak_off = float(y[int(np.argmax(I_off))])
    print(f"  peak at y = {peak_off:+.4f}")
    print()

    # --- 2. K != 0: AB-shifted fringes ---------------------------------
    print("[2] K != 0 (cylinder on / phi_AB rotates the fringes)")
    I_on = fringe_intensity(y, k, d, D, phi_ab=phi_ab)
    print(ascii_fringes(y, I_on))
    peak_on = float(y[int(np.argmax(I_on))])
    print(f"  peak at y = {peak_on:+.4f}")
    # ΔL(y) ≈ -2dy/D, so peaks satisfy y = D (phi_ab - 2 pi n) / (k * 2d).
    # The visible peak is the (n) that lands inside the screen — pick the
    # principal-branch shift by wrapping phi_ab to (-pi, pi].
    phi_wrapped = ((phi_ab + math.pi) % (2.0 * math.pi)) - math.pi
    expected_shift = phi_wrapped * D / (k * 2.0 * d)
    print(f"  observed shift = {peak_on - peak_off:+.4f}  "
          f"(principal-branch prediction {expected_shift:+.4f})")
    print()

    # --- 3. Pair extinction: delta = pi cancels the AB phase -----------
    pair = pair_extinction_AB(vs, r_loop, delta=math.pi)
    phi_ab_pair = pair["phi_pair"]
    print(f"[3] Anti-phase pair (delta = pi): phi_AB_pair = {phi_ab_pair:.6e} "
          f"(extinction factor = {pair['extinction_factor']:.3f})")
    I_pair = fringe_intensity(y, k, d, D, phi_ab=phi_ab_pair)
    print(ascii_fringes(y, I_pair))
    peak_pair = float(y[int(np.argmax(I_pair))])
    print(f"  peak at y = {peak_pair:+.4f}  "
          f"(matches K=0 baseline {peak_off:+.4f})")
    print()

    # --- 4. AB phase as a function of loop radius ----------------------
    print("[4] AB phase phi_AB(r) = 2 pi K(r) for a few enclosure radii:")
    for r in [1.1 * R, 1.5 * R, 2.0 * R, 3.0 * R, 5.0 * R, 10.0 * R]:
        phi_r = aharonov_bohm_phase(vs, r)
        peak_r = float(y[int(np.argmax(fringe_intensity(y, k, d, D, phi_r)))])
        print(f"  r = {r:5.2f}  phi_AB = {phi_r:+8.4f} rad  "
              f"fringe peak y = {peak_r:+.4f}")

    # --- 5. which-way ancilla: derive visibility from coupling ---------
    print("[5] Which-way ancilla: V from path-detector entanglement")
    print("    coupling kappa  ->  visibility V   distinguishability D   V^2+D^2")
    for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
        rel = englert_relation(kappa)
        print(f"      kappa = {kappa:.2f}        V = {rel['V']:.4f}       "
              f"D = {rel['D']:.4f}        {rel['V2_plus_D2']:.6f}")
    print()

    # --- 6. matplotlib plot --------------------------------------------
    save_path = _try_plot(vs, r_loop, k, d, D, phi_ab, phi_ab_pair)
    if save_path is not None:
        print(f"[6] plot saved to {save_path}")
    else:
        print("[6] matplotlib unavailable; skipping plot")


def _try_plot(
    vs: VanStockumInterior, r_loop: float, k: float, d: float, D: float,
    phi_ab: float, phi_ab_pair: float,
) -> str | None:
    """Render fringes to a PNG. Returns path on success, None if mpl absent."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    from pathlib import Path

    y_fine = np.linspace(-0.4, 0.4, 401)

    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    # (a) three fringe profiles ---------------------------------------
    ax = axes[0]
    ax.plot(y_fine, fringe_intensity(y_fine, k, d, D, 0.0),
            label="K = 0 (cylinder off)", lw=1.6)
    ax.plot(y_fine, fringe_intensity(y_fine, k, d, D, phi_ab),
            label=f"K ≠ 0  (φ$_{{AB}}$ = {phi_ab/math.pi:.3f}π)",
            lw=1.6)
    ax.plot(y_fine, fringe_intensity(y_fine, k, d, D, phi_ab_pair),
            label="pair δ=π  (φ$_{AB}$ → 0)",
            lw=1.6, ls="--")
    ax.set_xlabel("detector position y")
    ax.set_ylabel("|ψ|²")
    ax.set_title("(a) AB-shifted fringes")
    ax.legend(fontsize=8, loc="lower center")
    ax.set_ylim(-0.1, 4.3)

    # (b) AB phase rotates fringes as the loop radius changes ----------
    ax = axes[1]
    r_grid = np.linspace(1.05 * vs.R, 5.0 * vs.R, 240)
    phi_grid = np.array([aharonov_bohm_phase(vs, float(r)) for r in r_grid])
    I_grid = np.array([
        fringe_intensity(y_fine, k, d, D, float(p)) for p in phi_grid
    ])
    im = ax.imshow(
        I_grid, origin="lower", aspect="auto",
        extent=[y_fine[0], y_fine[-1], r_grid[0], r_grid[-1]],
        cmap="magma", vmin=0.0, vmax=4.0,
    )
    ax.set_xlabel("detector position y")
    ax.set_ylabel("loop radius r")
    ax.set_title("(b) fringe map vs enclosure radius")
    fig.colorbar(im, ax=ax, label="|ψ|²")

    # (c) observer effect: V derived from a which-way ancilla coupling --
    ax = axes[2]
    for kappa in (0.0, 0.25, 0.5, 0.75, 1.0):
        V = path_coherence(kappa)
        ax.plot(
            y_fine,
            fringe_intensity(y_fine, k, d, D, phi_ab, gamma=V),
            lw=1.4,
            label=f"κ = {kappa:.2f}  (V = {V:.2f})",
        )
    ax.set_xlabel("detector position y")
    ax.set_ylabel("|ψ|²")
    ax.set_title("(c) ancilla-driven visibility V(κ)")
    ax.legend(fontsize=7, loc="lower center", ncol=3)
    ax.set_ylim(-0.1, 4.3)

    # (d) Englert duality: V, D, V^2 + D^2 vs coupling -----------------
    ax = axes[3]
    sw = sweep_coupling(n=101)
    ax.plot(sw["coupling"], sw["V"], lw=1.6, label="V (visibility)")
    ax.plot(sw["coupling"], sw["D"], lw=1.6, label="D (which-way)")
    ax.plot(sw["coupling"], sw["V2_plus_D2"], lw=1.4, ls="--",
            label="V² + D²")
    ax.set_xlabel("coupling κ")
    ax.set_title("(d) Englert: V² + D² = 1")
    ax.legend(fontsize=8, loc="center right")
    ax.set_ylim(-0.05, 1.1)

    fig.suptitle(
        "Aharonov-Bohm double-slit on a van Stockum cylinder "
        "+ which-way ancilla",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out_path = Path(__file__).with_suffix(".png")
    fig.savefig(out_path, dpi=140)
    plt.close(fig)
    return str(out_path)


if __name__ == "__main__":
    main()
