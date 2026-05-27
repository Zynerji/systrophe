"""Demo: ER=EPR throat width vs which-way fringe visibility.

Walks the Werner-pair parameter w from 0 (no throat) to 1 (perfect throat),
computes the post-throat which-way visibility, and tests three candidate
upper bounds (A: naive Bekenstein vN, B: E_F-based, C: Englert-Bekenstein
hybrid).

Run:
    python examples/foundations/erepr_throat_visibility_demo.py
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

from systrophe.foundations.erepr_throat_visibility import (
    BOUNDS,
    evaluate_bounds,
    post_throat_visibility,
    throat_area_proxy,
    werner_concurrence,
    werner_entanglement_of_formation,
    werner_vn_entropy,
)


def main() -> None:
    print("ER=EPR throat / Werner-pair scan")
    print()
    print(" w     C(w)    S_vN     E_F      A(E_F)   V(theta=pi)")
    for w in np.linspace(0.0, 1.0, 11):
        C = werner_concurrence(float(w))
        SvN = werner_vn_entropy(float(w))
        EF = werner_entanglement_of_formation(float(w))
        A = throat_area_proxy(float(w), measure="E_F")
        V = post_throat_visibility(float(w), math.pi)
        print(f" {w:.2f}  {C:.4f}  {SvN:.4f}  {EF:.4f}  {A:.4f}   {V:.4f}")
    print()

    print("Bound verdicts on 21 x 21 grid over (w, theta) in [0,1] x [0, pi]:")
    res = evaluate_bounds()
    for name, v in res.verdicts.items():
        wW, tW = v.where_overshoot_max
        print(f"  {name:14s}  holds = {v.holds!s:5s}   "
              f"max overshoot = {v.max_overshoot:+.6f}   "
              f"worst at (w={wW:.2f}, theta={tW:.3f})")
    print()

    save_path = _make_plot(res)
    if save_path is not None:
        print(f"plot saved to {save_path}")
    else:
        print("matplotlib unavailable; skipping plot")


def _make_plot(res) -> str | None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    w, theta = res.w_grid, res.theta_grid
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))

    # (a) V_post(w, theta)
    ax = axes[0]
    im = ax.imshow(
        res.V_post, origin="lower", aspect="auto",
        extent=[theta[0], theta[-1], w[0], w[-1]],
        cmap="viridis", vmin=0.0, vmax=1.0,
    )
    ax.set_xlabel("theta")
    ax.set_ylabel("Werner w")
    ax.set_title("(a) V_post(w, theta)")
    fig.colorbar(im, ax=ax, label="V_post")

    # (b)-(d) overshoot heatmaps for each bound
    for i, (name, fn) in enumerate(BOUNDS.items(), start=1):
        ax = axes[i]
        Vb = np.array([[fn(float(ww), float(tt)) for tt in theta] for ww in w])
        over = res.V_post - Vb
        im = ax.imshow(
            over, origin="lower", aspect="auto",
            extent=[theta[0], theta[-1], w[0], w[-1]],
            cmap="RdBu_r",
            vmin=-max(abs(over).max(), 1e-6),
            vmax=max(abs(over).max(), 1e-6),
        )
        ax.set_xlabel("theta")
        ax.set_ylabel("Werner w")
        ax.set_title(f"({chr(ord('a')+i)}) V_post - bound[{name}]")
        fig.colorbar(im, ax=ax, label="overshoot")

    fig.suptitle(
        "ER=EPR throat conjecture: V_post vs three candidate "
        "upper bounds (red = bound violated)",
    )
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = Path(__file__).with_suffix(".png")
    fig.savefig(out, dpi=140)
    plt.close(fig)
    return str(out)


if __name__ == "__main__":
    main()
