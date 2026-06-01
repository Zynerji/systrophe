"""Generate the superradiance whitepaper figure from real batch-8 data."""
from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.superradiance import band_superradiance_at
from systrophe.qftcs.penrose_extraction import ergosurface_radii

HERE = Path(__file__).parent
RES = HERE.parent.parent / "experiments" / "marrakesh" / "results"
SHOTS = 8192

hw = json.load(open(RES / "marrakesh_batch8_hw_analysis.json"))
sim = json.load(open(RES / "marrakesh_batch8_sim_analysis.json"))
sd = {s["label"]: s for s in sim["per_circuit"]}

VS = VanStockumInterior(omega=2.0, R=1.0)
ergs = ergosurface_radii(VS, 1.05, 5.0, 4001)
r_erg = min(ergs, key=lambda e: abs(e - 3.0))

# Predicted P11(r) and F(r) on a fine grid
rs = np.linspace(2.0, 4.0, 400)
p11 = np.array([band_superradiance_at(VS, float(r)).pair_probability for r in rs])
Fs = np.array([float(VS.analytic_exterior_F(r)) for r in rs])

rows = hw["per_circuit"]
r_pts = np.array([s["r"] for s in rows])
F_pts = np.array([s["F"] for s in rows])
pred = np.array([s["P11_predicted"] for s in rows])
hwv = np.array([s["P11_observed"] for s in rows])
simv = np.array([sd[s["label"]]["P11_observed"] for s in rows])
ergo = F_pts < 0
err = np.sqrt(hwv * (1 - hwv) / SHOTS)

plt.rcParams.update({"font.size": 10, "axes.grid": True, "grid.alpha": 0.25})
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

# Panel (a): P11(r) with F-sign shading
ax1.axvspan(2.0, r_erg, color="tab:red", alpha=0.07)
ax1.axvspan(r_erg, 4.0, color="tab:blue", alpha=0.07)
ax1.plot(rs, p11, "k-", lw=1.4, label=r"prediction $\sin^2\theta(r)$")
ax1.errorbar(r_pts[ergo], hwv[ergo], yerr=err[ergo], fmt="o", color="tab:red",
             ms=6, capsize=2, label="hardware (F<0, ergoregion)")
ax1.errorbar(r_pts[~ergo], hwv[~ergo], yerr=err[~ergo], fmt="s", color="tab:blue",
             ms=6, capsize=2, label="hardware (F>0, passive)")
ax1.plot(r_pts, simv, "x", color="gray", ms=6, label="noiseless sim")
ax1.axvline(r_erg, color="k", ls=":", lw=1)
loc = hw["ergosurface_localization"]["estimated_transition_centre"]
ax1.axvline(loc, color="tab:green", ls="--", lw=1.2,
            label=f"catcher onset r={loc:.2f}")
ax1.annotate("ergosurface\n$F=0$", xy=(r_erg, 0.06), xytext=(r_erg + 0.18, 0.075),
             fontsize=8, arrowprops=dict(arrowstyle="->", lw=0.8))
ax1.set_xlabel(r"radius $r$  (units of $R$)")
ax1.set_ylabel(r"pair-creation probability $P_{11}$")
ax1.set_title(r"(a) Superradiant emission vs. radius")
ax1.set_xlim(2.0, 4.0); ax1.set_ylim(-0.005, 0.12)
ax1.legend(fontsize=7.5, loc="upper right")

# twin axis: F(r)
axt = ax1.twinx()
axt.plot(rs, Fs, color="tab:purple", lw=1.0, alpha=0.5)
axt.axhline(0, color="tab:purple", lw=0.6, ls="-", alpha=0.4)
axt.set_ylabel(r"$F(r)=g_{tt}$", color="tab:purple")
axt.tick_params(axis="y", labelcolor="tab:purple")
axt.grid(False)

# Panel (b): measured vs predicted
lim = 0.115
ax2.plot([0, lim], [0, lim], "k--", lw=1, alpha=0.6, label="$y=x$")
ax2.errorbar(pred[ergo], hwv[ergo], yerr=err[ergo], fmt="o", color="tab:red",
             ms=6, capsize=2, label="ergoregion (F<0)")
ax2.errorbar(pred[~ergo], hwv[~ergo], yerr=err[~ergo], fmt="s", color="tab:blue",
             ms=6, capsize=2, label="passive (F>0)")
ax2.set_xlabel(r"Bogoliubov prediction $\sin^2\theta$")
ax2.set_ylabel(r"hardware $P_{11}$ (ibm\_marrakesh)")
ax2.set_title(r"(b) Hardware vs. prediction")
ax2.set_xlim(-0.005, lim); ax2.set_ylim(-0.005, lim)
ax2.legend(fontsize=8, loc="upper left")
sep = hw["group_separation"]
ax2.text(0.055, 0.012,
         f"gap = {sep['observed_gap']:.3f}\nsurrogate $p$ = {sep['surrogate_p_value']:.4f}\n"
         f"max|obs$-$pred| = {hw['max_abs_error_vs_prediction']:.4f}",
         fontsize=7.5, bbox=dict(boxstyle="round", fc="white", ec="gray", alpha=0.9))

fig.tight_layout()
out = HERE / "superradiance_batch8.png"
fig.savefig(out, dpi=220, bbox_inches="tight")
print("wrote", out)
