"""The A/B harness. Two questions, decided with matched reservoirs:

  Q1  does NONLINEARITY add real compute?  (linear vs nonlinear)
  Q2  does helical/Mobius STRUCTURE beat plain-random connectivity?

Every arm shares size, spectral radius, leak, input scaling, ridge readout,
seed, and task data — only the variable-under-test differs.
"""
from __future__ import annotations
import numpy as np

from .reservoir import run_reservoir
from .tasks import narma10, parity_task, memory_capacity
from .readout import ridge_fit, ridge_pred, nrmse, address_lambda2

ARMS = [
    ("linear-random", "random", False),
    ("nonlinear-random", "random", True),
    ("nonlinear-helical", "helical", True),
    ("nonlinear-mobius", "mobius", True),
]


def evaluate(n=300, seeds=5, T=4200, split=2600, wash=200):
    """Return {arm: {metric: [per-seed values]}}."""
    res = {a[0]: {"narma": [], "p3": [], "p5": [], "mc": [], "l2": []} for a in ARMS}
    for s in range(seeds):
        rng = np.random.default_rng(100 + s)
        u_n, y_n = narma10(T, rng)
        u3, y3 = parity_task(T, 3, rng)
        u5, y5 = parity_task(T, 5, rng)
        u_m = rng.random((T, 1)) * 1.6 - 0.8
        for name, kind, nl in ARMS:
            def drive(u):
                return run_reservoir(u, n, kind, nl, np.random.default_rng(7000 + s))
            Xn, X3, X5, Xm = drive(u_n), drive(u3), drive(u5), drive(u_m)
            beta = ridge_fit(Xn[wash:split], y_n[wash:split])
            res[name]["narma"].append(nrmse(ridge_pred(Xn[split:], beta)[:, 0], y_n[split:, 0]))
            for tag, X, y in (("p3", X3, y3), ("p5", X5, y5)):
                beta = ridge_fit(X[wash:split], y[wash:split])
                pred = np.sign(ridge_pred(X[split:], beta)[:, 0])
                res[name][tag].append(float(np.mean(pred == y[split:, 0])))
            res[name]["mc"].append(
                memory_capacity(Xm[:split], Xm[split:], u_m[:split], u_m[split:], wash))
            res[name]["l2"].append(address_lambda2(Xn, wash, rng=rng))
    return res


def format_table(res) -> str:
    def ms(v):
        return f"{np.mean(v):.3f}+-{np.std(v):.3f}"
    lines = [
        f"{'arm':<20}{'NARMA10 NRMSE':>16}{'parity3':>12}{'parity5':>12}{'lin MC':>10}{'addr_lam2':>12}",
        "-" * 82,
    ]
    for name in res:
        d = res[name]
        lines.append(f"{name:<20}{ms(d['narma']):>16}{ms(d['p3']):>12}"
                     f"{ms(d['p5']):>12}{ms(d['mc']):>10}{ms(d['l2']):>12}")
    lines.append("\nlower NRMSE = better; parity 0.5 = chance, 1.0 = solved")
    return "\n".join(lines)
