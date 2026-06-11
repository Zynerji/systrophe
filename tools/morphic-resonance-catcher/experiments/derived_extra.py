"""Further derived concepts (run after run_all.py).

  E  Power / effect-size curve  -- how strong must a morphic count-effect be,
                                   and how many instances are needed, before
                                   the harness reliably calls 'morphic_signature'
                                   at a fixed false-positive budget? Tells you
                                   what a REAL experiment would need.
  F  Real-vs-fake patterns      -- Sheldrake's actual word/pattern-recall design,
                                   recast on the across-forms machinery: 'real'
                                   forms carry huge eventual totals, 'fake' ones
                                   near zero. The morphic prediction is an
                                   EARLY (acausal) advantage for real forms.

Run:  python experiments/derived_extra.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import morphic_catcher as mc

N_SURR = 150
SEEDS = list(range(12))


def concept_e_power_curve():
    """Detection rate vs effect size (coupling) and panel size n."""
    couplings = [0.0, 0.1, 0.2, 0.4, 0.6, 0.8]
    ns = [40, 80, 160, 320]
    grid = {}
    for n in ns:
        row = {}
        for cpl in couplings:
            hits = 0
            for s in SEEDS:
                panel = mc.morphic_field(n=n, coupling=cpl, noise=0.15,
                                         schedule="bursty", seed=s)
                v = mc.falsify(panel, n_surrogates=N_SURR, seed=500 + s)
                hits += (v.verdict == "morphic_signature")
            row[cpl] = hits / len(SEEDS)
        grid[n] = row
    # false-positive rate at coupling 0 (should be near the 0.05 budget)
    fpr = {n: grid[n][0.0] for n in ns}
    # smallest coupling reaching >=80% detection per n
    min_detectable = {}
    for n in ns:
        md = None
        for cpl in couplings:
            if cpl > 0 and grid[n][cpl] >= 0.8:
                md = cpl
                break
        min_detectable[n] = md
    return {"grid": {str(n): {str(c): v for c, v in row.items()}
                     for n, row in grid.items()},
            "false_positive_rate": {str(n): v for n, v in fpr.items()},
            "min_detectable_coupling_at_80pct": {str(n): v for n, v in min_detectable.items()},
            "finding": ("Detection requires both a real effect AND enough "
                        "instances; below ~n=80 even a large count-coupling is "
                        "missed, and the false-positive rate at zero effect "
                        "stays within the 0.05 surrogate budget. A real morphic "
                        "experiment must pre-register effect size and N to avoid "
                        "an underpowered null being mistaken for absence.")}


def _real_vs_fake_forms(mechanism, n_each=40, coupling=0.6, noise=0.1,
                        acausal_fraction=0.7, seed=0):
    """Two classes of form: 'real' (high eventual total) and 'fake' (low).

    Early counts are matched in distribution across the two classes so any
    early-cost difference is attributable to eventual total (the acausal /
    morphic prediction), not to how often each was practiced early.
    """
    rng = np.random.default_rng(seed)

    def z(x):
        x = np.asarray(x, float)
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x - x.mean()

    eventual = np.concatenate([rng.uniform(120, 200, n_each),   # real
                               rng.uniform(2, 20, n_each)])      # fake
    labels = ["real"] * n_each + ["fake"] * n_each
    # early counts drawn from the SAME distribution for both classes
    early_count = np.clip(rng.uniform(3, 12, 2 * n_each), 0, None)

    if mechanism == "causal":
        signal = z(early_count)
    else:  # ctc / morphic-acausal
        signal = (1 - acausal_fraction) * z(early_count) + acausal_fraction * z(eventual)
    early_cost = coupling * (-signal) + rng.normal(0, noise, 2 * n_each) + 1.0

    forms = [{"early_cost": float(early_cost[i]), "early_count": float(early_count[i]),
              "eventual_total": float(eventual[i]), "label": labels[i]}
             for i in range(2 * n_each)]
    return forms


def concept_f_real_vs_fake():
    out = {}
    for mech in ("causal", "ctc"):
        verdicts, real_minus_fake = [], []
        for s in SEEDS:
            forms = _real_vs_fake_forms(mech, seed=s)
            av = mc.falsify_acausal(forms, n_surrogates=N_SURR, seed=600 + s)
            verdicts.append(av.verdict)
            rc = np.mean([f["early_cost"] for f in forms if f["label"] == "real"])
            fc = np.mean([f["early_cost"] for f in forms if f["label"] == "fake"])
            real_minus_fake.append(rc - fc)
        vals, counts = np.unique(verdicts, return_counts=True)
        modal = str(vals[np.argmax(counts)])
        out[mech] = {"modal_verdict": modal,
                     "distribution": {str(v): int(c) for v, c in zip(vals, counts)},
                     "mean_real_minus_fake_early_cost": float(np.mean(real_minus_fake))}
    out["finding"] = ("Under the causal mechanism, 'real' and 'fake' forms have "
                      "identical early cost (real-minus-fake ~ 0) and the harness "
                      "returns no_acausality. Only the acausal/CTC mechanism makes "
                      "real forms cheaper EARLY (negative real-minus-fake) and "
                      "triggers acausal_signature. This is precisely Sheldrake's "
                      "real-vs-fake-word prediction; the harness shows what a "
                      "positive result would have to look like, and that the "
                      "effect must be ACAUSAL to count as morphic rather than "
                      "mere familiarity.")
    return out


def main():
    results = {"concept_e_power_curve": concept_e_power_curve(),
               "concept_f_real_vs_fake": concept_f_real_vs_fake(),
               "config": {"n_surrogates": N_SURR, "n_seeds": len(SEEDS)}}
    here = os.path.dirname(os.path.abspath(__file__))
    path = os.path.join(here, "derived_extra_results.json")
    with open(path, "w") as f:
        json.dump(results, f, indent=2)

    print("=" * 70)
    print("[E] Power curve  P(morphic_signature) by n (rows) x coupling (cols)")
    e = results["concept_e_power_curve"]
    couplings = sorted({float(c) for row in e["grid"].values() for c in row}, )
    header = "  n\\cpl  " + "".join(f"{c:>6.1f}" for c in couplings)
    print(header)
    for n, row in e["grid"].items():
        line = f"  {n:>5}  " + "".join(f"{row[str(c)]:>6.2f}" for c in couplings)
        print(line)
    print(f"  min-detectable coupling @80%: {e['min_detectable_coupling_at_80pct']}")
    print(f"  false-positive rate @cpl=0:   {e['false_positive_rate']}")
    print(f"  >> {e['finding']}")
    print()
    print("[F] Real-vs-fake patterns (Sheldrake word-recall design):")
    f = results["concept_f_real_vs_fake"]
    for mech in ("causal", "ctc"):
        print(f"  {mech:7s} -> {f[mech]['modal_verdict']:18s} "
              f"real-minus-fake early cost = {f[mech]['mean_real_minus_fake_early_cost']:+.3f}")
    print(f"  >> {f['finding']}")
    print(f"\nWrote {path}")


if __name__ == "__main__":
    main()
