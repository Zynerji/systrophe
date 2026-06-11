"""The falsification harness: panel in, adjudicated verdict out.

``falsify(panel)`` runs the full stack -- model-free catcher, count-vs-time
identifiability regression, acausality test, and surrogate nulls -- and
collapses them into one of five verdicts:

  no_structure          panel is consistent with independent learners.
  conventional_trend    real ordered structure, but it is a secular
                        CALENDAR-time trend, not count-coupling.
  unidentifiable        structure present, but cumulative count is too
                        collinear with time to attribute it to either
                        (the honest verdict for most real-world morphic data).
  morphic_signature     cost tracks cumulative COUNT beyond a time-trend,
                        and survives the order-shuffle null.

Acausality (the distinctive CTC-resonance fingerprint) is NOT decided here.
Within a single form, ``prior + future == const`` makes past/future counts
perfectly collinear and any nonlinear trend leaks into a spurious "future"
effect, so a single-panel acausality claim is not trustworthy. Acausality is
adjudicated only by :func:`falsify_acausal`, which works ACROSS forms with
varying eventual totals. ``acausal_test`` is reported as a diagnostic only.

Decision thresholds are deliberately conservative; the harness is built to
return the weakest claim the evidence supports.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detect import (
    catcher_verdict,
    count_vs_time_identifiability,
    acausal_test,
    acausal_across_forms,
)
from .nulls import order_shuffle_null


@dataclass
class MorphicVerdict:
    verdict: str
    catcher: dict
    identifiability: dict
    acausal: dict
    count_null: dict
    acausal_null: dict
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        ident = self.identifiability
        return (
            f"[{self.verdict}] {self.reason}\n"
            f"  catcher: {self.catcher['verdict']} "
            f"({self.catcher['n_sharp_features']} sharp)\n"
            f"  count vs time: count_t={ident['count_t']:+.2f} "
            f"time_t={ident['time_t']:+.2f} VIF={ident['count_time_vif']:.1f} "
            f"identifiable={ident['identifiable']}\n"
            f"  count-effect order-shuffle p={self.count_null['p_value']:.4f}\n"
            f"  acausal future_t={self.acausal['future_t']:+.2f} "
            f"order-shuffle p={self.acausal_null['p_value']:.4f}"
        )


def falsify(panel, n_surrogates: int = 200, seed: int = 0,
            alpha: float = 0.05) -> MorphicVerdict:
    """Adjudicate a panel. See module docstring for the verdict semantics."""
    catcher = catcher_verdict(panel)
    ident = count_vs_time_identifiability(panel)
    acaus = acausal_test(panel)

    # Decisive surrogate null (order-shuffle breaks cost<->order coupling).
    count_stat = lambda p: abs(count_vs_time_identifiability(p)["count_t"])
    count_null = order_shuffle_null(panel, count_stat,
                                    n_surrogates=n_surrogates, seed=seed)

    count_sig = count_null["p_value"] < alpha
    time_sig = abs(ident["time_t"]) > 3.0
    any_structure = (catcher["verdict"] == "novel_structure"
                     or count_sig or time_sig)

    if not any_structure:
        verdict, reason = "no_structure", (
            "No ordered structure beyond the independent-learner null.")
    elif not ident["identifiable"]:
        verdict, reason = "unidentifiable", (
            f"Structure present but count<->time VIF={ident['count_time_vif']:.1f} "
            ">= 10: cumulative count cannot be separated from a secular "
            "time-trend. This is the failure mode of the classic rat-maze data.")
    elif count_sig and abs(ident["count_t"]) > abs(ident["time_t"]):
        verdict, reason = "morphic_signature", (
            f"Cost tracks cumulative COUNT beyond time (count_t={ident['count_t']:+.2f} "
            f"> time_t={ident['time_t']:+.2f}, shuffle p={count_null['p_value']:.4f}).")
    elif time_sig:
        verdict, reason = "conventional_trend", (
            f"Real structure, but it is a secular time-trend (time_t={ident['time_t']:+.2f} "
            f"dominates count_t={ident['count_t']:+.2f}).")
    else:
        verdict, reason = "no_structure", (
            "Marginal structure did not survive the conservative thresholds.")

    return MorphicVerdict(
        verdict=verdict, catcher=catcher, identifiability=ident,
        acausal=acaus, count_null=count_null, acausal_null=None,
        reason=reason, meta=dict(panel.meta),
    )


@dataclass
class AcausalVerdict:
    verdict: str
    across_forms: dict
    null: dict
    reason: str = ""
    meta: dict = field(default_factory=dict)

    def summary(self) -> str:
        a = self.across_forms
        return (
            f"[{self.verdict}] {self.reason}\n"
            f"  eventual_total_t={a['eventual_total_t']:+.2f} "
            f"early_count_t={a['early_count_t']:+.2f} "
            f"VIF={a['count_eventual_vif']:.1f}\n"
            f"  shuffle p={self.null['p_value']:.4f}")


def falsify_acausal(forms, n_surrogates: int = 200, seed: int = 0,
                    alpha: float = 0.05) -> AcausalVerdict:
    """Adjudicate the CTC-resonance fingerprint across multiple forms.

    Tests whether a form's EARLY cost depends on its EVENTUAL total (future)
    beyond its early count. A label-shuffle null permutes the eventual totals
    across forms, destroying any genuine early<->eventual coupling while
    preserving both marginals. Returns ``acausal_signature`` only if the
    eventual-total effect is significant AND survives that null.
    """
    res = acausal_across_forms(forms)

    rng = np.random.default_rng(seed)
    eventual = np.array([f["eventual_total"] for f in forms], float)
    null = np.empty(n_surrogates)
    for k in range(n_surrogates):
        perm = rng.permutation(len(forms))
        shuffled = [dict(f, eventual_total=float(eventual[perm[i]]))
                    for i, f in enumerate(forms)]
        null[k] = abs(acausal_across_forms(shuffled)["eventual_total_t"])
    real_stat = abs(res["eventual_total_t"])
    p = float((np.sum(null >= real_stat) + 1) / (n_surrogates + 1))

    if res["count_eventual_vif"] >= 10.0:
        verdict, reason = "unidentifiable", (
            "early count and eventual total too collinear to separate.")
    elif p < alpha and res["acausal"]:
        verdict, reason = "acausal_signature", (
            f"Early cost depends on EVENTUAL total (eventual_t="
            f"{res['eventual_total_t']:+.2f}, shuffle p={p:.4f}); no causal "
            "mechanism reproduces this. This is the CTC-resonance fingerprint "
            "-- and exactly what no controlled morphic experiment has shown.")
    else:
        verdict, reason = "no_acausality", (
            "Early cost is explained by early count alone; no future "
            "dependence beyond the null.")

    return AcausalVerdict(
        verdict=verdict, across_forms=res,
        null={"p_value": p, "null_mean": float(null.mean()),
              "null_p95": float(np.percentile(null, 95)),
              "n_surrogates": n_surrogates},
        reason=reason,
        meta=dict(forms[0].get("meta", {})) if forms else {},
    )
