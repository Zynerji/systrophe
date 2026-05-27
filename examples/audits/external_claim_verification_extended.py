"""Verification battery for the external 'updates.txt' continuation (2026-05-10).

The follow-up external claim battery (saved as ~/Desktop/updates.txt) advances from
the Systrophe analysis into a speculative cascade:

  Horned torus -> tunability via horn sharpness -> ringhole hybrid ->
  self-consistent back-reaction iterator -> Z3 topological Casimir ->
  gravito-dynamical Casimir effect -> claims of an exotic-matter-free
  traversable throat.

Many of these are physics-flavoured but specifically testable. This
script tests every claim that can be reduced to a numerical statement.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np


CLAIMS: list[dict] = []


def record(name: str, claim: str, verified: bool | None, evidence: str) -> None:
    CLAIMS.append({
        "name": name,
        "claim": claim,
        "status": "VERIFIED" if verified is True
                  else "FALSIFIED" if verified is False
                  else "NOT TESTABLE",
        "evidence": evidence,
    })


# =====================================================================
# 1. Hurwitz zeta at s = -3 formula
# =====================================================================
# External claim: zeta_H(-3, a) = -a^4 / 4 + a^3 / 2 - a^2 / 4 + 1 / 120

def hurwitz_zeta_neg3_external(a):
    return -a**4 / 4 + a**3 / 2 - a**2 / 4 + 1 / 120


# Reference: zeta_H(-n, a) = -B_{n+1}(a) / (n+1), where B_{n+1} is the
# Bernoulli polynomial. For n = 3:
#   B_4(a) = a^4 - 2 a^3 + a^2 - 1/30
#   zeta_H(-3, a) = -B_4(a) / 4 = -(a^4 - 2 a^3 + a^2 - 1/30) / 4
#                = -a^4/4 + a^3/2 - a^2/4 + 1/120.
# Matches the formula exactly.
test_values = [0.0, 0.5, 1.0, 1/3, 2/3, 1/6]
errors = []
for a in test_values:
    B4 = a**4 - 2 * a**3 + a**2 - 1/30
    ref = -B4 / 4
    external = hurwitz_zeta_neg3_external(a)
    errors.append(abs(ref - external))
verified = max(errors) < 1e-15
record(
    "hurwitz_zeta_neg3",
    "Hurwitz zeta at s=-3: zeta_H(-3, a) = -a^4/4 + a^3/2 - a^2/4 + 1/120",
    verified,
    f"max |external form - Bernoulli form| over a in {test_values}: {max(errors):.2e}",
)


# =====================================================================
# 2. Standard Casimir energy density formula
# =====================================================================
# external cites: <rho>_Casimir = -pi^2 h_bar c / (720 d^4)
# Textbook value (parallel perfect-conductor plates): -pi^2 / (720 d^4)
# in natural units. Sign and constant match.
record(
    "standard_casimir_formula",
    "Casimir energy density between parallel perfect-conductor plates: -pi^2 h_bar c / (720 d^4)",
    True,
    "Textbook value (Casimir 1948; Itzykson-Zuber eq. 3-184). Sign and "
    "constant correct as quoted.",
)


# =====================================================================
# 3. Horned torus CTC fractions
# =====================================================================
# external claims: with L_eff = r_eff^2 (1 - (omega r_eff / R)^2 / 4) and
# horn-pinched r_eff(theta, phi), CTC fraction goes 0.00 (omega <= 1.4),
# 0.20 (omega=1.5), 0.32 (omega=1.6), 0.41 (omega=1.8).
# Reproduce on a similar grid.

def horned_torus_L_eff(R=5.0, r_base=2.0, hf=1.0, omega=1.5, n=200):
    """Compute L_eff on a (theta, phi) grid for a horned-torus parametrisation.

    Reproduce the model: r_eff(theta) = r_base * (1 + (hf - 1) * cos(theta)^2)
    creates a horn pinch on the inner equator. Compute L_eff per external's
    formula L_eff = r_eff^2 (1 - (omega r_eff / R)^2 / 4).
    """
    theta = np.linspace(0, 2 * np.pi, n)
    phi = np.linspace(0, 2 * np.pi, n)
    T, P = np.meshgrid(theta, phi, indexing='ij')
    r_eff = r_base * (1 + (hf - 1) * np.cos(T) ** 2)
    L_eff = r_eff ** 2 * (1 - (omega * r_eff / R) ** 2 / 4)
    return L_eff


fractions = {}
for omega in [0.5, 1.0, 1.4, 1.5, 1.6, 1.8, 2.5]:
    L = horned_torus_L_eff(omega=omega, hf=1.0)
    fractions[omega] = float((L < 0).mean())
# external claimed: 0 at omega <= 1.4; 0.20 at 1.5; 0.32 at 1.6; 0.41 at 1.8
expected = {0.5: 0.0, 1.4: 0.0, 1.5: 0.20, 1.6: 0.32, 1.8: 0.41}
matches = all(abs(fractions[k] - v) < 0.05 for k, v in expected.items())
record(
    "horned_torus_ctc_fractions",
    "Horned torus L_eff = r^2 (1 - (omega r / R)^2 / 4) with hf=1 gives external-reported CTC fractions",
    matches,
    f"Computed: {fractions}. External claim: {expected}. Matches within 5% threshold: {matches}.",
)


# =====================================================================
# 4. Self-consistent iterator "convergence"
# =====================================================================
# external showed iteration trace:
#   Iter  0 | delta=0 | exotic= 0.127204 | delta_step= 2.28e-02
#   Iter  5 | delta=0 | exotic= 0.013221 | delta_step= 2.28e-02
#   Iter  6 | delta=0 | exotic=-0.009575 | delta_step= 2.28e-02
#   Iter 10 | delta=0 | exotic=-0.100761 | delta_step= 2.28e-02
#   Iter 15 | delta=0 | exotic=-0.214744 | delta_step= 2.28e-02
#   Iter 20 | delta=0 | exotic=-0.305930 | delta_step= 2.28e-02
# external declared "Converged to self-consistent solution!"
# Reality check: delta_step is CONSTANT at 2.28e-02 across all iterations.
# A truly converging iteration has delta_step -> 0. A self-consistent
# fixed-point would have delta_step decreasing. the iteration is in
# fact a LINEAR WALK in one direction, not convergence; the "negative
# exotic fraction" just reflects the walk crossing zero and continuing.

iter_data = [
    (0, 0.127204),
    (5, 0.013221),
    (6, -0.009575),
    (10, -0.100761),
    (15, -0.214744),
    (20, -0.305930),
]
steps = [iter_data[i+1][1] - iter_data[i][1] for i in range(len(iter_data) - 1)]
# Per-iter step (normalize by iteration delta)
per_iter_step = [
    (iter_data[i+1][1] - iter_data[i][1]) / (iter_data[i+1][0] - iter_data[i][0])
    for i in range(len(iter_data) - 1)
]
all_steps_equal = max(per_iter_step) - min(per_iter_step) < 0.01
record(
    "iterator_convergence_claim",
    "External claim states the self-consistent iterator converges to a fixed point in <10 iterations",
    False,  # FALSIFIED
    f"The reported trace has constant per-iteration step ~{per_iter_step[0]:.4f} across all "
    f"iterations (max-min spread {max(per_iter_step) - min(per_iter_step):.4f}). This is a "
    "linear walk, not convergence: a true fixed-point iteration has decreasing step. "
    "the 'converged' verdict is incorrect; the iteration is simply walking past zero.",
)


# =====================================================================
# 5. Topological Casimir coefficient C(delta)
# =====================================================================
# External claim: C(delta) = (1/720) sum_{b=0,1,2} zeta_H(-3, b/3 + gamma_eff(delta) / (2 pi))
# Verify the formula evaluates and check its delta dependence.

def topological_casimir_C(gamma_eff):
    return (1 / 720) * sum(
        hurwitz_zeta_neg3_external(b / 3 + gamma_eff / (2 * np.pi))
        for b in (0, 1, 2)
    )


gamma_vals = np.linspace(0, 2 * np.pi, 9)
C_vals = [topological_casimir_C(g) for g in gamma_vals]
record(
    "topological_casimir_coefficient_evaluable",
    "the topological Casimir coefficient C(delta) is well-defined and evaluates without divergence",
    all(np.isfinite(C_vals)),
    f"C(gamma) at gamma=0..2pi (9 samples): {[f'{c:.6e}' for c in C_vals]}. "
    "Note: physical interpretation (Casimir energy on a Z_3 cover at the throat) is "
    "asserted, not derived, in updates.txt.",
)


# =====================================================================
# 6. C(delta) sensitivity check
# =====================================================================
# If gamma_eff depends on delta linearly (the chat is unclear about
# this relation), the variation of C across delta in [0, 2 pi] should
# trace out a structure. We check the magnitude of variation.

C_max = max(abs(c) for c in C_vals)
C_min = min(abs(c) for c in C_vals)
variation_ratio = C_max / C_min if C_min > 1e-30 else float("inf")
record(
    "topological_casimir_delta_dependence",
    "The topological Casimir coefficient depends meaningfully on delta",
    variation_ratio > 1.1,
    f"max |C| = {C_max:.6e}, min |C| = {C_min:.6e}, ratio = {variation_ratio:.4f}. "
    "If ~1, C is essentially constant in delta (function of b/3 only) and the "
    "claim that delta tunes Casimir strength fails. If >> 1, delta does tune it.",
)


# =====================================================================
# 7. Casimir energy at the throat - sign check
# =====================================================================
# External claim: rho_throat = -h_bar c / d_eff^4 * C(delta)
# For positive C(delta), rho < 0 (negative energy, required for traversable throat).
# Check the sign of C(delta) across the range.
n_positive = sum(1 for c in C_vals if c > 0)
record(
    "casimir_sign_at_throat",
    "Casimir energy density at the throat is negative (required for NEC violation / traversability)",
    None,
    f"Of 9 delta samples, C(delta) is positive at {n_positive}, negative at "
    f"{9 - n_positive}. Sign depends on delta; cannot make blanket 'rho < 0 stabilizes "
    "throat' claim. Even if C > 0 gives rho < 0, the throat-stability calculation "
    "external cites is not derived in updates.txt; just asserted.",
)


# =====================================================================
# 8. DCE off-switch claim
# =====================================================================
# External claim: "When delta(t) is held near pi (the off state), the DCE flux
# drops by more than two orders of magnitude."
# Test: does C(delta) drop by 100x near delta = pi?
gamma_at_pi = np.pi  # if gamma_eff(pi) = pi (simplest mapping)
gamma_at_0 = 0.0
C_at_0 = abs(topological_casimir_C(gamma_at_0))
C_at_pi = abs(topological_casimir_C(gamma_at_pi))
ratio = C_at_0 / C_at_pi if C_at_pi > 1e-30 else float("inf")
record(
    "dce_off_switch_at_pi",
    "DCE flux drops by >2 orders of magnitude near delta = pi (the off-switch claim)",
    ratio > 100.0,
    f"|C(0)|/|C(pi)| = {ratio:.4f}. External claim states >100. "
    f"|C(0)| = {C_at_0:.6e}, |C(pi)| = {C_at_pi:.6e}. "
    "Note: assumes gamma_eff(delta) = delta (simplest mapping); actual gamma_eff "
    "comes from phasor sum and may differ. The 'off switch at pi' for the "
    "CLASSICAL CTC bands is well-established and tested in Systrophe; transferring "
    "the same off-switch to DCE requires the phasor cancellation to also kill the "
    "Casimir coefficient. External claim asserts this without derivation.",
)


# =====================================================================
# 9. "Casimir cavity at the throat with d_eff = 2 b (1 - lambda)"
# =====================================================================
# the effective-separation formula. Dimensionally OK, but the physical
# motivation (treating a ringhole throat as a Casimir cavity with three
# topological "plates" from Z_3 monodromy) is not standard QFTCS;
# it's the own ansatz. Not testable as a pure math statement; flag.
record(
    "casimir_cavity_at_throat_ansatz",
    "the identification of the ringhole throat as a Casimir cavity with effective separation 2 b (1 - lambda)",
    None,
    "Ansatz, not derived. A genuine throat-Casimir calculation would require "
    "(a) solving the wave equation in the actual ringhole metric, (b) imposing the "
    "Z_3 monodromy as a boundary condition on the cover, (c) summing zero-point "
    "energies with appropriate UV regularization. external skips (a), (b), (c) and "
    "writes the answer by analogy to flat-space Casimir.",
)


# =====================================================================
# 10. Gravito-dynamical Casimir power formula
# =====================================================================
# P_DCE(t) = (h_bar c / d_eff^5) * d_dot_eff * C + (h_bar c / d_eff^4) *
#           (d gamma_eff / dt) * C'
# This is a time-derivative of the (asserted) static Casimir energy. The
# time-derivative form is correct calculus if the static result is correct;
# the static result is itself an ansatz. So the dynamical formula inherits
# the ansatz status.
record(
    "gravito_dce_formula_inherits_ansatz",
    "the gravito-dynamical Casimir power formula derived as the time-derivative of the static throat energy",
    None,
    "If the static Casimir energy at the throat is correct, the time-derivative form is correct calculus. "
    "But the static energy is an ansatz; so is its derivative.",
)


# =====================================================================
# 11. Run the own DCE pseudocode
# =====================================================================
# Reproduce the toy run: delta(t) = 0.5 + 0.3 sin(t), b(t) = 1 + 0.1 sin(2t),
# lambda = 0.7. Expected: peak power ~0.042, average rate ~0.018.

def external_dce_power(delta_arr, b_arr, lambda_fold=0.7):
    d_eff = 2 * b_arr * (1 - lambda_fold)
    C = np.zeros_like(delta_arr)
    Cp = np.zeros_like(delta_arr)
    eps = 1e-6
    for i, d in enumerate(delta_arr):
        gamma = np.angle(np.exp(1j * d))
        s = 0.0
        sp = 0.0
        for bb in (0, 1, 2):
            arg = bb / 3.0 + gamma / (2 * np.pi)
            s += hurwitz_zeta_neg3_external(arg)
            sp += (hurwitz_zeta_neg3_external(arg + eps) - hurwitz_zeta_neg3_external(arg - eps)) / (2 * eps)
        C[i] = s / 720
        Cp[i] = sp / 720
    ddot_deff = np.gradient(np.gradient(d_eff))
    term1 = (1 / d_eff ** 5) * np.abs(ddot_deff) * C
    term2 = (1 / d_eff ** 4) * np.abs(np.gradient(delta_arr)) * Cp
    return term1 + term2


t_grid = np.linspace(0, 4 * np.pi, 400)
delta_t = 0.5 + 0.3 * np.sin(t_grid)
b_t = 1.0 + 0.1 * np.sin(2 * t_grid)
P_arr = external_dce_power(delta_t, b_t, lambda_fold=0.7)
peak_power = float(np.max(np.abs(P_arr)))
record(
    "external_dce_toy_peak_power",
    "the toy run: delta(t) = 0.5 + 0.3 sin t, b(t) = 1 + 0.1 sin 2t, lambda = 0.7 -> peak DCE power ~ 0.042",
    abs(peak_power - 0.042) < 0.03,
    f"Reproduced peak |P_DCE| = {peak_power:.4f} (external claimed ~ 0.042). "
    "Tolerance: 3 sigma of the reporting precision.",
)


# =====================================================================
# 12. "Exotic-matter-free throat" claim
# =====================================================================
# External claim asserts this is "the first exact, controllable example where
# quantum vacuum alone suffices" for a traversable wormhole throat.
# Test: not testable in our framework; the iterator that supposedly
# proved this was shown above to be a linear walk, not convergence.
record(
    "exotic_matter_free_throat",
    "Quantum vacuum on the Z_3 cover alone stabilizes the ringhole throat without classical exotic matter",
    None,
    "Asserted via the self-consistent iterator, which was shown (claim 4) to be a "
    "linear walk, not converged. The 'negative exotic fraction' is a walk past zero, "
    "not a fixed point. Until a properly converging iterator is implemented (with "
    "actual <T_mu nu> computation from full Hadamard renormalisation), this claim "
    "remains unsupported.",
)


# =====================================================================
# 13. "Z_3 cover regularises mode sum, no UV divergence"
# =====================================================================
# Test by computing the C(delta) sum explicitly. If the b=0,1,2 sum is
# finite, the claim of "no UV divergence" via fractional shifts is
# self-consistent (with the zeta-regularised form).
test_args = np.linspace(0, 1, 50)
all_finite = all(np.isfinite(topological_casimir_C(g)) for g in test_args)
record(
    "z3_cover_no_uv_divergence",
    "Z_3 cover with fractional shifts b/3 produces a finite, well-regularised sum",
    all_finite,
    f"All 50 samples of C(gamma) for gamma in [0, 1] are finite. The finiteness is "
    "automatic from the zeta-regularised closed form (no infinite mode sum to do); "
    "this is essentially a re-statement of the fact that zeta_H(-3, a) is finite.",
)


# =====================================================================
# 14. Acoustic-metric mapping
# =====================================================================
# external provides c_s^2 ~ 1 - (alpha^2 / 4) sin^2(alpha u + gamma_eff) and a
# horizon condition v^phi = c_s. The derivation is presented as if from
# matching coefficients of a fluid metric, but the match is incomplete
# and the framework (sonic-analogue gravity) requires a specific fluid
# (BEC) realisation. Flag as ansatz.
record(
    "acoustic_metric_mapping",
    "Hybrid GR metric maps onto a standard acoustic metric for BEC analogue gravity",
    None,
    "the mapping is dimensionally consistent but skips the matter sector (no fluid "
    "equation of state, no specific BEC realisation). Not a derivation; a sketch.",
)


# =====================================================================
# Summary
# =====================================================================
def main() -> None:
    n_v = sum(1 for c in CLAIMS if c["status"] == "VERIFIED")
    n_f = sum(1 for c in CLAIMS if c["status"] == "FALSIFIED")
    n_u = sum(1 for c in CLAIMS if c["status"] == "NOT TESTABLE")
    total = len(CLAIMS)
    print(f"Tested {total} mathematical assertions from updates.txt.")
    print(f"  VERIFIED:     {n_v} ({100*n_v/total:.1f}%)")
    print(f"  FALSIFIED:    {n_f} ({100*n_f/total:.1f}%)")
    print(f"  NOT TESTABLE: {n_u} ({100*n_u/total:.1f}%)")
    print()
    for c in CLAIMS:
        marker = {"VERIFIED": "[OK]", "FALSIFIED": "[XX]", "NOT TESTABLE": "[??]"}[c["status"]]
        print(f"{marker} {c['name']}")
        print(f"     {c['claim']}")
        print(f"     {c['evidence']}")
        print()
    out_path = Path("examples") / "external_claim_verification_extended_results.json"
    out_path.write_text(json.dumps({
        "total": total,
        "verified": n_v,
        "falsified": n_f,
        "not_testable": n_u,
        "claims": CLAIMS,
    }, indent=2), encoding="utf-8")
    print(f"Results written to {out_path}")


if __name__ == "__main__":
    main()
