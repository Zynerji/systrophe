# D-CTC deep exploration — 30 derived phases

Phases A-I were the initial exploration (docs/DCTC_DEEP.md).
This document enumerates **30 additional phases** to push the
investigation toward first principles and algorithmic implications.

Already complete: **A, B, C, D, E, F, G, H, I** (9 phases).

The 30 below are organised into 6 groups by what they probe.

## Group I: Structural extensions (J–N)

**Phase J — dim_CTC scaling at fixed dim_CR.** Mirror of Phase H.
Sweep dim_CTC ∈ {2..6} at dim_CR=2; characterise how high-purity rate
decays. Predicted exponential in dim_CTC.

**Phase K — Three-Kraus channels (dim_CR=3).** With 3 Kraus operators,
need triple-overlap |⟨v_0|v_1⟩| · |⟨v_1|v_2⟩| · |⟨v_2|v_0⟩| > c.
Does triple overlap predict purity?

**Phase L — Minimal-dim case dim_CR=2, dim_CTC=2.** Phase A showed
max purity 0.989 there. Run 5000+ samples to characterise the
extreme tail with high statistical power.

**Phase M — σ_CR mixedness sweep.** σ_CR = (1−ε)|0⟩⟨0| + ε|1⟩⟨1|
parametrises from pure (ε=0) to maximally-mixed (ε=0.5). How does
high-purity fraction depend on ε?

**Phase N — ρ_init dependence.** Sample multiple random ρ_init for
the same U. Does the iteration converge to the same fixed point?
(Tests channel uniqueness.)

## Group II: Spectral structure (O–S)

**Phase O — Full spectrum statistics.** Distribution of *all*
eigenvalues of E (not just λ_2). Look for level statistics signatures
(GOE/GUE/Poisson) characteristic of random or chaotic systems.

**Phase P — Eigenvector localization (IPR).** Inverse Participation
Ratio of E's principal eigenvector. Localised eigenvectors (high IPR)
should correlate with high purity.

**Phase Q — Spectral form factor.** K(τ) = |Σ_n e^{i E_n τ}|² as a
function of τ for the channel eigenvalues. Diagnoses RMT class.

**Phase R — Ergodicity test.** From many ρ_init's, does iteration
visit a unique limit (ergodic) or multiple limits (non-ergodic)?

**Phase S — Lyapunov spectrum.** Eigenvalues of dE/dρ at the fixed
point; characterises local stability and convergence rate.

## Group III: Algebraic structure of U (T–Y)

**Phase T — Conditioning of U-blocks.** Look at the block structure
of U: U_{ab|cd} where a,c ∈ CR, b,d ∈ CTC. Compute singular value
distribution of each (CR_in→CR_out) block.

**Phase U — Operator Schmidt spectrum.** Treat U as a bipartite
operator on (CR, CR) ⊗ (CTC, CTC). Full Schmidt-rank spectrum.

**Phase V — Distance to ensemble centre.** Compute trace-distance
from U to the centre of its Haar ensemble. Does outlier distance
predict purity?

**Phase W — Distance to separable U.** Find argmin ||U - U_CR ⊗ U_CTC||
over (U_CR, U_CTC). Closer-to-separable should give higher purity
(less mixing through the channel).

**Phase X — Distance to Clifford.** Find nearest Clifford gate to U.
Closer-to-Clifford might boost or depress purity.

**Phase Y — Bell-state preservation.** Apply U to |Bell⟩ on (CR, CTC).
Measure resulting entanglement. Does U-preserved entanglement
correlate with high purity?

## Group IV: Iteration dynamics (Z–AD)

**Phase Z — Trajectory analysis.** Plot ρ_n in Bloch-sphere
projection (or equivalent low-dim viz) over the iteration. Look
for spirals, plateaus, jumps.

**Phase AA — Convergence-rate distribution detail.** Beyond log-
normal: is there a bimodal mixture? Tail truncation? Hidden modes?

**Phase AB — Perturbation sensitivity.** Perturb U by a small CPTP
noise; how robust is the high-purity fixed point? Estimates
basin-of-attraction width.

**Phase AC — Iteration under noise.** Add Markovian noise to each
iteration step. At what noise level does purity collapse?

**Phase AD — Anderson mixing acceleration.** Use Anderson mixing
on the iteration. Does it converge in fewer steps? By what factor?

## Group V: Algorithmic implications (AE–AH)

**Phase AE — State distinguisher.** Use high-purity D-CTC channel
to distinguish two close mixed states σ_a, σ_b. Compare to optimal
Helstrom bound. Aaronson-Watrous prediction: D-CTC channels can
distinguish what classical/quantum cannot in poly time.

**Phase AF — Classical capacity.** Compute Holevo capacity of the
D-CTC channel E. Does high-purity correlate with high capacity?

**Phase AG — Quantum capacity.** Compute coherent information
I_c(E, ρ) maximised over ρ. High-purity channels should have
distinctive Q_c values.

**Phase AH — Error-correction use.** Construct a code subspace
where the D-CTC channel acts as approximate identity. High-purity
channels should support smaller codes.

## Group VI: Cross-system applications (AI–AM)

**Phase AI — Z_3-monodromy D-CTC.** Use the Z_3 cycle-shift S as
σ_CR in the iteration. Does the structured σ_CR change purity
statistics?

**Phase AJ — D-CTC on LP background.** Apply the iteration to a
unitary built from the LP exterior structure (e.g., Floquet
quasi-energies). Does the cylinder geometry impose a privileged
fixed point?

**Phase AK — Acoustic-analog D-CTC.** Map the D-CTC iteration to
the acoustic-metric setting. Sound waves in a vortex see an
acoustic CTC; does the BdG mode structure produce a near-pure
fixed point experimentally?

**Phase AL — Polynomial-time PSPACE check.** Implement the
Aaronson-Watrous PSPACE-solving algorithm using our channels.
Verify the speedup happens precisely for high-purity channels.

**Phase AM — Chronology-protection D-CTC.** Use the back-reaction
self-consistent δ from `chronology_protection.py` to define σ_CR.
Does the chronology-protected configuration drive D-CTC purity to
floor (max-mixed) — i.e., does protection cause information loss?

---

## Execution strategy

Cost estimate per phase: ~5 min compute on this machine, plus
~10 min coding/analysis per phase. Total: ~30 hr unattended.

Priority order (highest-yield first):

1. **L** (extreme-tail statistics at d=2×2) — likely produces clean
   data on the absolute purity ceiling.
2. **M** (σ_CR mixedness sweep) — tests whether the high-purity
   class is robust to mixing.
3. **N** (ρ_init independence) — confirms uniqueness of fixed point.
4. **W** (distance to separable U) — directly tests the structural
   hypothesis "high-purity = near-separable U".
5. **P** (eigenvector IPR) — connects to Anderson localisation
   intuition.
6. **AE** (state distinguisher) — algorithmic payoff test.
7. **AB** (perturbation sensitivity) — basin-of-attraction.
8. **K** (3-Kraus generalisation) — extends the joint-eigenvector
   theory to dim_CR=3.
9. **U** (operator Schmidt spectrum) — alternative to W.
10. **AI** (Z_3 σ_CR) — connects to Systrophe Z_3 structure.

The remaining 20 are valuable but lower-yield given current findings.

Results: per-phase script under `examples/dctc_deep_phase_X.py` with
JSON output in `examples/dctc_deep_phase_X_results.json`.
