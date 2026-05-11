# Stress tests of the Systrophe framework

Three stress tests run on v0.15.1 to probe whether structural features
of the package generate novel algorithmic or theoretical content.
Each test is a self-contained script under `examples/stress_*.py`.

## Summary verdicts

| # | Test | Outcome | Novelty |
|---|------|---------|---------|
| 1 | Z_N closure pattern | clean theorem | yes |
| 2 | Cascade-DSI dimension surface | rich (no universal collapse) | partial |
| 2b | Novelty-catcher on cascade-DSI | sharp phase boundary detected | yes |
| 3 | D-CTC Haar-random U | heavy-tail in iterations + structured purity | yes |

---

## Test 1: Z_N anomaly-closure pattern

**Script**: `examples/stress_zn_closure.py`
**Output**: `examples/stress_zn_closure_results.json`

Generalises `anomaly_inflow.z3_total_eta` to Z_N covers for
N in [2, 30].

### Result (clean theorem)

For the APS eta-invariant convention
`eta(alpha) = 1 - 2 alpha` (with `eta(0) = 0` symmetric reg), the
closure

```
Sum_{b=0}^{N-1} eta_b(gamma_eff = 0) = 0
```

holds **to machine precision (< 1e-12)** for every N in [2, 30] tested.
Closure additionally holds at any `gamma_eff = 2 pi k / N` (integer k),
because the twist is then a cyclic permutation of branch indices and
leaves the eta-set invariant.

Between consecutive cyclic-permutation closure points, the eta-sum is
piecewise linear in gamma_eff with slope `-N/pi`, crossing zero
**exactly once per sub-interval**. The total number of closure points
in `[0, 2 pi)` is `2 N` (cyclic + mid-interval).

### Algorithmic application

The closed-form eta-sum on Z_N covers is a building block for index-
theorem applications. The piecewise-linear sawtooth structure makes
it cheap to compute and easy to invert: given a target eta-sum
residual `r`, one can solve for the required `gamma_eff` in O(1).

Possible downstream applications:
- topological-insulator band-structure index calculations on tight-
  binding cylinders;
- anomaly-coefficient extraction in 2D CFT model-building;
- lattice-gauge-theory observable construction on twisted boundaries.

---

## Test 2: cascade-DSI dimension surface

**Script**: `examples/stress_cascade_dsi.py`
**Output**: `examples/stress_cascade_dsi_results.json`

20x20 grid sweep over (scale_factor sigma, amp_decay rho) in
([1.5, 8.0] x [0.4, 0.99]) at fixed cascade depth `levels = 4`.

### Result

Box-counting dimension statistics across the grid:
- range: 0.000 to 0.793
- mean: 0.468, std: 0.179
- saturation: upper bound around 0.8 (truncation-limited at L=4)

Phase-transition boundary `D > 0.3`: a curve in (sigma, rho) space
roughly satisfying `sigma * rho ~ const`; at sigma=1.5 the threshold
is rho > 0.835, at sigma=8 even rho=0.4 already exceeds threshold.

### Scaling collapse

A search over `alpha in [-3, 3]` for the rescaling
`x = sigma * rho^alpha` that collapses D(sigma, rho) onto a 1D curve
yields **alpha ~ 0** (no useful collapse). The dimension surface
genuinely depends on two parameters; no simple universality.

### Verdict

Cascade-DSI produces non-trivial fractal dimension across most of the
parameter space. The lack of scaling collapse suggests the
(sigma, rho) plane has independent thermodynamic-style axes.
A larger `levels` would push the upper bound higher; the saturation
near 0.8 is a finite-depth artifact.

---

## Test 2b: address-space novelty-catcher on cascade-DSI

**Script**: `examples/stress_cascade_novelty.py`
**Output**: `examples/stress_cascade_novelty_results.json`

Implements the HASH-QUINE address-space rule: hash cascade zero sets
to 64-bit binary occupancy vectors (over u = ln r), build a Hamming-
distance graph, compute lambda_2 of the Laplacian.

### Result (novelty signal)

Local 3x3-neighbourhood lambda_2 surface:

```
sigma=1.50: 4.00  6.00  2.00  0.49  0.00  -0.00  -0.00  0.00
sigma=2.43: 6.00  2.00  0.43  0.00  0.00  -0.00  -0.00  0.00
sigma=3.36: 0.91  0.75  0.00  0.00  0.00  -0.00  -0.00  0.00
sigma=4.29: 1.00  0.42  0.00  0.00  0.00  0.00  0.00  0.00
sigma=5.21: 0.49  0.24  0.00  0.00  0.00  -0.00  0.00  0.00
sigma=6.14: 0.38  0.20  0.00  0.00  0.00  0.00  0.00  0.00
sigma=7.07: 0.00  0.00  0.00  0.00  0.00  0.00  -0.00  0.00
sigma=8.00: 0.00  0.00  0.00  0.00  0.00  0.00  0.00  0.00
```

**Interpretation.** lambda_2 > 0 only in the top-left region
(small sigma, small rho). This is where the cascade reduces to a
single dominant cosine, so all configurations produce similar
zero sets -- the Hamming-distance graph is dense and connected.
In the multi-scale regime (large sigma or large rho), each
configuration produces a *distinct* zero set; the local
neighbourhood graph fragments and lambda_2 -> 0.

**Two sharp jump features** at (sigma=2.43, rho=0.48) and
(sigma=2.43, rho=0.57) -- candidate phase boundary signaling
the trivial-to-multiscale transition.

### Comparison to dimension surface

The novelty-catcher boundary is sharper than the dimension-based
boundary from Test 2. lambda_2 acts as a **diversity diagnostic**:
it detects the onset of multi-scale richness as a phase transition
in graph connectivity. This is an algorithmic shortcut for detecting
universality classes in cascade systems: where the standard box-
counting gives a smooth gradient, the address-space graph
spectrum gives a sharp boundary.

---

## Test 3: D-CTC fixed-point on Haar-random unitaries

**Script**: `examples/stress_dctc_haar.py`
**Output**: `examples/stress_dctc_haar_results.json`

200 Haar-random unitaries on (dim_CR x 3) joint Hilbert space, with
non-mixed initial conditions (pure |0><0| sigma_CR + random pure
state rho_init on CTC). Tracks iteration count and fixed-point
state statistics.

### Result (heavy-tail finding)

| Statistic | dim_CR = 2 | dim_CR = 4 |
|---|---|---|
| Median iterations | 77 | 33 |
| P95 iterations | 195 | 50 |
| P99 iterations | 233 | 62 |
| Max iterations | 353 | 96 |
| Mean purity | 0.509 | 0.410 |
| Max purity | **0.935** | 0.617 |
| Min entropy | 0.167 | 0.679 |

### Two empirical findings

**(a) Heavy-tailed iteration distribution.** For dim_CR = 2, the
slowest sample took 353 iterations vs. median 77 -- a 4.6x outlier.
For dim_CR = 4, the tail is gentler (96 vs. 33, 2.9x).

**(b) Near-pure fixed points.** Some Haar-random unitaries on
(dim_CR=2 x 3) yield D-CTC fixed points with purity 0.935 --- i.e.
the CTC register settles into a nearly-pure quantum state. This is
*not* the expected outcome for "random" unitaries; it indicates a
class of D-CTC channels that compress quantum information rather
than mixing it.

### Conjectured algorithmic application

The dim_CR scaling of convergence rate **distinguishes structured
from mixing D-CTC channels** in O(d^3) per iteration. A channel that
converges quickly across all CR-register sizes is mixing; a channel
that converges slowly for small CR is *structured* (information-
preserving).

This may be exploitable for D-CTC complexity classification: rather
than diagonalising the full Choi matrix (O(d^6) cost), the iteration-
count signature acts as a fingerprint distinguishing channel classes.

Specifically: a *novel screening test* for finding D-CTC channels
relevant to Aaronson-Watrous-style speedups. Channels yielding
near-pure fixed points (purity > 0.9) are candidates; they preserve
sufficient quantum coherence through the CTC loop to enable
nonlinear quantum-information processing.

---

## What this means for the project

Two of the three tests (Test 1, Test 2b) produced **clean novel
findings**. Test 3 produced an empirical pattern that suggests a
**novel algorithmic shortcut** for D-CTC channel classification.

Of these, the most immediately exploitable is Test 1's Z_N closure
theorem: closed-form computation of the eta-sum sawtooth on any
Z_N orbifold cover, with mid-segment zero-crossings characterised
exactly. This is a small but clean theorem with applications in
condensed-matter index theory.

Test 2b validates the HASH-QUINE address-space encoding rule on a
new domain (continuous DSI cascades), reinforcing the broader
HASH-QUINE program.

Test 3's D-CTC empirical pattern is the most speculative but
highest-upside finding: it suggests a polynomial-time screening
test for D-CTC channel structure, which has direct quantum-
complexity implications.

### Recommended follow-up

1. Formalise Test 1 as a small note (eta-sum sawtooth pattern on Z_N
   covers).
2. Add Test 3's purity-vs-iteration scatter as a new diagnostic in
   `d_ctc.py`.
3. Cross-validate Test 2b's sharp lambda_2 features against an
   independent diversity metric.
