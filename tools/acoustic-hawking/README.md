# acoustic-hawking

**Unruh acoustic analog of the Lewis-Papapetrou exterior + Steinhauer
2019 BEC benchmark.**

Wraps `systrophe.analogs.acoustic_metric` and `systrophe.analogs.acoustic_hawking_spectrum`
in an LPAnalyser-style API:

* Locate the acoustic horizon (F = 0) of a van Stockum exterior.
* Compute its acoustic surface gravity and Hawking temperature.
* Self-check: acoustic T_H = gravitational T_H (Unruh identification).
* Audit CTC ↔ supersonic equivalence at sampled radii.
* Compute the phonon spectrum + total emission power.
* Plug BEC-vortex parameters in and compare predicted T_H to the
  Steinhauer 2019 measurement of 0.124 ± 0.012 nK.

## Why this exists

The acoustic-LP identification is the cleanest example of analog
gravity inside Systrophe: F = 0 is *simultaneously* the chronology
horizon (CTC band boundary) and the acoustic-horizon (sound cone
closes). This tool packages the diagnostic stack so a downstream
consumer can prototype BEC analog black-hole experiments without
re-reading the upstream Systrophe modules.

## Demo result

For van Stockum (omega=2, R=1):
* r_H ≈ 1.405, κ = 2.000, T_H = 0.318 (both acoustic and gravitational)
* CTC ↔ supersonic consistency: ✓ on 50 sampled radii

For a Rb-87 BEC vortex (omega=1000 s⁻¹, R=1 μm, n=1e18 m⁻³):
* T_predicted ≈ 0.084 nK
* T_steinhauer = 0.124 ± 0.012 nK
* σ deviation = 3.36 → **not consistent at 3σ**

The honest reading: at these naive parameters the LP analog
underestimates Steinhauer's measurement. A parameter sweep is needed
to find a region of (omega, R, n) that lands within the measurement
window — left as an exercise (this tool ships the comparison, not
the optimisation).

## API

```python
from acoustic_hawking import (
    AnalogHorizonAnalyser, compute_phonon_spectrum,
    benchmark_against_steinhauer_2019,
)

ah = AnalogHorizonAnalyser(omega=2.0, R=1.0)
rep = ah.report()                              # one-shot report
print(rep.horizon_r, rep.T_hawking_acoustic)

spec = compute_phonon_spectrum(
    omega=2.0, R=1.0, r_horizon=rep.horizon_r,
)
print(spec.total_emission_power)

bench = benchmark_against_steinhauer_2019(
    omega=1000.0, R=1e-6, n_density=1e18, atom_mass=1.443e-25,
)
print(bench.sigma_deviation, bench.consistent_with_measurement)
```

## Tests

15 tests, all offline:

```
PYTHONPATH=src:tools/acoustic-hawking python -m pytest \
    tools/acoustic-hawking/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
