# warp-drive-zoo

**Unified head-to-head comparator for canonical warp metrics.**

Wraps `systrophe.geometry.alcubierre`, `systrophe.geometry.bobrick_martire`, and
`systrophe.geometry.lentz_soliton` behind a common `WarpDrive` protocol so the
three can be compared side-by-side on the same axes:

* `nec_radial(x, rho)` — T_{kk} along outward radial null geodesic
* `energy_density(x, rho)` — T_tt at a point
* `total_negative_energy(box_half_size, n_grid)` — integral of
  `min(T_tt, 0)` over a cylindrical box (the exotic-matter requirement)
* `wall_location()` — drive-specific canonical wall sample point

`compare_drives([d1, d2, ...])` runs each of the three at its wall,
integrates each total negative energy, and returns a ranking. The
ranking reproduces the canonical literature picture:

* **Alcubierre**: NEC violated at wall; exotic-matter mass scales
  as v_s² (Pfenning-Ford 1997).
* **Bobrick-Martire** with m_ADM > 0: NEC respected; zero exotic
  matter (the breakthrough).
* **Bobrick-Martire** with m_ADM < 0: NEC violated; exotic-matter
  mass exceeds Alcubierre's.
* **Lentz**: NEC respected; positive-energy two-soliton.

## Demo output

```
drive                  wall (x,rho)         NEC       T_tt(wall)   |exotic E|  NEC_viol?
alcubierre             (1.00, 0.00)       -0.318      -1.59e-01    3.35e-01    True
bobrick_martire_pos    (1.00, 0.00)       +0.001      +1.34e-03    0.00e+00    False
bobrick_martire_neg    (1.00, 0.00)       -0.001      -1.34e-03    3.87e+00    True
lentz                  (0.00, 1.50)       +0.309      +1.54e-01    0.00e+00    False
```

## API

```python
from warp_drive_zoo import (
    AlcubierreDrive, BobrickMartireDrive, LentzDrive, compare_drives,
)

drives = [
    AlcubierreDrive(v_s=1.0, R=1.0, sigma=8.0),
    BobrickMartireDrive(m_ADM=1.0, R=1.0, sigma=4.0),
    LentzDrive(v_s=0.7, sigma=4.0),
]
cmp_ = compare_drives(drives, box_half_size=2.5, n_grid=20)
print(cmp_.nec_violated)
print(cmp_.ranking_by_exotic_matter)
```

## Quantum-Realizability Scorer (Ford-Roman QI budget axis)

`qi_scorer` + `qi_registry` add a uniform **`qi_normalized_score`** across the
whole spacetime + warp registry:

```
qi_normalized_score = |most-negative T_kk excursion| / |Ford-Roman QI bound(tau)|
score < 1  ->  REALIZABLE     (classically sourceable)
score > 1  ->  QI-FORBIDDEN   (needs exotic matter)
```

`ford_roman_qi_bound(tau)` generalises
`systrophe.qftcs.anec_bound.anec_quantum_inequality_bound` off the
van-Stockum hardcoding to a source-agnostic temporal sampling time `tau`
(same `3/(32 pi^2)` constant; passing `tau = r_max - r_min` reproduces the
old number exactly). Where no Hadamard vacuum is available, `T_kk` is built
from the published exotic density (Alcubierre closed form; Morris-Thorne
throat `b'/(8 pi r_t^2)`).

Scored at the macroscopic sampling time (the scale over which the negative
energy must be sustained):

```
van_stockum      score=  0.0000  realizable   (positive-energy dust; all 4 ECs hold)
godel            score=  0.0000  realizable   (positive-energy dust + Lambda)
gott             score=  0.0000  realizable   (positive-tension strings; vacuum exterior)
kerr             score=  0.0000  realizable   (exact vacuum, T_mu_nu = 0)
alcubierre       score= 16.7552  FORBIDDEN    (wall density -(v_s^2/32pi)(df/dr)^2 < 0)
wormhole_throat  score=  4.1888  FORBIDDEN    (Morris-Thorne local NEC violation)
```

Alcubierre's geometrized exotic-energy principal (1 m luminal bubble,
kappa = 1/12) is **1.01e43 J = 0.059 Jupiter mass-energies** — the same
~Jupiter-mass floor cited in `knopp_ratchet.feasibility_report` /
`warp_geometry`, corroborating the QI-forbidden verdict. The address-space
novelty catcher + shuffled-label surrogate (`qi_lambda2_separation`)
separates the classical cluster from the exotic cluster at **z = 2.29**
(true within-group Hamming 10.0 vs surrogate 15.07 +/- 2.22; full graph
disconnected at radius 6).

```python
from warp_drive_zoo import (
    score_full_registry, qi_lambda2_separation, compare_drives_with_qi,
)
reg = score_full_registry()                  # list[QIScore] on one axis
sep = qi_lambda2_separation(reg)             # catcher + surrogate null
```

## Tests

14 base tests + 21 QI-scorer tests, all offline, fast:

```
PYTHONPATH=src:tools/warp-drive-zoo python -m pytest \
    tools/warp-drive-zoo/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
