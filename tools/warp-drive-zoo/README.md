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

## Tests

14 tests, all offline, fast (< 2 s):

```
PYTHONPATH=src:tools/warp-drive-zoo python -m pytest \
    tools/warp-drive-zoo/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
