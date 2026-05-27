# Knopp Drive API reference

The Knopp Drive lives in `systrophe.knopp.knopp_drive` with supporting modules
in `systrophe.knopp.knopp_traversal`, `systrophe.geometry.tipler_krasnikov_hybrid`,
`systrophe.geometry.krasnikov_tube`, `systrophe.qftcs.feedback_amplified_shell`,
`systrophe.geometry.horn_toroidal_warp`, `systrophe.geometry.krasnikov_pair`, and
`systrophe.geometry.krasnikov_ring`.

## High-level interface — `KnoppDrive`

```python
from systrophe.knopp.knopp_drive import KnoppDrive

drive = KnoppDrive(
    Q=100.0,            # cavity quality factor
    epsilon_horn=0.2,   # horn-twist amplitude in [0, 1)
    theta_0_horn=0.0,   # steering axis (radians)
    omega=1.0,          # Tipler seed angular velocity
    R_cylinder=1.0,     # Tipler seed radius
    alpha_wall=4.0,     # Krasnikov wall sharpness
    sigma_shell=4.0,    # bubble shell thickness
    v_s=1.0,            # apparent shell velocity
    R_bubble=1.0,       # outer bubble radius
)
```

### Methods

| Method | Returns | Purpose |
|---|---|---|
| `budget(r_orbit=None)` | `KnoppDriveBudget` | Instantaneous engineering budget at the given (or configured) orbit radius. |
| `journey(distance, n_steps=80)` | `KnoppTraversalReport` | End-to-end traversal of given distance with full energy accounting. |
| `is_pfenning_ford_compatible(distance=1.0)` | `bool` | True iff the Knopp Drive respects the P-F quantum inequality at every point along the journey. |
| `steering_vector()` | `(p_x, p_y)` | Steering dipole from the horn-toroidal twist. |
| `is_inside_band(r_orbit=None)` | `bool` | True iff the orbit lies inside a Tipler CTC band (zero exotic matter). |
| `summarise(r_orbit=None)` | `str` | One-line budget summary suitable for logging. |

### Headline shortcut

Inside any Tipler CTC band the composite exotic-matter requirement
collapses to exactly zero:

```python
drive = KnoppDrive(Q=100.0, epsilon_horn=0.2)
b = drive.budget(r_orbit=1.5)
assert b.composite_E_neg == 0.0  # zero exotic matter inside band
```

For an Earth--Mars equivalent journey (`L=0.52` AU geometric units):

```python
report = drive.journey(distance=0.52)
assert report.exotic_matter_total == 0.0   # entire journey inside band
assert report.inside_band_fraction == 1.0
```

## Functional interface

```python
from systrophe.knopp.knopp_drive import (
    knopp_budget,
    KnoppDriveConfig,
    summarise_knopp_budget,
    novelty_scan,
)

cfg = KnoppDriveConfig(Q=100.0, epsilon_horn=0.2, r_orbit=1.5)
b = knopp_budget(cfg)
print(summarise_knopp_budget(b))
# Knopp Drive @ r=1.50: E_neg=-0.0000e+00, P_drive=+6.7742e-07, ...
```

`KnoppDriveConfig` is an immutable dataclass with these fields:

| Field | Default | Meaning |
|---|---|---|
| `omega` | 1.0 | Tipler seed angular velocity |
| `R_cylinder` | 1.0 | Tipler seed cylinder radius |
| `alpha_wall` | 4.0 | Krasnikov wall sharpness |
| `Q` | 10.0 | Feedback cavity quality factor |
| `sigma_shell` | 4.0 | Bubble shell thickness |
| `epsilon_horn` | 0.2 | Horn-twist amplitude |
| `theta_0_horn` | 0.0 | Horn-twist axis (radians) |
| `r_orbit` | 1.5 | Tube worldline radius in the LP frame |
| `v_s` | 1.0 | Apparent shell velocity |
| `R_bubble` | 1.0 | Outer bubble radius |

`KnoppDriveBudget` is an immutable dataclass returned by `knopp_budget`:

| Field | Meaning |
|---|---|
| `config` | The originating `KnoppDriveConfig`. |
| `tipler_gate_factor` | $(1-c\,T(r))_{+}$; zero inside CTC band. |
| `feedback_factor` | $1/Q^{2}$. |
| `horn_amplification` | $1+\epsilon$ (worst-case angular weighting). |
| `krasnikov_bare_E_neg` | Integrated Krasnikov wall NEC without any subtraction. |
| `composite_E_neg` | Final $|E_{\mathrm{neg}}|^{\mathrm{Knopp}}$ (zero inside band). |
| `sustained_drive_power` | $P_{\mathrm{drive}}$ at saturation. |
| `pfenning_ford_compatible` | True iff respect Pfenning-Ford. |
| `steering_vector_pxpy` | $(p_{x},p_{y})$ horn-twist dipole. |
| `steering_magnitude` | $\sqrt{p_{x}^{2}+p_{y}^{2}}$. |
| `natural_frequency` | $f_{0}=c/(2\pi\sigma)$. |
| `cavity_lifetime_tau` | $\tau=Q/f_{0}$. |
| `saturation_field_energy_value` | $|E_{\mathrm{shell}}|=|E_{\mathrm{Krasnikov}}|/Q$. |
| `parametric_gain_required` | $g=\log Q/\tau$. |

## Traversal — `knopp_traversal`

```python
from systrophe.knopp.knopp_traversal import knopp_traversal, knopp_traversal_Q_sweep

# Single-distance traversal
report = knopp_traversal(cfg, distance=10.0, n_steps=80)
print(f"E_total = {report.total_energy_budget:.3e}")
print(f"inside_band_fraction = {report.inside_band_fraction:.2f}")

# Sweep Q to find the P-F bound
sweep = knopp_traversal_Q_sweep(distance=10.0, Q_range=(1.0, 2000.0))
print(f"P-F flip at Q ~ {sweep['flip_Q']}")
```

`KnoppTraversalReport` fields:

| Field | Meaning |
|---|---|
| `distance` | Apparent traversal distance. |
| `v_s_apparent` | Apparent shell velocity. |
| `coord_time_total` | External coordinate-time elapsed. |
| `proper_time_total` | Craft proper-time elapsed. |
| `n_band_segments` | Number of distinct CTC bands crossed. |
| `inside_band_fraction` | Fraction of journey inside any CTC band. |
| `exotic_matter_total` | Integrated $\lvert E_{\mathrm{neg}}\rvert$. |
| `sustained_drive_power` | $P_{\mathrm{drive}}$ at saturation. |
| `total_energy_budget` | $P_{\mathrm{drive}}\cdot t_{\mathrm{coord}}$. |
| `pfenning_ford_compatible` | P-F flag for the full journey. |

## Catcher integration

Every module exposes a `novelty_scan(...)` function that runs the
address-space novelty catcher on its natural parameter sweep:

```python
from systrophe.knopp.knopp_drive import novelty_scan

res = novelty_scan(
    r_orbit_range=(1.05, 12.0), n_r=30,
    Q_range=(2.0, 50.0), n_Q=8,
    epsilon_range=(0.0, 0.8), n_eps=8,
)
print(f"verdict: {res['novelty_verdict']}")
# verdict: novel_structure   <- the CTC-band exit is the catcher hit
```

## Hardware-confirmed result

The Knopp Drive's headline shortcut was hardware-confirmed on IBM
Quantum's 156-qubit `ibm_marrakesh` Heron-r2 processor (Marrakesh
batch 6, job `d8183b7oha1c73bk1n60`, 2026-05-11). HW results
reproduce simulator predictions to TV $\le 0.05$ at every $r$ in an
8-point sweep across the first Tipler CTC band exit; the catcher
flags `r3 -> r4` (the band exit) as a sharp Hamming transition
(step$=12$). See `experiments/marrakesh_batch_6_knopp_drive.py` and
`paper/knopp_drive.pdf` Section "Hardware confirmation on
\texttt{ibm\_marrakesh}".

## See also

- `paper/knopp_drive.pdf` — full whitepaper (~11 pages, 4 figures)
- `examples/knopp_drive_walkthrough.py` — six-configuration demo
- `examples/knopp_drive_earth_mars.py` — Earth-Mars journey
- `examples/warp_drive_comparison.py` — comparison vs Alcubierre / Krasnikov / Lentz / B-M
- `experiments/marrakesh_batch_6_knopp_drive.py` — hardware experiment
- `tests/test_knopp_drive.py` — unit tests
- `tests/test_knopp_drive_integration.py` — integration tests
