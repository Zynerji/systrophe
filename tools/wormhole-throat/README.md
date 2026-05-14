# wormhole-throat

**Traversable-wormhole-throat scanner on the van Stockum exterior + Brown-Maclay Casimir energy density at each throat.**

Wraps `systrophe.wormhole_throat` and `systrophe.casimir_throat` in
an LPAnalyser-style explorer.

## What a throat is here

For the Morris-Thorne-style wormhole interpretation of the van Stockum
exterior, the *throat* is the locus where the effective shape function
b_eff(r) = r, equivalently L(r) = 0 — i.e. the CTC-band boundaries.

This tool:

* scans for those candidate throats,
* checks the flaring-out condition b'(r_t) < 1 at each,
* reports the redshift function φ(r_t) (real only where F > 0),
* gives the Z₃-cover quotient interpretation (cylinder axis as the
  fixed locus of a 3-fold rotation symmetry),
* computes the Brown-Maclay Casimir tensor at the throat for a given
  plate-separation d and reports the topological/standard ratio.

## API

```python
from wormhole_throat import (
    WormholeThroatExplorer, casimir_at_throat,
)

we = WormholeThroatExplorer(omega=2.0, R=1.0)
candidates = we.candidate_throats()          # list of r where L(r) = 0
reports = we.report_all()                    # ThroatReport per candidate

cas = casimir_at_throat(omega=2.0, R=1.0, r_throat=candidates[0],
                          plate_separation_d=1.0)
print(cas.brown_maclay_energy_density_flat)  # negative, scales -1/d^4
print(cas.topological_coefficient)           # Z_3 holonomy contribution
```

## Tests

12 tests, all offline, fast (< 2 s):

```
PYTHONPATH=src:tools/wormhole-throat python -m pytest \
    tools/wormhole-throat/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
