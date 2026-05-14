# floquet-engineer

**Time-periodic Z₃ Floquet quasi-energies + CTC stability gap.**

Wraps `systrophe.floquet_mobius` and `systrophe.floquet_engineering`
in a single FloquetEngineer object.

Use case: drive the three Z₃ branches of a Möbius-Z₃ cover with a
time-periodic perturbation, ask whether the CTC sector gets gapped
out (Floquet stabilisation of chronology protection).

## API

```python
from floquet_engineer import FloquetEngineer
import numpy as np

fe = FloquetEngineer(branch_energies=[1.0, 1.1, 1.2], hopping=0.1)

# Single-point analysis
res = fe.analyze(drive_amp=0.5, omega_drive=1.0)
print(res.quasi_energies)

# Sanity checks
print(fe.static_limit(omega_drive=1.0))
print(fe.z3_symmetry(drive_amp=0.2, omega_drive=1.0))

# 2D sweep over (drive_amp, omega_drive)
rep = fe.sweep(
    drive_amps=np.linspace(0.05, 0.6, 10),
    omega_drives=np.linspace(0.5, 2.0, 10),
)
print(rep.max_gap, rep.stabilisation_efficacy)
print(rep.resonances)
```

## Tests

11 tests, all offline:

```
PYTHONPATH=src:tools/floquet-engineer python -m pytest \
    tools/floquet-engineer/tests/ -q
```

## License

MIT, inherited from the Systrophe parent package.
