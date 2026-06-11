# schumann-elf-catcher

**The Systrophe address-space catcher stack, reparameterized for the
Schumann-resonance / ELF band, with a surrogate (phase-randomized) null.**

Point it at a raw ELF magnetometer recording and it reports four things:

1. **Transient bursts** — chunked address-space catcher over a whitened →
   constant-Q time-frequency representation. Flags windows whose per-band
   log-energy fingerprint changes sharply (Q-bursts / ELF transients).
2. **A surrogate null for every burst flag** — phase-randomized surrogates
   preserve the power spectrum exactly (induction-coil response, the Schumann
   comb, mains lines) but destroy phase coherence, so genuine transients
   vanish. A small p-value means the transient structure is *not* explained
   by the recording's spectrum alone. **Without this, a "detection" is
   meaningless.**
3. **Slow amplitude trends** — `systrophe.catchers.growth_catcher` on a per-bin
   band-RMS track, with a built-in permutation null (z, p).
4. **Spectral regime change-points** — `systrophe.catchers.novelty_catcher` on per-bin
   sub-band spectra.

## Provenance

- Detection primitive: the address-space rank/value-thermometer encoding from
  `systrophe.catchers.novelty_catcher` (the framework's load-bearing catcher).
- Pipeline shape: reuses `tools/gw-burst-catcher`'s `whiten_strain` +
  `q_transform` front end, reparameterized from the LIGO kHz band to the ELF
  band (~6–45 Hz at ~256 Hz sample rate).
- Two changes the GW tool lacks but honest ELF anomaly hunting needs:
  **chunked streaming** (the O(n²) catcher can't take a whole hour at once)
  and a **phase-randomized surrogate null** (the GW tool reports a raw
  Hamming step with no null).

## Data: Sierra Nevada ELF station

Validated against the public Sierra Nevada ELF station archive
(Rodríguez-Camacho et al., *Computers & Geosciences* 2022), Zenodo records
[6348691](https://zenodo.org/records/6348691) (2014) and siblings, CC-BY-4.0.
Each hourly file is 921,600 little-endian int16 samples at **256 Hz**
(sampling period 3906 µs) = exactly one hour; sensor 0 = NS, sensor 1 = EW.
A companion `*_info.txt` carries the sampling period and first-sample UTC time.

Because the archive ships as one ~27 GB ZIP per year, `pull_one_hour.py` uses
HTTP range requests (`remotezip`) to extract a single hourly file without
downloading the whole archive.

## Quick start

```python
from schumann_elf_catcher import load_sierra_nevada, burst_null_test

rec = load_sierra_nevada(".../smplGRTU1_sensor_1_1403081849")
res = burst_null_test(rec.samples, rec.sample_rate, n_surrogates=120)
print(res.max_hamming_step, res.null_p_value)   # statistic + null p-value
```

Or the full report:

```
python run_sierra_nevada.py <data_file> --surrogates 120 --json out.json
```

## Validation

`tests/test_adapter.py` (synthetic, no network): phase randomization
preserves the PSD; an injected ELF transient is localized in time; the
surrogate null **supports** a real injected burst (small p) and **rejects**
pure colored noise (large p); the growth catcher flags an amplitude ramp.

## What this is NOT

Not a calibrated SR observatory pipeline. No instrument-response
deconvolution, no glitch veto, no absolute false-alarm rate, no
multi-station coincidence (single-channel here). Every flag is a *candidate*
that the surrogate null supports or rejects. Confirming a real, "hidden"
signal needs a second independent station in coincidence and instrument-level
vetoes. There is no mechanism here for detecting encoded messages or any such
thing — only statistical structure in the EM field.

## License

MIT, inherited from the Systrophe parent package.
