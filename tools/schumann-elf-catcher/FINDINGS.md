# Findings — schumann-elf-catcher on Sierra Nevada ELF (first real hour)

**Recording:** Sierra Nevada ELF station, 2014-03-08 18:49:29.177 UTC, 1 hour,
256 Hz, both horizontal magnetometer axes (sensor_0 = NS, sensor_1 = EW),
sample-aligned. Source: Zenodo 6348691 (CC-BY-4.0), pulled one hour at a time
via ranged ZIP (`pull_one_hour.py`).

## Detector

Systrophe address-space catcher reparameterized for the ELF band (6-45 Hz):
whiten (Welch PSD) -> constant-Q time-frequency -> sliding 500 ms / 100 ms
windows -> value-thermometer address per 24 sub-bands -> consecutive Hamming
step. Headline statistic = largest Hamming step. **Calibrated against a
phase-randomized surrogate null (n=120)** that preserves the exact power
spectrum (coil response, Schumann comb, mains) and destroys only phase.

Synthetic validation (tests/test_adapter.py, 4/4): injected ELF transient
localized within 2.5 s; null supports a real injection (p<=0.10) and rejects
pure colored noise (p>0.10).

## Results

| Quantity | EW (sensor_1) | NS (sensor_0) |
|---|---|---|
| Largest Hamming step | 18 @ 1741.6 s | 20 @ 2604.8 s |
| Surrogate-null p-value | 0.0083 (0/120) | 0.0083 (0/120) |
| Slow amplitude trend | stationary | stationary |
| Regime change-point (no null) | 1920 s | 3150 s |

**Cross-axis coincidence (the load-bearing check):**

- Zero-lag Pearson r of the two step trajectories = 0.164 (best lag = 0).
- At EW's peak, NS is at ~99th pct; at NS's peak, EW is at ~99th pct.
- Windows above each axis's 99th percentile: EW=228, NS=245, **joint=27 vs
  ~1.6 expected by chance (~17x excess)**.

## Interpretation

1. Each axis carries statistically-real, phase-coherent transient structure
   that the power spectrum alone cannot reproduce (p<0.01). "Hidden" in the
   legitimate sense: invisible to a PSD/spectrogram view.
2. The ~17x excess of *time-coincident* transients across two physically
   separate orthogonal coils is the signature of genuine broadband ELF
   transients (Q-bursts / energetic lightning-sprite events), not per-channel
   electronic glitches.
3. The *loudest* event differs per axis because a transient's H-field projects
   onto NS vs EW by arrival azimuth -- characteristic of real EM transients;
   a common-mode digitizer glitch would be identical on both (r would be ~1).

## Honest limits

- Both axes share the GRTU digitizer + timing, so two-axis coincidence does
  not fully exclude common-mode electronics. Gold standard = a second
  independent station (the tool inherits a two-station TOF coincidence hook
  from gw-burst-catcher).
- No instrument-response deconvolution, glitch veto, or absolute FAR.
- Regime change-points lack a surrogate null; reported as candidates only.
- No slow trend is expected within a single hour (diurnal variation needs
  many hours).
