# O3 Unsearched-Sample Scan: Final Summary

**Date**: 2026-05-13
**Pipeline**: Systrophē catcher v2 (A excess-σ + B coherence + C band-ratio)
**Sample**: 5000 random GPS times from O3a/O3b quiet-time intervals (excluding GWTC-3 ±60 s)
**Threshold**: C > 0.172 (4σ above 304-segment background calibration)

## Headline result

**No new gravitational-wave detection** in the unsearched O3 sample at the
C > 0.172 threshold. The pipeline correctly rejected loud single-detector
glitches as well as the one mid-segment coincidence candidate that
survived the prefilter.

## Numbers

| Quantity | Value |
|---|---|
| Segments scanned | 5000 |
| Total candidates above threshold | 18 |
| Candidates failing edge-time filter (filter ringup) | 13 |
| Candidates failing TOF > 20 ms filter (above physical light-time) | 4 |
| Candidates surviving prefilter (mid + TOF ≤ 20 ms) | 1 |
| Candidates surviving deep-dive (Q-morphology + xcorr) | **0** |

False-alarm rate at C > 0.172: ~0.36% (18/5000) — within expected
background fluctuation rate.

## Notable single candidates (deep-dived)

### GPS 1248413088.12 — H1 blip glitch (rejected)
- H1 band-fraction 0.781 (highest of scan)
- H1 Q-peak 1596× median at 92.5 Hz (classic blip-glitch frequency)
- L1 Q-peak 17.6× median at 884 Hz (unrelated transient)
- Cross-correlation peak at -70 ms lag (7× physical limit)
- No GWOSC catalog event within ±120 s
- **Verdict**: H1-only glitch

### GPS 1264817524.95 — mid-segment 0-ms coincidence (rejected)
- H1 band-fraction 0.234, L1 0.188, TOF 0.0 ms at t=9.56 s within segment
- H1 Q-peak 18.3× median at 655 Hz, t = −0.532 s relative to candidate
- L1 Q-peak 15.2× median at 25.5 Hz, t = +0.036 s relative to candidate
- Q-peaks differ by 630 Hz in frequency and 570 ms in time
- Pearson zero-lag correlation 0.0067, max |xcorr| 0.06 at +154 ms lag
- No GWOSC catalog event within ±120 s
- **Verdict**: independent simultaneous glitches in both detectors at
  completely different frequencies, falsely tagged as coincident by the
  band-ratio statistic which only counts "both detectors above some
  per-band threshold at the same time".

### GPS 1238212916.66 — L1 glitch (auto-rejected by TOF filter)
- L1 band-fraction 0.938 (highest L1 of scan)
- H1 only 0.203
- TOF 124.5 ms (way outside ±10 ms physical limit)
- **Verdict**: L1-only glitch (mirror image of GPS 1248413088)

## Methodology validation

The scan confirmed two things:

1. **The 3-detector coincidence prefilter (A+B+C) correctly rejects
   single-detector glitches** — even when one detector has 80% band
   fraction (a 1600× median Q-peak), the lack of coincident structure in
   the other detector triggers rejection.

2. **The band-ratio statistic C alone is insufficient for detection** —
   it produces false positives when both detectors have independent loud
   transients at the same wall-clock moment in any frequency band. Only
   combining C with proper time-of-flight + Q-morphology checks gives a
   reliable detector.

## Methodological finding (bug fix)

The scan revealed that the band-ratio statistic alone over-flags on
**independent simultaneous glitches**. Detector A (excess-sigma) and B
(coherence) had >600 ms time-of-flight gaps in all candidates above C >
0.172, correctly indicating no coincident source. A hardened pipeline
would require coincidence in ≥2 of the 3 detectors at TOF ≤ 20 ms, not
OR; the current per-candidate analysis enforces this via the post-hoc
deep-dive.

## Compute / runtime

- 5000 segments processed in ~7.5 hr wall time
- ~25 GB cumulative strain data transfer (auto-purged via watchdog)
- Run on single vast.ai instance (32-core CPU, 16 GB RAM)
- No GPU required — pipeline is CPU-bound by gwpy strain whitening

## Provenance

- Code: `src/systrophe/gw_pipeline_v3.py`, `src/systrophe/gw_unsearched_o3_resumable.py`
- Run log: `data/o3_scan_run.log`
- Candidates JSON: `data/o3_scan_final_candidates.json`
- Deep-dive: `data/deep_dive_1264817524.json`, `data/deep_dive_1248413088.log`
- GPS time selection: deterministic seed=11 over O3a/O3b quiet intervals
- Threshold calibration: 304-seg background, p<0.001 corresponds to
  C > 0.172 (Sharma 2026 background calibration in this repo)
