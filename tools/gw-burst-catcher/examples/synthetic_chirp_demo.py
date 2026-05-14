"""Offline smoke demo: inject a chirp, locate it via the catcher."""

from __future__ import annotations

import pathlib
import sys

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from gw_burst_catcher import inject_chirp, make_gaussian_noise
from gw_burst_catcher.detection import detect_burst_in_strain


def main():
    sr = 1024
    t_truth = 4.0
    strain = make_gaussian_noise(duration_s=8.0, sample_rate=sr, seed=0)
    info = inject_chirp(strain, sample_rate=sr, t_inject_s=t_truth,
                          f_start=50.0, f_end=250.0,
                          amplitude=10.0, duration_s=0.3)
    print(f"Injected chirp: t={info.t_inject_s}s  "
          f"f={info.f_start}->{info.f_end}Hz  A={info.amplitude}")
    res = detect_burst_in_strain(strain, sample_rate=sr)
    err = abs(res.max_hamming_step_time_s - t_truth)
    print(f"  detected at:   t={res.max_hamming_step_time_s:.3f}s  (error {err*1000:.0f}ms)")
    print(f"  verdict:       {res.verdict}")
    print(f"  max_hamming:   {res.max_hamming_step}")
    print(f"  n_sharp_features: {res.n_sharp_features}")
    print(f"  n_windows:     {res.n_windows}")


if __name__ == "__main__":
    main()
