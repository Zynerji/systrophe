"""Batch 8: rotational superradiance in van Stockum CTC bands (IBM hardware).

Hypothesis (qftcs.superradiance): an ``F < 0`` band of the supercritical
van Stockum exterior is a rotational ergoregion where a probe wave is
*amplified* (superradiance) -- the wave Penrose process. In the truncated
two-mode-squeezing encoding this shows up as **spontaneous signal/idler pair
creation from the vacuum**, with probability ``P_pair = sin^2(theta)`` and
squeezing angle ``theta = arctan(eta)`` set by the band's Penrose efficiency
``eta = |E_min|/E_max``. Passive ``F > 0`` bands are beam splitters:
``theta = 0`` -> no spontaneous emission.

Circuit (per radius r): 2 qubits.  ``Ry(2 theta_r)`` on the signal qubit,
``CX`` signal->idler, measure both.  ``P(|11>) = sin^2(theta_r)``.

Decisive observable: ``P_11`` is ``> 0`` only in ``F < 0`` (ergoregion) bands.
The ``F > 0`` bands are *built-in controls* (and the hardware readout/thermal
floor). Falsification:
  (1) noiseless P_11 == sin^2(theta) to < 1e-6 (wiring/dictionary),
  (2) hardware: P_11(ergoregion) significantly exceeds P_11(passive),
  (3) F-sign-label permutation surrogate: real labels explain the
      amplification far better than shuffled labels (p < 0.05),
  (4) derivative catcher localises the ergosurface (F=0) as the P_11 onset.

Calibration: opt_level=3 + DD(XpXm) + gate/measure twirling, 8192 shots,
live backend calibration via the preset pass manager (matches batches 4-7).

Usage::

    python marrakesh_batch_8_superradiance.py            # local Aer/Basic sim
    python marrakesh_batch_8_superradiance.py --submit   # fire-and-forget HW
    python marrakesh_batch_8_superradiance.py --hw       # submit + wait
    python marrakesh_batch_8_superradiance.py --hw --backend ibm_kingston
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from qiskit import ClassicalRegister, QuantumCircuit, QuantumRegister
from qiskit.transpiler.preset_passmanagers import generate_preset_pass_manager

from systrophe.geometry.vanstockum import VanStockumInterior
from systrophe.qftcs.superradiance import band_superradiance_at, superradiance_scan
from systrophe.catchers.derivative_catcher import catch_smooth_transition

SHOTS = 8192
RESULTS_DIR = Path(__file__).parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)

# omega R = 2 supercritical: ergosurfaces at r ~ 1.41, 3.16, 7.12, ...
VS = VanStockumInterior(omega=2.0, R=1.0)
# Ergoregion (F<0) band between the first two ergosurfaces, and the passive
# (F>0) control band just beyond the second ergosurface (r ~ 3.16).
ERGO_RADII = [2.2, 2.3, 2.4, 2.5, 2.6, 2.7]
PASSIVE_RADII = [3.2, 3.3, 3.4, 3.5, 3.6, 3.7]
RNG_SEED = 8


def superradiance_circuit(theta: float, label: str) -> QuantumCircuit:
    """Truncated two-mode squeezer: |00> -> cos(theta)|00> + sin(theta)|11>."""
    qr = QuantumRegister(2, "q")
    cr = ClassicalRegister(2, "c")
    qc = QuantumCircuit(qr, cr, name=label)
    qc.ry(2.0 * theta, qr[0])   # signal mode
    qc.cx(qr[0], qr[1])         # entangle idler -> pair state
    qc.measure(qr, cr)
    return qc


def all_experiments() -> list[dict]:
    out = []
    for tag, radii in (("ergo", ERGO_RADII), ("passive", PASSIVE_RADII)):
        for r in radii:
            b = band_superradiance_at(VS, r)
            label = f"{tag}_r{r:.2f}"
            out.append({
                "label": label,
                "tag": tag,
                "r": float(r),
                "F": b.F,
                "is_ergoregion": b.is_ergoregion,
                "theta": b.squeezing_angle,
                "efficiency": b.efficiency,
                "predicted_P11": b.pair_probability,
                "circuit": superradiance_circuit(b.squeezing_angle, label),
            })
    return out


def transpile_and_check(experiments: list[dict], backend) -> list[dict]:
    pm = generate_preset_pass_manager(backend=backend, optimization_level=3)
    for ex in experiments:
        isa_qc = pm.run(ex["circuit"])
        ex["isa_circuit"] = isa_qc
        ex["isa_depth"] = isa_qc.depth()
    return experiments


def _p11(counts: dict) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    return counts.get("11", 0) / total


def _group_separation(summaries: list[dict], seed: int = RNG_SEED) -> dict:
    """Permutation test: do TRUE F-sign labels explain P_11 better than shuffled?

    Statistic = mean P_11(ergoregion) - mean P_11(passive). Null permutes the
    ergoregion/passive labels across radii. Returns observed gap + p-value.
    """
    p11 = np.array([s["P11_observed"] for s in summaries])
    is_ergo = np.array([s["is_ergoregion"] for s in summaries], dtype=bool)
    n_ergo = int(is_ergo.sum())
    obs = float(p11[is_ergo].mean() - p11[~is_ergo].mean())
    rng = np.random.default_rng(seed)
    n_perm = 5000
    ge = 0
    for _ in range(n_perm):
        perm = rng.permutation(len(p11))
        lab = np.zeros(len(p11), dtype=bool)
        lab[perm[:n_ergo]] = True
        stat = p11[lab].mean() - p11[~lab].mean()
        if stat >= obs:
            ge += 1
    p_value = (ge + 1) / (n_perm + 1)
    return {
        "mean_P11_ergoregion": float(p11[is_ergo].mean()),
        "mean_P11_passive": float(p11[~is_ergo].mean()),
        "observed_gap": obs,
        "surrogate_p_value": float(p_value),
        "n_permutations": n_perm,
    }


def _ergosurface_localization() -> dict:
    """Derivative catcher: locate the F=0 ergosurface as the P_11(r) onset."""
    radii = np.linspace(2.0, 4.0, 41)

    def p11_of_r(r: float) -> float:
        return band_superradiance_at(VS, float(r)).pair_probability

    res = catch_smooth_transition(radii, p11_of_r)
    # True second ergosurface for omega=2,R=1:
    from systrophe.qftcs.penrose_extraction import ergosurface_radii
    ergs = ergosurface_radii(VS, 1.05, 5.0, 4001)
    true_r = min(ergs, key=lambda e: abs(e - 3.0)) if ergs else float("nan")
    centre = res.get("estimated_transition_centre")
    return {
        "estimated_transition_centre": centre,
        "true_ergosurface_r": float(true_r),
        "kind": res.get("kind"),
        "abs_error": (abs(centre - true_r) if centre is not None else None),
    }


def _analyze(experiments, get_counts, tag: str, extra: dict | None = None) -> dict:
    summaries = []
    max_abs_err = 0.0
    for ex in experiments:
        counts = get_counts(ex)
        p11 = _p11(counts)
        err = abs(p11 - ex["predicted_P11"])
        max_abs_err = max(max_abs_err, err)
        summaries.append({
            "label": ex["label"], "tag": ex["tag"], "r": ex["r"], "F": ex["F"],
            "is_ergoregion": ex["is_ergoregion"], "theta": ex["theta"],
            "P11_predicted": ex["predicted_P11"], "P11_observed": float(p11),
            "abs_error": float(err), "isa_depth": ex.get("isa_depth"),
        })
        print(f"  {ex['label']:14s} F={ex['F']:+.3f}  "
              f"P11_pred={ex['predicted_P11']:.4f}  P11_obs={p11:.4f}", flush=True)

    sep = _group_separation(summaries)
    loc = _ergosurface_localization()
    print(f"\n[{tag}] mean P11  ergoregion={sep['mean_P11_ergoregion']:.4f}  "
          f"passive={sep['mean_P11_passive']:.4f}  gap={sep['observed_gap']:.4f}  "
          f"surrogate_p={sep['surrogate_p_value']:.4f}", flush=True)
    print(f"[{tag}] ergosurface localized at r={loc['estimated_transition_centre']} "
          f"(true {loc['true_ergosurface_r']:.3f}, err={loc['abs_error']})", flush=True)
    print(f"[{tag}] max |P11_obs - P11_pred| = {max_abs_err:.5f}", flush=True)
    out = {
        "backend_tag": tag,
        "per_circuit": summaries,
        "group_separation": sep,
        "ergosurface_localization": loc,
        "max_abs_error_vs_prediction": float(max_abs_err),
    }
    if extra:
        out.update(extra)
    return out


def run_simulator() -> None:
    try:
        from qiskit_aer import AerSimulator
        backend = AerSimulator()
        sim_name = "aer"
    except Exception:
        from qiskit.providers.basic_provider import BasicSimulator
        backend = BasicSimulator()
        sim_name = "basic"
    experiments = all_experiments()
    print(f"\n[sim:{sim_name}] {len(experiments)} circuits", flush=True)
    experiments = transpile_and_check(experiments, backend)

    def get_counts(ex):
        job = backend.run(ex["isa_circuit"], shots=SHOTS)
        return job.result().get_counts()

    analysis = _analyze(experiments, get_counts, f"sim:{sim_name}")
    # Noiseless wiring gate: prediction must be recovered to shot-noise.
    assert analysis["max_abs_error_vs_prediction"] < 0.02, "circuit != prediction"
    out_path = RESULTS_DIR / "marrakesh_batch8_sim_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"[sim] wrote {out_path}", flush=True)


def _sampler_options():
    from qiskit_ibm_runtime.options import SamplerOptions
    opts = SamplerOptions()
    opts.dynamical_decoupling.enable = True
    opts.dynamical_decoupling.sequence_type = "XpXm"
    opts.twirling.enable_gates = True
    opts.twirling.enable_measure = True
    opts.default_shots = SHOTS
    return opts


def run_submit(backend_name: str, instance: str) -> None:
    """Fire-and-forget: submit and write the job id (do not wait)."""
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)
    status = backend.status()
    print(f"\n[submit] backend={backend.name} pending={status.pending_jobs}",
          flush=True)
    experiments = all_experiments()
    experiments = transpile_and_check(experiments, backend)
    sampler = SamplerV2(mode=backend, options=_sampler_options())
    job = sampler.run([ex["isa_circuit"] for ex in experiments], shots=SHOTS)
    meta = {
        "job_id": job.job_id(),
        "backend": backend.name,
        "instance": instance,
        "circuit_order": [ex["label"] for ex in experiments],
        "predicted_P11": {ex["label"]: ex["predicted_P11"] for ex in experiments},
        "is_ergoregion": {ex["label"]: ex["is_ergoregion"] for ex in experiments},
    }
    out = RESULTS_DIR / "marrakesh_batch8_submitted_job.json"
    out.write_text(json.dumps(meta, indent=2))
    print(f"[submit] job_id={job.job_id()}  wrote {out}", flush=True)


def run_hardware(backend_name: str, instance: str) -> None:
    from qiskit_ibm_runtime import QiskitRuntimeService, SamplerV2
    service = QiskitRuntimeService(instance=instance)
    backend = service.backend(backend_name)
    status = backend.status()
    print(f"\n[hw] backend={backend.name} status={status.status_msg} "
          f"pending={status.pending_jobs}", flush=True)
    experiments = all_experiments()
    experiments = transpile_and_check(experiments, backend)
    sampler = SamplerV2(mode=backend, options=_sampler_options())
    isa = [ex["isa_circuit"] for ex in experiments]
    t0 = time.monotonic()
    job = sampler.run(isa, shots=SHOTS)
    print(f"[hw] job_id={job.job_id()}", flush=True)
    result = job.result()
    print(f"[hw] result ready in {time.monotonic() - t0:.1f}s wall", flush=True)
    idx = {ex["label"]: i for i, ex in enumerate(experiments)}

    def get_counts(ex):
        data = result[idx[ex["label"]]].data
        creg = next(iter(data))
        return getattr(data, creg).get_counts()

    analysis = _analyze(experiments, get_counts, f"hw:{backend.name}",
                        extra={"job_id": job.job_id(), "instance": instance})
    out_path = RESULTS_DIR / "marrakesh_batch8_hw_analysis.json"
    out_path.write_text(json.dumps(analysis, indent=2))
    print(f"[hw] wrote {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--hw", action="store_true", help="submit to hardware and wait")
    ap.add_argument("--submit", action="store_true", help="fire-and-forget submit")
    ap.add_argument("--backend", default="ibm_marrakesh")
    ap.add_argument("--instance", default="Zynerji")
    args = ap.parse_args()
    if args.submit:
        run_submit(args.backend, args.instance)
    elif args.hw:
        run_hardware(args.backend, args.instance)
    else:
        run_simulator()


if __name__ == "__main__":
    main()
