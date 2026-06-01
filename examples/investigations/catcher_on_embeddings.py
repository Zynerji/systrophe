"""Micro-experiment: does the address-space rank-Hamming / lambda_2 novelty
catcher do anything useful on LLM-embedding-like data?

Two falsifiable questions, numpy-only, < 60 s, no torch/LLM.

Q1 (RETRIEVAL): does a rank-Hamming (copula) distance preserve nearest-
    neighbour cluster structure comparably to cosine, on unit-norm Gaussian
    clusters that have been RANDOMLY ROTATED (so structure is NOT axis-aligned)?
    Methods compared on kNN cluster purity:
      - cosine (the embedding-space standard)
      - raw-coordinate rank-Hamming (catcher's per-axis thermometer metric)
      - rank-Hamming on a random-projection subspace
      - rank-Hamming on a PCA subspace
    The catcher metric == sum over axes of |quantised_rank_i - quantised_rank_j|
    (per the catcher_theory identity). We compute it via the REAL catcher
    thermometer primitive, then verify equals the rank-L1 closed form.

Q2 (OOD / DRIFT): can the REAL catcher's calibrated CDF / lambda_2 test detect
    a SUBTLE distribution shift in a batch of embeddings that a cosine-to-
    centroid drift score misses? We measure detection rate vs the catcher's
    analytic hypergeometric false-alarm rate (and a surrogate null), for:
      - small mean shift along ONE latent (rotated, non-axis) direction
      - covariance inflation (scale) along one latent direction
      - heavy-tail injection (t-distributed) in one latent direction

All randomness seeded. Honest negatives are reported as-is.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

# --- import the REAL catcher modules -----------------------------------------
_REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(_REPO / "src"))

from systrophe.catchers.novelty_catcher import (  # noqa: E402
    _rank_thermometer_address,
    catch_novelty_in_distributions,
    hamming_distance,
    lambda_2_of_hamming_graph,
    scan_novelty,
)
from systrophe.catchers.catcher_theory import (  # noqa: E402
    absolute_floor_for_n_bits,
    hypergeometric_step_far,
    path_power_fiedler,
)


# =============================================================================
# Shared helpers
# =============================================================================

def random_rotation(d: int, rng: np.random.Generator) -> np.ndarray:
    """Haar-random orthogonal d x d matrix (QR of a Gaussian)."""
    A = rng.normal(size=(d, d))
    Q, R = np.linalg.qr(A)
    # fix signs so Q is genuinely Haar
    Q *= np.sign(np.diag(R))
    return Q


def unit_norm(X: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(X, axis=1, keepdims=True)
    n[n == 0] = 1.0
    return X / n


def rank_transform(X: np.ndarray) -> np.ndarray:
    """Per-axis global rank (the 'copula' transform). Column c -> ranks of X[:,c].
    This is exactly the quantity the catcher thermometer-codes per axis."""
    return np.argsort(np.argsort(X, axis=0, kind="stable"), axis=0, kind="stable")


def thermometer_codes(X: np.ndarray, bits_per_comp: int) -> np.ndarray:
    """Build the catcher's concatenated per-axis thermometer address for every
    row, using the REAL _rank_thermometer_address primitive.

    Returns an (n_rows, d * bits_per_comp) int matrix. Hamming distance between
    two rows then == sum_c |q_ic - q_jc| (catcher_theory identity)."""
    n, d = X.shape
    out = np.zeros((n, d * bits_per_comp), dtype=int)
    for c in range(d):
        col = X[:, c]
        for i in range(n):
            out[i, c * bits_per_comp:(c + 1) * bits_per_comp] = \
                _rank_thermometer_address(col, i, bits_per_comp)
    return out


def knn_purity(dist_or_sim: np.ndarray, labels: np.ndarray, k: int,
               is_similarity: bool) -> float:
    """Mean fraction of the k nearest neighbours (excluding self) that share
    the query's label. Higher = better cluster preservation."""
    n = len(labels)
    M = dist_or_sim.copy().astype(float)
    if is_similarity:
        np.fill_diagonal(M, -np.inf)
        nn = np.argsort(-M, axis=1)[:, :k]
    else:
        np.fill_diagonal(M, np.inf)
        nn = np.argsort(M, axis=1)[:, :k]
    same = labels[nn] == labels[:, None]
    return float(same.mean())


# =============================================================================
# Q1 — RETRIEVAL: cluster purity, cosine vs rank-Hamming variants
# =============================================================================

def experiment_q1(seed: int = 0) -> dict:
    rng = np.random.default_rng(seed)
    d = 64
    n_clusters = 6
    per_cluster = 40
    n = n_clusters * per_cluster
    k = 10
    bits_per_comp = 8  # thermometer resolution per axis

    # cluster centres on the sphere, well separated; small within-cluster sigma
    centres = unit_norm(rng.normal(size=(n_clusters, d)))
    sigma = 0.35
    X = np.empty((n, d))
    labels = np.empty(n, dtype=int)
    for c in range(n_clusters):
        pts = centres[c] + sigma * rng.normal(size=(per_cluster, d))
        X[c * per_cluster:(c + 1) * per_cluster] = pts
        labels[c * per_cluster:(c + 1) * per_cluster] = c
    X = unit_norm(X)

    # APPLY A RANDOM ROTATION so structure is NOT axis-aligned (the hard case
    # for any per-axis / coordinate-wise method).
    Rot = random_rotation(d, rng)
    Xr = X @ Rot
    Xr = unit_norm(Xr)

    # --- cosine baseline (on rotated data; cosine is rotation-invariant) ---
    cos = Xr @ Xr.T
    purity_cosine = knn_purity(cos, labels, k, is_similarity=True)

    # --- raw-coordinate rank-Hamming (catcher metric on the rotated coords) ---
    therm_raw = thermometer_codes(Xr, bits_per_comp)
    # pairwise Hamming == catcher metric
    # compute via fast XOR-popcount on the 0/1 matrix
    def pairwise_hamming(B: np.ndarray) -> np.ndarray:
        # B is (n, m) 0/1; hamming = m - 2*B@B.T + rowsum_i + rowsum_j ... use
        # (a!=b) = a + b - 2ab summed over bits.
        s = B.sum(axis=1)
        inner = B @ B.T
        return s[:, None] + s[None, :] - 2 * inner
    H_raw = pairwise_hamming(therm_raw)
    purity_rankhamming_raw = knn_purity(H_raw, labels, k, is_similarity=False)

    # verify the catcher metric == sum_c |q_ic - q_jc| (rank-L1 closed form)
    q = (rank_transform(Xr) / max(n - 1, 1) * bits_per_comp).astype(int)
    # rank-L1 between rows 0 and 1 as a spot check across a few pairs
    max_err = 0
    for (i, j) in [(0, 1), (5, 137), (200, 13), (n - 1, n - 2)]:
        rank_l1 = int(np.abs(q[i] - q[j]).sum())
        ham = int(H_raw[i, j])
        max_err = max(max_err, abs(rank_l1 - ham))

    # --- rank-Hamming on a RANDOM-PROJECTION subspace ---
    proj_dim = 16
    G = rng.normal(size=(d, proj_dim)) / np.sqrt(d)
    Xrp = unit_norm(Xr @ G)
    therm_rp = thermometer_codes(Xrp, bits_per_comp)
    H_rp = pairwise_hamming(therm_rp)
    purity_rankhamming_rp = knn_purity(H_rp, labels, k, is_similarity=False)

    # --- rank-Hamming on a PCA subspace (top proj_dim PCs) ---
    Xc = Xr - Xr.mean(axis=0, keepdims=True)
    U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
    Xpca = unit_norm(Xc @ Vt[:proj_dim].T)
    therm_pca = thermometer_codes(Xpca, bits_per_comp)
    H_pca = pairwise_hamming(therm_pca)
    purity_rankhamming_pca = knn_purity(H_pca, labels, k, is_similarity=False)

    # --- control: cosine on the random-projection subspace (sanity) ---
    purity_cosine_rp = knn_purity(Xrp @ Xrp.T, labels, k, is_similarity=True)

    # chance level
    chance = 1.0 / n_clusters

    return {
        "n": n, "d": d, "n_clusters": n_clusters, "k": k,
        "bits_per_comp": bits_per_comp, "proj_dim": proj_dim,
        "chance": chance,
        "catcher_metric_equals_rankL1_max_err": max_err,
        "purity_cosine_full": purity_cosine,
        "purity_rankhamming_raw64": purity_rankhamming_raw,
        "purity_rankhamming_randproj16": purity_rankhamming_rp,
        "purity_rankhamming_pca16": purity_rankhamming_pca,
        "purity_cosine_randproj16": purity_cosine_rp,
    }


# =============================================================================
# Q2 — OOD / DRIFT: catcher CDF/lambda_2 test vs cosine-to-centroid
# =============================================================================

def _cosine_to_centroid_scores(ref: np.ndarray, batch: np.ndarray) -> np.ndarray:
    """Per-point cosine similarity to the reference centroid (the standard
    cheap drift / OOD heuristic)."""
    c = unit_norm(ref.mean(axis=0, keepdims=True))[0]
    b = unit_norm(batch)
    return b @ c


def _catcher_drift_stat(ref: np.ndarray, batch: np.ndarray,
                        bits_per_comp: int) -> float:
    """The catcher's per-axis rank-CDF discrepancy as a single scalar.

    The catcher detects a distribution shift as a discontinuity of the
    empirical quantile function (a Kolmogorov-type statistic on per-axis
    ranks). We pool ref+batch, take per-axis global ranks (the catcher's
    quantised-rank thermometer code), and measure the MAX over axes of the
    mean quantised-rank gap between the ref block and the batch block — i.e.
    how far the batch's rank-position differs from uniform interleaving.
    Weakest-link (min lambda_2) theory says the most-shifted axis governs;
    a shift along a latent (rotated) direction must show up as a per-axis
    rank displacement, which is what this measures.
    """
    pooled = np.vstack([ref, batch])
    n_ref = len(ref)
    n_tot = len(pooled)
    q = (rank_transform(pooled) / max(n_tot - 1, 1) * bits_per_comp)
    # mean quantised rank of batch rows minus that of ref rows, per axis
    ref_mean = q[:n_ref].mean(axis=0)
    batch_mean = q[n_ref:].mean(axis=0)
    return float(np.max(np.abs(batch_mean - ref_mean)))


def _make_base(rng, n, d):
    """Anisotropic embedding-like cloud with a 'rogue' dominant direction and a
    Haar rotation (semantic info in arbitrary linear directions)."""
    # anisotropic covariance: one big eigenvalue (rogue dim), rest moderate
    scales = np.ones(d)
    scales[0] = 4.0
    Z = rng.normal(size=(n, d)) * scales
    Rot = random_rotation(d, rng)
    X = Z @ Rot
    return unit_norm(X), Rot


def _energy_distance(A: np.ndarray, B: np.ndarray) -> float:
    """Gold-standard multivariate two-sample energy distance (full magnitude,
    original space): 2 E|a-b| - E|a-a'| - E|b-b'|. Used as an upper-bound
    CONTROL: if even this misses a shift, the catcher's miss is not a weakness
    of the catcher but a property of the shift being genuinely undetectable."""
    from scipy.spatial.distance import cdist
    ab = cdist(A, B).mean()
    aa = cdist(A, A).mean()
    bb = cdist(B, B).mean()
    return float(2 * ab - aa - bb)


def experiment_q2(seed: int = 1, n_trials: int = 200) -> dict:
    rng = np.random.default_rng(seed)
    d = 32
    n_ref = 300
    n_batch = 80
    bits_per_comp = 16

    # one shared geometry (rotation + a fixed latent shift direction)
    _, Rot = _make_base(rng, 10, d)
    latent_dir = Rot[:, 1]  # a non-axis-aligned latent direction
    latent_dir = latent_dir / np.linalg.norm(latent_dir)

    # catcher analytic per-axis FAR for a single adjacent-pair step is not the
    # right FAR for a two-sample drift test, so we CALIBRATE the catcher stat's
    # null empirically (surrogate) AND report the analytic hypergeometric step
    # FAR for context. Detection rate is measured at a threshold set to give a
    # target empirical FAR.
    target_far = 0.05

    def base_cloud(local_rng):
        scales = np.ones(d); scales[0] = 4.0
        Z = local_rng.normal(size=(n_ref + n_batch, d)) * scales
        X = unit_norm(Z @ Rot)
        return X[:n_ref], X[n_ref:]

    # --- 1. Build NULL distributions (no shift) for both stats ---
    null_catcher = []
    null_cosine = []
    rng_null = np.random.default_rng(seed + 100)
    for _ in range(n_trials):
        ref, batch = base_cloud(rng_null)
        null_catcher.append(_catcher_drift_stat(ref, batch, bits_per_comp))
        cs = _cosine_to_centroid_scores(ref, batch)
        # cosine drift stat = |mean batch cos-to-centroid - mean ref cos-to-centroid|
        ref_cs = _cosine_to_centroid_scores(ref, ref)
        null_cosine.append(abs(cs.mean() - ref_cs.mean()))
    null_catcher = np.array(null_catcher)
    null_cosine = np.array(null_cosine)
    thr_catcher = np.quantile(null_catcher, 1 - target_far)
    thr_cosine = np.quantile(null_cosine, 1 - target_far)

    # empirical FAR achieved at those thresholds (sanity, should ~target)
    far_catcher = float((null_catcher > thr_catcher).mean())
    far_cosine = float((null_cosine > thr_cosine).mean())

    # analytic hypergeometric per-pair step FAR.
    # IMPORTANT: hypergeometric_step_far models a SINGLE component whose max
    # step is bits_per_comp. The catcher's PER-AXIS floor convention is
    # n_bits ~ 4*bits_per_comp (see verify_catcher_theory), NOT the full
    # concatenated width d*bits_per_comp. If you (wrongly) pass the full width
    # the floor (128) exceeds the single-axis max step (16) and the analytic
    # FAR UNDERFLOWS TO EXACTLY 0 — a real foot-gun worth reporting.
    full_bits = d * bits_per_comp
    floor_full_wrong = absolute_floor_for_n_bits(full_bits)
    analytic_far_full_wrong = hypergeometric_step_far(
        n_ref + n_batch, bits_per_comp, floor_full_wrong)
    per_axis_bits = 4 * bits_per_comp
    floor = absolute_floor_for_n_bits(per_axis_bits)
    analytic_step_far = hypergeometric_step_far(
        n_ref + n_batch, bits_per_comp, floor)

    # --- 2. Detection rate under each SHIFT type ---
    # energy-distance CONTROL null threshold (gold standard upper bound)
    null_energy = []
    rng_en = np.random.default_rng(seed + 500)
    n_en = min(n_trials, 120)  # energy dist is O(n^2); cap trials for speed
    for _ in range(n_en):
        ref, batch = base_cloud(rng_en)
        null_energy.append(_energy_distance(ref, batch))
    thr_energy = np.quantile(null_energy, 1 - target_far)
    far_energy = float((np.array(null_energy) > thr_energy).mean())

    def detect_rate(shift_fn, rng_seed):
        lr = np.random.default_rng(rng_seed)
        det_c = 0
        det_cos = 0
        det_en = 0
        for t in range(n_trials):
            ref, batch = base_cloud(lr)
            batch = shift_fn(batch, lr)
            if _catcher_drift_stat(ref, batch, bits_per_comp) > thr_catcher:
                det_c += 1
            cs = _cosine_to_centroid_scores(ref, batch)
            ref_cs = _cosine_to_centroid_scores(ref, ref)
            if abs(cs.mean() - ref_cs.mean()) > thr_cosine:
                det_cos += 1
            if t < n_en and _energy_distance(ref, batch) > thr_energy:
                det_en += 1
        return det_c / n_trials, det_cos / n_trials, det_en / n_en

    # (a) small mean shift along the latent direction
    mean_shift_mag = 0.25
    def shift_mean(batch, lr):
        b = batch + mean_shift_mag * latent_dir[None, :]
        return unit_norm(b)

    # (b) covariance inflation along the latent direction
    def shift_cov(batch, lr):
        proj = batch @ latent_dir
        b = batch + 0.8 * (proj[:, None]) * latent_dir[None, :]  # stretch axis
        return unit_norm(b)

    # (c) heavy-tail injection along the latent direction (t_3 noise)
    def shift_tail(batch, lr):
        t = lr.standard_t(df=3, size=len(batch)) * 0.35
        b = batch + t[:, None] * latent_dir[None, :]
        return unit_norm(b)

    dr_mean = detect_rate(shift_mean, seed + 200)
    dr_cov = detect_rate(shift_cov, seed + 300)
    dr_tail = detect_rate(shift_tail, seed + 400)

    return {
        "d": d, "n_ref": n_ref, "n_batch": n_batch,
        "bits_per_comp": bits_per_comp, "n_trials": n_trials,
        "target_far": target_far,
        "far_catcher_empirical": far_catcher,
        "far_cosine_empirical": far_cosine,
        "far_energy_empirical": far_energy,
        "analytic_hypergeom_step_far_perAxis": analytic_step_far,
        "analytic_far_full_width_UNDERFLOW": analytic_far_full_wrong,
        "per_axis_floor_bits": floor,
        "full_width_floor_bits_foot_gun": floor_full_wrong,
        "mean_shift": {"mag": mean_shift_mag, "detect_catcher": dr_mean[0],
                       "detect_cosine": dr_mean[1], "detect_energy_GOLD": dr_mean[2]},
        "cov_inflation": {"detect_catcher": dr_cov[0], "detect_cosine": dr_cov[1],
                          "detect_energy_GOLD": dr_cov[2]},
        "heavy_tail": {"detect_catcher": dr_tail[0], "detect_cosine": dr_tail[1],
                       "detect_energy_GOLD": dr_tail[2]},
    }


# =============================================================================
# Q2b — sanity: drive the REAL scan_novelty / catch_novelty_in_distributions
#       end-to-end on a drifting embedding stream (uses shipped code path).
# =============================================================================

def experiment_q2b_real_catcher(seed: int = 7) -> dict:
    """Feed a stream of per-axis-mean 'embedding summary' vectors through the
    REAL scan_novelty, with a drift injected partway, and report whether the
    shipped catcher flags it (verdict / sharp features / lambda_2)."""
    rng = np.random.default_rng(seed)
    d = 32
    n_steps = 30
    drift_at = 20
    Rot = random_rotation(d, rng)
    latent = Rot[:, 3]

    def step_output(p):
        idx = int(round(p))
        lr = np.random.default_rng(seed * 1000 + idx)
        scales = np.ones(d); scales[0] = 4.0
        Z = lr.normal(size=(200, d)) * scales
        X = unit_norm(Z @ Rot)
        if idx >= drift_at:
            X = unit_norm(X + 0.5 * latent[None, :])
        return X.mean(axis=0)  # 32-dim per-step summary -> catcher ranks axes

    res = scan_novelty(
        np.arange(n_steps, dtype=float), step_output,
        n_bits=32, parameter_label="stream_step", data_adaptive=True,
    )
    # also run catch_novelty_in_distributions on per-step histograms of |proj|
    dists = []
    for idx in range(n_steps):
        lr = np.random.default_rng(seed * 1000 + idx)
        scales = np.ones(d); scales[0] = 4.0
        Z = lr.normal(size=(200, d)) * scales
        X = unit_norm(Z @ Rot)
        if idx >= drift_at:
            X = unit_norm(X + 0.5 * latent[None, :])
        proj = X @ latent
        h, _ = np.histogram(proj, bins=32, range=(-1, 1))
        dists.append(h.astype(float) / max(h.sum(), 1))
    cd = catch_novelty_in_distributions(dists, n_bits=32)

    return {
        "drift_injected_at_step": drift_at,
        "scan_novelty_verdict": res.verdict,
        "scan_novelty_n_sharp": len(res.sharp_features),
        "scan_novelty_sharp_steps": [s["parameter_value"]
                                     for s in res.sharp_features],
        "scan_novelty_lambda2": res.lambda_2_at_radius,
        "catch_in_dists_verdict": cd["verdict"],
        "catch_in_dists_n_sharp": len(cd["sharp_features"]),
        "catch_in_dists_sharp": [s["between"] for s in cd["sharp_features"]],
    }


# =============================================================================
def main():
    t0 = time.time()
    print("=" * 72)
    print("CATCHER-ON-EMBEDDINGS micro-experiment (numpy-only)")
    print("=" * 72)

    print("\n--- Q1: RETRIEVAL (kNN cluster purity; data RANDOMLY ROTATED) ---")
    q1 = experiment_q1(seed=0)
    for kk, vv in q1.items():
        if isinstance(vv, float):
            print(f"  {kk:38s} = {vv:.4f}")
        else:
            print(f"  {kk:38s} = {vv}")

    print("\n--- Q2: OOD/DRIFT detection rate vs FAR (catcher vs cosine-to-centroid) ---")
    q2 = experiment_q2(seed=1, n_trials=200)
    for kk, vv in q2.items():
        print(f"  {kk:38s} = {vv}")

    print("\n--- Q2b: REAL scan_novelty / catch_novelty_in_distributions on drift stream ---")
    q2b = experiment_q2b_real_catcher(seed=7)
    for kk, vv in q2b.items():
        print(f"  {kk:38s} = {vv}")

    print(f"\n[elapsed {time.time() - t0:.1f} s]")


if __name__ == "__main__":
    main()
