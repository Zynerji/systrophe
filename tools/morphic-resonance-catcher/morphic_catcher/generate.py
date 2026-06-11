"""Generative panel models for the morphic-resonance harness.

A *panel* is a set of independent "learners" / "instances" indexed in
temporal order, each of which acquires a fixed target form at some
acquisition cost (e.g. trials-to-criterion; lower = easier). The models
differ only in *how* a learner's cost is produced:

  independent_learners  -- no coupling at all (the null world).
  secular_trend         -- cost falls with CALENDAR time (conventional
                           improvement: better methods, contamination).
  morphic_field         -- cost falls with the CUMULATIVE COUNT of prior
                           realizations (Sheldrake's mechanism), with a
                           tunable -- and deliberately bursty -- instantiation
                           schedule so count can be decoupled from time.
  local_diffusion       -- cost falls via a conventional communication
                           network (a confound that must be ruled out).
  ctc_resonance         -- the morphic field as a Deutsch-CTC /
                           Schwinger-Keldysh self-consistent fixed point
                           (see ctc.py). Its distinctive, falsifiable
                           prediction is an ACAUSAL (future-count) advantage.

Every model returns the same :class:`Panel` so the detector and nulls are
model-agnostic.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Panel:
    """A temporally ordered panel of acquisition outcomes.

    Attributes
    ----------
    cost : (N,) float
        Acquisition cost of the target form for each instance, in temporal
        order. Lower = the form was easier to acquire.
    calendar_index : (N,) float
        Position in calendar time (monotone increasing; not necessarily the
        integer index -- bursts cluster instances in time).
    cumulative_prior : (N,) float
        Number of *prior* realizations of the form available to each
        instance at its acquisition time (the causal morphic "field strength").
    cumulative_future : (N,) float
        Number of realizations that occur *after* each instance. Only the
        CTC-resonance model lets cost depend on this; for every causal model
        it is a pure placebo regressor (used by the acausality test).
    meta : dict
        Generator name + parameters, for the findings log.
    """

    cost: np.ndarray
    calendar_index: np.ndarray
    cumulative_prior: np.ndarray
    cumulative_future: np.ndarray
    meta: dict = field(default_factory=dict)

    def __post_init__(self):
        self.cost = np.asarray(self.cost, float)
        self.calendar_index = np.asarray(self.calendar_index, float)
        self.cumulative_prior = np.asarray(self.cumulative_prior, float)
        self.cumulative_future = np.asarray(self.cumulative_future, float)

    @property
    def n(self) -> int:
        return len(self.cost)


# --------------------------------------------------------------------------- #
# Instantiation schedules: when, in calendar time, each instance occurs.
# A bursty schedule is what makes the morphic count effect *identifiable*
# (cumulative count grows in steps, decoupling it from linear calendar time).
# --------------------------------------------------------------------------- #

def _uniform_schedule(n: int) -> np.ndarray:
    """Evenly spaced calendar times -> count is collinear with time."""
    return np.linspace(0.0, 1.0, n)


def _bursty_schedule(n: int, n_bursts: int, rng: np.random.Generator) -> np.ndarray:
    """Instances cluster into ``n_bursts`` epochs of VERY UNEQUAL size.

    Equal-size bursts do not decouple count from time (cumulative count still
    tracks calendar order globally; VIF ~ 14). Drawing burst sizes from a
    sparse Dirichlet makes the instantiation RATE vary by orders of magnitude
    -- a big early wave then a long sparse tail -- so calendar time and
    cumulative count genuinely separate (VIF ~ 3). This is the only regime in
    which a morphic count-effect is statistically distinguishable from a
    secular time-trend, and it mirrors how real activity (publications,
    syntheses) actually arrives.
    """
    centres = np.sort(rng.uniform(0.02, 0.98, n_bursts))
    weights = rng.dirichlet(np.ones(n_bursts) * 0.3)
    sizes = np.maximum(1, np.round(weights * n).astype(int))
    times = []
    for c, s in zip(centres, sizes):
        times.extend([c] * int(s))
    times = np.array(times[:n], dtype=float)
    if len(times) < n:  # rounding shortfall -> pad into the last burst
        times = np.concatenate([times, np.full(n - len(times), centres[-1])])
    times = times + rng.normal(0.0, 1e-3, size=n)
    return np.sort(times)


def _schedule(n, schedule, n_bursts, rng):
    if schedule == "uniform":
        return _uniform_schedule(n)
    if schedule == "bursty":
        return _bursty_schedule(n, n_bursts, rng)
    raise ValueError(f"unknown schedule {schedule!r}")


def _cumulatives(calendar_index: np.ndarray):
    """Prior / future realization counts for each instance.

    With instances sorted by calendar time, the i-th instance has ``i`` prior
    realizations and ``N-1-i`` future ones.
    """
    n = len(calendar_index)
    prior = np.arange(n, dtype=float)
    future = (n - 1) - np.arange(n, dtype=float)
    return prior, future


# --------------------------------------------------------------------------- #
# Models
# --------------------------------------------------------------------------- #

def independent_learners(
    n: int = 120, base_cost: float = 1.0, noise: float = 0.15,
    schedule: str = "bursty", n_bursts: int = 6, seed: int = 0,
) -> Panel:
    """No coupling: every instance pays an i.i.d. cost. The null world."""
    rng = np.random.default_rng(seed)
    t = _schedule(n, schedule, n_bursts, rng)
    prior, future = _cumulatives(t)
    cost = base_cost + rng.normal(0.0, noise, size=n)
    return Panel(cost, t, prior, future,
                 meta={"model": "independent_learners", "noise": noise,
                       "schedule": schedule, "seed": seed})


def secular_trend(
    n: int = 120, base_cost: float = 1.0, trend: float = 0.6,
    noise: float = 0.15, schedule: str = "bursty", n_bursts: int = 6,
    seed: int = 0,
) -> Panel:
    """Cost falls with CALENDAR time -- conventional improvement.

    Contains real, detectable structure, but it is a time-trend, NOT
    count-coupling. A correct harness must call this ``conventional_trend``
    (or ``unidentifiable`` when the schedule makes count ~ time).
    """
    rng = np.random.default_rng(seed)
    t = _schedule(n, schedule, n_bursts, rng)
    prior, future = _cumulatives(t)
    cost = base_cost - trend * t + rng.normal(0.0, noise, size=n)
    return Panel(cost, t, prior, future,
                 meta={"model": "secular_trend", "trend": trend,
                       "noise": noise, "schedule": schedule, "seed": seed})


def morphic_field(
    n: int = 120, base_cost: float = 1.0, coupling: float = 0.6,
    noise: float = 0.15, schedule: str = "bursty", n_bursts: int = 6,
    seed: int = 0,
) -> Panel:
    """Cost falls with the CUMULATIVE COUNT of prior realizations.

    This is Sheldrake's mechanism: field strength == how many times the
    form has already been instantiated. With a bursty schedule the count
    decouples from calendar time, so the effect is *identifiable*.
    """
    rng = np.random.default_rng(seed)
    t = _schedule(n, schedule, n_bursts, rng)
    prior, future = _cumulatives(t)
    # Normalize count to [0,1] so ``coupling`` is comparable to ``trend``.
    cnorm = prior / max(prior.max(), 1.0)
    cost = base_cost - coupling * cnorm + rng.normal(0.0, noise, size=n)
    return Panel(cost, t, prior, future,
                 meta={"model": "morphic_field", "coupling": coupling,
                       "noise": noise, "schedule": schedule, "seed": seed})


def local_diffusion(
    n: int = 120, base_cost: float = 1.0, coupling: float = 0.6,
    noise: float = 0.15, n_neighbors: int = 4, schedule: str = "bursty",
    n_bursts: int = 6, seed: int = 0,
) -> Panel:
    """Cost falls because earlier learners *tell* later ones (a network).

    A conventional channel that mimics a count effect. Included so the harness
    can be probed for its ability (or inability) to distinguish morphic
    coupling from ordinary diffusion -- the non-locality identifiability
    question (Concept C).
    """
    rng = np.random.default_rng(seed)
    t = _schedule(n, schedule, n_bursts, rng)
    prior, future = _cumulatives(t)
    cost = np.empty(n)
    known = 0.0  # diffusing "how well the form is known in the network"
    for i in range(n):
        cost[i] = base_cost - coupling * known + rng.normal(0.0, noise)
        # local update: only the most recent few learners propagate knowledge
        recent = min(i + 1, n_neighbors)
        known = min(1.0, known + recent / (n_neighbors * n))
    return Panel(cost, t, prior, future,
                 meta={"model": "local_diffusion", "coupling": coupling,
                       "n_neighbors": n_neighbors, "noise": noise,
                       "schedule": schedule, "seed": seed})


def multiform_forms(
    mechanism: str = "causal", n_forms: int = 60, base_cost: float = 1.0,
    coupling: float = 0.6, acausal_fraction: float = 0.6, noise: float = 0.1,
    seed: int = 0,
) -> list[dict]:
    """Generate a panel of FORMS for the across-forms acausality test.

    Each form has an early window (first-k instances) and a total lifetime.
    The temporal profile is randomized per form so that ``early_count`` and
    ``eventual_total`` are decoupled (some forms front-loaded, some
    back-loaded) -- which is exactly what makes a future effect identifiable.

    mechanism:
      "causal" -- early cost depends only on the form's EARLY count (any
                  conventional learning: cumulative culture, diffusion, ...).
      "ctc"    -- early cost depends on (1-a)*early_count + a*eventual_total,
                  i.e. information about the form's eventual establishment is
                  looped back to its early instances (the CTC fixed point).
    Returns a list of form dicts consumable by ``detect.acausal_across_forms``.
    """
    rng = np.random.default_rng(seed)

    def z(x):
        x = np.asarray(x, float)
        s = x.std()
        return (x - x.mean()) / s if s > 0 else x - x.mean()

    eventual = rng.uniform(5.0, 200.0, size=n_forms)
    # front-load fraction in [0.1, 0.9], independent of eventual -> decouples
    front = rng.uniform(0.1, 0.9, size=n_forms)
    early_count = front * eventual + rng.normal(0.0, 1.0, size=n_forms)
    early_count = np.clip(early_count, 0.0, None)

    if mechanism == "causal":
        signal = z(early_count)
    elif mechanism == "ctc":
        signal = (1.0 - acausal_fraction) * z(early_count) \
                 + acausal_fraction * z(eventual)
    else:
        raise ValueError(f"unknown mechanism {mechanism!r}")

    early_cost = base_cost - coupling * signal + rng.normal(0.0, noise, size=n_forms)
    return [
        {"early_cost": float(early_cost[i]), "early_count": float(early_count[i]),
         "eventual_total": float(eventual[i]),
         "meta": {"mechanism": mechanism, "acausal_fraction": acausal_fraction}}
        for i in range(n_forms)
    ]


def ctc_resonance(
    n: int = 120, base_cost: float = 1.0, coupling: float = 0.6,
    acausal_fraction: float = 0.5, noise: float = 0.15,
    schedule: str = "bursty", n_bursts: int = 6, seed: int = 0,
    gamma: float = 0.25, n_iter: int = 64,
) -> Panel:
    """The morphic field as a Deutsch-CTC self-consistent fixed point.

    See :mod:`morphic_catcher.ctc`. The field is solved as a global
    self-consistent loop rather than accumulated causally, which lets an
    instance's cost depend on realizations BOTH before and after it. The
    fraction of the field that is "looped from the future" is
    ``acausal_fraction``; at 0 it reduces to ordinary causal cumulative
    reinforcement (== :func:`morphic_field`).

    This model exists to give the harness's acausality test something with a
    real future-count signal to detect.
    """
    from .ctc import solve_ctc_field

    rng = np.random.default_rng(seed)
    t = _schedule(n, schedule, n_bursts, rng)
    prior, future = _cumulatives(t)

    field_strength = solve_ctc_field(
        n=n, acausal_fraction=acausal_fraction, gamma=gamma, n_iter=n_iter,
    )
    cost = base_cost - coupling * field_strength + rng.normal(0.0, noise, size=n)
    return Panel(cost, t, prior, future,
                 meta={"model": "ctc_resonance", "coupling": coupling,
                       "acausal_fraction": acausal_fraction, "gamma": gamma,
                       "noise": noise, "schedule": schedule, "seed": seed})
