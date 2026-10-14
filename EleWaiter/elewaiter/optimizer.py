"""Tune the AI dispatcher's cost-function weights with a Genetic Algorithm or
Simulated Annealing. The simulator itself is the fitness function (no labelled
data needed).

Methodology (to keep the result honest)
---------------------------------------
* Training uses a FIXED set of scenarios x seeds x intensities, so the
  "best fitness" curve is comparable across generations (unlike re-seeding
  every generation, which makes the curve meaningless).
* Evaluate the tuned weights on DIFFERENT seeds than were used for training.
"""
from __future__ import annotations

import math
import random
from typing import Callable, Dict, List, Optional, Sequence, Tuple

from .benchmark import run_once
from .dispatchers.ai import DEFAULT_WEIGHTS, WEIGHT_BOUNDS
from .metrics import SimulationResult
from .models import BuildingConfig

TRAIN_SCENARIOS = ("emergency_surge", "mass_casualty", "visiting_hours")
TRAIN_SEEDS = (1000, 1001, 1002)
TRAIN_INTENSITIES = (1.0, 1.8)
TRAIN_DURATION = 300.0
TRAIN_ALERT_COVERAGE = 1.0
TRAIN_TRANSFER_SPACE = 3


def composite(r: SimulationResult) -> float:
    """Lower is better. Includes full emergency delivery time and missed trips."""
    n = max(1, r.n_total)
    f = r.avg_wait + 0.5 * r.p95_wait + 0.3 * r.wait_std
    f += 3.0 * (r.em_avg_wait or 0.0) + 1.0 * (r.em_p95_wait or 0.0)
    f += 1.5 * (r.em_avg_journey or 0.0) + 0.5 * (r.em_p95_journey or 0.0)
    if r.em_within_target is not None:
        f += 20.0 * (1.0 - r.em_within_target)
    f += 200.0 * r.starvation_count / n          # 1 % starving = 2 points
    f += 20.0 * r.left_behind / n
    f += 500.0 * r.n_unserved / n
    return f


def fitness(weights: Dict[str, float], config: BuildingConfig,
            scenarios: Sequence[str] = TRAIN_SCENARIOS, seeds: Sequence[int] = TRAIN_SEEDS,
            intensities: Sequence[float] = TRAIN_INTENSITIES,
            duration: float = TRAIN_DURATION, alert_coverage: float = TRAIN_ALERT_COVERAGE,
            transfer_space: int = TRAIN_TRANSFER_SPACE) -> float:
    tot, k = 0.0, 0
    for sc in scenarios:
        for seed in seeds:
            for inten in intensities:
                tot += composite(run_once(config, sc, "EleWaiter AI", seed, duration, inten,
                                          weights, alert_coverage=alert_coverage,
                                          transfer_space=transfer_space))
                k += 1
    return tot / max(1, k)


# ----------------------------------------------------------------------- helpers
def _clip(k: str, v: float) -> float:
    lo, hi = WEIGHT_BOUNDS[k]
    return max(lo, min(hi, v))


def _mutate(ind: Dict[str, float], rng: random.Random, rate: float, scale: float = 0.15):
    out = dict(ind)
    for k in out:
        if rng.random() < rate:
            lo, hi = WEIGHT_BOUNDS[k]
            out[k] = _clip(k, out[k] + rng.gauss(0, (hi - lo) * scale))
    return out


def _crossover(a, b, rng):
    return {k: (a[k] if rng.random() < 0.5 else b[k]) for k in a}


def _key(w: Dict[str, float]):
    return tuple(round(w[k], 6) for k in sorted(WEIGHT_BOUNDS))


# ------------------------------------------------------------------ genetic algorithm
def genetic_algorithm(config: BuildingConfig, pop_size: int = 14, generations: int = 10,
                      seed: int = 7, fit_kwargs: Optional[dict] = None,
                      progress: Optional[Callable[[int, float], None]] = None
                      ) -> Tuple[Dict[str, float], List[float]]:
    """Returns (best weights, best-fitness-per-generation)."""
    rng = random.Random(seed)
    fk = fit_kwargs or {}
    cache: Dict[tuple, float] = {}

    def score(w):
        k = _key(w)
        if k not in cache:
            cache[k] = fitness(w, config, **fk)
        return cache[k]

    pop = [dict(DEFAULT_WEIGHTS)]
    while len(pop) < pop_size:
        if len(pop) % 3 == 0:       # a few fully random individuals for diversity
            pop.append({k: rng.uniform(*WEIGHT_BOUNDS[k]) for k in WEIGHT_BOUNDS})
        else:
            pop.append(_mutate(DEFAULT_WEIGHTS, rng, rate=0.7, scale=0.2))

    history: List[float] = []
    for gen in range(generations):
        scored = sorted(((score(w), w) for w in pop), key=lambda x: x[0])
        history.append(scored[0][0])
        if progress:
            progress(gen + 1, scored[0][0])
        elites = [w for _, w in scored[:max(2, pop_size // 5)]]
        nxt = list(elites)
        while len(nxt) < pop_size:
            a = min(rng.sample(scored, 3), key=lambda x: x[0])[1]     # tournament selection
            b = min(rng.sample(scored, 3), key=lambda x: x[0])[1]
            nxt.append(_mutate(_crossover(a, b, rng), rng, rate=0.3))
        pop = nxt
    best = min(pop, key=score)
    history.append(score(best))
    return best, history


# ------------------------------------------------------------- simulated annealing
def simulated_annealing(config: BuildingConfig, iterations: int = 120, t_start: float = 0.08,
                        t_end: float = 0.004, seed: int = 7, fit_kwargs: Optional[dict] = None,
                        progress: Optional[Callable[[int, float], None]] = None
                        ) -> Tuple[Dict[str, float], List[float]]:
    """Temperature is relative: a move that is `t` (fractionally) worse is
    accepted with probability ~1/e. Returns (best weights, best-so-far history)."""
    rng = random.Random(seed)
    fk = fit_kwargs or {}
    cur = dict(DEFAULT_WEIGHTS)
    cur_s = fitness(cur, config, **fk)
    best, best_s = dict(cur), cur_s
    history = [best_s]
    cooling = (t_end / t_start) ** (1.0 / max(1, iterations - 1))
    T = t_start
    for i in range(iterations):
        cand = _mutate(cur, rng, rate=0.4, scale=0.12)
        s = fitness(cand, config, **fk)
        delta = (s - cur_s) / max(cur_s, 1e-9)
        if delta < 0 or rng.random() < math.exp(-delta / T):
            cur, cur_s = cand, s
            if s < best_s:
                best, best_s = dict(cand), s
        T *= cooling
        history.append(best_s)
        if progress:
            progress(i + 1, best_s)
    return best, history
