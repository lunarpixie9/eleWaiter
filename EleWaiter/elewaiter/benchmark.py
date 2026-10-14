"""Run dispatchers against scenarios over many random seeds."""
from __future__ import annotations

from typing import Dict, Iterable, Optional

import pandas as pd

from .dispatchers import (AIDispatcher, Collective, DEFAULT_WEIGHTS, FCFS, Nearest,
                          PriorityNearest, load_tuned_weights)
from .metrics import SimulationResult
from .models import BuildingConfig
from .simulator import ElevatorSimulator
from .traffic import SCENARIOS, TrafficGenerator

ALGORITHMS = ["FCFS", "Nearest", "Collective (LOOK-ETA)", "Priority-Nearest",
              "EleWaiter AI", "AI (no emergency logic)"]
CORE_ALGORITHMS = ["FCFS", "Nearest", "Collective (LOOK-ETA)", "Priority-Nearest", "EleWaiter AI"]


def make_dispatcher(name: str, weights: Optional[Dict[str, float]] = None):
    if name == "FCFS":
        return FCFS()
    if name == "Nearest":
        return Nearest()
    if name.startswith("Collective"):
        return Collective()
    if name == "Priority-Nearest":
        return PriorityNearest()
    if name == "EleWaiter AI":
        return AIDispatcher(weights, emergency=True)
    if name == "AI (no emergency logic)":
        return AIDispatcher(weights, emergency=False)
    raise ValueError(f"unknown algorithm: {name}")


def run_once(config: BuildingConfig, scenario: str, algorithm: str, seed: int = 42,
             duration: float = 600.0, intensity: float = 1.0,
             weights: Optional[Dict[str, float]] = None,
             emergency_rate: Optional[float] = None, alert_coverage: float = 1.0,
             transfer_space: int = 1) -> SimulationResult:
    """Every algorithm sees the identical passenger stream for a given seed."""
    pax = TrafficGenerator(config, seed=seed).generate(scenario, duration, intensity,
                                                    emergency_rate, alert_coverage, transfer_space)
    spec = SCENARIOS[scenario]
    selected_weights = weights
    if algorithm in ("EleWaiter AI", "AI (no emergency logic)") and selected_weights is None:
        selected_weights = load_tuned_weights(config.name) or dict(DEFAULT_WEIGHTS)
    sim = ElevatorSimulator(config, pax, make_dispatcher(algorithm, selected_weights), scenario=scenario,
                            disabled_lifts=spec.disabled_lifts, failure_at=spec.failure_at,
                            failure_lifts=spec.failure_lifts)
    res = sim.run()
    res.algorithm = algorithm
    return res


def run_benchmark(config: BuildingConfig, scenarios: Iterable[str], algorithms: Iterable[str],
                  seeds: Iterable[int], duration: float = 600.0, intensity: float = 1.0,
                  weights: Optional[Dict[str, float]] = None, alert_coverage: float = 1.0,
                  transfer_space: int = 1) -> pd.DataFrame:
    rows = []
    for sc in scenarios:
        for seed in seeds:
            for algo in algorithms:
                r = run_once(config, sc, algo, seed, duration, intensity, weights,
                             alert_coverage=alert_coverage, transfer_space=transfer_space)
                row = r.row()
                row["seed"] = seed
                rows.append(row)
    return pd.DataFrame(rows)


SUMMARY_COLS = ["avg_wait", "p95_wait", "max_wait", "wait_std", "starvation_count",
                "left_behind", "travel_distance", "n_unserved", "n_emergency", "em_avg_wait",
                "em_p95_wait", "em_avg_journey", "em_p95_journey", "em_within_target",
                "routine_avg_wait", "n_clinical_alerts", "infer_precision", "infer_recall"]


def summarize(df: pd.DataFrame) -> pd.DataFrame:
    """Mean over seeds, per scenario and algorithm."""
    cols = [c for c in SUMMARY_COLS if c in df.columns]
    return df.groupby(["scenario", "algorithm"], sort=False)[cols].mean().reset_index()


def win_rate(df: pd.DataFrame, algo: str, rival: str, metric: str) -> pd.DataFrame:
    """Per scenario: share of seeds where `algo` has a lower `metric` than `rival`."""
    out = []
    for sc, g in df.groupby("scenario", sort=False):
        a = g[g.algorithm == algo].set_index("seed")[metric]
        b = g[g.algorithm == rival].set_index("seed")[metric]
        j = pd.concat([a, b], axis=1, keys=["a", "b"]).dropna()
        out.append({"scenario": sc, "rival": rival, "metric": metric,
                    "wins": int((j.a < j.b).sum()), "seeds": len(j)})
    return pd.DataFrame(out)
