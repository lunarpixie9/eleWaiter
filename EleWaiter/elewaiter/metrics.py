"""Metrics. Emergency metrics use GROUND TRUTH (`is_emergency`), which the
dispatchers never see, so they measure real outcomes rather than the system's
own guesses."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

import numpy as np

from .models import Urgency

EMERGENCY_WAIT_TARGET = 20.0   # seconds: an emergency should board within this


@dataclass
class SimulationResult:
    scenario: str = ""
    algorithm: str = ""
    building: str = ""
    n_total: int = 0
    n_served: int = 0
    n_unserved: int = 0
    # overall
    avg_wait: float = 0.0
    p95_wait: float = 0.0
    max_wait: float = 0.0
    wait_std: float = 0.0            # fairness: spread of waiting times
    avg_journey: float = 0.0         # arrival -> exit
    starvation_count: int = 0        # waited >= starvation limit
    left_behind: int = 0             # bypass events caused by a full lift
    travel_distance: float = 0.0     # floors travelled by all lifts
    forced_assignments: int = 0      # times the starvation guard intervened
    # emergencies (ground truth)
    n_emergency: int = 0
    em_avg_wait: Optional[float] = None
    em_p95_wait: Optional[float] = None
    em_max_wait: Optional[float] = None
    em_avg_journey: Optional[float] = None
    em_p95_journey: Optional[float] = None
    em_within_target: Optional[float] = None   # share boarding within the target
    routine_avg_wait: Optional[float] = None
    n_clinical_alerts: int = 0
    # context-inference quality
    infer_precision: Optional[float] = None    # inferred CRITICAL that were real emergencies
    infer_recall: Optional[float] = None       # real emergencies inferred CRITICAL
    wait_times: List[float] = field(default_factory=list)
    em_wait_times: List[float] = field(default_factory=list)
    timeline: List[dict] = field(default_factory=list)

    ROW_FIELDS = (
        "n_total", "n_served", "n_unserved", "avg_wait", "p95_wait", "max_wait", "wait_std", "avg_journey",
        "starvation_count", "left_behind", "travel_distance", "forced_assignments",
        "n_emergency", "em_avg_wait", "em_p95_wait", "em_max_wait", "em_avg_journey",
        "em_p95_journey", "em_within_target", "routine_avg_wait", "n_clinical_alerts",
        "infer_precision", "infer_recall",
    )

    def row(self) -> Dict[str, float]:
        d = {"scenario": self.scenario, "algorithm": self.algorithm, "building": self.building}
        d.update({k: getattr(self, k) for k in self.ROW_FIELDS})
        return d


def _stat(fn, arr):
    return float(fn(arr)) if len(arr) else None


def build_result(sim) -> SimulationResult:
    done = sim.completed
    waits = np.array([p.wait_time for p in done], dtype=float)
    journeys = np.array([p.journey_time for p in done], dtype=float)
    em = [p for p in done if p.is_emergency]
    em_w = np.array([p.wait_time for p in em], dtype=float)
    em_j = np.array([p.journey_time for p in em], dtype=float)
    routine_w = np.array([p.wait_time for p in done if not p.is_emergency], dtype=float)

    inferred_crit = [p for p in sim.passengers if p.urgency == Urgency.CRITICAL]
    true_em = [p for p in sim.passengers if p.is_emergency]
    precision = (sum(p.is_emergency for p in inferred_crit) / len(inferred_crit)) if inferred_crit else None
    recall = (sum(p.urgency == Urgency.CRITICAL for p in true_em) / len(true_em)) if true_em else None

    res = SimulationResult(
        scenario=sim.scenario, building=sim.config.name,
        n_total=len(sim.passengers), n_served=len(done),
        n_unserved=len(sim.passengers) - len(done),
        avg_wait=float(waits.mean()) if len(waits) else 0.0,
        p95_wait=float(np.percentile(waits, 95)) if len(waits) else 0.0,
        max_wait=float(waits.max()) if len(waits) else 0.0,
        wait_std=float(waits.std()) if len(waits) else 0.0,
        avg_journey=float(journeys.mean()) if len(journeys) else 0.0,
        starvation_count=int((waits >= sim.starvation_limit).sum()) if len(waits) else 0,
        left_behind=sim.left_behind_total,
        travel_distance=float(sum(l.travelled for l in sim.lifts)),
        forced_assignments=sim.forced_assignments,
        n_emergency=len(true_em),
        em_avg_wait=_stat(np.mean, em_w),
        em_p95_wait=_stat(lambda a: np.percentile(a, 95), em_w),
        em_max_wait=_stat(np.max, em_w),
        em_avg_journey=_stat(np.mean, em_j),
        em_p95_journey=_stat(lambda a: np.percentile(a, 95), em_j),
        em_within_target=float((em_w <= EMERGENCY_WAIT_TARGET).mean()) if len(em_w) else None,
        routine_avg_wait=_stat(np.mean, routine_w),
        n_clinical_alerts=sum(p.clinical_alert for p in sim.passengers),
        infer_precision=precision, infer_recall=recall,
        wait_times=waits.tolist(), em_wait_times=em_w.tolist(), timeline=sim.timeline,
    )
    return res
