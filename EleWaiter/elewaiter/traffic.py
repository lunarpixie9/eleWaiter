"""Hospital passenger traffic with Poisson arrivals and scenario-specific flows.

Emergencies are ground truth (Passenger.is_emergency). By default, every
emergency receives a simulated clinical alert visible to the dispatcher.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Tuple

import numpy as np

from .models import BuildingConfig, Passenger

BASE_RATE_PER_LIFT = 0.10   # passengers / second / lift at intensity 1.0


@dataclass(frozen=True)
class ScenarioSpec:
    key: str
    label: str
    description: str
    disabled_lifts: Tuple[int, ...] = ()
    failure_at: Optional[float] = None
    failure_lifts: Tuple[int, ...] = ()
    rate_factor: float = 1.0
    emergency_rate: float = 0.0     # emergencies / second


SCENARIOS: Dict[str, ScenarioSpec] = {s.key: s for s in [
    ScenarioSpec("emergency_surge", "Emergency surge",
                 "Normal hospital traffic plus a steady stream of ER -> ICU/OT/imaging patient transfers "
                 "and occasional code-response trips.", rate_factor=1.0, emergency_rate=0.012),
    ScenarioSpec("mass_casualty", "Mass-casualty event",
                 "Moderate traffic, then 14 emergency transfers and code responses arrive within about a minute.",
                 rate_factor=0.9, emergency_rate=0.004),
    ScenarioSpec("visiting_hours", "Visiting hours",
                 "Heavy lobby -> ward visitor flow (the 'up-peak' of a hospital) with sporadic emergencies.",
                 rate_factor=1.3, emergency_rate=0.005),
    ScenarioSpec("shift_change", "Shift change",
                 "Staff flowing between lobby and all floors in both directions, with sporadic emergencies.",
                 rate_factor=1.2, emergency_rate=0.005),
    ScenarioSpec("lift_failure", "Lift removed from service",
                 "Emergency-surge traffic; one lift stops accepting calls at 120 s and delivers anyone already aboard.",
                 failure_at=120.0, failure_lifts=(0,), rate_factor=1.0, emergency_rate=0.010),
]}

HOSPITAL_SCENARIOS = list(SCENARIOS)


class TrafficGenerator:
    def __init__(self, config: BuildingConfig, seed: int = 42):
        self.cfg = config
        self.seed = seed
        self.rng = np.random.default_rng(seed)
        self.lobby = config.lobby
        self.all_floors = list(range(1, config.num_floors + 1))
        self.upper = [f for f in self.all_floors if f != self.lobby]

    # ---------------------------------------------------------------- helpers
    def _role(self, role: str) -> List[int]:
        return self.cfg.floors_with_role(role) or self.upper

    def _pick(self, floors: List[int]) -> int:
        return int(self.rng.choice(floors))

    def _times(self, rate: float, duration: float) -> List[float]:
        out, t = [], 0.0
        if rate <= 0:
            return out
        while True:
            t += float(self.rng.exponential(1.0 / rate))
            if t >= duration:
                return out
            out.append(t)

    def _pair_other_than(self, o: int, pool: List[int]) -> int:
        choices = [f for f in pool if f != o] or [f for f in self.all_floors if f != o]
        return self._pick(choices)

    # ------------------------------------------------------------ routine flows
    def _lobby_up(self):
        return self.lobby, self._pick(self.upper)

    def _lobby_down(self):
        return self._pick(self.upper), self.lobby

    def _staff(self):
        o = self._pick(self.upper)
        return o, self._pair_other_than(o, self.upper)

    def _visitor_up(self):
        # visitors go to wards (and, rarely, ICU)
        pool = self._role("ward")
        if self.cfg.floors_with_role("icu") and self.rng.random() < 0.08:
            pool = self.cfg.floors_with_role("icu")
        return self.lobby, self._pick(pool)

    def _visitor_down(self):
        return self._pick(self._role("ward")), self.lobby

    def _outpatient_up(self):
        return self.lobby, self._pick(self._role("clinic") + self.cfg.floors_with_role("imaging"))

    def _outpatient_down(self):
        return self._pick(self._role("clinic") + self.cfg.floors_with_role("imaging")), self.lobby

    # ---------------------------------------------------------- emergency flows
    def _er_transfer(self):
        """Patient moved from the ER to ICU / OT / imaging (ground-truth emergency)."""
        er = self._role("er")
        dest = (self.cfg.floors_with_role("icu") + self.cfg.floors_with_role("ot")
                + self.cfg.floors_with_role("imaging")) or self.upper
        o = self._pick(er)
        return o, self._pair_other_than(o, dest)

    def _post_op(self):
        """Patient moved from the OT to ICU."""
        ot, icu = self._role("ot"), self._role("icu")
        if not any(origin != destination for origin in ot for destination in icu):
            # A shared OT/ICU floor has no lift transfer; generate an ER transfer instead.
            return self._er_transfer()
        o = self._pick(ot)
        return o, self._pair_other_than(o, icu)

    def _code_response(self):
        """Staff responding to a code call on a ward floor. Real emergency, but
        context rules will often MISS it (a ward is a NORMAL zone)."""
        pool = self._role("ward") + self._role("clinic")
        o = self._pick(self.upper)
        return o, self._pair_other_than(o, pool)

    # -------------------------------------------------------------- generation
    def _mix(self, rate: float, duration: float, mix: List[Tuple[float, Callable]]):
        weights = np.array([w for w, _ in mix], dtype=float)
        weights /= weights.sum()
        events = []
        for t in self._times(rate, duration):
            fn = mix[int(self.rng.choice(len(mix), p=weights))][1]
            o, d = fn()
            events.append((t, o, d, False, False))
        return events

    def _emergencies(self, rate: float, duration: float):
        if rate <= 0 or not self.cfg.critical_floors():
            return []
        events = []
        kinds = [(0.55, self._er_transfer, True), (0.2, self._post_op, True),
                 (0.25, self._code_response, False)]
        w = np.array([k[0] for k in kinds]); w /= w.sum()
        for t in self._times(rate, duration):
            _, fn, transfer = kinds[int(self.rng.choice(len(kinds), p=w))]
            o, d = fn()
            events.append((t, o, d, True, transfer))
        return events

    def _burst(self, start: float, span: float, n_transfers: int, n_codes: int, duration: float):
        if not self.cfg.critical_floors():
            return []
        events = []
        for _ in range(n_transfers):
            o, d = self._er_transfer()
            events.append((start + float(self.rng.uniform(0, span)), o, d, True, True))
        for _ in range(n_codes):
            o, d = self._code_response()
            events.append((start + float(self.rng.uniform(0, span)), o, d, True, False))
        return [e for e in events if e[0] < duration]

    def generate(self, scenario: str, duration: float = 600.0, intensity: float = 1.0,
                 emergency_rate: Optional[float] = None, alert_coverage: float = 1.0,
                 transfer_space: int = 1) -> List[Passenger]:
        if not 0.0 <= alert_coverage <= 1.0:
            raise ValueError("alert_coverage must be between 0 and 1")
        if not 1 <= transfer_space <= self.cfg.capacity:
            raise ValueError("transfer_space must fit in a lift")
        spec = SCENARIOS[scenario]
        rate = BASE_RATE_PER_LIFT * self.cfg.num_lifts * spec.rate_factor * intensity
        # emergencies are an external process: `intensity` scales routine traffic only
        em_rate = spec.emergency_rate if emergency_rate is None else emergency_rate

        if scenario in ("emergency_surge", "lift_failure", "mass_casualty"):
            mix = [(0.30, self._visitor_up), (0.20, self._visitor_down),
                   (0.12, self._outpatient_up), (0.10, self._outpatient_down),
                   (0.28, self._staff)]
        elif scenario == "visiting_hours":
            mix = [(0.50, self._visitor_up), (0.22, self._visitor_down),
                   (0.08, self._outpatient_up), (0.05, self._outpatient_down),
                   (0.15, self._staff)]
        elif scenario == "shift_change":
            mix = [(0.30, self._lobby_up), (0.30, self._lobby_down), (0.40, self._staff)]
        else:
            raise ValueError(f"unknown hospital scenario: {scenario}")

        events = self._mix(rate, duration, mix)
        emergency_events = self._emergencies(em_rate, duration)
        if scenario == "mass_casualty":
            emergency_events += self._burst(60.0, 60.0, 10, 4, duration)
        if not emergency_events and em_rate > 0 and duration > 0 and self.cfg.critical_floors():
            # A short random run can otherwise contain no emergency at all.
            o, d = self._er_transfer()
            emergency_events.append((duration / 2, o, d, True, True))
        events += emergency_events

        events.sort(key=lambda e: e[0])
        # Separate RNG keeps the passenger stream identical when alert coverage changes.
        alerts = np.random.default_rng(self.seed + 1000003)
        return [Passenger(i, o, d, t, is_emergency=em,
                          clinical_alert=bool(em and alerts.random() < alert_coverage),
                          space_units=transfer_space if transfer else 1)
                for i, (t, o, d, em, transfer) in enumerate(events)]
