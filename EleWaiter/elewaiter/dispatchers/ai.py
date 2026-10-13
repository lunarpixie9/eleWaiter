"""EleWaiter AI dispatcher.

Core ideas
----------
1. Multi-objective cost (see planner.normal_cost): the passenger's own wait and
   ride, scaled by an urgency weight, versus the delay imposed on others, lift
   crowding and predicted overflow (left-behind risk).
2. Urgency weight W = 1 + a_urg*urgency + a_age*log(1+waited/15) + a_lb*left_behind
   -> emergencies first, long waiters (fairness / anti-starvation) next, and
   passengers who were already left behind get a boost.
3. Emergency express mode: a CRITICAL trip gets one lift committed to it,
   running directly to the caller then directly to the destination.
4. A* search: pending calls are assigned JOINTLY (not one at a time), because
   giving the best lift to one caller changes what is best for the next.
5. Dynamic re-optimisation: the best few already-assigned, not-yet-boarded
   passengers are re-planned too, with a switching penalty to avoid thrashing.
"""
from __future__ import annotations

import heapq
import json
import math
import os
from typing import Dict, Optional, Tuple

from ..models import LiftState, Urgency
from ..planner import (LiftModel, express_cost, model_from_lift, normal_cost)
from .base import Assignment, Dispatcher

DEFAULT_WEIGHTS: Dict[str, float] = {
    "w_wait": 1.0,       # weight on the passenger's own waiting time
    "w_ride": 0.5,       # ... and ride time
    "w_delay": 0.35,     # delay imposed on people already planned
    "w_load": 8.0,       # crowding (seconds-equivalent at a full lift)
    "w_over": 60.0,      # predicted overflow = would be left behind
    "a_urg": 1.5,        # urgency boost per urgency level
    "a_age": 0.8,        # anti-starvation: boost grows with waiting time
    "a_lb": 0.5,         # boost per time already left behind
    "w_switch": 3.0,     # penalty for moving an existing assignment
    "w_release": 15.0,   # cost of bumping a pickup when a lift goes express
    "x_express": 1.0,    # multiplier on express cost (>1 = reluctant to go express)
    "w_backlog": 0.4,    # extra express cost per waiting passenger per active lift
    "w_protect": 0.15,   # benefit of a lane: ride protected from later detours
}

WEIGHT_BOUNDS: Dict[str, Tuple[float, float]] = {
    "w_wait": (0.3, 3.0), "w_ride": (0.1, 2.0), "w_delay": (0.05, 1.5),
    "w_load": (0.0, 20.0), "w_over": (10.0, 150.0), "a_urg": (0.0, 6.0),
    "a_age": (0.0, 3.0), "a_lb": (0.0, 2.0), "w_switch": (0.0, 10.0),
    "w_release": (0.0, 40.0), "x_express": (0.3, 3.0),
    "w_backlog": (0.0, 2.0), "w_protect": (0.0, 1.0),
}

_TUNED_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "tuned_weights.json")


def load_tuned_weights() -> Optional[Dict[str, float]]:
    try:
        with open(_TUNED_PATH) as fh:
            w = json.load(fh)
        return {**DEFAULT_WEIGHTS, **{k: float(v) for k, v in w.items() if k in DEFAULT_WEIGHTS}}
    except (OSError, ValueError):
        return None


class AIDispatcher(Dispatcher):
    reoptimize = True

    def __init__(self, weights: Optional[Dict[str, float]] = None, emergency: bool = True,
                 batch: int = 4, max_reopt: int = 3, max_nodes: int = 120,
                 name: Optional[str] = None):
        self.w = {**DEFAULT_WEIGHTS, **(weights if weights is not None else (load_tuned_weights() or {}))}
        self.emergency = emergency
        self.batch = batch
        self.max_reopt = max_reopt
        self.max_nodes = max_nodes
        self.priority_boarding = emergency
        self._bpl = 0.0
        self.name = name or ("EleWaiter AI" if emergency else "AI (no emergency logic)")

    # ------------------------------------------------------------------ weights
    def pweight(self, p, now: float) -> float:
        W = 1.0
        if self.emergency:
            W += self.w["a_urg"] * int(p.urgency)
        W += self.w["a_age"] * math.log1p(max(0.0, now - p.arrival_time) / 15.0)
        W += self.w["a_lb"] * p.left_behind
        return W

    def _is_express(self, p) -> bool:
        return self.emergency and p.urgency == Urgency.CRITICAL

    # ------------------------------------------------------------------- costs
    def _option(self, p, W: float, m: LiftModel):
        """Best (cost, new_model, express?) for lift m serving passenger p, or None.

        * A lift already running an emergency lane accepts ONLY other critical
          passengers (batching them into the same trip).
        * For a CRITICAL passenger on a free lift both options are priced and the
          cheaper wins: start a lane (express) or normal priority service. So the
          dispatcher rushes emergencies when that is cheap and falls back to
          weighted priority when rushing would stall the rest of the building.
        """
        if not m.ok:
            return None
        critical = self._is_express(p)
        if m.express:                                   # lane already running
            if not critical or m.onboard >= m.cap:
                return None
            c = normal_cost(m, p.origin, p.destination, W, self.w)
            return c, m.with_pickup(p.origin, p.destination), True
        best = None
        if critical and m.onboard < m.cap:
            c = self.w["x_express"] * express_cost(m, p.origin, p.destination, W, self.w, self._bpl)
            best = (c, m.as_express(), True)
        c = normal_cost(m, p.origin, p.destination, W, self.w)
        if p.assigned_lift is not None and p.assigned_lift != m.id:
            c += self.w["w_switch"]
        if best is None or c < best[0]:
            best = (c, m.with_pickup(p.origin, p.destination), False)
        return best

    # ------------------------------------------------------------------ search
    def _search(self, chunk, Ws, models: Tuple[LiftModel, ...]):
        """A* over joint assignments of `chunk` (already ordered by urgency).

        State  : (index of next passenger, tuple of lift models after assignments)
        g      : accumulated cost of the assignments made so far
        h      : sum over remaining passengers of their cheapest option on the
                 *current* models (a heuristic lower bound: adding stops to a lift
                 rarely makes a later call cheaper)
        Falls back to greedy completion if `max_nodes` expansions are exceeded.
        """
        n = len(chunk)
        h = [0.0] * (n + 1)
        for j in range(n - 1, -1, -1):
            best = None
            for m in models:
                o = self._option(chunk[j], Ws[j], m)
                if o is not None and (best is None or o[0] < best):
                    best = o[0]
            h[j] = h[j + 1] + (best if best is not None else 0.0)

        counter = 0
        heap = [(h[0], 0, 0, 0.0, models, ())]
        expansions = 0
        while heap:
            _, _, idx, g, ms, picks = heapq.heappop(heap)
            if idx == n:
                return picks, ms
            expansions += 1
            if expansions > self.max_nodes:
                return self._greedy_finish(chunk, Ws, idx, ms, picks)
            p = chunk[idx]
            expanded = False
            for k, m in enumerate(ms):
                opt = self._option(p, Ws[idx], m)
                if opt is None:
                    continue
                c, nm, ex = opt
                counter += 1
                heapq.heappush(heap, (g + c + h[idx + 1], counter, idx + 1, g + c,
                                      ms[:k] + (nm,) + ms[k + 1:],
                                      picks + ((p.id, m.id, ex),)))
                expanded = True
            if not expanded:    # nobody can take this passenger right now
                counter += 1
                heapq.heappush(heap, (g + h[idx + 1], counter, idx + 1, g, ms, picks))
        return (), models

    def _greedy_finish(self, chunk, Ws, idx, ms, picks):
        for j in range(idx, len(chunk)):
            best = None
            for k, m in enumerate(ms):
                o = self._option(chunk[j], Ws[j], m)
                if o is not None and (best is None or o[0] < best[0]):
                    best = (o[0], o[1], o[2], k, m.id)
            if best is None:
                continue
            _, nm, ex, k, lid = best
            ms = ms[:k] + (nm,) + ms[k + 1:]
            picks = picks + ((chunk[j].id, lid, ex),)
        return picks, ms

    # ------------------------------------------------------------------ API
    def assign(self, unassigned, reassignable, lifts, now, config) -> Assignment:
        out = Assignment()
        active = max(1, sum(1 for l in lifts if l.state != LiftState.OUT_OF_SERVICE))
        self._bpl = (len(unassigned) + sum(len(l.assigned) for l in lifts)) / active
        un = sorted(unassigned, key=lambda p: (-self.pweight(p, now), p.arrival_time))
        re = sorted(reassignable, key=lambda p: (-self.pweight(p, now), p.arrival_time))
        batch = un + re[: self.max_reopt]
        if not batch:
            return out
        exclude = {p.id for p in batch}
        models = tuple(model_from_lift(l, now, exclude) for l in lifts)
        for i in range(0, len(batch), self.batch):
            chunk = batch[i:i + self.batch]
            Ws = [self.pweight(p, now) for p in chunk]
            picks, models = self._search(chunk, Ws, models)
            for pid, lid, ex in picks:
                out.lift_for[pid] = lid
                if ex:
                    out.express.add(pid)
        return out
