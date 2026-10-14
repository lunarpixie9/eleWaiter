"""Conventional dispatchers used as benchmarks.

FCFS, Nearest and Collective are urgency-blind (as in most real buildings).
PriorityNearest is a deliberately strong, simple 'emergency-aware' baseline: it
shows what a hand-written rule achieves, so the AI has to beat that - not only
naive rules.
"""
from __future__ import annotations

from ..models import LiftState, Urgency
from ..planner import eta_table, model_from_lift
from .base import Assignment, Dispatcher


def _eligible(lifts, p=None):
    """Conventional rule: in service, not full, not busy with an emergency."""
    return [l for l in lifts if l.state != LiftState.OUT_OF_SERVICE
            and not l.retiring and not l.is_full() and not l.in_lane()
            and (p is None or l.has_room_for(p))]


class FCFS(Dispatcher):
    """Serve calls in arrival order; give each to the lift with the fewest stops."""
    name = "FCFS"

    def assign(self, unassigned, reassignable, lifts, now, config):
        out = Assignment()
        extra = {l.id: 0 for l in lifts}
        for p in sorted(unassigned, key=lambda p: p.arrival_time):
            el = _eligible(lifts, p)
            if not el:
                continue
            best = min(el, key=lambda l: (len(l.stop_floors()) + extra[l.id], l.id))
            out.lift_for[p.id] = best.id
            extra[best.id] += 1
        return out


class Nearest(Dispatcher):
    """Send the lift that is closest (in floors) to the caller."""
    name = "Nearest"

    def assign(self, unassigned, reassignable, lifts, now, config):
        out = Assignment()
        for p in sorted(unassigned, key=lambda p: p.arrival_time):
            el = _eligible(lifts, p)
            if el:
                out.lift_for[p.id] = min(el, key=lambda l: (abs(l.floor - p.origin), l.id)).id
        return out


class Collective(Dispatcher):
    """Collective control with ETA: choose the lift that will reach the caller
    soonest given its current direction and stops (LOOK-aware nearest car)."""
    name = "Collective (LOOK-ETA)"

    def assign(self, unassigned, reassignable, lifts, now, config):
        out = Assignment()
        for p in sorted(unassigned, key=lambda p: p.arrival_time):
            el = _eligible(lifts, p)
            if not el:
                continue
            best, best_t = None, 1e18
            for l in el:
                m = model_from_lift(l, now, exclude={q.id for q in unassigned})
                _, tab, _ = eta_table(m, m.stop_set | {p.origin, p.destination})
                t = tab[p.origin]
                if t < best_t:
                    best, best_t = l, t
            out.lift_for[p.id] = best.id
        return out


class PriorityNearest(Dispatcher):
    """Nearest-lift, plus hand-written emergency rules: a CRITICAL trip gets the
    nearest free lift in express mode, and urgent passengers board first."""
    name = "Priority-Nearest"
    priority_boarding = True

    def assign(self, unassigned, reassignable, lifts, now, config):
        out = Assignment()
        order = sorted(unassigned, key=lambda p: (-int(p.urgency), p.arrival_time))
        taken_express = set()
        for p in order:
            el = [l for l in _eligible(lifts, p) if l.id not in taken_express]
            if not el:
                continue
            best = min(el, key=lambda l: (abs(l.floor - p.origin), l.id))
            out.lift_for[p.id] = best.id
            if p.urgency == Urgency.CRITICAL:
                out.express.add(p.id)
                taken_express.add(best.id)
        return out
