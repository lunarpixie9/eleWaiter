"""Discrete-event lift simulator.

Design choices that matter for correctness
------------------------------------------
* Lifts move ONE FLOOR AT A TIME (a STEP event per floor). At every floor the
  lift re-decides, so it can stop en route, never teleports, and always spends
  1/speed seconds per floor travelled.
* Stops are derived from state (passengers aboard + passengers assigned), so
  there is no separate stop list that can go stale.
* Capacity is enforced per passenger while boarding. A lift that is full passes
  a waiting passenger (bypass); that passenger is "left behind" and re-dispatched.
* Movement follows LOOK/collective control. Emergency "express" mode overrides
  it: the lift heads straight to the emergency caller, picks up nobody else, and
  drops them at their destination (onboard passengers are still dropped when
  their floor is on the way).
* Starvation guard: anyone waiting longer than `starvation_limit` is promoted -
  force-assigned to a usable lift if needed, and that lift heads for them first.
"""
from __future__ import annotations

import heapq
from typing import List, Sequence

from .metrics import SimulationResult, build_result
from .models import BuildingConfig, LiftCar, LiftState, Passenger

ARRIVE, STEP, DOORS, TICK = 0, 1, 2, 3


class ElevatorSimulator:
    def __init__(self, config: BuildingConfig, passengers: List[Passenger], dispatcher,
                 scenario: str = "", disabled_lifts: Sequence[int] = (),
                 starvation_limit: float = 45.0, tick: float = 3.0,
                 max_overrun: float = 1800.0):
        self.config = config
        self.passengers = passengers
        self.pax = {p.id: p for p in passengers}
        self.dispatcher = dispatcher
        self.scenario = scenario
        self.starvation_limit = starvation_limit
        self.tick = tick
        self.lifts = [
            LiftCar(id=i, floor=config.lobby, capacity=config.capacity, speed=config.speed,
                    door_time=config.door_time,
                    state=LiftState.OUT_OF_SERVICE if i in set(disabled_lifts) else LiftState.IDLE)
            for i in range(config.num_lifts)
        ]
        self.now = 0.0
        self.eq: list = []
        self._seq = 0
        self.waiting: List[Passenger] = []
        self.completed: List[Passenger] = []
        self.timeline: List[dict] = []
        self.left_behind_total = 0
        self.forced_assignments = 0
        last = max((p.arrival_time for p in passengers), default=0.0)
        self.t_max = last + max_overrun
        self._dispatching = False
        self._redo = False

    # ------------------------------------------------------------------ engine
    def _push(self, t: float, kind: int, a: int = 0):
        self._seq += 1
        heapq.heappush(self.eq, (t, self._seq, kind, a))

    def run(self) -> SimulationResult:
        for p in self.passengers:
            self._push(p.arrival_time, ARRIVE, p.id)
        self._push(0.0, TICK)
        while self.eq:
            t, _, kind, a = heapq.heappop(self.eq)
            if t > self.t_max:
                break
            self.now = t
            if kind == ARRIVE:
                p = self.pax[a]
                p.urgency = self.config.infer_urgency(p.origin, p.destination, t)
                self.waiting.append(p)
                self._dispatch()
            elif kind == STEP:
                self._on_step(self.lifts[a])
            elif kind == DOORS:
                self._proceed(self.lifts[a])
            else:
                self._on_tick()
        return build_result(self)

    # ---------------------------------------------------------------- dispatch
    def _locked(self, p: Passenger) -> bool:
        lift = self.lifts[p.assigned_lift]
        return p.express or (lift.state == LiftState.DOORS_OPEN and lift.floor == p.origin)

    def _dispatch(self):
        if self._dispatching:
            self._redo = True
            return
        self._dispatching = True
        try:
            for _ in range(6):
                self._redo = False
                again = self._dispatch_once()
                if not (again or self._redo):
                    break
        finally:
            self._dispatching = False

    def _dispatch_once(self) -> bool:
        unassigned = [p for p in self.waiting if p.assigned_lift is None]
        reassign = []
        if self.dispatcher.reoptimize:
            reassign = [p for p in self.waiting if p.assigned_lift is not None and not self._locked(p)]
        if not unassigned and not reassign:
            return False
        result = self.dispatcher.assign(unassigned, reassign, self.lifts, self.now, self.config)
        released, touched = False, set()
        for pid, lid in result.lift_for.items():
            p = self.pax[pid]
            if p.assigned_lift == lid:
                continue
            if p.assigned_lift is not None:
                old = self.lifts[p.assigned_lift]
                old.assigned.remove(p)
                touched.add(old.id)
            p.assigned_lift = lid
            self.lifts[lid].assigned.append(p)
            touched.add(lid)
        for pid in result.express:
            p = self.pax[pid]
            if p.assigned_lift is None or p.express:
                continue
            lift = self.lifts[p.assigned_lift]
            if lift.is_full():
                continue
            new_lane = not lift.in_lane()
            p.express = True
            if new_lane:            # starting a lane: release routine pickups
                for q in list(lift.assigned):
                    if not q.express:
                        lift.assigned.remove(q)
                        q.assigned_lift = None
                        released = True
        for lid in touched:
            if self.lifts[lid].state == LiftState.IDLE:
                self._proceed(self.lifts[lid])
        return released

    def _release(self, p: Passenger, left_behind: bool):
        lift = self.lifts[p.assigned_lift]
        lift.assigned.remove(p)
        p.assigned_lift = None
        p.express = False
        if left_behind:
            p.left_behind += 1
            self.left_behind_total += 1

    # ------------------------------------------------------------- starvation
    def _on_tick(self):
        self._starvation_guard()
        self._dispatch()
        if self.eq or self.waiting:
            self._push(self.now + self.tick, TICK)

    def _starvation_guard(self):
        for p in list(self.waiting):
            if self.now - p.arrival_time < self.starvation_limit:
                continue
            lift = self.lifts[p.assigned_lift] if p.assigned_lift is not None else None
            usable = lift is not None and lift.state != LiftState.OUT_OF_SERVICE and not lift.is_full()
            if usable and p.promoted:
                continue
            if not usable:
                if lift is not None:
                    self._release(p, left_behind=False)
                cands = [l for l in self.lifts if l.state != LiftState.OUT_OF_SERVICE
                         and not l.is_full() and not l.in_lane()]
                if not cands:
                    continue
                lift = min(cands, key=lambda l: (abs(l.floor - p.origin), l.id))
                p.assigned_lift = lift.id
                lift.assigned.append(p)
            p.promoted = True
            self.forced_assignments += 1
            if lift.state == LiftState.IDLE:
                self._proceed(lift)

    # --------------------------------------------------------------- movement
    def _next_direction(self, lift: LiftCar, stops) -> int:
        f = lift.floor
        lane = [t for t in lift.lane_targets() if t != f]
        if lane:                # emergency lane: head for the nearest critical target
            target = min(lane, key=lambda x: abs(x - f))
            return 1 if target > f else -1
        if not lift.is_full():
            promo = [p.origin for p in lift.assigned if p.promoted]
            if promo:
                target = min(promo, key=lambda x: abs(x - f))
                if target != f:
                    return 1 if target > f else -1
        above = [s for s in stops if s > f]
        below = [s for s in stops if s < f]
        d = lift.direction
        if d > 0 and above:
            return 1
        if d < 0 and below:
            return -1
        if above and below:
            return 1 if (min(above) - f) <= (f - max(below)) else -1
        if above:
            return 1
        if below:
            return -1
        return 0

    def _proceed(self, lift: LiftCar):
        """Lift is at a floor with doors closed: open here, move on, or go idle."""
        stops = lift.stop_floors()
        if lift.floor in stops:
            self._open_doors(lift)
            return
        d = self._next_direction(lift, stops)
        if d == 0:
            lift.state = LiftState.IDLE
            lift.direction = 0
            return
        lift.direction = d
        lift.state = LiftState.MOVING
        lift.next_arrival_at = self.now + 1.0 / lift.speed
        self._push(lift.next_arrival_at, STEP, lift.id)

    def _on_step(self, lift: LiftCar):
        lift.floor += lift.direction
        lift.travelled += 1.0
        released = False
        if lift.is_full():      # full lift bypasses hall calls on this floor
            for p in [q for q in lift.assigned if q.origin == lift.floor]:
                self._release(p, left_behind=True)
                released = True
        self._proceed(lift)
        if released:
            self._dispatch()

    def _open_doors(self, lift: LiftCar):
        lift.state = LiftState.DOORS_OPEN
        f = lift.floor
        exiting = [p for p in lift.passengers if p.destination == f]
        for p in exiting:
            lift.passengers.remove(p)
            p.exit_time = self.now
            self.completed.append(p)
        cands = [p for p in lift.assigned if p.origin == f]
        lane = lift.in_lane()
        if lane:                # lane lifts board only express (critical) passengers
            for p in [q for q in cands if not q.express]:
                self._release(p, left_behind=False)
            cands = [p for p in cands if p.express]
        if self.dispatcher.priority_boarding:
            cands.sort(key=lambda p: (-int(p.urgency), p.arrival_time))
        else:
            cands.sort(key=lambda p: p.arrival_time)
        boarded = []
        for p in cands:
            if lift.is_full():
                break
            lift.assigned.remove(p)
            lift.passengers.append(p)
            self.waiting.remove(p)
            p.board_time = self.now
            boarded.append(p)
        for p in cands[len(boarded):]:          # no room: left behind
            self._release(p, left_behind=True)
        dwell = lift.door_time + self.config.board_time * (len(exiting) + len(boarded))
        lift.doors_close_at = self.now + dwell
        self._push(lift.doors_close_at, DOORS, lift.id)
        self.timeline.append({"time": round(self.now, 1), "lift": lift.id, "floor": f,
                              "load": len(lift.passengers), "boarded": len(boarded),
                              "exited": len(exiting), "lane": lane,
                              "routine_boarded_in_lane": sum(1 for p in boarded if lane and not p.express)})
        self._dispatch()
