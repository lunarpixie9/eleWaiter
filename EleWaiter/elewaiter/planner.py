"""Fast analytic model of a lift's future, used by the AI dispatcher.

A `LiftModel` is an immutable-ish snapshot of one lift (position, direction,
stops, who is aboard, who is already assigned). From it we estimate, in seconds:

  * when the lift will reach a new passenger's floor (wait),
  * how long that passenger will then ride (ride),
  * how much the new stops delay everyone already planned (imposed delay),
  * how full the lift will be when it reaches the pickup floor (predicted load).

Lifts are assumed to follow LOOK / collective control: keep going in the current
direction while there are stops ahead, then reverse.
"""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Set

from .models import LiftCar, LiftState


# --------------------------------------------------------------------------- route
def look_order(floor: int, direction: int, stops: Iterable[int]) -> List[int]:
    """Order in which a LOOK lift would visit `stops`."""
    stops = set(stops)
    if direction > 0:
        up = sorted(s for s in stops if s >= floor)
        down = sorted((s for s in stops if s < floor), reverse=True)
        return up + down
    if direction < 0:
        down = sorted((s for s in stops if s <= floor), reverse=True)
        up = sorted(s for s in stops if s > floor)
        return down + up
    up = sorted(s for s in stops if s >= floor)
    down = sorted((s for s in stops if s < floor), reverse=True)
    if up and (not down or (up[0] - floor) <= (floor - down[0])):
        return up + down
    return down + up


class LiftModel:
    __slots__ = ("id", "floor", "dir", "t0", "speed", "door", "cap", "onboard",
                 "exits", "ppl", "pend", "express", "ok", "_stops", "_tab")

    def __init__(self, id, floor, direction, t0, speed, door, cap, onboard,
                 exits, ppl, pend, express=False, ok=True):
        self.id = id
        self.floor = floor
        self.dir = direction
        self.t0 = t0                # seconds until the lift can start moving from `floor`
        self.speed = speed
        self.door = door
        self.cap = cap
        self.onboard = onboard      # people currently inside
        self.exits = exits          # {floor: n onboard passengers leaving there}
        self.ppl = ppl              # {floor: n people involved in a stop there}
        self.pend = pend            # ((origin, dest), ...) assigned, not yet boarded
        self.express = express      # being rushed for an emergency -> not a candidate
        self.ok = ok                # in service
        self._stops: Optional[Set[int]] = None
        self._tab = None

    # -- cached views ---------------------------------------------------------
    @property
    def stop_set(self) -> Set[int]:
        if self._stops is None:
            s = set(self.exits)
            for o, d, _ in self.pend:
                s.add(o)
                s.add(d)
            self._stops = s
        return self._stops

    def table(self):
        if self._tab is None:
            self._tab = eta_table(self, self.stop_set)
        return self._tab

    # -- state transitions (return NEW models) -------------------------------
    def with_pickup(self, o: int, d: int, units: int = 1) -> "LiftModel":
        ppl = dict(self.ppl)
        ppl[o] = ppl.get(o, 0) + 1
        ppl[d] = ppl.get(d, 0) + 1
        return LiftModel(self.id, self.floor, self.dir, self.t0, self.speed, self.door,
                         self.cap, self.onboard, self.exits, ppl,
                         self.pend + ((o, d, units),), self.express, self.ok)

    def as_express(self) -> "LiftModel":
        """Lift committed to an emergency: other pickups are released."""
        return LiftModel(self.id, self.floor, self.dir, self.t0, self.speed, self.door,
                         self.cap, self.onboard, self.exits, self.ppl, (), True, self.ok)


def eta_table(m: LiftModel, stops: Iterable[int]):
    """Return (order, {floor: arrival_time}, time_when_route_finished)."""
    order = look_order(m.floor, m.dir, stops)
    t, pos, tab = m.t0, m.floor, {}
    sp, dr = m.speed, m.door
    for s in order:
        t += abs(s - pos) / sp
        tab[s] = t
        t += dr
        pos = s
    return order, tab, t


def model_from_lift(lift: LiftCar, now: float, exclude: Set[int] = frozenset()) -> LiftModel:
    if lift.state == LiftState.MOVING:
        floor = lift.floor + lift.direction
        t0 = max(0.0, lift.next_arrival_at - now)
    elif lift.state == LiftState.DOORS_OPEN:
        floor, t0 = lift.floor, max(0.0, lift.doors_close_at - now)
    else:
        floor, t0 = lift.floor, 0.0
    exits: Dict[int, int] = {}
    ppl: Dict[int, int] = {}
    for p in lift.passengers:
        exits[p.destination] = exits.get(p.destination, 0) + p.space_units
        ppl[p.destination] = ppl.get(p.destination, 0) + p.space_units
    pend = []
    for p in lift.assigned:
        if p.id in exclude:
            continue
        pend.append((p.origin, p.destination, p.space_units))
        ppl[p.origin] = ppl.get(p.origin, 0) + p.space_units
        ppl[p.destination] = ppl.get(p.destination, 0) + p.space_units
    return LiftModel(lift.id, floor, lift.direction, t0, lift.speed, lift.door_time,
                     lift.capacity, lift.load_units(), exits, ppl, tuple(pend),
                     express=lift.in_lane(),
                     ok=lift.state != LiftState.OUT_OF_SERVICE and not lift.retiring)


# --------------------------------------------------------------------------- costs
def predicted_load(m: LiftModel, tab: Dict[int, float], t_pickup: float) -> int:
    """People inside the lift when it reaches the pickup floor."""
    load = m.onboard
    for s, c in m.exits.items():
        if tab[s] < t_pickup:
            load -= c
    for po, pd, units in m.pend:
        if tab[po] <= t_pickup:
            load += units
            if tab[po] < tab[pd] <= t_pickup:
                load -= units
    return load


def normal_cost(m: LiftModel, o: int, d: int, W: float, w: Dict[str, float],
                units: int = 1) -> float:
    """Cost of letting lift `m` serve a call o->d under normal (LOOK) operation.

    cost = W * (w_wait*wait + w_ride*ride)       <- the passenger's OWN experience,
                                                    scaled by their urgency weight W
         + w_delay * imposed_delay               <- delay inflicted on others
         + w_load * predicted_load_fraction      <- crowding
         + w_over * W   if the lift will be full when it gets there
    Because W multiplies only the passenger's own terms, urgent passengers are
    willing to impose more delay on others - so urgency genuinely changes WHICH
    lift is chosen (a constant per-passenger bonus would not).
    """
    _, tab0, _ = m.table()
    stops1 = m.stop_set | {o, d}
    _, tab1, tend1 = eta_table(m, stops1)
    to, td = tab1[o], tab1[d]
    ride = (td - to) if td > to else (tend1 - to) + abs(o - d) / m.speed
    imposed = 0.0
    for s in m.stop_set:
        imposed += (tab1[s] - tab0[s]) * m.ppl.get(s, 1)
    load = predicted_load(m, tab1, to)
    cost = W * (w["w_wait"] * to + w["w_ride"] * ride) + w["w_delay"] * imposed \
        + w["w_load"] * max(0, load) / m.cap
    if load + units > m.cap:
        cost += w["w_over"] * W
    return cost


def express_cost(m: LiftModel, o: int, d: int, W: float, w: Dict[str, float],
                 bpl: float = 0.0, units: int = 1) -> float:
    """Cost of starting an emergency lane on lift `m`: straight to `o`, then `d`,
    with no routine pickups. Passengers already aboard are still dropped if their
    floor is on the way. `bpl` = waiting passengers per active lift: taking a lift
    out of service delays everyone already waiting, so rushing is costlier when
    the building is busy."""
    f, sp, dr = m.floor, m.speed, m.door
    lo, hi = (f, o) if f <= o else (o, f)
    drops_to_o = [c for s, c in m.exits.items() if lo < s < hi]
    wait = m.t0 + abs(f - o) / sp + dr * len(drops_to_o)
    lo2, hi2 = (o, d) if o <= d else (d, o)
    drops_to_d = sum(1 for s in m.exits if lo2 < s < hi2)
    ride = dr + abs(o - d) / sp + dr * drops_to_d
    onboard_after = m.onboard - sum(drops_to_o)
    still_aboard = max(0, m.onboard - sum(drops_to_o))
    imposed = still_aboard * 0.5 * (wait + ride) + len(m.pend) * w["w_release"] \
        + w["w_backlog"] * bpl * 0.5 * (wait + ride)
    cost = W * (w["w_wait"] * wait + w["w_ride"] * ride) + w["w_delay"] * imposed \
        + w["w_load"] * onboard_after / m.cap
    if onboard_after + units > m.cap:
        cost += w["w_over"] * W
    # benefit of a lane: the emergency ride is protected from later detours
    cost -= w["w_protect"] * W * ride
    return cost
