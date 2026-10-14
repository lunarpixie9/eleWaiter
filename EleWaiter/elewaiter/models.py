"""Core data models for EleWaiter."""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import Dict, List, Optional, Set, Tuple


class Zone(IntEnum):
    """Clinical/operational importance of a floor (set once per building)."""
    LOW = 0        # storage, parking, admin
    NORMAL = 1     # lobby, wards, clinics
    HIGH = 2       # imaging, labs, maternity
    CRITICAL = 3   # emergency dept, ICU, operating theatres


class Urgency(IntEnum):
    """Urgency the dispatcher INFERS for a trip (it never sees ground truth)."""
    ROUTINE = 0
    ELEVATED = 1
    CRITICAL = 2


class LiftState(IntEnum):
    IDLE = 0
    MOVING = 1
    DOORS_OPEN = 2
    OUT_OF_SERVICE = 3


@dataclass
class Floor:
    number: int
    name: str
    zone: Zone = Zone.NORMAL
    role: str = "other"          # lobby | er | icu | ot | imaging | ward | clinic | other
    # (start, end) simulation-time windows in which the zone is upgraded one level
    boost_windows: Tuple[Tuple[float, float], ...] = ()
    roles: Tuple[str, ...] = ()   # multiple departments can share a small hospital floor

    def zone_at(self, t: float) -> Zone:
        z = self.zone
        if z < Zone.CRITICAL:
            for a, b in self.boost_windows:
                if a <= t < b:
                    return Zone(z + 1)
        return z


@dataclass
class BuildingConfig:
    name: str
    kind: str                       # "hospital"
    num_floors: int
    num_lifts: int
    capacity: int = 12
    speed: float = 1.0              # floors per second
    door_time: float = 3.0          # seconds a stop costs (doors open+close)
    board_time: float = 0.5         # extra dwell per boarding/exiting passenger
    floors: Dict[int, Floor] = field(default_factory=dict)
    lobby: int = 1

    # ---- zone helpers -----------------------------------------------------
    def zone_at(self, floor: int, t: float = 0.0) -> Zone:
        f = self.floors.get(floor)
        return f.zone_at(t) if f else Zone.NORMAL

    def floors_with_role(self, role: str) -> List[int]:
        return sorted(n for n, f in self.floors.items() if f.role == role or role in f.roles)

    def critical_floors(self) -> List[int]:
        return sorted(n for n, f in self.floors.items() if f.zone == Zone.CRITICAL)

    # ---- urgency inference (context only, no extra hardware) -------------
    def infer_urgency(self, origin: int, dest: int, t: float = 0.0) -> Urgency:
        """Infer how urgent a trip is from WHERE it starts/ends and WHEN.

        CRITICAL : leaves a critical unit for another HIGH/CRITICAL area
                   (e.g. ER -> ICU / OT / imaging: a patient transfer).
        ELEVATED : touches a HIGH or CRITICAL zone (e.g. staff going to ICU).
        ROUTINE  : everything else.
        """
        zo, zd = self.zone_at(origin, t), self.zone_at(dest, t)
        if zo == Zone.CRITICAL and zd >= Zone.HIGH:
            return Urgency.CRITICAL
        if max(zo, zd) >= Zone.HIGH:
            return Urgency.ELEVATED
        return Urgency.ROUTINE


@dataclass
class Passenger:
    id: int
    origin: int
    destination: int
    arrival_time: float
    is_emergency: bool = False          # simulation truth; never read by dispatchers
    clinical_alert: bool = False        # simulated authorized alert visible to dispatch
    space_units: int = 1                # patient, team and equipment travel together
    # --- runtime state -----------------------------------------------------
    urgency: Urgency = Urgency.ROUTINE  # inferred by the simulator on arrival
    assigned_lift: Optional[int] = None
    board_time: Optional[float] = None
    exit_time: Optional[float] = None
    left_behind: int = 0
    promoted: bool = False              # starvation guard flag
    express: bool = False               # riding in an emergency lane

    @property
    def wait_time(self) -> Optional[float]:
        return None if self.board_time is None else self.board_time - self.arrival_time

    @property
    def journey_time(self) -> Optional[float]:
        return None if self.exit_time is None else self.exit_time - self.arrival_time


@dataclass
class LiftCar:
    id: int
    floor: int = 1
    capacity: int = 12
    speed: float = 1.0
    door_time: float = 3.0
    state: LiftState = LiftState.IDLE
    direction: int = 0                  # last travel direction: +1 / -1 / 0
    passengers: List[Passenger] = field(default_factory=list)   # onboard
    assigned: List[Passenger] = field(default_factory=list)     # waiting, assigned here
    doors_close_at: float = 0.0
    next_arrival_at: float = 0.0
    travelled: float = 0.0
    retiring: bool = False              # failed lift finishes onboard drop-offs only

    def load_units(self) -> int:
        return sum(p.space_units for p in self.passengers)

    def has_room_for(self, p: Passenger) -> bool:
        return self.load_units() + p.space_units <= self.capacity

    def is_full(self) -> bool:
        return self.load_units() >= self.capacity

    def in_lane(self) -> bool:
        """Emergency lane: the lift is committed to express (critical) passengers and
        takes no routine pickups until every express passenger is delivered."""
        return any(p.express for p in self.passengers) or any(p.express for p in self.assigned)

    def lane_targets(self) -> List[int]:
        t = [p.destination for p in self.passengers if p.express]
        t += [p.origin for p in self.assigned if p.express]
        return t

    def stop_floors(self) -> Set[int]:
        """Floors the lift must visit: drop-offs, plus pickups if it has room."""
        s = {p.destination for p in self.passengers}
        if not self.is_full() and not self.retiring:
            s.update(p.origin for p in self.assigned)
        return s
