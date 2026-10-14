"""Hospital building presets."""
from __future__ import annotations

from typing import Callable, Dict

from .models import BuildingConfig, Floor, Zone


def custom_hospital(num_floors: int, num_lifts: int, capacity: int,
                    er_floor: int, icu_floor: int, ot_floor: int,
                    imaging_floor: int) -> BuildingConfig:
    """Build a hospital layout from the simple frontend floor assignments."""
    if not 3 <= num_floors <= 30:
        raise ValueError("hospital must have between 3 and 30 floors")
    if not 1 <= num_lifts <= 20 or not 3 <= capacity <= 40:
        raise ValueError("lift count or capacity is outside the supported range")
    clinical = (er_floor, icu_floor, ot_floor, imaging_floor)
    if any(not 2 <= n <= num_floors for n in clinical):
        raise ValueError("clinical departments must be above the lobby")
    if len(set(clinical)) < 2:
        raise ValueError("clinical departments must use at least two different floors so transfers need a lift")

    floors = {1: Floor(1, "Main Lobby", Zone.NORMAL, "lobby")}
    departments = (
        (er_floor, "Emergency Dept", Zone.CRITICAL, "er"),
        (icu_floor, "ICU", Zone.CRITICAL, "icu"),
        (ot_floor, "Operating Theatres", Zone.CRITICAL, "ot"),
        (imaging_floor, "Imaging & Labs", Zone.HIGH, "imaging"),
    )
    for n in range(2, num_floors + 1):
        assigned = [(name, zone, role) for floor, name, zone, role in departments if floor == n]
        if assigned:
            name = " / ".join(item[0] for item in assigned)
            zone = max(item[1] for item in assigned)
            role = assigned[0][2]
            roles = tuple(item[2] for item in assigned)
        else:
            role = "clinic" if n % 3 == 1 else "ward"
            name, zone = ("Outpatient Clinic" if role == "clinic" else "General Ward"), Zone.NORMAL
            roles = ()
        floors[n] = Floor(n, name, zone, role, roles=roles)
    return BuildingConfig(f"{num_floors}-floor Custom Hospital", "hospital", num_floors,
                          num_lifts, capacity=capacity, floors=floors)


def _mk(rows):
    """rows: (number, name, zone, role[, boost_windows])"""
    out = {}
    for r in rows:
        n, name, zone, role = r[:4]
        win = r[4] if len(r) > 4 else ()
        out[n] = Floor(n, name, zone, role, win)
    return out


def hospital_10() -> BuildingConfig:
    rows = [
        (1, "Main Lobby", Zone.NORMAL, "lobby"),
        (2, "Emergency Dept", Zone.CRITICAL, "er"),
        (3, "Imaging & Labs", Zone.HIGH, "imaging"),
        (4, "Outpatient A", Zone.NORMAL, "clinic"),
        (5, "Outpatient B", Zone.NORMAL, "clinic"),
        (6, "General Ward A", Zone.NORMAL, "ward"),
        (7, "General Ward B", Zone.NORMAL, "ward"),
        (8, "Maternity", Zone.HIGH, "ward"),
        (9, "ICU", Zone.CRITICAL, "icu"),
        (10, "Operating Theatres", Zone.CRITICAL, "ot"),
    ]
    return BuildingConfig("10-floor Hospital", "hospital", 10, 3, capacity=12,
                          speed=1.0, door_time=3.0, floors=_mk(rows))


def hospital_15() -> BuildingConfig:
    rows = [
        (1, "Main Lobby", Zone.NORMAL, "lobby"),
        (2, "Emergency Dept", Zone.CRITICAL, "er"),
        (3, "Radiology", Zone.HIGH, "imaging"),
        (4, "Pharmacy & Labs", Zone.HIGH, "imaging"),
        (5, "Outpatient A", Zone.NORMAL, "clinic"),
        (6, "Outpatient B", Zone.NORMAL, "clinic"),
        (7, "General Ward A", Zone.NORMAL, "ward"),
        (8, "General Ward B", Zone.NORMAL, "ward"),
        (9, "General Ward C", Zone.NORMAL, "ward"),
        (10, "Paediatrics", Zone.NORMAL, "ward"),
        (11, "Maternity", Zone.HIGH, "ward"),
        (12, "Cardiac Cath Lab", Zone.CRITICAL, "ot"),
        (13, "Surgical Ward", Zone.NORMAL, "ward"),
        (14, "ICU / NICU", Zone.CRITICAL, "icu"),
        (15, "Operating Theatres", Zone.CRITICAL, "ot"),
    ]
    return BuildingConfig("15-floor Hospital", "hospital", 15, 4, capacity=14,
                          speed=1.2, door_time=3.0, floors=_mk(rows))


PRESETS: Dict[str, Callable[[], BuildingConfig]] = {
    "10-floor Hospital": hospital_10,
    "15-floor Hospital": hospital_15,
}
