"""Building presets. Hospitals first; office/college show the system generalises."""
from __future__ import annotations

from typing import Callable, Dict

from .models import BuildingConfig, Floor, Zone


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


def office_12() -> BuildingConfig:
    rows = [(1, "Lobby", Zone.NORMAL, "lobby"), (2, "Cafeteria", Zone.NORMAL, "other")]
    rows += [(n, f"Office Floor {n}", Zone.NORMAL, "other") for n in range(3, 12)]
    rows.append((12, "Executive Floor", Zone.HIGH, "other"))
    return BuildingConfig("12-floor Office", "office", 12, 4, capacity=13,
                          speed=1.5, door_time=2.5, floors=_mk(rows))


def college_8() -> BuildingConfig:
    # lecture floors get a priority boost just before class starts (time-of-day context)
    cls = ((100.0, 180.0),)
    rows = [
        (1, "Entrance", Zone.NORMAL, "lobby"),
        (2, "Canteen", Zone.NORMAL, "other"),
        (3, "Lecture Hall A", Zone.NORMAL, "other", cls),
        (4, "Lecture Hall B", Zone.NORMAL, "other", cls),
        (5, "Lecture Hall C", Zone.NORMAL, "other", cls),
        (6, "Labs", Zone.NORMAL, "other"),
        (7, "Library", Zone.NORMAL, "other"),
        (8, "Admin", Zone.LOW, "other"),
    ]
    return BuildingConfig("8-floor College", "college", 8, 3, capacity=10,
                          speed=1.2, door_time=2.5, floors=_mk(rows))


PRESETS: Dict[str, Callable[[], BuildingConfig]] = {
    "10-floor Hospital": hospital_10,
    "15-floor Hospital": hospital_15,
    "12-floor Office": office_12,
    "8-floor College": college_8,
}
