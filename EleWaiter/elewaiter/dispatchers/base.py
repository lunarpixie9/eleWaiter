"""Dispatcher interface shared by baselines and the AI dispatcher."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Set


@dataclass
class Assignment:
    lift_for: Dict[int, int] = field(default_factory=dict)   # passenger id -> lift id
    express: Set[int] = field(default_factory=set)           # passenger ids to rush


class Dispatcher:
    """A dispatcher decides which lift serves which waiting passenger.

    unassigned   : passengers with no lift yet (must be handled)
    reassignable : passengers already assigned but not yet boarded; dispatchers
                   with `reoptimize = True` may move them to a better lift.
    """
    name = "base"
    reoptimize = False          # may it change existing assignments?
    priority_boarding = False   # board higher-urgency passengers first when space is short

    def assign(self, unassigned, reassignable, lifts, now, config) -> Assignment:
        raise NotImplementedError
