from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

MatchType = Literal["Singles", "Doubles"]
EventType = Literal["point", "good_shot", "bad_shot", "service_fault"]


@dataclass
class Player:
    player_id: str
    name: str
    created_at: str
    is_active: bool = True
