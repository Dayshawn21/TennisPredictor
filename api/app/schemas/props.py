from __future__ import annotations

from datetime import datetime
from typing import List, Tuple

from pydantic import BaseModel


class PropReq(BaseModel):
    prop_id: str
    game_id: int | None = None
    player_id: int
    player: str | None = None
    position: str | None = None
    team_id: int | None = None
    opp_team_id: int | None = None
    date_utc: datetime | None = None
    line: float | None = None
    american_odds: int | None = None
    market: str | None = None


class Rec(BaseModel):
    """Response schema for a single recommendation."""
    prop_id: str
    label: str
    predicted_value: float
    mode_exact: int
    pi68: Tuple[int, int]
    pi90: Tuple[int, int]
    p_over_line: float
    ev_per_dollar: float
    kelly: float
    confidence: str | int
    pmf_window: List[Tuple[int, float]] | dict[str, float]
    american_odds: int
    line: float
    reasons: List[str]