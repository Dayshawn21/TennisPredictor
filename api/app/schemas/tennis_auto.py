from __future__ import annotations

from datetime import date
from typing import Dict, Optional

from pydantic import BaseModel, Field


class TennisPredictAutoRequest(BaseModel):
    player1: str
    player2: str
    surface: str = Field(..., description="e.g. hard, clay, grass, indoor_hard (treated as hard)")
    match_date: Optional[date] = Field(
        None,
        description="If omitted, uses today's date. Snapshot uses latest <= match_date.",
    )
    tournament: Optional[str] = None
    match_id: Optional[str] = None


class TennisPredictAutoResponse(BaseModel):
    player1: str
    player2: str
    surface: str
    match_date: date

    winner_side: int = Field(..., description="1 => player1, 2 => player2")
    pick: str = Field(..., description="player1 or player2")

    p_player1_model: float
    p_player2_model: float
    p_player1: float
    p_player2: float

    confidence_tier: str
    edge_pct: float

    h2h_p1_wins: int
    h2h_p2_wins: int
    h2h_surface_matches: int
    h2h_applied: bool
    h2h_adjustment: float

    model: str
    features: Dict[str, float]


class TennisPlayerStatsResponse(BaseModel):
    player: str
    as_of: date
    snapshots: Dict[str, Dict[str, Optional[float]]]
