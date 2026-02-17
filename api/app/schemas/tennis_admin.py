from __future__ import annotations

from datetime import date
from typing import Dict, List, Optional

from pydantic import BaseModel


class EloHealthRow(BaseModel):
    match_date: date
    tour: str
    tournament_name: Optional[str] = None
    tournament_round: Optional[str] = None
    player1_name: Optional[str] = None
    player1_id: Optional[int] = None
    player2_name: Optional[str] = None
    player2_id: Optional[int] = None
    status: Optional[str] = None
    p1_elo: Optional[float] = None
    p2_elo: Optional[float] = None
    p1_elo_source: Optional[str] = None
    p2_elo_source: Optional[str] = None
    elo_status: str


class EloHealthResponse(BaseModel):
    as_of: date
    counts: Dict[str, int]
    rows: List[EloHealthRow]
