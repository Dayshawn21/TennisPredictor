# app/schemas/tennis_predictions.py
from __future__ import annotations

from datetime import date
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class H2HMatch(BaseModel):
    date: Optional[str] = None
    tournament: Optional[str] = None
    round: Optional[str] = None
    surface: Optional[str] = None
    home: Optional[str] = None
    away: Optional[str] = None
    score: Optional[str] = None
    winner_code: Optional[int] = None


class EloPrediction(BaseModel):
    match_id: Optional[Any] = None
    match_key: Optional[str] = None
    match_date: Optional[date] = None
    match_start_utc: Optional[str] = None

    tour: Optional[str] = None
    tournament: Optional[str] = None
    round: Optional[str] = None
    surface: Optional[str] = None
    best_of: Optional[int] = None

    p1_name: Optional[str] = None
    p2_name: Optional[str] = None

    # ✅ you are using these in the router
    p1_player_id: Optional[int] = None
    p2_player_id: Optional[int] = None

    p1_ta_id: Optional[int] = None
    p2_ta_id: Optional[int] = None

    # Model winner probs / fair odds
    p1_win_prob: Optional[float] = None
    p2_win_prob: Optional[float] = None
    p1_fair_american: Optional[int] = None
    p2_fair_american: Optional[int] = None

    # Market odds (SofaScore)
    p1_market_odds_american: Optional[int] = None
    p2_market_odds_american: Optional[int] = None
    odds_fetched_at: Optional[str] = None

    # Market probabilities
    p1_market_implied_prob: Optional[float] = None
    p2_market_implied_prob: Optional[float] = None
    p1_market_no_vig_prob: Optional[float] = None
    p2_market_no_vig_prob: Optional[float] = None

    # H2H
    h2h_p1_wins: Optional[int] = None
    h2h_p2_wins: Optional[int] = None
    h2h_total_matches: Optional[int] = None
    h2h_surface_p1_wins: Optional[int] = None
    h2h_surface_p2_wins: Optional[int] = None
    h2h_surface_matches: Optional[int] = None
    h2h_matches: Optional[List[H2HMatch]] = None

    # Blended
    p1_blended_win_prob: Optional[float] = None
    p2_blended_win_prob: Optional[float] = None
    predicted_winner_blended: Optional[str] = None  # "p1" or "p2"
    edge_no_vig: Optional[float] = None
    edge_blended: Optional[float] = None
    bet_eligible: Optional[bool] = None
    bet_side: Optional[str] = None
    kelly_fraction_capped: Optional[float] = None
    value_bet_summary: Optional[str] = None

    # Missing data info
    missing_reason: Optional[str] = None

    # Debug inputs
    inputs: Dict[str, Any] = Field(default_factory=dict)

    # Projections
    predicted_winner: Optional[str] = None
    projected_total_games: Optional[float] = None
    projected_spread_p1: Optional[float] = None
    projected_sets: Optional[str] = None
    pAserve: Optional[float] = None
    pBserve: Optional[float] = None


class EloPredictionsResponse(BaseModel):
    as_of: date
    source: str
    cached: bool
    count: int
    items: List[EloPrediction]
