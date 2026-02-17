# FILE: api/app/routers/tennis_value_bets.py
"""
Detect value betting opportunities where model disagrees with market.
"""

from __future__ import annotations

from datetime import date, timedelta
from typing import List

from fastapi import APIRouter, Query
from pydantic import BaseModel

from app.db_session import engine
from app.routers.tennis_predictions_today_enhanced import (
    predictions_today,
    EloPrediction
)

router = APIRouter(tags=["Tennis Value Bets"])


class ValueBet(BaseModel):
    match: EloPrediction
    value_side: str  # "p1" or "p2"
    model_prob: float
    market_prob: float
    edge_percent: float
    kelly_fraction: float
    confidence: str  # "low", "medium", "high"


class ValueBetsResponse(BaseModel):
    as_of: date
    count: int
    total_opportunities: int
    value_bets: List[ValueBet]


def calculate_kelly_criterion(prob: float, odds_american: int) -> float:
    """
    Calculate Kelly Criterion bet sizing.
    
    Args:
        prob: Your model's probability (0-1)
        odds_american: Market odds in American format
    
    Returns:
        Kelly fraction (0-1), capped at 0.25 for safety
    """
    # Convert American odds to decimal
    if odds_american > 0:
        decimal_odds = (odds_american / 100) + 1
    else:
        decimal_odds = (100 / abs(odds_american)) + 1
    
    # Kelly formula: f = (bp - q) / b
    # where b = decimal odds - 1, p = win probability, q = 1 - p
    b = decimal_odds - 1
    q = 1 - prob
    
    kelly = (b * prob - q) / b
    
    # Cap at 25% for safety (fractional Kelly)
    return max(0, min(kelly, 0.25))


@router.get("/tennis/value-bets/today", response_model=ValueBetsResponse)
async def value_bets_today(
    min_edge: float = Query(0.05, description="Minimum edge (5% = 0.05)"),
    min_confidence: float = Query(0.55, description="Minimum model confidence"),
    days_ahead: int = Query(0, ge=0, le=7),
):
    """
    Find value betting opportunities where model significantly disagrees with market.
    
    Criteria for value bet:
    - Model probability differs from market by at least min_edge
    - Model is confident (prob > min_confidence or prob < 1 - min_confidence)
    - Market odds are available
    """
    
    # Get all predictions
    predictions = await predictions_today(
        days_ahead=days_ahead,
        include_incomplete=True,
        bust_cache=True
    )
    
    value_bets = []
    total_with_odds = 0
    
    for pred in predictions["items"]:  # FIX: predictions is a dict with "items" key
        # Skip if no market odds
        individual = pred.inputs.get('individual_predictions', {})
        market_prob = individual.get('market')
        
        if market_prob is None:
            continue
        
        total_with_odds += 1
        
        model_prob = pred.p1_win_prob
        if model_prob is None:
            continue
        
        # Calculate edge for both sides
        p1_edge = model_prob - market_prob
        p2_edge = (1 - model_prob) - (1 - market_prob)
        
        # Check P1 value
        if (p1_edge >= min_edge and 
            model_prob >= min_confidence and
            pred.inputs.get('has_odds')):
            
            # Get P1 odds from match data
            async with engine.begin() as conn:
                from sqlalchemy import text
                odds_row = await conn.execute(
                    text("SELECT p1_odds_american FROM tennis_matches WHERE match_id = :mid"),
                    {"mid": pred.match_id}
                )
                odds_result = odds_row.first()
                p1_odds = odds_result[0] if odds_result else None
            
            if p1_odds:
                kelly = calculate_kelly_criterion(model_prob, p1_odds)
                
                # Determine confidence level
                if p1_edge >= 0.15:
                    confidence = "high"
                elif p1_edge >= 0.10:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                value_bets.append(ValueBet(
                    match=pred,
                    value_side="p1",
                    model_prob=model_prob,
                    market_prob=market_prob,
                    edge_percent=p1_edge * 100,
                    kelly_fraction=kelly,
                    confidence=confidence
                ))
        
        # Check P2 value
        if (p2_edge >= min_edge and 
            (1 - model_prob) >= min_confidence and
            pred.inputs.get('has_odds')):
            
            async with engine.begin() as conn:
                from sqlalchemy import text
                odds_row = await conn.execute(
                    text("SELECT p2_odds_american FROM tennis_matches WHERE match_id = :mid"),
                    {"mid": pred.match_id}
                )
                odds_result = odds_row.first()
                p2_odds = odds_result[0] if odds_result else None
            
            if p2_odds:
                kelly = calculate_kelly_criterion(1 - model_prob, p2_odds)
                
                if p2_edge >= 0.15:
                    confidence = "high"
                elif p2_edge >= 0.10:
                    confidence = "medium"
                else:
                    confidence = "low"
                
                value_bets.append(ValueBet(
                    match=pred,
                    value_side="p2",
                    model_prob=1 - model_prob,
                    market_prob=1 - market_prob,
                    edge_percent=p2_edge * 100,
                    kelly_fraction=kelly,
                    confidence=confidence
                ))
    
    # Sort by edge
    value_bets.sort(key=lambda x: x.edge_percent, reverse=True)
    
    return ValueBetsResponse(
        as_of=date.today(),
        count=len(value_bets),
        total_opportunities=total_with_odds,
        value_bets=value_bets
    )