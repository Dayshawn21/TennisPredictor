from __future__ import annotations

from typing import Literal, Optional

from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.services.tennis.props_service import predict_prop_payload

router = APIRouter(tags=["Tennis Props"])

PropType = Literal["aces", "break_points_won"]


class TennisPropRequest(BaseModel):
    player: Literal["player1", "player2"] = "player1"
    prop_type: PropType
    line: float = Field(..., description="Sportsbook line, e.g. 9.5")

    expected_games: float = Field(..., description="Expected total games in match, e.g. 22.0")
    expected_return_games: Optional[float] = Field(
        None,
        description="Expected return games for the player (often ~ expected_games/2)",
    )

    p_last10_aces_pg: Optional[float] = None
    p_surf_last10_aces_pg: Optional[float] = None

    opp_last10_aces_allowed_pg: Optional[float] = None
    opp_surf_last10_aces_allowed_pg: Optional[float] = None

    p_last10_bp_won_prg: Optional[float] = None
    p_surf_last10_bp_won_prg: Optional[float] = None

    opp_last10_bp_won_allowed_psg: Optional[float] = None
    opp_surf_last10_bp_won_allowed_psg: Optional[float] = None


class TennisPropResponse(BaseModel):
    player: str
    prop_type: str
    line: float
    expected: float
    p_over: float
    p_under: float
    confidence_tier: str


@router.post("/props", response_model=TennisPropResponse)
def predict_prop(req: TennisPropRequest) -> TennisPropResponse:
    payload = predict_prop_payload(req.model_dump())
    return TennisPropResponse(**payload)
