from __future__ import annotations

from datetime import date
from typing import Optional

from fastapi import APIRouter, Depends
from sqlalchemy.ext.asyncio import AsyncSession

from app.db_session import get_db
from app.schemas.tennis_auto import (
    TennisPlayerStatsResponse,
    TennisPredictAutoRequest,
    TennisPredictAutoResponse,
)
from app.services.tennis.auto_prediction_service import (
    get_player_stats as get_player_stats_service,
    predict_auto as predict_auto_service,
)

router = APIRouter(tags=["Tennis"])


@router.post("/predict-auto", response_model=TennisPredictAutoResponse)
async def predict_auto(req: TennisPredictAutoRequest, db: AsyncSession = Depends(get_db)):
    return await predict_auto_service(req=req, db=db)


@router.get("/player/stats", response_model=TennisPlayerStatsResponse)
async def get_player_stats(
    player: str,
    surface: Optional[str] = None,
    as_of: Optional[date] = None,
    db: AsyncSession = Depends(get_db),
):
    return await get_player_stats_service(
        player=player,
        surface=surface,
        as_of=as_of,
        db=db,
    )
