from __future__ import annotations

from datetime import date
from typing import Any, Dict

from fastapi import APIRouter, Query

from app.schemas.tennis_admin import EloHealthResponse
from app.services.tennis.admin_elo_service import (
    fix_fixtures_mapping as fix_fixtures_mapping_service,
    get_elo_health,
)

router = APIRouter(prefix="/tennis/admin", tags=["Tennis Admin"])


@router.get("/elo-health", response_model=EloHealthResponse)
async def elo_health(match_date: date = Query(...)) -> EloHealthResponse:
    return await get_elo_health(match_date=match_date)


@router.post("/fix-fixtures-mapping")
async def fix_fixtures_mapping(match_date: date = Query(...)) -> Dict[str, Any]:
    return await fix_fixtures_mapping_service(match_date=match_date)
