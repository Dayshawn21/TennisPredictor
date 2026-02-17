from __future__ import annotations

from datetime import date
from fastapi import APIRouter, Query

from app.services.tennis.fixtures_ingest_service import (
    ingest_fixtures as ingest_fixtures_service,
    ingest_fixtures_daily as ingest_fixtures_daily_service,
)

router = APIRouter(prefix="/tennis/api-tennis", tags=["Tennis (API-Tennis)"])


@router.post("/fixtures/ingest")
async def ingest_fixtures(
    start: date = Query(..., description="YYYY-MM-DD"),
    end: date = Query(..., description="YYYY-MM-DD"),
):
    return await ingest_fixtures_service(start=start, end=end)


@router.post("/fixtures/ingest-daily")
async def ingest_fixtures_daily(
    forward_days: int = Query(7, ge=0, le=30, description="How many days ahead to ingest"),
    backfill_days: int = Query(7, ge=0, le=30, description="How many days back to re-ingest (late corrections)"),
):
    return await ingest_fixtures_daily_service(
        forward_days=forward_days,
        backfill_days=backfill_days,
    )
