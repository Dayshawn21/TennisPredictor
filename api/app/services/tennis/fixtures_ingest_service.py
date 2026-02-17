from __future__ import annotations

import logging
from datetime import date, timedelta
from typing import Any, Dict

from app.ingest.tennis.api_tennis_fixtures_ingest import ingest_api_tennis_fixtures

logger = logging.getLogger(__name__)


async def ingest_fixtures(start: date, end: date) -> Dict[str, Any]:
    logger.info("tennis_api_tennis.ingest_fixtures start start=%s end=%s", start.isoformat(), end.isoformat())
    counts = await ingest_api_tennis_fixtures(start, end)
    payload = {"start": start.isoformat(), "end": end.isoformat(), "counts": counts}
    logger.info("tennis_api_tennis.ingest_fixtures done total=%s", counts.get("TOTAL"))
    return payload


async def ingest_fixtures_daily(forward_days: int, backfill_days: int) -> Dict[str, Any]:
    today = date.today()

    upcoming_start = today
    upcoming_end = today + timedelta(days=forward_days)
    backfill_end = today - timedelta(days=1)
    backfill_start = today - timedelta(days=backfill_days)

    logger.info(
        "tennis_api_tennis.ingest_fixtures_daily start today=%s forward_days=%s backfill_days=%s",
        today.isoformat(),
        forward_days,
        backfill_days,
    )
    upcoming = await ingest_api_tennis_fixtures(upcoming_start, upcoming_end)
    backfill = await ingest_api_tennis_fixtures(backfill_start, backfill_end)

    combined = {
        "ATP": upcoming["ATP"] + backfill["ATP"],
        "WTA": upcoming["WTA"] + backfill["WTA"],
        "TOTAL": upcoming["TOTAL"] + backfill["TOTAL"],
    }

    payload = {
        "today": today.isoformat(),
        "upcoming": {
            "start": upcoming_start.isoformat(),
            "end": upcoming_end.isoformat(),
            "counts": upcoming,
        },
        "backfill": {
            "start": backfill_start.isoformat(),
            "end": backfill_end.isoformat(),
            "counts": backfill,
        },
        "combined_counts": combined,
    }
    logger.info(
        "tennis_api_tennis.ingest_fixtures_daily done upcoming_total=%s backfill_total=%s combined_total=%s",
        upcoming.get("TOTAL"),
        backfill.get("TOTAL"),
        combined.get("TOTAL"),
    )
    return payload
