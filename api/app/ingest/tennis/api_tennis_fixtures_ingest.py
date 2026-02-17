# api/app/ingest/tennis/api_tennis_fixtures_ingest.py
from __future__ import annotations

import json
import os
import logging
from datetime import date, datetime
from typing import Any, Dict, List, Optional

from sqlalchemy import text

from app.db_session import engine
from app.ingest.tennis.api_tennis_schema import ENSURE_API_TENNIS_TABLES
from app.services.tennis.api_tennis_client import ApiTennisClient
from app.services.tennis.api_tennis_event_types import get_atp_wta_singles_keys

logger = logging.getLogger(__name__)


def parse_date(value: Any) -> Optional[date]:
    if value is None or value == "":
        return None
    if isinstance(value, date) and not isinstance(value, datetime):
        return value
    if isinstance(value, datetime):
        return value.date()
    if isinstance(value, str):
        try:
            return datetime.strptime(value.strip(), "%Y-%m-%d").date()
        except ValueError:
            return None
    return None


def _pick(row: Dict[str, Any], *keys: str) -> Optional[Any]:
    for k in keys:
        v = row.get(k)
        if v is not None and str(v).strip() != "":
            return v
    return None


def _to_bigint(value: Any) -> Optional[int]:
    if value is None:
        return None
    try:
        s = str(value).strip()
        if s == "":
            return None
        if "." in s:
            s = s.split(".", 1)[0]
        return int(s)
    except Exception:
        return None


def _build_score_raw(row: Dict[str, Any]) -> Optional[str]:
    for k in ("event_final_result", "event_score", "final_result", "score"):
        v = row.get(k)
        if v:
            return str(v).strip()
    return None


def _infer_status(row: Dict[str, Any]) -> Optional[str]:
    for k in ("event_status", "status", "event_live"):
        v = row.get(k)
        if v is None:
            continue
        return str(v).strip()
    return None


def _infer_winner(row: Dict[str, Any]) -> Optional[str]:
    v = row.get("event_winner")
    if v:
        return str(v).strip()
    v2 = row.get("winner_name")
    if v2:
        return str(v2).strip()
    return None


async def ensure_tables() -> None:
    async with engine.begin() as conn:
        for stmt in ENSURE_API_TENNIS_TABLES:
            await conn.execute(text(stmt))


UPSERT_FIXTURE = text("""
INSERT INTO api_tennis_fixtures (
  event_key, match_key,
  tour, match_date, match_time, timezone,
  tournament_name, tournament_round, surface,
  player1_name, player2_name,
  player1_api_id, player2_api_id,
  player1_id, player2_id,
  status, score_raw, winner_name,
  raw_payload,
  updated_at
)
VALUES (
  :event_key, :match_key,
  :tour, :match_date, :match_time, :timezone,
  :tournament_name, :tournament_round, :surface,
  :player1_name, :player2_name,

  -- FORCE TYPES so asyncpg/Postgres can prepare the statement
  CAST(:player1_api_id AS bigint),
  CAST(:player2_api_id AS bigint),

  -- resolve canonical ids from tennis_player_sources(source='api_tennis')
  (SELECT s.player_id
   FROM tennis_player_sources s
   WHERE s.source = 'api_tennis'
     AND :player1_api_id IS NOT NULL
     AND s.source_player_id ~ '^[0-9]+$'
     AND s.source_player_id::bigint = CAST(:player1_api_id AS bigint)
   LIMIT 1),

  (SELECT s.player_id
   FROM tennis_player_sources s
   WHERE s.source = 'api_tennis'
     AND :player2_api_id IS NOT NULL
     AND s.source_player_id ~ '^[0-9]+$'
     AND s.source_player_id::bigint = CAST(:player2_api_id AS bigint)
   LIMIT 1),

  :status, :score_raw, :winner_name,
  CAST(:raw_payload AS jsonb),
  now()
)
ON CONFLICT (event_key)
DO UPDATE SET
  match_key        = EXCLUDED.match_key,
  tour             = EXCLUDED.tour,
  match_date       = EXCLUDED.match_date,
  match_time       = EXCLUDED.match_time,
  timezone         = EXCLUDED.timezone,
  tournament_name  = EXCLUDED.tournament_name,
  tournament_round = EXCLUDED.tournament_round,
  surface          = EXCLUDED.surface,
  player1_name     = EXCLUDED.player1_name,
  player2_name     = EXCLUDED.player2_name,

  -- always update API keys
  player1_api_id   = EXCLUDED.player1_api_id,
  player2_api_id   = EXCLUDED.player2_api_id,

  -- preserve canonical ids; also repair polluted rows where player_id == api_id
  player1_id = COALESCE(
      NULLIF(api_tennis_fixtures.player1_id, api_tennis_fixtures.player1_api_id),
      EXCLUDED.player1_id
  ),
  player2_id = COALESCE(
      NULLIF(api_tennis_fixtures.player2_id, api_tennis_fixtures.player2_api_id),
      EXCLUDED.player2_id
  ),

  status           = EXCLUDED.status,
  score_raw        = EXCLUDED.score_raw,
  winner_name      = EXCLUDED.winner_name,
  raw_payload      = EXCLUDED.raw_payload,
  updated_at       = now();
""")



async def ingest_api_tennis_fixtures(date_start: date, date_stop: date) -> Dict[str, int]:
    logger.info("api_tennis_fixtures_ingest.start start=%s end=%s", date_start.isoformat(), date_stop.isoformat())
    await ensure_tables()

    client = ApiTennisClient()
    tz = os.getenv("API_TENNIS_TZ", "America/Chicago")
    keys = await get_atp_wta_singles_keys(client)

    counts = {"ATP": 0, "WTA": 0, "TOTAL": 0}

    async with engine.begin() as conn:
        for tour in ("ATP", "WTA"):
            event_type_key = keys[tour]

            payload = await client.get_fixtures(
                date_start=date_start.isoformat(),
                date_stop=date_stop.isoformat(),
                event_type_key=event_type_key,
                timezone=tz,
            )

            rows: List[Dict[str, Any]] = payload.get("result", []) or []
            if not rows:
                continue

            batch_params: List[Dict[str, Any]] = []

            for r in rows:
                event_key = _pick(r, "event_key", "match_key", "event_id")
                if not event_key:
                    continue

                event_key_s = str(event_key).strip()
                match_key = f"api_tennis:{event_key_s}"

                match_date = parse_date(_pick(r, "event_date", "match_date", "date"))
                match_time = _pick(r, "event_time", "time")
                tournament = _pick(r, "tournament_name", "tournament")
                rnd = _pick(r, "tournament_round", "round")
                surface = _pick(r, "surface", "event_surface")

                p1 = _pick(r, "event_first_player", "player1", "home_player", "first_player")
                p2 = _pick(r, "event_second_player", "player2", "away_player", "second_player")
                if not p1 or not p2:
                    continue

                # IMPORTANT: keep these as Python ints (or None)
                p1_api_id = _to_bigint(_pick(r, "first_player_key"))
                p2_api_id = _to_bigint(_pick(r, "second_player_key"))

                status = _infer_status(r)
                score_raw = _build_score_raw(r)
                winner_name = _infer_winner(r)

                batch_params.append(
                    {
                        "event_key": event_key_s,
                        "match_key": match_key,
                        "tour": tour,
                        "match_date": match_date,
                        "match_time": match_time,
                        "timezone": tz,
                        "tournament_name": tournament,
                        "tournament_round": rnd,
                        "surface": surface,
                        "player1_name": p1,
                        "player2_name": p2,
                        "player1_api_id": p1_api_id,  # ✅ int/None
                        "player2_api_id": p2_api_id,  # ✅ int/None
                        "status": status,
                        "score_raw": score_raw,
                        "winner_name": winner_name,
                        "raw_payload": json.dumps(r, default=str),
                    }
                )

                counts[tour] += 1
                counts["TOTAL"] += 1

            if batch_params:
                await conn.execute(UPSERT_FIXTURE, batch_params)

    logger.info(
        "api_tennis_fixtures_ingest.done start=%s end=%s total=%s",
        date_start.isoformat(),
        date_stop.isoformat(),
        counts["TOTAL"],
    )
    return counts


if __name__ == "__main__":
    import asyncio
    import logging
    from datetime import timedelta

    logging.basicConfig(level=logging.INFO)

    start_s = os.getenv("START_DATE")
    end_s = os.getenv("END_DATE")

    if start_s and end_s:
        start = date.fromisoformat(start_s)
        end = date.fromisoformat(end_s)
    else:
        start = date.today()
        end = start + timedelta(days=7)

    print(f"▶ ingest_api_tennis_fixtures: {start} -> {end}")
    counts = asyncio.run(ingest_api_tennis_fixtures(start, end))
    print("✅ done:", counts)
