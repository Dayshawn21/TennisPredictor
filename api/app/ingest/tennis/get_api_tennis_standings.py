from __future__ import annotations

import os
import json
import asyncio
from typing import Any, Dict, List, Optional

import pandas as pd
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

# Reuse your proven SSL/url normalizer
from app.ingest.tennis.sofascore_stats_finalize import normalize_asyncpg_url_and_ssl


# Point to your combined file (or run twice with different paths)
CSV_PATH = r"C:\Users\dayshawn\Desktop\SportApp\api\standings_atp.csv"
# If you have a combined file, keep one path.
# If you have two separate files, call this module twice with different CSV_PATH or use env var override.


CREATE_TABLE_SQL = text("""
CREATE TABLE IF NOT EXISTS api_tennis_players (
  player_key   bigint PRIMARY KEY,
  league       text NULL,
  display_name text NOT NULL,
  country      text NULL,
  points       int NULL,
  place        int NULL,
  raw_row      jsonb NULL,
  created_at   timestamptz DEFAULT now(),
  updated_at   timestamptz DEFAULT now()
);
""")

# Helpful indexes (optional, but nice)
ENSURE_INDEXES_SQL = [
    text("CREATE INDEX IF NOT EXISTS idx_api_tennis_players_league ON api_tennis_players(league);"),
    text("CREATE INDEX IF NOT EXISTS idx_api_tennis_players_display_name ON api_tennis_players(display_name);"),
]

# Upsert uses CAST(:raw_payload AS jsonb) so we can pass a JSON string safely
UPSERT_SQL = text("""
INSERT INTO api_tennis_players (
  player_key, league, display_name, country, points, place, raw_row, updated_at
)
VALUES (
  :player_key,
  :league,
  :display_name,
  :country,
  :points,
  :place,
  CAST(:raw_payload AS jsonb),
  now()
)
ON CONFLICT (player_key) DO UPDATE
SET
  league       = EXCLUDED.league,
  display_name = EXCLUDED.display_name,
  country      = EXCLUDED.country,
  points       = EXCLUDED.points,
  place        = EXCLUDED.place,
  raw_row      = EXCLUDED.raw_row,
  updated_at   = now();
""")


def _clean_str(x: Any) -> Optional[str]:
    if x is None:
        return None
    s = str(x).strip()
    return s if s != "" and s.lower() != "nan" else None


def _to_int(x: Any) -> Optional[int]:
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        if isinstance(x, str) and x.strip() == "":
            return None
        return int(float(x))
    except Exception:
        return None


def _to_bigint_player_key(x: Any) -> Optional[int]:
    """
    CSV can contain "2382.0" or 2382.0; normalize to int.
    """
    try:
        if x is None or (isinstance(x, float) and pd.isna(x)):
            return None
        return int(float(x))
    except Exception:
        return None


def _csv_path() -> str:
    # Allow override: set CSV_PATH in env without editing file
    return os.getenv("API_TENNIS_PLAYERS_CSV", CSV_PATH)


def _chunk(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    return [items[i:i + size] for i in range(0, len(items), size)]


async def main() -> None:
    db_url = os.getenv("DATABASE_URL_ASYNC") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Missing DATABASE_URL or DATABASE_URL_ASYNC")

    normalized_url, connect_args = normalize_asyncpg_url_and_ssl(db_url)

    path = _csv_path()
    df = pd.read_csv(path)

    print("CSV:", path)
    print("CSV columns:", list(df.columns))
    print("Rows:", len(df))

    # Expected columns from your output:
    # ['place','player','player_key','league','movement','country','points']

    engine = create_async_engine(
        normalized_url,
        future=True,
        echo=False,
        connect_args=connect_args,
        pool_pre_ping=True,
    )

    # Build parameter rows
    params: List[Dict[str, Any]] = []
    for _, r in df.iterrows():
        pk = _to_bigint_player_key(r.get("player_key"))
        name = _clean_str(r.get("player"))  # CSV uses "player"
        if pk is None or not name:
            continue

        row_dict = r.to_dict()
        params.append(
            {
                "player_key": pk,
                "league": _clean_str(r.get("league")),
                "display_name": name,
                "country": _clean_str(r.get("country")),
                "points": _to_int(r.get("points")),
                "place": _to_int(r.get("place")),
                "raw_payload": json.dumps(row_dict, default=str),
            }
        )

    async with engine.begin() as conn:
        await conn.execute(CREATE_TABLE_SQL)
        for stmt in ENSURE_INDEXES_SQL:
            await conn.execute(stmt)

        # executemany in chunks
        chunk_size = int(os.getenv("API_TENNIS_UPSERT_CHUNK", "500"))
        for part in _chunk(params, chunk_size):
            await conn.execute(UPSERT_SQL, part)

    await engine.dispose()
    print(f"✅ Loaded api_tennis_players: inserted/updated={len(params)}")


if __name__ == "__main__":
    asyncio.run(main())
