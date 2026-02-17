#!/usr/bin/env python3
"""
Build a small fatigue cache table for fast /predictions/today/enhanced queries.

Computes (per player_id, as_of_date):
  - last_match_date
  - rest_days
  - went_distance
  - matches_10d
  - sets_10d

Usage (from api/):
  python -m app.ingest.tennis.build_fatigue_cache --days-ahead 0
  python -m app.ingest.tennis.build_fatigue_cache --start-date 2026-02-01 --end-date 2026-02-03
"""

from __future__ import annotations

import argparse
import asyncio
import datetime as dt
import os

from dotenv import load_dotenv
from sqlalchemy import text
from sqlalchemy.ext.asyncio import create_async_engine

load_dotenv()


SQL_CREATE = """
CREATE TABLE IF NOT EXISTS tennis_player_fatigue_cache (
  player_id INT NOT NULL,
  as_of_date DATE NOT NULL,
  last_match_date DATE,
  rest_days INT,
  went_distance BOOLEAN,
  matches_10d INT,
  sets_10d INT,
  updated_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  PRIMARY KEY (player_id, as_of_date)
);
"""

SQL_DELETE_RANGE = """
DELETE FROM tennis_player_fatigue_cache
WHERE as_of_date BETWEEN :start_date AND :end_date;
"""

SQL_INSERT_RANGE = """
WITH target_matches AS (
  SELECT DISTINCT
    match_date,
    p1_canonical_id AS p1_id,
    p2_canonical_id AS p2_id
  FROM tennis_matches
  WHERE match_date BETWEEN :start_date AND :end_date
    AND upper(tour) IN ('ATP','WTA')
),
targets AS (
  SELECT match_date AS as_of_date, p1_id AS player_id
  FROM target_matches
  WHERE p1_id IS NOT NULL
  UNION
  SELECT match_date AS as_of_date, p2_id AS player_id
  FROM target_matches
  WHERE p2_id IS NOT NULL
),
last_match AS (
  SELECT
    t.player_id,
    t.as_of_date,
    pm.match_date AS last_match_date,
    (t.as_of_date - pm.match_date)::int AS rest_days,
    (sp.sets_played = bo.best_of_pm) AS went_distance
  FROM targets t
  LEFT JOIN LATERAL (
    SELECT pm.match_date, pm.score_raw, pm.tour, pm.tournament, pm."round"
    FROM tennis_matches pm
    WHERE (pm.p1_canonical_id = t.player_id OR pm.p2_canonical_id = t.player_id)
      AND pm.match_date <= t.as_of_date
      AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
      AND COALESCE(pm.score_raw,'') <> ''
    ORDER BY pm.match_date DESC
    LIMIT 1
  ) pm ON TRUE
  LEFT JOIN LATERAL (
    SELECT
      CASE
        WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
          split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
          split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
        ELSE
          (SELECT count(*)
           FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
           WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
          )
      END AS sets_played
  ) sp ON TRUE
  LEFT JOIN LATERAL (
    SELECT
      CASE
        WHEN upper(pm.tour) = 'ATP'
         AND (
              pm.tournament ILIKE '%Australian Open%'
           OR pm.tournament ILIKE '%Roland Garros%'
           OR pm.tournament ILIKE '%French Open%'
           OR pm.tournament ILIKE '%Wimbledon%'
           OR pm.tournament ILIKE '%US Open%'
         )
         AND COALESCE(pm."round",'') !~* 'qual'
        THEN 5 ELSE 3
      END AS best_of_pm
  ) bo ON TRUE
),
recent_10d AS (
  SELECT
    t.player_id,
    t.as_of_date,
    count(*)::int AS matches_10d,
    COALESCE(sum(sp.sets_played), 0)::int AS sets_10d
  FROM targets t
  JOIN tennis_matches pm
    ON (pm.p1_canonical_id = t.player_id OR pm.p2_canonical_id = t.player_id)
   AND pm.match_date >= (t.as_of_date - interval '10 days')
   AND pm.match_date <= t.as_of_date
   AND COALESCE(lower(pm.status),'') IN ('finished','completed','ended')
   AND COALESCE(pm.score_raw,'') <> ''
  LEFT JOIN LATERAL (
    SELECT
      CASE
        WHEN COALESCE(pm.score_raw,'') ~ '^\s*\d+\s*-\s*\d+\s*$' THEN
          split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 1)::int +
          split_part(regexp_replace(pm.score_raw,'\s','','g'), '-', 2)::int
        ELSE
          (SELECT count(*)
           FROM unnest(regexp_split_to_array(COALESCE(pm.score_raw,''), '\s+')) t(tok)
           WHERE regexp_replace(tok, '\(.*\)', '', 'g') ~ '^\d+\-\d+$'
          )
      END AS sets_played
  ) sp ON TRUE
  GROUP BY 1,2
)
INSERT INTO tennis_player_fatigue_cache (
  player_id, as_of_date, last_match_date, rest_days, went_distance, matches_10d, sets_10d
)
SELECT
  t.player_id,
  t.as_of_date,
  lm.last_match_date,
  lm.rest_days,
  lm.went_distance,
  r.matches_10d,
  r.sets_10d
FROM targets t
LEFT JOIN last_match lm
  ON lm.player_id = t.player_id AND lm.as_of_date = t.as_of_date
LEFT JOIN recent_10d r
  ON r.player_id = t.player_id AND r.as_of_date = t.as_of_date
ON CONFLICT (player_id, as_of_date)
DO UPDATE SET
  last_match_date = EXCLUDED.last_match_date,
  rest_days = EXCLUDED.rest_days,
  went_distance = EXCLUDED.went_distance,
  matches_10d = EXCLUDED.matches_10d,
  sets_10d = EXCLUDED.sets_10d,
  updated_at = now();
"""


async def main_async() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start-date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--end-date", default=None, help="YYYY-MM-DD")
    ap.add_argument("--days-ahead", type=int, default=0, help="Build for today..today+N")
    args = ap.parse_args()

    today = dt.date.today()
    if args.start_date and args.end_date:
        start_date = dt.date.fromisoformat(args.start_date)
        end_date = dt.date.fromisoformat(args.end_date)
    else:
        start_date = today
        end_date = today + dt.timedelta(days=int(args.days_ahead))

    db_url = os.getenv("DATABASE_URL_ASYNC") or os.getenv("DATABASE_URL")
    if not db_url:
        raise RuntimeError("Set DATABASE_URL_ASYNC or DATABASE_URL in your .env")

    engine = create_async_engine(db_url, echo=False, future=True, connect_args={"ssl": "require"})

    async with engine.begin() as conn:
        await conn.execute(text(SQL_CREATE))
        await conn.execute(text(SQL_DELETE_RANGE), {"start_date": start_date, "end_date": end_date})
        await conn.execute(text(SQL_INSERT_RANGE), {"start_date": start_date, "end_date": end_date})

    await engine.dispose()


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
