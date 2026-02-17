from __future__ import annotations

import re
from datetime import date
from typing import Any, Dict, Optional, Tuple

from sqlalchemy import text
from sqlalchemy.exc import DBAPIError
from sqlalchemy.ext.asyncio import AsyncSession

_SNAPSHOT_KEYS = [
    "last5_hold",
    "last5_break",
    "last10_hold",
    "last10_break",
    "surf_last10_hold",
    "surf_last10_break",
    "last10_aces_pg",
    "surf_last10_aces_pg",
    "last10_df_pg",
    "surf_last10_df_pg",
    "last10_tb_match_rate",
    "last10_tb_win_pct",
    "surf_last10_tb_match_rate",
    "surf_last10_tb_win_pct",
]


def _normalize_name(name: str) -> str:
    lowered = name.lower()
    lowered = re.sub(r"[^\w\s]", "", lowered)
    lowered = re.sub(r"\s+", " ", lowered).strip()
    return lowered


async def find_player_id(db: AsyncSession, name: str) -> Optional[int]:
    r = await db.execute(text("SELECT id FROM tennis_players WHERE name = :name LIMIT 1"), {"name": name})
    row = r.first()
    if row:
        return int(row[0])

    r2 = await db.execute(
        text("SELECT id FROM tennis_players WHERE lower(name) = lower(:name) LIMIT 1"),
        {"name": name},
    )
    row2 = r2.first()
    if row2:
        return int(row2[0])

    last = (_normalize_name(name).split() or [name])[-1]
    r3 = await db.execute(
        text(
            """
            SELECT id
            FROM tennis_players
            WHERE lower(name) LIKE :pat
            ORDER BY length(name) ASC
            LIMIT 1
            """
        ),
        {"pat": f"%{last}%"},
    )
    row3 = r3.first()
    if row3:
        return int(row3[0])

    return None


async def load_snapshot(db: AsyncSession, player_id: int, surface: str, as_of: date) -> Optional[Dict[str, Any]]:
    r = await db.execute(
        text(
            """
            SELECT
              last5_hold, last5_break,
              last10_hold, last10_break,
              surf_last10_hold, surf_last10_break,
              last10_aces_pg, surf_last10_aces_pg,
              last10_df_pg, surf_last10_df_pg,
              last10_tb_match_rate, last10_tb_win_pct,
              surf_last10_tb_match_rate, surf_last10_tb_win_pct
            FROM tennis_player_snapshots
            WHERE player_id = :player_id
              AND surface = :surface
              AND as_of_date <= :as_of
            ORDER BY as_of_date DESC
            LIMIT 1
            """
        ),
        {"player_id": player_id, "surface": surface, "as_of": as_of},
    )
    row = r.first()
    if not row:
        return None
    return dict(zip(_SNAPSHOT_KEYS, row))


async def load_snapshot_by_surface(db: AsyncSession, player_id: int, surface: str, as_of: date) -> Dict[str, Any]:
    snapshot = await load_snapshot(db=db, player_id=player_id, surface=surface, as_of=as_of)
    return snapshot or {}


async def get_h2h_from_db(
    db: AsyncSession,
    p1_canonical_id: int,
    p2_canonical_id: int,
    surface: str,
    as_of: date,
    lookback_years: int = 5,
) -> Tuple[int, int, int]:
    try:
        async with db.begin_nested():
            q = text(
                """
                WITH h2h AS (
                  SELECT match_date, surface, winner_canonical_id
                  FROM tennis_matches
                  WHERE match_date < :as_of
                    AND match_date >= (:as_of - make_interval(years => :lookback_years))
                    AND (
                      (p1_canonical_id = :p1 AND p2_canonical_id = :p2)
                      OR
                      (p1_canonical_id = :p2 AND p2_canonical_id = :p1)
                    )
                )
                SELECT
                  COALESCE(SUM(CASE WHEN winner_canonical_id = :p1 THEN 1 ELSE 0 END), 0) AS p1_wins,
                  COALESCE(SUM(CASE WHEN winner_canonical_id = :p2 THEN 1 ELSE 0 END), 0) AS p2_wins,
                  COALESCE(SUM(CASE WHEN surface = :surface THEN 1 ELSE 0 END), 0) AS surface_matches
                FROM h2h
                """
            )
            r = await db.execute(
                q,
                {
                    "as_of": as_of,
                    "lookback_years": lookback_years,
                    "p1": p1_canonical_id,
                    "p2": p2_canonical_id,
                    "surface": surface,
                },
            )
            row = r.first()
            if not row:
                return 0, 0, 0
            return int(row[0]), int(row[1]), int(row[2])
    except DBAPIError as e:
        msg = str(e.orig) if getattr(e, "orig", None) is not None else str(e)
        if "tennis_matches" in msg and ("does not exist" in msg or "undefined" in msg):
            return 0, 0, 0
        if "winner_canonical_id" in msg and ("does not exist" in msg or "undefined" in msg):
            return 0, 0, 0
        raise
