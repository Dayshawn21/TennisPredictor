from __future__ import annotations

import logging
from datetime import date
from typing import Any, Dict

from sqlalchemy import text

from app.db_session import engine
from app.schemas.tennis_admin import EloHealthResponse, EloHealthRow

logger = logging.getLogger(__name__)


async def get_elo_health(match_date: date) -> EloHealthResponse:
    logger.info("tennis_admin.get_elo_health start match_date=%s", match_date.isoformat())

    sql_rows = text(
        """
        SELECT
          match_date, tour, tournament_name, tournament_round,
          player1_name, player1_id,
          player2_name, player2_id,
          status,
          p1_elo, p1_elo_source,
          p2_elo, p2_elo_source,
          elo_status::text AS elo_status
        FROM v_api_fixtures_elo_unified
        WHERE match_date = :d
        ORDER BY tour, tournament_name, tournament_round;
        """
    )
    sql_counts = text(
        """
        SELECT elo_status::text AS elo_status, COUNT(*)::int AS n
        FROM v_api_fixtures_elo_unified
        WHERE match_date = :d
        GROUP BY 1
        ORDER BY n DESC;
        """
    )

    async with engine.begin() as conn:
        rows_res = await conn.execute(sql_rows, {"d": match_date})
        count_res = await conn.execute(sql_counts, {"d": match_date})
        rows = [dict(r._mapping) for r in rows_res]
        counts_list = [dict(r._mapping) for r in count_res]

    counts: Dict[str, int] = {r["elo_status"]: r["n"] for r in counts_list}
    response = EloHealthResponse(
        as_of=match_date,
        counts=counts,
        rows=[EloHealthRow(**r) for r in rows],
    )
    logger.info(
        "tennis_admin.get_elo_health done match_date=%s rows=%s statuses=%s",
        match_date.isoformat(),
        len(response.rows),
        len(response.counts),
    )
    return response


async def fix_fixtures_mapping(match_date: date) -> Dict[str, Any]:
    logger.info("tennis_admin.fix_fixtures_mapping start match_date=%s", match_date.isoformat())

    fix_sql = text(
        """
        UPDATE api_tennis_fixtures f
        SET
          player1_id = COALESCE(
            NULLIF(f.player1_id, f.player1_api_id),
            s1.player_id
          ),
          player2_id = COALESCE(
            NULLIF(f.player2_id, f.player2_api_id),
            s2.player_id
          )
        FROM tennis_player_sources s1
        JOIN tennis_player_sources s2
          ON s2.source = 'api_tennis'
         AND s2.source_player_id = f.player2_api_id::text
        WHERE f.match_date = :d
          AND s1.source = 'api_tennis'
          AND s1.source_player_id = f.player1_api_id::text
          AND (
            f.player1_id IS NULL OR f.player1_id = f.player1_api_id
            OR
            f.player2_id IS NULL OR f.player2_id = f.player2_api_id
          );
        """
    )
    count_sql = text(
        """
        SELECT
          COUNT(*) AS fixtures_on_date,
          SUM(CASE WHEN player1_id IS NULL OR player1_id = player1_api_id THEN 1 ELSE 0 END) AS bad_p1,
          SUM(CASE WHEN player2_id IS NULL OR player2_id = player2_api_id THEN 1 ELSE 0 END) AS bad_p2
        FROM api_tennis_fixtures
        WHERE match_date = :d;
        """
    )

    async with engine.begin() as conn:
        before = (await conn.execute(count_sql, {"d": match_date})).mappings().one()
        result = await conn.execute(fix_sql, {"d": match_date})
        after = (await conn.execute(count_sql, {"d": match_date})).mappings().one()

    payload = {
        "match_date": str(match_date),
        "rows_updated": result.rowcount,
        "before": dict(before),
        "after": dict(after),
    }
    logger.info(
        "tennis_admin.fix_fixtures_mapping done match_date=%s rows_updated=%s",
        match_date.isoformat(),
        payload["rows_updated"],
    )
    return payload
