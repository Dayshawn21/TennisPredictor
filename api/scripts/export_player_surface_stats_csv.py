from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
from pathlib import Path

from sqlalchemy import text

from app.db_session import engine


HEADERS = [
    "player",
    "ta_id",
    "player_name",
    "svc_matches",
    "svc_aces_per_game",
    "svc_dfs_per_game",
    "svc_ace_to_df_ratio",
    "svc_first_serve_pct",
    "svc_first_serve_win_pct",
    "svc_second_serve_win_pct",
    "svc_service_pts_win_pct",
    "svc_bp_save_pct",
    "svc_hold_pct",
    "tour",
    "window",
    "surface",
    "ret_matches",
    "ret_opp_aces_per_game",
    "ret_opp_dfs_per_game",
    "ret_opp_first_serve_pct",
    "ret_first_return_win_pct",
    "ret_second_return_win_pct",
    "ret_return_pts_win_pct",
    "ret_bp_win_pct",
    "ret_opp_hold_pct",
]

ENSURE_PROFILE_TABLE_SQL = text(
    """
    CREATE TABLE IF NOT EXISTS tennis_player_surface_profile_stats (
      player_id BIGINT NOT NULL,
      tour TEXT NOT NULL,
      window_name TEXT NOT NULL,
      surface TEXT NOT NULL,
      svc_matches DOUBLE PRECISION NULL,
      svc_aces_per_game DOUBLE PRECISION NULL,
      svc_dfs_per_game DOUBLE PRECISION NULL,
      svc_ace_to_df_ratio DOUBLE PRECISION NULL,
      svc_first_serve_pct DOUBLE PRECISION NULL,
      svc_first_serve_win_pct DOUBLE PRECISION NULL,
      svc_second_serve_win_pct DOUBLE PRECISION NULL,
      svc_service_pts_win_pct DOUBLE PRECISION NULL,
      svc_bp_save_pct DOUBLE PRECISION NULL,
      svc_hold_pct DOUBLE PRECISION NULL,
      ret_matches DOUBLE PRECISION NULL,
      ret_opp_aces_per_game DOUBLE PRECISION NULL,
      ret_opp_dfs_per_game DOUBLE PRECISION NULL,
      ret_opp_first_serve_pct DOUBLE PRECISION NULL,
      ret_first_return_win_pct DOUBLE PRECISION NULL,
      ret_second_return_win_pct DOUBLE PRECISION NULL,
      ret_return_pts_win_pct DOUBLE PRECISION NULL,
      ret_bp_win_pct DOUBLE PRECISION NULL,
      ret_opp_hold_pct DOUBLE PRECISION NULL,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW(),
      PRIMARY KEY (player_id, tour, window_name, surface)
    )
    """
)


EXPORT_SQL = text(
    """
    WITH base AS (
      SELECT
        m.sofascore_event_id,
        upper(m.tour) AS tour,
        CASE
          WHEN lower(coalesce(m.surface,'')) LIKE '%clay%' THEN 'clay'
          WHEN lower(coalesce(m.surface,'')) LIKE '%grass%' THEN 'grass'
          WHEN lower(coalesce(m.surface,'')) LIKE '%hard%' THEN 'hard'
          WHEN lower(coalesce(m.surface,'')) LIKE '%carpet%' THEN 'carpet'
          ELSE 'hard'
        END AS surface,
        CASE
          WHEN s.player_id = m.p1_sofascore_player_id THEN m.p1_canonical_id
          WHEN s.player_id = m.p2_sofascore_player_id THEN m.p2_canonical_id
          ELSE NULL
        END AS player_id,
        CASE
          WHEN s.player_id = m.p1_sofascore_player_id THEN m.p2_canonical_id
          WHEN s.player_id = m.p2_sofascore_player_id THEN m.p1_canonical_id
          ELSE NULL
        END AS opp_player_id,
        s.ace, s.df, s.svpt, s.first_in, s.first_won, s.second_won, s.sv_gms, s.bp_saved, s.bp_faced
      FROM tennis_matches m
      JOIN tennis_match_player_stats s
        ON s.sofascore_event_id = m.sofascore_event_id
      WHERE m.match_date >= :start_date
        AND upper(m.tour) IN ('ATP','WTA')
        AND coalesce(lower(m.status),'') IN ('finished','completed','ended','final')
        AND m.p1_canonical_id IS NOT NULL
        AND m.p2_canonical_id IS NOT NULL
    ),
    joined AS (
      SELECT
        b.*,
        o.ace AS opp_ace,
        o.df AS opp_df,
        o.svpt AS opp_svpt,
        o.first_in AS opp_first_in,
        o.first_won AS opp_first_won,
        o.second_won AS opp_second_won,
        o.sv_gms AS opp_sv_gms,
        o.bp_saved AS opp_bp_saved,
        o.bp_faced AS opp_bp_faced
      FROM base b
      JOIN base o
        ON o.sofascore_event_id = b.sofascore_event_id
       AND o.player_id = b.opp_player_id
    ),
    agg AS (
      SELECT
        j.player_id,
        j.tour,
        CAST(:window_name AS text) AS window,
        j.surface,
        count(*)::float AS svc_matches,
        sum(coalesce(j.ace,0))::float / nullif(sum(coalesce(j.sv_gms,0)),0) AS svc_aces_per_game,
        sum(coalesce(j.df,0))::float / nullif(sum(coalesce(j.sv_gms,0)),0) AS svc_dfs_per_game,
        sum(coalesce(j.ace,0))::float / nullif(sum(coalesce(j.df,0)),0) AS svc_ace_to_df_ratio,
        sum(coalesce(j.first_in,0))::float / nullif(sum(coalesce(j.svpt,0)),0) AS svc_first_serve_pct,
        sum(coalesce(j.first_won,0))::float / nullif(sum(coalesce(j.first_in,0)),0) AS svc_first_serve_win_pct,
        sum(coalesce(j.second_won,0))::float / nullif(sum(coalesce(j.svpt,0)-coalesce(j.first_in,0)),0) AS svc_second_serve_win_pct,
        sum(coalesce(j.first_won,0)+coalesce(j.second_won,0))::float / nullif(sum(coalesce(j.svpt,0)),0) AS svc_service_pts_win_pct,
        sum(coalesce(j.bp_saved,0))::float / nullif(sum(coalesce(j.bp_faced,0)),0) AS svc_bp_save_pct,
        (sum(coalesce(j.sv_gms,0))::float - sum(greatest(coalesce(j.bp_faced,0)-coalesce(j.bp_saved,0),0))::float)
          / nullif(sum(coalesce(j.sv_gms,0)),0) AS svc_hold_pct,
        count(*)::float AS ret_matches,
        sum(coalesce(j.opp_ace,0))::float / nullif(sum(coalesce(j.opp_sv_gms,0)),0) AS ret_opp_aces_per_game,
        sum(coalesce(j.opp_df,0))::float / nullif(sum(coalesce(j.opp_sv_gms,0)),0) AS ret_opp_dfs_per_game,
        sum(coalesce(j.opp_first_in,0))::float / nullif(sum(coalesce(j.opp_svpt,0)),0) AS ret_opp_first_serve_pct,
        1.0 - (sum(coalesce(j.opp_first_won,0))::float / nullif(sum(coalesce(j.opp_first_in,0)),0)) AS ret_first_return_win_pct,
        1.0 - (sum(coalesce(j.opp_second_won,0))::float / nullif(sum(coalesce(j.opp_svpt,0)-coalesce(j.opp_first_in,0)),0)) AS ret_second_return_win_pct,
        1.0 - ((sum(coalesce(j.opp_first_won,0)+coalesce(j.opp_second_won,0))::float) / nullif(sum(coalesce(j.opp_svpt,0)),0)) AS ret_return_pts_win_pct,
        sum(greatest(coalesce(j.opp_bp_faced,0)-coalesce(j.opp_bp_saved,0),0))::float / nullif(sum(coalesce(j.opp_bp_faced,0)),0) AS ret_bp_win_pct,
        (sum(coalesce(j.opp_sv_gms,0))::float - sum(greatest(coalesce(j.opp_bp_faced,0)-coalesce(j.opp_bp_saved,0),0))::float)
          / nullif(sum(coalesce(j.opp_sv_gms,0)),0) AS ret_opp_hold_pct
      FROM joined j
      GROUP BY j.player_id, j.tour, j.surface
    ),
    profile AS (
      SELECT
        p.player_id,
        upper(p.tour) AS tour,
        p.window_name AS window,
        lower(p.surface) AS surface,
        p.svc_matches, p.svc_aces_per_game, p.svc_dfs_per_game, p.svc_ace_to_df_ratio,
        p.svc_first_serve_pct, p.svc_first_serve_win_pct, p.svc_second_serve_win_pct,
        p.svc_service_pts_win_pct, p.svc_bp_save_pct, p.svc_hold_pct,
        p.ret_matches, p.ret_opp_aces_per_game, p.ret_opp_dfs_per_game, p.ret_opp_first_serve_pct,
        p.ret_first_return_win_pct, p.ret_second_return_win_pct, p.ret_return_pts_win_pct,
        p.ret_bp_win_pct, p.ret_opp_hold_pct
      FROM tennis_player_surface_profile_stats p
      WHERE p.window_name = :window_name
        AND upper(p.tour) IN ('ATP','WTA')
    ),
    merged AS (
      SELECT
        coalesce(p.player_id, a.player_id) AS player_id,
        coalesce(p.tour, a.tour) AS tour,
        coalesce(p.window, a.window) AS window,
        coalesce(p.surface, a.surface) AS surface,
        coalesce(p.svc_matches, a.svc_matches) AS svc_matches,
        coalesce(p.svc_aces_per_game, a.svc_aces_per_game) AS svc_aces_per_game,
        coalesce(p.svc_dfs_per_game, a.svc_dfs_per_game) AS svc_dfs_per_game,
        coalesce(p.svc_ace_to_df_ratio, a.svc_ace_to_df_ratio) AS svc_ace_to_df_ratio,
        coalesce(p.svc_first_serve_pct, a.svc_first_serve_pct) AS svc_first_serve_pct,
        coalesce(p.svc_first_serve_win_pct, a.svc_first_serve_win_pct) AS svc_first_serve_win_pct,
        coalesce(p.svc_second_serve_win_pct, a.svc_second_serve_win_pct) AS svc_second_serve_win_pct,
        coalesce(p.svc_service_pts_win_pct, a.svc_service_pts_win_pct) AS svc_service_pts_win_pct,
        coalesce(p.svc_bp_save_pct, a.svc_bp_save_pct) AS svc_bp_save_pct,
        coalesce(p.svc_hold_pct, a.svc_hold_pct) AS svc_hold_pct,
        coalesce(p.ret_matches, a.ret_matches) AS ret_matches,
        coalesce(p.ret_opp_aces_per_game, a.ret_opp_aces_per_game) AS ret_opp_aces_per_game,
        coalesce(p.ret_opp_dfs_per_game, a.ret_opp_dfs_per_game) AS ret_opp_dfs_per_game,
        coalesce(p.ret_opp_first_serve_pct, a.ret_opp_first_serve_pct) AS ret_opp_first_serve_pct,
        coalesce(p.ret_first_return_win_pct, a.ret_first_return_win_pct) AS ret_first_return_win_pct,
        coalesce(p.ret_second_return_win_pct, a.ret_second_return_win_pct) AS ret_second_return_win_pct,
        coalesce(p.ret_return_pts_win_pct, a.ret_return_pts_win_pct) AS ret_return_pts_win_pct,
        coalesce(p.ret_bp_win_pct, a.ret_bp_win_pct) AS ret_bp_win_pct,
        coalesce(p.ret_opp_hold_pct, a.ret_opp_hold_pct) AS ret_opp_hold_pct
      FROM agg a
      FULL OUTER JOIN profile p
        ON p.player_id = a.player_id
       AND p.tour = a.tour
       AND p.window = a.window
       AND p.surface = a.surface
    )
    , ta_map AS (
      SELECT
        tps.player_id,
        CASE
          WHEN tps.source = 'tennisabstract_elo_wta' THEN 'WTA'
          ELSE 'ATP'
        END AS tour,
        CASE
          WHEN NULLIF(btrim(tps.source_player_id), '') ~ '^[0-9]+$'
          THEN NULLIF(btrim(tps.source_player_id), '')::bigint
          ELSE NULL
        END AS ta_id
      FROM tennis_player_sources tps
      WHERE tps.source IN ('tennisabstract_elo_atp', 'tennisabstract_elo_wta')
    )
    SELECT
      m.player_id::text AS player,
      tm.ta_id::text AS ta_id,
      tp.name AS player_name,
      m.svc_matches, m.svc_aces_per_game, m.svc_dfs_per_game, m.svc_ace_to_df_ratio,
      m.svc_first_serve_pct, m.svc_first_serve_win_pct, m.svc_second_serve_win_pct,
      m.svc_service_pts_win_pct, m.svc_bp_save_pct, m.svc_hold_pct,
      m.tour, m.window, m.surface,
      m.ret_matches, m.ret_opp_aces_per_game, m.ret_opp_dfs_per_game, m.ret_opp_first_serve_pct,
      m.ret_first_return_win_pct, m.ret_second_return_win_pct, m.ret_return_pts_win_pct,
      m.ret_bp_win_pct, m.ret_opp_hold_pct
    FROM merged m
    LEFT JOIN ta_map tm
      ON tm.player_id = m.player_id
     AND tm.tour = m.tour
    LEFT JOIN tennis_players tp
      ON tp.id = m.player_id
    WHERE :min_matches <= 1 OR (coalesce(m.svc_matches, 0) >= :min_matches AND coalesce(m.ret_matches, 0) >= :min_matches)
    ORDER BY m.tour, m.surface, player
    """
)


def _default_start_date(today: dt.date) -> dt.date:
    return dt.date(today.year, 1, 1)


async def export_csv(start_date: dt.date, min_matches: int, window_name: str, out_path: Path) -> int:
    async with engine.begin() as conn:
        await conn.execute(ENSURE_PROFILE_TABLE_SQL)
        res = await conn.execute(
            EXPORT_SQL,
            {
                "start_date": start_date,
                "min_matches": int(min_matches),
                "window_name": window_name,
            },
        )
        rows = [dict(r._mapping) for r in res.fetchall()]

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(HEADERS)
        for r in rows:
            w.writerow([(r.get(h) if r.get(h) is not None else "") for h in HEADERS])
    return len(rows)


def main() -> None:
    today = dt.date.today()
    p = argparse.ArgumentParser(description="Export player_surface_stats.csv from DB")
    p.add_argument("--start-date", default=_default_start_date(today).isoformat(), help="YYYY-MM-DD")
    p.add_argument("--window-name", default="12 month")
    p.add_argument("--min-matches", type=int, default=1)
    p.add_argument("--output", default="app/data/player_surface_stats.csv")
    args = p.parse_args()

    start_date = dt.date.fromisoformat(args.start_date)
    out_path = Path(args.output)
    count = asyncio.run(
        export_csv(
            start_date=start_date,
            min_matches=int(args.min_matches),
            window_name=str(args.window_name),
            out_path=out_path,
        )
    )
    print(f"wrote_rows={count} output={out_path}")


if __name__ == "__main__":
    main()
