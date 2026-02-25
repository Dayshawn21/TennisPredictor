from __future__ import annotations

import argparse
import asyncio
import csv
import datetime as dt
from pathlib import Path
from typing import Any, Optional

from sqlalchemy import text

from app.db_session import engine


ENSURE_SQL = text(
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


UPSERT_SQL = text(
    """
    INSERT INTO tennis_player_surface_profile_stats (
      player_id, tour, window_name, surface,
      svc_matches, svc_aces_per_game, svc_dfs_per_game, svc_ace_to_df_ratio,
      svc_first_serve_pct, svc_first_serve_win_pct, svc_second_serve_win_pct, svc_service_pts_win_pct,
      svc_bp_save_pct, svc_hold_pct,
      ret_matches, ret_opp_aces_per_game, ret_opp_dfs_per_game, ret_opp_first_serve_pct,
      ret_first_return_win_pct, ret_second_return_win_pct, ret_return_pts_win_pct, ret_bp_win_pct, ret_opp_hold_pct,
      updated_at
    )
    VALUES (
      :player_id, :tour, :window_name, :surface,
      :svc_matches, :svc_aces_per_game, :svc_dfs_per_game, :svc_ace_to_df_ratio,
      :svc_first_serve_pct, :svc_first_serve_win_pct, :svc_second_serve_win_pct, :svc_service_pts_win_pct,
      :svc_bp_save_pct, :svc_hold_pct,
      :ret_matches, :ret_opp_aces_per_game, :ret_opp_dfs_per_game, :ret_opp_first_serve_pct,
      :ret_first_return_win_pct, :ret_second_return_win_pct, :ret_return_pts_win_pct, :ret_bp_win_pct, :ret_opp_hold_pct,
      NOW()
    )
    ON CONFLICT (player_id, tour, window_name, surface)
    DO UPDATE SET
      svc_matches = COALESCE(EXCLUDED.svc_matches, tennis_player_surface_profile_stats.svc_matches),
      svc_aces_per_game = COALESCE(EXCLUDED.svc_aces_per_game, tennis_player_surface_profile_stats.svc_aces_per_game),
      svc_dfs_per_game = COALESCE(EXCLUDED.svc_dfs_per_game, tennis_player_surface_profile_stats.svc_dfs_per_game),
      svc_ace_to_df_ratio = COALESCE(EXCLUDED.svc_ace_to_df_ratio, tennis_player_surface_profile_stats.svc_ace_to_df_ratio),
      svc_first_serve_pct = COALESCE(EXCLUDED.svc_first_serve_pct, tennis_player_surface_profile_stats.svc_first_serve_pct),
      svc_first_serve_win_pct = COALESCE(EXCLUDED.svc_first_serve_win_pct, tennis_player_surface_profile_stats.svc_first_serve_win_pct),
      svc_second_serve_win_pct = COALESCE(EXCLUDED.svc_second_serve_win_pct, tennis_player_surface_profile_stats.svc_second_serve_win_pct),
      svc_service_pts_win_pct = COALESCE(EXCLUDED.svc_service_pts_win_pct, tennis_player_surface_profile_stats.svc_service_pts_win_pct),
      svc_bp_save_pct = COALESCE(EXCLUDED.svc_bp_save_pct, tennis_player_surface_profile_stats.svc_bp_save_pct),
      svc_hold_pct = COALESCE(EXCLUDED.svc_hold_pct, tennis_player_surface_profile_stats.svc_hold_pct),
      ret_matches = COALESCE(EXCLUDED.ret_matches, tennis_player_surface_profile_stats.ret_matches),
      ret_opp_aces_per_game = COALESCE(EXCLUDED.ret_opp_aces_per_game, tennis_player_surface_profile_stats.ret_opp_aces_per_game),
      ret_opp_dfs_per_game = COALESCE(EXCLUDED.ret_opp_dfs_per_game, tennis_player_surface_profile_stats.ret_opp_dfs_per_game),
      ret_opp_first_serve_pct = COALESCE(EXCLUDED.ret_opp_first_serve_pct, tennis_player_surface_profile_stats.ret_opp_first_serve_pct),
      ret_first_return_win_pct = COALESCE(EXCLUDED.ret_first_return_win_pct, tennis_player_surface_profile_stats.ret_first_return_win_pct),
      ret_second_return_win_pct = COALESCE(EXCLUDED.ret_second_return_win_pct, tennis_player_surface_profile_stats.ret_second_return_win_pct),
      ret_return_pts_win_pct = COALESCE(EXCLUDED.ret_return_pts_win_pct, tennis_player_surface_profile_stats.ret_return_pts_win_pct),
      ret_bp_win_pct = COALESCE(EXCLUDED.ret_bp_win_pct, tennis_player_surface_profile_stats.ret_bp_win_pct),
      ret_opp_hold_pct = COALESCE(EXCLUDED.ret_opp_hold_pct, tennis_player_surface_profile_stats.ret_opp_hold_pct),
      updated_at = NOW()
    """
)


FIELDS = [
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


def _to_float(v: Any) -> Optional[float]:
    if v is None:
        return None
    s = str(v).strip()
    if not s:
        return None
    try:
        return float(s)
    except Exception:
        return None


async def import_csv(path: Path, window_name: str) -> int:
    with path.open("r", encoding="utf-8", newline="") as f:
        rows = list(csv.DictReader(f))

    upserts = 0
    async with engine.begin() as conn:
        await conn.execute(ENSURE_SQL)
        for r in rows:
            p = str(r.get("player") or "").strip()
            if not p.isdigit():
                continue
            tour = str(r.get("tour") or "").strip().upper()
            if tour not in {"ATP", "WTA"}:
                continue
            surface = str(r.get("surface") or "").strip().lower()
            if not surface:
                continue

            params = {
                "player_id": int(p),
                "tour": tour,
                "window_name": (str(r.get("window") or "").strip() or window_name),
                "surface": surface,
            }
            for f_name in FIELDS:
                params[f_name] = _to_float(r.get(f_name))

            await conn.execute(UPSERT_SQL, params)
            upserts += 1

    return upserts


def main() -> None:
    ap = argparse.ArgumentParser(description="Import player surface profile CSV into DB override table")
    ap.add_argument("--input", default="app/data/player_surface_profile_stats.csv")
    ap.add_argument("--window-name", default="")
    args = ap.parse_args()

    in_path = Path(args.input)
    if not in_path.exists():
        print(f"input_missing={in_path}")
        return

    window_name = (args.window_name or "").strip()
    if not window_name:
        window_name = f"{dt.date.today().year} ytd"

    count = asyncio.run(import_csv(in_path, window_name))
    print(f"upserts={count} input={in_path}")


if __name__ == "__main__":
    main()
