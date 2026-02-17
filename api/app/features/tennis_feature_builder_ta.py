from __future__ import annotations

from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
from collections import defaultdict
from dataclasses import dataclass
from datetime import date
from typing import List, Dict

import asyncpg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv(
    "DATABASE_URL",
    "postgresql://neondb_owner:npg_3bxYFijyoeD4@ep-dawn-wave-a8928yr9-pooler.eastus2.azure.neon.tech/neondb?sslmode=require",
)

if not DATABASE_URL:
    raise RuntimeError("DATABASE_URL is not set")


# -----------------------------
# Dataclass for one match row
# -----------------------------

@dataclass
class MatchRow:
    match_id: str
    match_date: date
    tour: str
    surface: str | None
    winner: str
    loser: str
    score: str | None

    # winner stats
    w_svpt: int | None
    w_1stin: int | None
    w_1stwon: int | None
    w_2ndwon: int | None
    w_svgms: int | None
    w_bpsaved: int | None
    w_bpfaced: int | None
    w_ace: int | None
    w_df: int | None

    # loser stats
    l_svpt: int | None
    l_1stin: int | None
    l_1stwon: int | None
    l_2ndwon: int | None
    l_svgms: int | None
    l_bpsaved: int | None
    l_bpfaced: int | None
    l_ace: int | None
    l_df: int | None


# -----------------------------
# Helpers
# -----------------------------

def pct(numer: int | None, denom: int | None) -> float | None:
    if numer is None or denom in (None, 0):
        return None
    return 100.0 * numer / denom


async def load_all_matches(conn: asyncpg.Connection) -> List[MatchRow]:
    """
    Load all finished matches that have TA stats attached.
    We alias column names so they match the MatchRow fields exactly.
    """
    rows = await conn.fetch(
        """
        SELECT
            m.match_id,
            m.match_date,
            m.tour,
            m.surface,
            m.p1_name AS winner,
            m.p2_name AS loser,
            m.score   AS score,

            -- Winner stats
            s.w_svpt                       AS w_svpt,
            s.w_1stIn                      AS w_1stin,
            s.w_1stWon                     AS w_1stwon,
            s.w_2ndWon                     AS w_2ndwon,
            s.w_SvGms                      AS w_svgms,
            s.w_bpSaved                    AS w_bpsaved,
            s.w_bpFaced                    AS w_bpfaced,
            s.w_ace                        AS w_ace,
            s.w_df                         AS w_df,

            -- Loser stats
            s.l_svpt                       AS l_svpt,
            s.l_1stIn                      AS l_1stin,
            s.l_1stWon                     AS l_1stwon,
            s.l_2ndWon                     AS l_2ndwon,
            s.l_SvGms                      AS l_svgms,
            s.l_bpSaved                    AS l_bpsaved,
            s.l_bpFaced                    AS l_bpfaced,
            s.l_ace                        AS l_ace,
            s.l_df                         AS l_df

        FROM tennis_matches m
        JOIN tennis_match_stats_ta s
          ON s.match_id = m.match_id
        WHERE m.status = 'finished'
        ORDER BY m.match_date
        """
    )

    return [MatchRow(**dict(r)) for r in rows]


def compute_hold_and_break_for_player(match: MatchRow, player: str) -> tuple[float | None, float | None]:
    """
    For a given match and player, compute:
      - hold_pct: % of service games held
      - break_pct: % of opponent service games broken
    Based on bp_faced / bp_saved.
    """
    if player == match.winner:
        my_svgms = match.w_svgms or 0
        my_bpfaced = match.w_bpfaced or 0
        my_bpsaved = match.w_bpsaved or 0

        opp_svgms = match.l_svgms or 0
        opp_bpfaced = match.l_bpfaced or 0
        opp_bpsaved = match.l_bpsaved or 0
    else:
        my_svgms = match.l_svgms or 0
        my_bpfaced = match.l_bpfaced or 0
        my_bpsaved = match.l_bpsaved or 0

        opp_svgms = match.w_svgms or 0
        opp_bpfaced = match.w_bpfaced or 0
        opp_bpsaved = match.w_bpsaved or 0

    hold_pct = None
    if my_svgms:
        my_breaks_conceded = my_bpfaced - my_bpsaved
        my_holds = my_svgms - my_breaks_conceded
        hold_pct = pct(my_holds, my_svgms)

    break_pct = None
    if opp_svgms:
        opp_breaks_conceded = opp_bpfaced - opp_bpsaved
        break_pct = pct(opp_breaks_conceded, opp_svgms)

    return hold_pct, break_pct


def compute_aces_per_svc_game(match: MatchRow, player: str) -> float | None:
    """
    Aces per service game for this match.
    """
    if player == match.winner:
        svgms = match.w_svgms or 0
        aces = match.w_ace or 0
    else:
        svgms = match.l_svgms or 0
        aces = match.l_ace or 0

    if svgms == 0:
        return None
    return float(aces) / float(svgms)


def compute_df_per_svc_game(match: MatchRow, player: str) -> float | None:
    """
    Double-faults per service game for this match.
    """
    if player == match.winner:
        svgms = match.w_svgms or 0
        dfs = match.w_df or 0
    else:
        svgms = match.l_svgms or 0
        dfs = match.l_df or 0

    if svgms == 0:
        return None
    return float(dfs) / float(svgms)


def tiebreak_stats_for_player_in_match(match: MatchRow, player: str) -> tuple[int, int, int]:
    """
    For one match and a player, return:
      - had_tb_match (0/1): did this match contain at least one tiebreak?
      - tb_sets_played: number of tiebreak sets this player participated in
      - tb_sets_won: how many of those tiebreak sets this player won

    Tennis Abstract scores are from winner's perspective, e.g.:
      "7-6(5) 6-4"
      "6-7(4) 7-6(3) 7-5"
    """
    if not match.score:
        return 0, 0, 0

    score_str = match.score.strip()
    if not score_str:
        return 0, 0, 0

    had_tb_match = 0
    tb_played = 0
    tb_won = 0

    is_player_winner = (player == match.winner)

    # Split into set tokens by space
    for token in score_str.split():
        # Ignore weird tokens like "RET", "W/O"
        if token.upper() in ("RET", "W/O", "DEF"):
            continue

        base = token.split("(")[0]  # "7-6(5)" -> "7-6"
        base = base.strip()

        if base.startswith("7-6") or base.startswith("6-7"):
            had_tb_match = 1
            tb_played += 1

            # From winner perspective:
            #  - "7-6"  => winner of match won this TB
            #  - "6-7"  => winner of match lost this TB
            if base.startswith("7-6"):
                if is_player_winner:
                    tb_won += 1
            else:  # "6-7"
                if not is_player_winner:
                    tb_won += 1

    return had_tb_match, tb_played, tb_won


# -----------------------------
# Main feature builder
# -----------------------------

async def build_features() -> None:
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        matches = await load_all_matches(conn)
        logger.info("Loaded %d matches with stats", len(matches))

        # History of matches per player (in date order)
        history: Dict[str, List[MatchRow]] = defaultdict(list)
        inserted = 0

        for match in matches:
            p1 = match.winner  # by our ingest convention
            p2 = match.loser

            # ---- Helper: hold/break rolling ----

            def last_n_hold_break(player: str, n: int, surface: str | None = None) -> tuple[float | None, float | None]:
                rows = [
                    m for m in history[player]
                    if surface is None or m.surface == surface
                ]
                rows = rows[-n:]
                if not rows:
                    return None, None

                hold_vals: List[float] = []
                break_vals: List[float] = []

                for m in rows:
                    h, b = compute_hold_and_break_for_player(m, player)
                    if h is not None:
                        hold_vals.append(h)
                    if b is not None:
                        break_vals.append(b)

                hold_avg = sum(hold_vals) / len(hold_vals) if hold_vals else None
                break_avg = sum(break_vals) / len(break_vals) if break_vals else None
                return hold_avg, break_avg

            # ---- Helper: aces per service game rolling ----

            def last_n_aces_per_svc_game(player: str, n: int, surface: str | None = None) -> float | None:
                rows = [
                    m for m in history[player]
                    if surface is None or m.surface == surface
                ]
                rows = rows[-n:]
                if not rows:
                    return None

                vals: List[float] = []
                for m in rows:
                    v = compute_aces_per_svc_game(m, player)
                    if v is not None:
                        vals.append(v)

                if not vals:
                    return None
                return sum(vals) / len(vals)

            # ---- Helper: double-faults per service game rolling ----

            def last_n_df_per_svc_game(player: str, n: int, surface: str | None = None) -> float | None:
                rows = [
                    m for m in history[player]
                    if surface is None or m.surface == surface
                ]
                rows = rows[-n:]
                if not rows:
                    return None

                vals: List[float] = []
                for m in rows:
                    v = compute_df_per_svc_game(m, player)
                    if v is not None:
                        vals.append(v)

                if not vals:
                    return None
                return sum(vals) / len(vals)

            # ---- Helper: tiebreak rolling ----

            def last_n_tiebreak_stats(player: str, n: int, surface: str | None = None) -> tuple[float | None, float | None]:
                """
                Returns:
                  - tb_match_rate: % of matches (0-100) that had at least one tiebreak
                  - tb_win_pct: % of tiebreak sets won (0-100)
                """
                rows = [
                    m for m in history[player]
                    if surface is None or m.surface == surface
                ]
                rows = rows[-n:]
                if not rows:
                    return None, None

                matches_total = 0
                matches_with_tb = 0
                tb_sets_played = 0
                tb_sets_won = 0

                for m in rows:
                    had_tb, tb_played, tb_won = tiebreak_stats_for_player_in_match(m, player)
                    matches_total += 1
                    if had_tb:
                        matches_with_tb += 1
                    tb_sets_played += tb_played
                    tb_sets_won += tb_won

                tb_match_rate = pct(matches_with_tb, matches_total)
                tb_win_pct = pct(tb_sets_won, tb_sets_played) if tb_sets_played > 0 else None
                return tb_match_rate, tb_win_pct

            # --- Hold / break features ---

            p1_last5_hold, p1_last5_break = last_n_hold_break(p1, 5)
            p1_last10_hold, p1_last10_break = last_n_hold_break(p1, 10)

            p2_last5_hold, p2_last5_break = last_n_hold_break(p2, 5)
            p2_last10_hold, p2_last10_break = last_n_hold_break(p2, 10)

            p1_surf_last10_hold, p1_surf_last10_break = last_n_hold_break(p1, 10, match.surface)
            p2_surf_last10_hold, p2_surf_last10_break = last_n_hold_break(p2, 10, match.surface)

            # --- Aces per service game (overall + surface) ---

            p1_last10_aces_pg = last_n_aces_per_svc_game(p1, 10)
            p2_last10_aces_pg = last_n_aces_per_svc_game(p2, 10)

            p1_surf_last10_aces_pg = last_n_aces_per_svc_game(p1, 10, match.surface)
            p2_surf_last10_aces_pg = last_n_aces_per_svc_game(p2, 10, match.surface)

            # --- Double-faults per service game (overall + surface) ---

            p1_last10_df_pg = last_n_df_per_svc_game(p1, 10)
            p2_last10_df_pg = last_n_df_per_svc_game(p2, 10)

            p1_surf_last10_df_pg = last_n_df_per_svc_game(p1, 10, match.surface)
            p2_surf_last10_df_pg = last_n_df_per_svc_game(p2, 10, match.surface)

            # --- Tiebreak features (overall + surface) ---

            p1_last10_tb_match_rate, p1_last10_tb_win_pct = last_n_tiebreak_stats(p1, 10)
            p2_last10_tb_match_rate, p2_last10_tb_win_pct = last_n_tiebreak_stats(p2, 10)

            p1_surf_last10_tb_match_rate, p1_surf_last10_tb_win_pct = last_n_tiebreak_stats(p1, 10, match.surface)
            p2_surf_last10_tb_match_rate, p2_surf_last10_tb_win_pct = last_n_tiebreak_stats(p2, 10, match.surface)

            # Insert / update into tennis_features_ta
            await conn.execute(
                """
            INSERT INTO tennis_features_ta (
    match_id, match_date, tour, surface, p1_name, p2_name, p1_won,
    p1_last5_hold_pct, p1_last5_break_pct,
    p1_last10_hold_pct, p1_last10_break_pct,
    p2_last5_hold_pct, p2_last5_break_pct,
    p2_last10_hold_pct, p2_last10_break_pct,
    p1_surface_last10_hold_pct, p1_surface_last10_break_pct,
    p2_surface_last10_hold_pct, p2_surface_last10_break_pct,
    p1_last10_aces_per_svc_game,
    p2_last10_aces_per_svc_game,
    p1_surface_last10_aces_per_svc_game,
    p2_surface_last10_aces_per_svc_game,
    p1_last10_df_per_svc_game,
    p2_last10_df_per_svc_game,
    p1_surface_last10_df_per_svc_game,
    p2_surface_last10_df_per_svc_game,
    p1_last10_tb_match_rate,
    p1_last10_tb_win_pct,
    p2_last10_tb_match_rate,
    p2_last10_tb_win_pct,
    p1_surface_last10_tb_match_rate,
    p1_surface_last10_tb_win_pct,
    p2_surface_last10_tb_match_rate,
    p2_surface_last10_tb_win_pct
)
VALUES (
    $1,$2,$3,$4,$5,$6,$7,
    $8,$9,$10,$11,$12,$13,$14,$15,$16,$17,$18,$19,
    $20,$21,$22,$23,$24,$25,$26,$27,$28,$29,$30,$31,$32,$33,$34,$35
)
                ON CONFLICT (match_id)
                DO UPDATE SET
                    p1_last5_hold_pct                   = EXCLUDED.p1_last5_hold_pct,
                    p1_last5_break_pct                  = EXCLUDED.p1_last5_break_pct,
                    p1_last10_hold_pct                  = EXCLUDED.p1_last10_hold_pct,
                    p1_last10_break_pct                 = EXCLUDED.p1_last10_break_pct,
                    p2_last5_hold_pct                   = EXCLUDED.p2_last5_hold_pct,
                    p2_last5_break_pct                  = EXCLUDED.p2_last5_break_pct,
                    p2_last10_hold_pct                  = EXCLUDED.p2_last10_hold_pct,
                    p2_last10_break_pct                 = EXCLUDED.p2_last10_break_pct,
                    p1_surface_last10_hold_pct          = EXCLUDED.p1_surface_last10_hold_pct,
                    p1_surface_last10_break_pct         = EXCLUDED.p1_surface_last10_break_pct,
                    p2_surface_last10_hold_pct          = EXCLUDED.p2_surface_last10_hold_pct,
                    p2_surface_last10_break_pct         = EXCLUDED.p2_surface_last10_break_pct,
                    p1_last10_aces_per_svc_game         = EXCLUDED.p1_last10_aces_per_svc_game,
                    p2_last10_aces_per_svc_game         = EXCLUDED.p2_last10_aces_per_svc_game,
                    p1_surface_last10_aces_per_svc_game = EXCLUDED.p1_surface_last10_aces_per_svc_game,
                    p2_surface_last10_aces_per_svc_game = EXCLUDED.p2_surface_last10_aces_per_svc_game,
                    p1_last10_df_per_svc_game           = EXCLUDED.p1_last10_df_per_svc_game,
                    p2_last10_df_per_svc_game           = EXCLUDED.p2_last10_df_per_svc_game,
                    p1_surface_last10_df_per_svc_game   = EXCLUDED.p1_surface_last10_df_per_svc_game,
                    p2_surface_last10_df_per_svc_game   = EXCLUDED.p2_surface_last10_df_per_svc_game,
                    p1_last10_tb_match_rate             = EXCLUDED.p1_last10_tb_match_rate,
                    p1_last10_tb_win_pct                = EXCLUDED.p1_last10_tb_win_pct,
                    p2_last10_tb_match_rate             = EXCLUDED.p2_last10_tb_match_rate,
                    p2_last10_tb_win_pct                = EXCLUDED.p2_last10_tb_win_pct,
                    p1_surface_last10_tb_match_rate     = EXCLUDED.p1_surface_last10_tb_match_rate,
                    p1_surface_last10_tb_win_pct        = EXCLUDED.p1_surface_last10_tb_win_pct,
                    p2_surface_last10_tb_match_rate     = EXCLUDED.p2_surface_last10_tb_match_rate,
                    p2_surface_last10_tb_win_pct        = EXCLUDED.p2_surface_last10_tb_win_pct,
                    updated_at                          = now()
                """,
                match.match_id,
                match.match_date,
                match.tour,
                match.surface,
                p1,
                p2,
                True,  # p1_won (winner)

                p1_last5_hold,
                p1_last5_break,
                p1_last10_hold,
                p1_last10_break,

                p2_last5_hold,
                p2_last5_break,
                p2_last10_hold,
                p2_last10_break,

                p1_surf_last10_hold,
                p1_surf_last10_break,
                p2_surf_last10_hold,
                p2_surf_last10_break,

                p1_last10_aces_pg,
                p2_last10_aces_pg,
                p1_surf_last10_aces_pg,
                p2_surf_last10_aces_pg,

                p1_last10_df_pg,
                p2_last10_df_pg,
                p1_surf_last10_df_pg,
                p2_surf_last10_df_pg,

                p1_last10_tb_match_rate,
                p1_last10_tb_win_pct,
                p2_last10_tb_match_rate,
                p2_last10_tb_win_pct,
                p1_surf_last10_tb_match_rate,
                p1_surf_last10_tb_win_pct,
                p2_surf_last10_tb_match_rate,
                p2_surf_last10_tb_win_pct,
            )

            # Update histories AFTER using them
            history[p1].append(match)
            history[p2].append(match)

            inserted += 1
            if inserted % 5000 == 0:
                logger.info("Processed %d matches", inserted)

        logger.info("Feature build complete. Processed %d matches.", inserted)

    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(build_features())
