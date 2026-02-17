# FILE: api/app/services/tennis/calculate_player_rolling_stats.py
"""
Calculate rolling stats for all active players based on recent match history.
This allows the XGBoost simple model to make predictions on current matches.
"""

from __future__ import annotations
from dotenv import load_dotenv
load_dotenv()

import os
import asyncio
import logging
import re
from typing import Dict, Optional
from datetime import date, timedelta
import asyncpg

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

DATABASE_URL = os.getenv("DATABASE_URL")
if DATABASE_URL and DATABASE_URL.startswith("postgresql+asyncpg://"):
    DATABASE_URL = DATABASE_URL.replace("postgresql+asyncpg://", "postgresql://")


def normalize_name(name: Optional[str]) -> Optional[str]:
    if not name:
        return None
    # Trim + collapse whitespace
    return re.sub(r"\s+", " ", name.strip())


def norm_surface(surface: Optional[str]) -> Optional[str]:
    """
    Normalize any surface string to one of: Hard / Clay / Grass / None
    """
    if not surface:
        return None
    s = surface.strip().lower()
    if "hard" in s:
        return "Hard"
    if "clay" in s:
        return "Clay"
    if "grass" in s:
        return "Grass"
    return None


def parse_winner_from_score(score: str) -> bool | None:
    """Parse score to determine if P1 won."""
    if not score:
        return None

    try:
        sets = score.strip().split()
        p1_sets_won = 0
        p2_sets_won = 0

        for set_score in sets:
            if "-" not in set_score:
                continue
            set_score = set_score.split("(")[0]  # drop TB like 7-6(5)
            parts = set_score.split("-")
            if len(parts) != 2:
                continue

            p1_games = int(parts[0])
            p2_games = int(parts[1])

            if p1_games > p2_games:
                p1_sets_won += 1
            elif p2_games > p1_games:
                p2_sets_won += 1

        if p1_sets_won > p2_sets_won:
            return True
        if p2_sets_won > p1_sets_won:
            return False
        return None
    except (ValueError, IndexError):
        return None


async def get_recent_matches(conn) -> list[dict]:
    """Get all finished matches from last 12 months."""
    cutoff_date = date.today() - timedelta(days=365)

    query = """
    SELECT
        m.match_date,
        UPPER(TRIM(m.tour)) AS tour,
        m.surface,
        m.p1_name,
        m.p2_name,
        COALESCE(NULLIF(TRIM(m.score), ''), NULLIF(TRIM(m.score_raw), '')) AS score
    FROM tennis_matches m
    WHERE COALESCE(LOWER(m.status), '') IN ('finished', 'completed', 'ended')
      AND m.match_date >= $1
      AND UPPER(TRIM(m.tour)) IN ('ATP', 'WTA')
      AND COALESCE(NULLIF(TRIM(m.score), ''), NULLIF(TRIM(m.score_raw), '')) IS NOT NULL
    ORDER BY m.match_date
    """

    rows = await conn.fetch(query, cutoff_date)

    atp_count = sum(1 for r in rows if (r["tour"] or "").upper() == "ATP")
    wta_count = sum(1 for r in rows if (r["tour"] or "").upper() == "WTA")

    logger.info("Fetched %d recent matches since %s", len(rows), cutoff_date)
    logger.info("  ATP: %d matches", atp_count)
    logger.info("  WTA: %d matches", wta_count)

    return [dict(r) for r in rows]


def calculate_rolling_stats(matches: list[dict]) -> Dict[str, Dict]:
    """Calculate rolling stats for each player."""
    logger.info("Calculating rolling stats for all players...")

    player_stats: Dict[str, Dict] = {}
    skipped_count = 0

    for match in matches:
        p1_won = parse_winner_from_score(match["score"])
        if p1_won is None:
            skipped_count += 1
            continue

        p1_name = normalize_name(match["p1_name"])
        p2_name = normalize_name(match["p2_name"])
        tour = (match.get("tour") or "").upper().strip()
        surface = norm_surface(match.get("surface"))

        if not p1_name or not p2_name:
            skipped_count += 1
            continue

        # Initialize players
        for player in (p1_name, p2_name):
            if player not in player_stats:
                player_stats[player] = {
                    "tour_counts": {"ATP": 0, "WTA": 0},
                    "all_matches": [],
                    "surface_matches": {"Hard": [], "Clay": [], "Grass": []},
                }

        if tour in ("ATP", "WTA"):
            player_stats[p1_name]["tour_counts"][tour] += 1
            player_stats[p2_name]["tour_counts"][tour] += 1

        # Record overall results
        player_stats[p1_name]["all_matches"].append(1 if p1_won else 0)
        player_stats[p2_name]["all_matches"].append(0 if p1_won else 1)

        # Record surface results
        if surface in ("Hard", "Clay", "Grass"):
            player_stats[p1_name]["surface_matches"][surface].append(1 if p1_won else 0)
            player_stats[p2_name]["surface_matches"][surface].append(0 if p1_won else 1)

    final_stats: Dict[str, Dict] = {}

    for player, data in player_stats.items():
        recent_20 = data["all_matches"][-20:]

        # pick predominant tour for reference (optional)
        tc = data["tour_counts"]
        player_tour = "ATP" if tc["ATP"] >= tc["WTA"] else "WTA"

        final_stats[player] = {
            "tour": player_tour,
            "win_rate_last_20": (sum(recent_20) / len(recent_20)) if recent_20 else 0.5,
            "matches_played": len(data["all_matches"]),
            "surface_stats": {},
        }

        for surf in ("Hard", "Clay", "Grass"):
            results = data["surface_matches"][surf]
            recent_10 = results[-10:]
            final_stats[player]["surface_stats"][surf] = {
                "win_rate_last_10": (sum(recent_10) / len(recent_10)) if recent_10 else 0.5
            }

    logger.info("Calculated stats for %d players", len(final_stats))
    logger.info("Skipped %d matches with unparseable/missing data", skipped_count)

    return final_stats


async def save_to_cache_table(conn, player_stats: Dict[str, Dict]) -> None:
    """Save player stats to a cache table for fast lookup."""
    await conn.execute("""
        CREATE TABLE IF NOT EXISTS tennis_player_rolling_stats (
            player_name TEXT PRIMARY KEY,
            tour TEXT,
            win_rate_last_20 NUMERIC,
            matches_played INT,
            hard_win_rate_last_10 NUMERIC,
            clay_win_rate_last_10 NUMERIC,
            grass_win_rate_last_10 NUMERIC,
            updated_at TIMESTAMPTZ DEFAULT NOW()
        )
    """)

    await conn.execute("TRUNCATE TABLE tennis_player_rolling_stats")

    for player, stats in player_stats.items():
        surface_stats = stats.get("surface_stats", {})

        await conn.execute(
            """
            INSERT INTO tennis_player_rolling_stats
            (player_name, tour, win_rate_last_20, matches_played,
             hard_win_rate_last_10, clay_win_rate_last_10, grass_win_rate_last_10)
            VALUES ($1, $2, $3, $4, $5, $6, $7)
            """,
            player,
            stats.get("tour"),
            stats["win_rate_last_20"],
            stats["matches_played"],
            surface_stats.get("Hard", {}).get("win_rate_last_10"),
            surface_stats.get("Clay", {}).get("win_rate_last_10"),
            surface_stats.get("Grass", {}).get("win_rate_last_10"),
        )

    logger.info("✅ Saved %d player stats to database", len(player_stats))


async def main():
    conn = await asyncpg.connect(DATABASE_URL)
    try:
        matches = await get_recent_matches(conn)
        player_stats = calculate_rolling_stats(matches)
        await save_to_cache_table(conn, player_stats)
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(main())
